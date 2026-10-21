"""Offline ACT inference API; returns positions and never sends robot commands."""
from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from robot_bringup.act_data import ACTION_SOURCE
from robot_bringup.act_model import ACT, ACTConfig, CHECKPOINT_VERSION
from robot_bringup.demonstration_images import JOINT_NAMES


def image_tensor(image, image_size):
    """Same resize for training/inference. Array inputs must be uint8 RGB HWC."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected an H x W x 3 uint8 RGB image (convert OpenCV BGR first)')
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise ValueError('Expected a PIL image or uint8 RGB array')
    height, width = image_size
    image = image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.array(image, dtype=np.float32, copy=True) / 255.
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


class ACTPolicy:
    def __init__(self, checkpoint, device='cpu', precision='auto'):
        self.device = inference_device(device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if payload.get('format_version') != CHECKPOINT_VERSION or payload.get('action_source') != ACTION_SOURCE:
            raise ValueError('ACT checkpoint format 2 required; retrain with --preset medium')
        if payload['joint_names'] != JOINT_NAMES:
            raise ValueError('Checkpoint joint order differs from this robot')
        self.config = ACTConfig(**payload['model_config'])
        self.image_size = tuple(payload['image_size'])
        if (len(self.image_size) != 2 or
                any(not isinstance(size, int) or isinstance(size, bool) or size < 32 for size in self.image_size)):
            raise ValueError('Checkpoint image_size must contain integer height and width >= 32')
        self.fps = float(payload['fps'])
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Checkpoint fps must be positive and finite')
        self.control_dt = 1. / self.fps
        statistics = {}
        for name in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std'):
            if name not in payload['normalization']:
                raise ValueError('Checkpoint is missing normalization: ' + name)
            value = torch.as_tensor(payload['normalization'][name], dtype=torch.float32)
            if value.shape != (self.config.state_dim,) or not torch.isfinite(value).all():
                raise ValueError('Checkpoint normalization must contain seven finite values: ' + name)
            if name.endswith('_std') and torch.any(value <= 0):
                raise ValueError('Checkpoint normalization standard deviations must be positive: ' + name)
            statistics[name] = value.to(self.device)
        self.statistics = statistics
        self.precision = inference_precision(self.device, precision, not payload.get('run_config', {}).get('no_amp', False))
        self.amp_dtype = {'fp32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}[self.precision]
        self.model = ACT(self.config, pretrained=False)
        self.model.load_state_dict(payload['model'])
        self.model.to(self.device).eval().requires_grad_(False)
        self.training_step = payload.get('global_step')

    def describe(self):
        """JSON-compatible settings of the loaded snapshot, not live training state."""
        return dict(checkpoint=str(self.checkpoint), format_version=CHECKPOINT_VERSION,
                    model_config=asdict(self.config), model_parameters=sum(p.numel() for p in self.model.parameters()),
                    image_size=list(self.image_size), joint_names=list(JOINT_NAMES),
                    fps=self.fps, control_dt=self.control_dt, device=str(self.device), precision=self.precision,
                    latent='zero_prior', decoder_output='last_layer', normalization_source='checkpoint',
                    training_step=self.training_step, action_source=ACTION_SOURCE)

    def warmup(self):
        """Exercise the actual inference path before ROS starts; discard all actions."""
        started = time.monotonic()
        image = Image.new('RGB', (self.image_size[1], self.image_size[0]))
        self.predict(dict.fromkeys(self.config.cameras, image), self.statistics['qpos_mean'].cpu().numpy())
        return time.monotonic() - started

    @torch.inference_mode()
    def predict(self, images, qpos):
        """Return K x 7 absolute joint targets in radians, at dt, 2dt, ... Kdt.

        images: dict(camera name -> PIL RGB or uint8 RGB array).
        qpos: current positions in JOINT_NAMES order. No final pose argument.
        """
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise ValueError('qpos must contain seven finite current joint positions')
        missing = [name for name in self.config.cameras if name not in images]
        if missing:
            raise ValueError('Missing RGB views: ' + ', '.join(missing))
        pixels = torch.stack([image_tensor(images[name], self.image_size) for name in self.config.cameras])
        state = torch.from_numpy(qpos).to(self.device)
        state = (state - self.statistics['qpos_mean']) / self.statistics['qpos_std']
        # Autocast is thread-local: enter it HERE, inside the rollout's worker.
        # No actions/is_pad enter the model, so only the z=0 policy is evaluated.
        with (torch.autocast('cuda', dtype=self.amp_dtype) if self.amp_dtype is not None else nullcontext()):
            prediction, _, _ = self.model(state[None], pixels[None].to(self.device))
        if prediction.shape != (1, self.config.chunk_size, self.config.state_dim):
            raise RuntimeError('ACT returned an unexpected action shape')
        actions = prediction[0].float() * self.statistics['action_std'] + self.statistics['action_mean']
        if not torch.isfinite(actions).all():
            raise RuntimeError('Nonfinite action prediction')
        return actions.cpu().numpy()


def inference_device(requested):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if str(requested) == 'auto' else torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Inference device must be auto, cpu, cuda or cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; use --device cpu or a compatible GPU PyTorch build')
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index {index} is unavailable')
        device = torch.device('cuda', index)
    return device


def inference_precision(device, requested, training_amp=True):
    if requested not in ('auto', 'fp32', 'bf16', 'fp16'):
        raise ValueError('Inference precision must be auto, fp32, bf16 or fp16')
    if device.type == 'cpu':
        if requested not in ('auto', 'fp32'):
            raise ValueError('CPU inference uses fp32; bf16/fp16 options require CUDA')
        return 'fp32'
    if requested == 'auto':
        return ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if training_amp else 'fp32'
    if requested == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support bf16; use auto, fp16 or fp32')
    return requested
