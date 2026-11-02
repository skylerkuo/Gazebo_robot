"""Force-conditioned ACT: raw state(6) + wrist wrench(6), three RGBs, TCP actions(7)."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys

INSERT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(INSERT_ROOT / "tools/lerobot_reader_python"),
                str(INSERT_ROOT / "tools/dataset_python")]

import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

CAMERAS = tuple(f"observation.images.{camera}_camera" for camera in ("left", "center", "right"))
STATE = "observation.state"
WRENCH = "observation.environment_state"


@dataclass
class TrainConfig:
    dataset_root: str = "/mnt/d/robot_datasets/aic_cable_insertion_gazebo_v3_act"
    output_dir: str = "/mnt/d/robot_models/aic_ftact"
    seed: int = 42
    device: str = "cuda"
    amp: str = "bf16"
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    num_threads: int = 4
    steps: int = 50000
    log_every: int = 50
    eval_every: int = 1000
    eval_batches: int = 100  # 0 evaluates the complete validation split.
    eval_samples: int | None = None  # Fixed count across batch-size changes; 0 means full split.
    save_every: int = 1000
    learning_rate: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0
    chunk_size: int = 20
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    dropout: float = .1
    kl_weight: float = 10.0
    pretrained_backbone: bool = True

    def validate(self):
        for key in ("batch_size", "gradient_accumulation_steps", "steps", "log_every", "eval_every",
                    "save_every", "chunk_size", "num_threads"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.num_workers < 0 or self.eval_batches < 0:
            raise ValueError("num_workers and eval_batches must be nonnegative")
        if self.eval_samples is not None and self.eval_samples < 0:
            raise ValueError("eval_samples must be nonnegative")
        if self.amp not in ("bf16", "fp16", "none"):
            raise ValueError("amp must be bf16, fp16, or none")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for a CPU run")
        if self.device.startswith("cuda") and self.amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16; use --amp fp16 or --amp none")
        if self.device == "cpu" and self.amp != "none":
            raise ValueError("CPU runs require --amp none")


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configure_data_cache(root):
    """Set caches before importing the official reader; no dataset downloads are permitted."""
    preparation = Path(root) / "_preparation"
    os.environ["HF_HOME"] = str(preparation / "reader_hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(preparation / "reader_cache")
    os.environ["HF_LEROBOT_HOME"] = str(preparation / "reader_lerobot_home")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def read_data_contract(root):
    root = Path(root).resolve()
    preparation = root / "_preparation"
    for name in ("validation.json", "lerobot_reader_validation.json"):
        if not json.loads((preparation / name).read_text()).get("passed"):
            raise ValueError(f"Dataset has not passed {name}")
    info = json.loads((root / "meta/info.json").read_text())
    expected = {STATE: [6], WRENCH: [6], "action": [7],
                **{key: [256, 288, 3] for key in CAMERAS}}
    if info["fps"] != 20 or any(info["features"][key]["shape"] != shape for key, shape in expected.items()):
        raise ValueError("Expected the prepared 20 Hz AIC dataset, three 256x288 RGBs, state6/wrench6/action7")
    splits = json.loads((preparation / "splits.json").read_text())["splits"]
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for key in ("episode_ids", "source_episode_ids"):
            if set(splits[a][key]) & set(splits[b][key]):
                raise ValueError("Dataset split leakage")
    stats = json.loads((preparation / "train_stats.json").read_text())
    if stats != json.loads((root / "meta/stats.json").read_text()):
        raise ValueError("Canonical normalization statistics are not the verified training-only statistics")
    digest = hashlib.sha256()
    for path in (preparation / "settings.json", preparation / "splits.json",
                 preparation / "train_stats.json", root / "meta/info.json"):
        digest.update(path.read_bytes())
    settings = json.loads((preparation / "settings.json").read_text())
    return stats, splits, {"metadata_sha256": digest.hexdigest(), **settings}


class AICDataset(Dataset):
    """Use the official reader for RGB decoding and action windows/padding at episode boundaries."""
    def __init__(self, root, episodes, chunk_size):
        configure_data_cache(root)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.reader = LeRobotDataset("local/aic-cable-insertion-clean", root=Path(root),
            episodes=list(episodes), delta_timestamps={"action": [i / 20 for i in range(chunk_size)]},
            download_videos=False, video_backend="pyav", token=False)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        sample = self.reader[index]
        keys = (*CAMERAS, STATE, WRENCH, "action", "action_is_pad", "episode_index", "frame_index")
        return {key: sample[key] for key in keys}


class EpochBatchSampler(Sampler):
    """Epoch permutations with an explicit consumed-batch cursor, independent of worker prefetch."""
    def __init__(self, size, batch_size, seed, epoch=0, start_batch=0, start_sample=None):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.start_batch = epoch, start_batch
        self.start_sample = start_sample

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        offset = self.start_batch * self.batch_size if self.start_sample is None else self.start_sample
        for start in range(offset, self.size, self.batch_size):
            yield order[start:start + self.batch_size]

    def __len__(self):
        offset = self.start_batch * self.batch_size if self.start_sample is None else self.start_sample
        return max(0, math.ceil((self.size - offset) / self.batch_size))


class FTNormalizer(nn.Module):
    """Per-channel train-only normalization, with mandatory state-then-wrench concatenation."""
    def __init__(self, stats, cameras=CAMERAS):
        super().__init__()
        self.cameras = tuple(cameras)
        for field in ("mean", "std"):
            self.register_buffer(f"state_{field}", torch.tensor(stats[STATE][field] + stats[WRENCH][field], dtype=torch.float32))
            self.register_buffer(f"action_{field}", torch.tensor(stats["action"][field], dtype=torch.float32))
            self.register_buffer(f"image_{field}", torch.stack([
                torch.tensor(stats[key][field], dtype=torch.float32) for key in self.cameras]))

    def observations(self, batch):
        if WRENCH not in batch:
            raise ValueError("FTACT requires observation.environment_state with six force/torque values")
        state, wrench = batch[STATE], batch[WRENCH]
        if state.ndim != 2 or state.shape[-1] != 6 or wrench.shape != state.shape:
            raise ValueError("Expected batched state [B,6] and wrist wrench [B,6]")
        combined = torch.cat((state.float(), wrench.float()), dim=-1)
        output = {STATE: (combined - self.state_mean) / self.state_std.clamp_min(1e-6)}
        for index, key in enumerate(self.cameras):
            image = batch[key]
            if image.ndim != 4 or image.shape[1:] != (3, 256, 288):
                raise ValueError(f"{key} must have shape [B,3,256,288] with RGB floats in [0,1]")
            output[key] = (image.float() - self.image_mean[index]) / self.image_std[index].clamp_min(1e-6)
        return output

    def actions(self, action):
        return (action.float() - self.action_mean) / self.action_std.clamp_min(1e-6)

    def unnormalize_actions(self, action):
        return action.float() * self.action_std.clamp_min(1e-6) + self.action_mean


def unit_tcp_quaternions(actions):
    actions = actions.float().clone()
    norm = actions[..., 3:].norm(dim=-1, keepdim=True)
    if not torch.isfinite(actions).all() or (norm < 1e-8).any():
        raise ValueError("Predicted TCP pose is nonfinite or has a degenerate quaternion")
    actions[..., 3:] /= norm
    return actions


class ForceTorqueACT(nn.Module):
    """Official ACT with a single 12-D state token; no separate ENV token or force-free mode."""
    def __init__(self, config, stats, *, initialize_backbone=True, use_vae=True, cameras=CAMERAS):
        super().__init__()
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        self.cameras = tuple(cameras)
        self.normalizer = FTNormalizer(stats, self.cameras)
        policy_config = ACTConfig(
            input_features={STATE: PolicyFeature(FeatureType.STATE, (12,)),
                            **{key: PolicyFeature(FeatureType.VISUAL, (3, 256, 288)) for key in self.cameras}},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
            device=config.device, push_to_hub=False, use_amp=config.amp != "none",
            chunk_size=config.chunk_size, n_action_steps=1,
            vision_backbone="resnet18",
            pretrained_backbone_weights=("ResNet18_Weights.IMAGENET1K_V1"
                if initialize_backbone and config.pretrained_backbone else None),
            dim_model=config.dim_model, n_heads=config.n_heads, dim_feedforward=config.dim_feedforward,
            n_encoder_layers=config.n_encoder_layers, n_decoder_layers=config.n_decoder_layers,
            use_vae=use_vae, latent_dim=config.latent_dim, n_vae_encoder_layers=config.n_vae_encoder_layers,
            dropout=config.dropout, kl_weight=config.kl_weight,
            optimizer_lr=config.learning_rate, optimizer_lr_backbone=config.backbone_lr,
            optimizer_weight_decay=config.weight_decay)
        self.policy = ACTPolicy(policy_config)

    def forward(self, batch):
        prepared = self.normalizer.observations(batch)
        prepared["action"] = self.normalizer.actions(batch["action"])
        prepared["action_is_pad"] = batch["action_is_pad"]
        return self.policy(prepared)

    @torch.no_grad()
    def normalized_action_chunk(self, batch):
        # Do not pass ground-truth actions to inference, even when present in an offline sample.
        return self.policy.predict_action_chunk(self.normalizer.observations(batch)).float()

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return unit_tcp_quaternions(self.normalizer.unnormalize_actions(self.normalized_action_chunk(batch)))


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get('format')=='aic-sfp0-alternating-act-residual-flow-multiview-v1':
        from alternating_residual_flow_multiview import load_alternating_checkpoint
        return load_alternating_checkpoint(payload,device)
    if payload.get('format')=='aic-sfp0-stacked-residual-flow-multiview-v1':
        from stacked_residual_flow_multiview import load_stacked_checkpoint
        return load_stacked_checkpoint(payload,device)
    if payload.get('format')=='aic-sfp0-supervised-residual-flow-multiview-v1':
        from residual_flow_multiview import load_residual_checkpoint
        return load_residual_checkpoint(payload,device)
    if payload.get('format')=='aic-sfp0-flow-matching-multiview-v1':
        from flow_multiview import load_flow_multiview_checkpoint
        return load_flow_multiview_checkpoint(payload,device)
    if payload.get('format')=='aic-sfp0-act-pace-pred-multiview-v1':
        from pace_pred_multiview import load_pace_pred_checkpoint
        return load_pace_pred_checkpoint(payload,device)
    if payload.get('format')=='aic-sfp0-act-pace-multiview-v1':
        from pace_multiview import load_pace_multiview_checkpoint
        return load_pace_multiview_checkpoint(payload,device)
    if payload.get('format') == 'aic-sfp0-act-multiview-v1':
        from act_multiview import load_multiview_checkpoint
        return load_multiview_checkpoint(payload, device)
    if payload.get('format') == 'aic-sfp0-act-pace-v1':
        from pace_act import load_pace_checkpoint
        return load_pace_checkpoint(payload, device)
    if payload.get('format') == 'aic-sfp0-flow-matching-v1':
        from flow_matching import load_flow_checkpoint
        return load_flow_checkpoint(payload, device)
    if payload.get("format") == "aic-sfp0-act-v2":
        from ftact_v2 import load_sfp_checkpoint
        return load_sfp_checkpoint(payload, device)
    if payload.get("format") != "aic-ftact-v1":
        raise ValueError("Not an aic-ftact-v1 checkpoint")
    config = TrainConfig(**payload["config"])
    config.device = device
    if device == "cpu":
        config.amp = "none"
    config.validate()
    model = ForceTorqueACT(config, payload["stats"], initialize_backbone=False)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload
