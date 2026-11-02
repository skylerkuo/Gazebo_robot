"""Second supervised residual Flow around an embedded, frozen ACT + Flow.

This is an experimental extension of the first-stage residual implementation,
not an implementation of critic weighting or online ACT absorption.
"""
import copy
from dataclasses import dataclass
import math

import torch
from torch import nn

from act_multiview import FOUR_CAMERAS
from flow_matching import anchor_noise
from flow_multiview import MultiViewFlow, MultiViewFlowConfig, initialize_flow_encoder_from_act
from residual_flow_multiview import (
    FORMAT as BASE_FORMAT, ResidualFlowConfig, SupervisedResidualFlow,
    residual_readiness,
)
from ftact import STATE

FORMAT = 'aic-sfp0-stacked-residual-flow-multiview-v1'


@dataclass
class StackedResidualFlowConfig(ResidualFlowConfig):
    policy_type: str = 'stacked_residual_flow'
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_stacked_residual_flow_expert_v1'
    base_policy_config: dict | None = None
    base_policy_sha256: str | None = None
    base_policy_step: int | None = None
    base_noise_seed: int = 10042

    def validate(self, *, legacy_inference=False):
        MultiViewFlowConfig.validate(self, legacy_inference=legacy_inference)
        if self.policy_type != 'stacked_residual_flow':
            raise ValueError('Select the stacked_residual_flow policy type')
        if not math.isfinite(self.residual_scale) or self.residual_scale <= 0:
            raise ValueError('residual_scale must be finite and positive')
        if not self.expert_only or self.retain_model_motion or self.correction_repeat != 1:
            raise ValueError('Second Flow requires expert-only complete windows, once each')
        if self.sigma_min != 0:
            raise ValueError('Second residual Flow uses the straight path sigma_min=0')
        if isinstance(self.base_noise_seed, bool) or not isinstance(self.base_noise_seed, int):
            raise ValueError('base_noise_seed must be an integer')


def bind_base_policy(config, payload):
    """Snapshot the first-stage architecture/provenance once at a fresh start."""
    if (payload.get('format') != BASE_FORMAT or payload.get('smoke_test')
            or payload.get('camera_order') != list(FOUR_CAMERAS)
            or payload.get('action_offset') != 1):
        raise ValueError('Second Flow base must be a real four-view ACT + residual Flow checkpoint')
    config.base_policy_config = copy.deepcopy(payload['config'])
    config.base_policy_sha256 = payload['_initialization_sha256']
    config.base_policy_step = payload['step']
    for key in ('base_act_config', 'base_act_sha256', 'base_act_step'):
        setattr(config, key, copy.deepcopy(payload['config'][key]))


def stacked_readiness(config):
    report = residual_readiness(config)
    report['supervision'] = 'complete expert future-20 windows; ACT + first Flow frozen'
    report['base_noise_seed'] = config.base_noise_seed
    return report


class StackedResidualFlow(SupervisedResidualFlow):
    def __init__(self, config, stats, *, initialize_backbone=True):
        # Build a fresh trainable Flow, rather than another copy of base ACT alone.
        MultiViewFlow.__init__(self, config, stats, initialize_backbone=initialize_backbone)
        if config.base_policy_config is None:
            raise ValueError('Missing embedded ACT + first Flow configuration')
        base_config = ResidualFlowConfig(**copy.deepcopy(config.base_policy_config))
        base_config.device, base_config.amp = config.device, config.amp
        base_config.validate()
        self.base_policy = SupervisedResidualFlow(base_config, stats, initialize_backbone=False)
        self.base_policy.requires_grad_(False).eval()
        self.policy.model.flow_base_action_input_proj = nn.Linear(7, config.dim_model)
        self.residual_enabled = True

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'base_policy'):
            self.base_policy.eval()
        return self

    def reset_sampler(self, seed):
        super().reset_sampler(seed)
        # Separate reproducible streams for the two stochastic Flow layers.
        self.base_noise_generator = torch.Generator(device='cpu').manual_seed(
            (int(seed) ^ 0x5A17C9E3) % (2**63 - 1))

    @torch.no_grad()
    def base_normalized_action(self, batch):
        has_episode, has_frame = 'episode_index' in batch, 'frame_index' in batch
        if has_episode != has_frame:
            raise ValueError('Frozen-base anchor noise requires both episode and frame indices')
        if has_episode:
            # Same fixed first-stage sample for TRAIN, VAL and disabled-layer baseline.
            noise = anchor_noise(batch['episode_index'].detach().cpu().tolist(),
                batch['frame_index'].detach().cpu().tolist(), self.flow_config.base_noise_seed)
        else:
            # Online inference has observations only and never reads future labels.
            noise = torch.randn((len(batch[STATE]), 20, 7), generator=self.base_noise_generator)
        actions = self.base_policy.predict_action_chunk(batch, noise=noise)
        return self.normalizer.actions(actions).detach()

    def validation_metadata(self):
        return {**super().validation_metadata(),
            'generation': 'ACT_plus_frozen_residual_flow_plus_second_residual_flow',
            'residual_layers': 2, 'base_policy_frozen': True,
            'base_policy_step': self.flow_config.base_policy_step,
            'base_policy_sha256': self.flow_config.base_policy_sha256,
            'base_noise_seed': self.flow_config.base_noise_seed,
            'base_ode_steps': self.base_policy.sampling_steps,
            'base_ode_solver': self.base_policy.sampling_solver}


def initialize_stacked_from_residual(model, payload):
    model.base_policy.load_state_dict(payload['model'], strict=True)
    model.base_policy.requires_grad_(False).eval()
    report = initialize_flow_encoder_from_act(model, payload['model'])
    return {**report, 'frozen_base_step': payload['step'],
        'scope': 'full frozen ACT + first Flow; observation encoder copied into fresh second Flow'}


def load_stacked_checkpoint(payload, device):
    config = StackedResidualFlowConfig(**payload['config'])
    config.device = device
    if device == 'cpu':
        config.amp = 'none'
    config.validate()
    if payload.get('camera_order') != list(FOUR_CAMERAS) or payload.get('action_offset') != 1:
        raise ValueError('Second Flow camera/action contract differs')
    model = StackedResidualFlow(config, payload['stats'], initialize_backbone=False)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), payload
