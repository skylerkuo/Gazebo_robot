"""Offline alternating ACT behavior cloning and detached residual Flow updates.

Uses the successful-action BC/relabeling idea of ACT--Flow, without its online
collection or critic. There is one residual Flow, not a stacked second layer.
"""
from dataclasses import dataclass, fields
import math

import torch

from act_multiview import FOUR_CAMERAS, MultiViewConfig
from flow_multiview import MultiViewFlowConfig
from residual_flow_multiview import (
    ResidualFlowConfig, SupervisedResidualFlow, initialize_residual_from_act,
)

FORMAT = 'aic-sfp0-alternating-act-residual-flow-multiview-v1'


@dataclass
class AlternatingResidualFlowConfig(ResidualFlowConfig):
    policy_type: str = 'alternating_residual_flow'
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_alternating_residual_flow_expert_v1'
    act_learning_rate: float = 1e-5
    act_backbone_lr: float = 1e-6
    from_scratch: bool = False
    lr_schedule_steps: int | None = None

    def validate(self, *, legacy_inference=False):
        MultiViewFlowConfig.validate(self, legacy_inference=legacy_inference)
        if self.policy_type != 'alternating_residual_flow':
            raise ValueError('Select the alternating_residual_flow policy type')
        if not self.expert_only or self.retain_model_motion or self.correction_repeat != 1:
            raise ValueError('Alternating supervision requires expert-only complete windows, once each')
        if self.sigma_min != 0:
            raise ValueError('Alternating residual Flow requires sigma_min=0')
        if self.lr_schedule_steps is not None and (isinstance(self.lr_schedule_steps, bool)
                or not isinstance(self.lr_schedule_steps, int) or self.lr_schedule_steps < 1):
            raise ValueError('lr_schedule_steps must be a positive integer')
        if self.from_scratch and (self.init_from is not None or self.pretrained_backbone
                or self.base_act_sha256 is not None or self.base_act_step not in (None, 0)):
            raise ValueError('From-scratch training cannot load an ACT checkpoint or pretrained backbone')
        if self.from_scratch and self.base_act_config is not None and (
                self.base_act_config.get('init_from') is not None
                or self.base_act_config.get('pretrained_backbone', True)):
            raise ValueError('From-scratch embedded ACT must also have no pretrained weights')
        for key in ('residual_scale', 'act_learning_rate', 'act_backbone_lr'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')


def validate_expert_batch(batch):
    target = batch['action']
    mask = batch.get('action_mask')
    if (target.shape[1:] != (20, 7) or mask is None or mask.dtype != torch.bool
            or mask.shape != target.shape[:-1] or not mask.all()
            or batch.get('action_is_pad', torch.zeros(1)).any()
            or not torch.isfinite(target).all()):
        raise ValueError('Alternating training requires complete valid expert future-20 actions')


class AlternatingResidualFlow(SupervisedResidualFlow):
    def __init__(self, config, stats, *, initialize_backbone=True):
        if config.from_scratch:
            config.validate()
            if config.base_act_config is None:
                configure_scratch_act(config)
            initialize_backbone = False
        super().__init__(config, stats, initialize_backbone=initialize_backbone)
        self.base_act.requires_grad_(True)
        self.base_act.multiview_config.position_scale_m = config.position_scale_m
        self.register_buffer('act_update_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('flow_update_count', torch.zeros((), dtype=torch.long))

    def act_loss(self, batch):
        validate_expert_batch(batch)
        # ACT's dropout and BN are active only during its own BC forward pass.
        self.base_act.train(self.training)
        try:
            return self.base_act(batch)
        finally:
            self.base_act.eval()

    def forward(self, batch):
        validate_expert_batch(batch)
        # Inherited @no_grad ACT prediction is recomputed here, after ACT.step().
        # Labels and conditioning use the same deterministic, current ACT output.
        self.base_act.eval()
        return super().forward(batch)

    def validation_metadata(self):
        return {**super().validation_metadata(),
            'generation': 'alternating_ACT_plus_supervised_residual_flow',
            'base_act_frozen': False, 'base_act_step': None,
            'act_initialization_step': self.flow_config.base_act_step,
            'act_initialization_sha256': self.flow_config.base_act_sha256,
            'from_scratch': self.flow_config.from_scratch,
            'act_updates': int(self.act_update_count.item()),
            'flow_updates': int(self.flow_update_count.item()),
            'update_order': 'ACT_BC_then_relabel_then_Flow_MSE',
            'flow_gradient_to_act': False}


def configure_scratch_act(config):
    """Create ACT architecture from this run's settings without reading weights."""
    config.base_act_config = {field.name: getattr(config, field.name) for field in fields(MultiViewConfig)}
    config.base_act_config.update(policy_type='act', init_from=None, pretrained_backbone=False,
        learning_rate=config.act_learning_rate, backbone_lr=config.act_backbone_lr)
    config.base_act_sha256, config.base_act_step = None, 0


def initialize_alternating_from_act(model, payload):
    if model.flow_config.from_scratch:
        raise ValueError('A from-scratch model cannot import ACT weights')
    if payload.get('format') in (FORMAT,'aic-sfp0-supervised-residual-flow-multiview-v1'):
        # Import the complete learned generator, including both vision encoders.
        # Counters/optimizers belong to this new split, so only counters reset here.
        weights=dict(payload['model'])
        for name in ('act_update_count','flow_update_count'):
            weights[name]=getattr(model,name).detach().new_zeros(())
        model.load_state_dict(weights,strict=True)
        model.base_act.requires_grad_(True).eval()
        return {'scope':'full trained ACT + Flow pair retained; both trainable',
            'source_format':payload['format'],'source_pair_step':payload['step'],
            'source_base_act_step':payload['config'].get('base_act_step'),
            'flow_weights_preserved':True,'optimizer_restarted':True,
            'act_initialization_step':payload['step'],'act_frozen':False}
    report = initialize_residual_from_act(model, payload)
    report.pop('frozen_base_step', None)
    model.base_act.requires_grad_(True).eval()
    return {**report, 'scope': 'trainable ACT initialization plus copied Flow observation encoder; fresh Flow generator',
            'act_initialization_step': payload['step'], 'act_frozen': False}


def load_alternating_checkpoint(payload, device):
    config = AlternatingResidualFlowConfig(**payload['config'])
    config.device = device
    if device == 'cpu':
        config.amp = 'none'
    config.validate()
    if payload.get('camera_order') != list(FOUR_CAMERAS) or payload.get('action_offset') != 1:
        raise ValueError('Alternating policy camera/action contract differs')
    model = AlternatingResidualFlow(config, payload['stats'], initialize_backbone=False)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), payload
