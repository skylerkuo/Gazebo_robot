"""Four-view action CFM: Lipman et al. Eq. (22)-(23), AIC masked adaptation.

The probability path/objective follow the paper's per-example conditional OT
Gaussian construction. Observation conditioning, Transformer architecture,
masked action labels and quaternion projection are robotics adaptations.
"""
from dataclasses import dataclass
import math

import torch

from act_multiview import MultiViewConfig, FOUR_CAMERAS
from ftact_v2 import Float32Linear
from flow_matching import ForceTorqueFlow, anchor_noise, ot_path, validate_solver

FORMAT = 'aic-sfp0-flow-matching-multiview-v1'


@dataclass
class MultiViewFlowConfig(MultiViewConfig):
    policy_type: str = 'flow'
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_flow_expert_v1'
    sigma_min: float = 1e-4
    ode_steps: int = 10
    ode_solver: str = 'euler'
    validation_noise_seed: int = 10042

    def validate(self, *, legacy_inference=False):
        super().validate(legacy_inference=legacy_inference)
        if self.policy_type not in ('flow','residual_flow','stacked_residual_flow','alternating_residual_flow'):raise ValueError('Flow config requires a Flow policy type')
        valid_sigma = 0 <= self.sigma_min < 1 if self.policy_type in ('residual_flow','stacked_residual_flow','alternating_residual_flow') else 0 < self.sigma_min < 1
        if not math.isfinite(self.sigma_min) or not valid_sigma:
            raise ValueError('Invalid sigma_min for this Flow policy')
        validate_solver(self.ode_steps, self.ode_solver)
        if self.dim_model < 4 or self.dim_model % 2:
            raise ValueError('Flow time embedding requires an even dim_model >= 4')
        if isinstance(self.validation_noise_seed, bool) or not isinstance(self.validation_noise_seed, int):
            raise ValueError('Validation noise seed must be an integer')


def masked_velocity_mse(prediction, target, mask):
    with torch.autocast(prediction.device.type, enabled=False):
        if prediction.shape != target.shape or prediction.shape[-2:] != (20,7):
            raise ValueError('Flow velocity must have shape [B,20,7]')
        if mask.shape != prediction.shape[:-1] or mask.dtype != torch.bool or not mask.any():
            raise ValueError('Flow loss requires a boolean [B,20] mask with valid targets')
        if not torch.isfinite(prediction).all() or not torch.isfinite(target[mask]).all():
            raise FloatingPointError('Nonfinite Flow velocity or valid target')
        target = torch.where(mask[...,None],target.float(),0.)
        error = (prediction.float()-target).square().masked_fill(~mask[...,None],0.)
        return error.sum()/(mask.sum()*7)


class MultiViewFlow(ForceTorqueFlow):
    def __init__(self, config, stats, *, initialize_backbone=True):
        super().__init__(config,stats,initialize_backbone=initialize_backbone,cameras=FOUR_CAMERAS)
        # CFM operates after an invertible affine action transform. Equal XYZ
        # scales preserve equal physical position weights instead of inheriting
        # the old parent's much larger Z std. Observation/rotation stats stay fixed.
        with torch.no_grad():self.normalizer.action_std[:3].fill_(config.position_scale_m)
        # Output is velocity, not ACT's normalized pose; always a fresh head.
        self.policy.model.action_head = Float32Linear(config.dim_model,7)

    def forward(self, batch):
        target = self.normalizer.actions(batch['action']).float()
        if target.shape[1:] != (20,7) or batch.get('action_is_pad',torch.zeros(1)).any():
            raise ValueError('Flow training requires complete future-20 chunks')
        mask = batch.get('action_mask', torch.ones(target.shape[:-1],dtype=torch.bool,device=target.device))
        if mask.shape != target.shape[:-1] or mask.dtype != torch.bool or not mask.any():
            raise ValueError('Flow requires a boolean action mask with valid steps')
        if not torch.isfinite(target[mask]).all():raise FloatingPointError('Nonfinite valid action')
        # Never condition other queries on an excluded demonstration target.
        # Neutral endpoints for masked tokens are an explicit missing-label adaptation.
        target = torch.where(mask[...,None],target,0.)
        noise = torch.randn_like(target)
        tau = torch.rand(len(target),device=target.device)
        state, velocity_target = ot_path(target,noise,tau,self.flow_config.sigma_min)
        prediction = self.velocity(state,tau,self.encode_observations(batch))
        loss = masked_velocity_mse(prediction,velocity_target,mask)
        return loss, {'flow_mse':loss.detach().item(),'flow_valid_steps':mask.sum().item()}

    def validation_action_chunk(self, batch):
        noise = anchor_noise(batch['episode_index'].detach().cpu().tolist(),
            batch['frame_index'].detach().cpu().tolist(), self.flow_config.validation_noise_seed)
        return self.normalized_action_chunk(batch,noise=noise)

    def validation_metadata(self):
        return {'generation':'conditional_flow_matching', 'ode_steps':self.sampling_steps,
            'ode_solver':self.sampling_solver,'validation_noise_seed':self.flow_config.validation_noise_seed,
            'validation_samples_per_anchor':1,'best_selection_metric':'loss',
            'selection_note':'generated action common-scale loss; not train velocity MSE'}


def initialize_flow_encoder_from_act(model, weights):
    prefixes = ('policy.model.backbone.', 'policy.model.encoder.')
    names = [name for name in model.state_dict() if name.startswith(prefixes)
             or name.startswith('policy.model.encoder_')]
    if not names or any(name not in weights for name in names):
        raise ValueError('ACT source is missing observation encoder weights')
    transferred = {name:weights[name] for name in names}
    result = model.load_state_dict(transferred,strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(model.state_dict())-set(names):
        raise ValueError('Flow transfer may load only the ACT observation encoder')
    return {'transferred_tensors':len(names),'scope':'ResNet and observation Encoder only; fresh velocity decoder/head'}


def load_flow_multiview_checkpoint(payload, device):
    config = MultiViewFlowConfig(**payload['config']);config.device=device
    if device=='cpu':config.amp='none'
    config.validate()
    if payload.get('camera_order') != list(FOUR_CAMERAS) or payload.get('action_offset') != 1:
        raise ValueError('Four-view Flow camera/action contract differs')
    model = MultiViewFlow(config,payload['stats'],initialize_backbone=False)
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),payload
