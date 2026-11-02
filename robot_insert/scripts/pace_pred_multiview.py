"""PACE Phase/Event + paper Eq. (6)-(7), Appendix A.4 contact prediction.

Future FT is a training/validation target only. Current observations and causal
Phase/Event histories are the sole action-generation inputs. The deterministic
ACT backbone and common-scale imitation loss remain unchanged.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from act_multiview import FOUR_CAMERAS
from act_objectives import common_scale_loss
from ftact import WRENCH
from ftact_v2 import Float32Linear
from pace_multiview import MultiViewPACEConfig, MultiViewPACE, MultiViewPACEDataset

FORMAT = 'aic-sfp0-act-pace-pred-multiview-v1'
PRED_TARGET = 'supervision.future_wrench_delta'
PRED_MASK = 'supervision.future_wrench_mask'


@dataclass
class MultiViewPACEPredConfig(MultiViewPACEConfig):
    policy_type: str = 'pace_pred'
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_pace_pred_expert_v1'
    pred_loss_weight: float = .05

    def validate(self, *, legacy_inference=False):
        super().validate(legacy_inference=legacy_inference)
        if self.policy_type != 'pace_pred':
            raise ValueError('Pred config requires policy_type=pace_pred')
        if not math.isfinite(self.pred_loss_weight) or self.pred_loss_weight < 0:
            raise ValueError('pred_loss_weight must be finite and nonnegative')


def future_wrench_targets(numeric, frame):
    """Match action t+1:t+20 with FT[i]-FT[i-1]; mask discontinuous pairs."""
    window = slice(frame, frame+21)
    wrench = np.asarray(numeric[WRENCH][window], dtype=np.float32)
    times = np.asarray(numeric['simulation_time'][window], dtype=np.float64)
    sources = np.asarray(numeric['action_source'][window])
    segments = np.asarray(numeric['action_segment'][window])
    if wrench.shape != (21, 6) or not np.isfinite(wrench).all() or times.shape != (21,):
        raise ValueError('Pred targets require 21 finite wrench readings within one episode')
    valid = (np.isclose(np.diff(times), .05, rtol=0, atol=1e-5)
             & (sources[1:] == sources[:-1]) & (segments[1:] == segments[:-1])
             & np.isin(sources[1:], ['model', 'expert']))
    motion = numeric.get('__action_mask', numeric.get('__action_weights'))
    if motion is not None:
        motion = np.asarray(motion[window]) > 0
        valid &= motion[1:] & motion[:-1]
    delta = np.diff(wrench, axis=0)
    delta[~valid] = 0
    return delta, valid


class MultiViewPACEPredDataset(MultiViewPACEDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ep, frame = self.anchors[index]
        delta, mask = future_wrench_targets(self.numeric[ep], frame)
        sample[PRED_TARGET] = torch.from_numpy(delta.copy())
        sample[PRED_MASK] = torch.from_numpy(mask.copy())
        return sample


def masked_prediction_error(prediction, target, mask, std):
    with torch.autocast(prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.float()
        if (prediction.shape != target.shape or prediction.shape[-2:] != (20, 6)
                or mask.shape != prediction.shape[:-1] or mask.dtype != torch.bool):
            raise ValueError('Pred expects [B,20,6] targets and boolean [B,20] validity')
        if not torch.isfinite(prediction).all() or not torch.isfinite(target[mask]).all():
            raise FloatingPointError('Nonfinite contact prediction/valid target')
        target = torch.where(mask[..., None], target, 0.) / std.float().clamp_min(1e-6)
        squared = (prediction-target).square().masked_fill(~mask[..., None], 0.)
        # No valid future pair is allowed to train the head or the backbone.
        steps = mask.sum()
        error_sum = squared.sum()
        mse = error_sum/(steps*6).clamp_min(1)
        return mse, error_sum, steps


class MultiViewPACEPred(MultiViewPACE):
    def __init__(self, config, stats, *, initialize_backbone=True):
        super().__init__(config, stats, initialize_backbone=initialize_backbone)
        self.policy.model.pace_pred_head = nn.Sequential(
            nn.Linear(config.dim_model, 512), nn.ReLU(), Float32Linear(512, 6))

    def prediction_objective(self, batch, hidden):
        predicted = self.policy.model.pace_pred_head(hidden)
        return masked_prediction_error(predicted, batch[PRED_TARGET], batch[PRED_MASK],
                                       self.normalizer.state_std[6:])

    def forward(self, batch):
        if batch.get('action_is_pad') is not None and batch['action_is_pad'].any():
            raise ValueError('Pred training requires complete future action windows')
        action, event, hidden = self._predict(batch, return_hidden=True)
        imitation, metrics = common_scale_loss(action, batch['action'], self.normalizer,
            self.pace_config.position_scale_m, batch.get('action_mask', batch.get('action_weights')))
        pred_mse, _, valid_steps = self.prediction_objective(batch, hidden)
        loss = imitation+self.pace_config.pred_loss_weight*pred_mse
        return loss, {k: v.detach().item() for k, v in {
            **metrics, **event, 'imitation_loss': imitation, 'pred_mse': pred_mse,
            'pred_valid_steps': valid_steps}.items()}

    def validation_predictions(self, batch):
        action, _, hidden = self._predict(batch, return_hidden=True)
        _, error_sum, steps = self.prediction_objective(batch, hidden)
        return action, error_sum.item(), steps.item()


def load_pace_pred_checkpoint(payload, device):
    config = MultiViewPACEPredConfig(**payload['config']); config.device = device
    if device == 'cpu': config.amp = 'none'
    config.validate()
    if payload.get('camera_order') != list(FOUR_CAMERAS):
        raise ValueError('Pred checkpoint camera order differs')
    model = MultiViewPACEPred(config, payload['stats'], initialize_backbone=False)
    model.load_state_dict(payload['model'], strict=True)
    # Only Phase/Event and the action generator are needed at deployment.
    del model.policy.model.pace_pred_head
    return model.to(device).eval(), payload
