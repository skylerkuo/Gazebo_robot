"""SFP0 ACT + PACE Phase/Event, adapted to absolute TCP xyzw commands.

PACE: https://arxiv.org/abs/2610.16308, Eq. (1)-(5), Appendix A.
This controlled adaptation retains our existing deterministic ACT backbone:
no CVAE, Pred. head, extra state history, or previous expert command input.
"""
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ftact import CAMERAS, STATE, WRENCH, unit_tcp_quaternions
from ftact_v2 import SFPConfig, SFP0ACT, TaskDataset, Float32Linear

FORMAT = 'aic-sfp0-act-pace-v1'
PAPER = 'https://arxiv.org/abs/2610.16308'
PHASE = 'observation.pace_contact_state'
HISTORY = 'observation.pace_wrench_history'


@dataclass
class PACEConfig(SFPConfig):
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_act_pace_v1'
    save_every: int = 500
    phase_alpha: float = .05
    tactile_history: int = 4
    correction_horizon: int = 5
    position_scale_m: float = .01

    def validate(self):
        super().validate()
        if not math.isfinite(self.phase_alpha) or not 0 < self.phase_alpha <= 1:
            raise ValueError('phase_alpha must be finite and in (0, 1]')
        if isinstance(self.tactile_history, bool) or not isinstance(self.tactile_history, int) or self.tactile_history < 2:
            raise ValueError('tactile_history must be an integer >= 2')
        if isinstance(self.correction_horizon, bool) or not isinstance(self.correction_horizon, int) or not 1 <= self.correction_horizon <= self.chunk_size:
            raise ValueError('correction_horizon must fit inside the action chunk')
        if not math.isfinite(self.position_scale_m) or self.position_scale_m <= 0:
            raise ValueError('position_scale_m must be finite and positive')


class TactileContext:
    """Causal 20 Hz state; reset at each segment/task, never each random batch.

    The first reading initializes the EMA and left-pads history. Online missing
    ticks use the last known wrench, not interpolated future observations.
    Duplicate timestamps do not update state. All calculations stay in FP32.
    """
    def __init__(self, alpha=.05, history=4, fps=20):
        if not 0 < alpha <= 1 or history < 2 or fps <= 0:
            raise ValueError('Invalid tactile context settings')
        self.alpha, self.history, self.fps = alpha, history, fps
        self.reset()

    def reset(self):
        self.state, self.previous, self.last_stamp = None, None, None
        self.values = deque(maxlen=self.history)

    def _append(self, wrench):
        self.state = ((1 - self.alpha) * self.state + self.alpha * wrench).astype(np.float32)
        self.values.append(wrench.copy())

    def update(self, wrench, stamp=None):
        wrench = np.asarray(wrench, dtype=np.float32)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError('PACE requires a finite six-axis wrench')
        if stamp is not None and not math.isfinite(stamp):
            raise ValueError('Nonfinite tactile timestamp')
        if self.last_stamp is not None and stamp is None:
            raise ValueError('Cannot omit timestamps after starting a timed stream')
        if self.state is None:
            self.state = wrench.copy()
            self.values.extend(wrench.copy() for _ in range(self.history))
        else:
            if stamp is not None and self.last_stamp is not None:
                delta = stamp - self.last_stamp
                if delta < -1e-6:
                    raise ValueError('Tactile clock went backwards; reset at the new task')
                if delta <= 1e-6:
                    return self.features()
                if delta > 1.:
                    raise ValueError('Tactile stream gap exceeds one second')
                ticks = max(1, round(delta * self.fps))
                for _ in range(ticks - 1):
                    self._append(self.previous)
            self._append(wrench)
        self.previous = wrench.copy()
        self.last_stamp = stamp
        return self.features()

    def features(self):
        return {PHASE: self.state.copy(), HISTORY: np.stack(self.values)}


class PACETaskDataset(TaskDataset):
    """Same anchors/targets as ACT; FT context uses only segment start through t."""
    def __init__(self, manifest, split, *, smoke=False, alpha=.05, history=4):
        super().__init__(manifest, split, smoke=smoke)
        self.contexts = {}
        self.history = history
        for segment in self.segments:
            wrench = read_episode_wrench(manifest['dataset_root'], segment['episode_index'])
            if wrench.shape != (segment['episode_length'], 6) or not np.isfinite(wrench).all():
                raise ValueError('Invalid wrench column in the prepared episode')
            start, end = segment['start'], segment['end']
            values = wrench[start:end].copy()
            states = np.empty_like(values)
            context = TactileContext(alpha, history)
            for i, value in enumerate(values):
                states[i] = context.update(value)[PHASE]
            self.contexts[segment['episode_index']] = (start, values, states)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        episode, frame = self.anchors[index]
        start, values, states = self.contexts[episode]
        local = frame - start
        indices = np.maximum(0, np.arange(local - self.history + 1, local + 1))
        sample[PHASE] = torch.from_numpy(states[local].copy())
        sample[HISTORY] = torch.from_numpy(values[indices].copy())
        if not torch.allclose(sample[HISTORY][-1], sample[WRENCH].float(), atol=1e-6, rtol=1e-6):
            raise ValueError('PACE history does not end at the current observation')
        return sample


def read_episode_wrench(root, episode):
    # Read only the numeric column; never decode historical images or use future actions.
    import pyarrow.parquet as pq
    path = Path(root) / f'data/chunk-000/file-{episode:03d}.parquet'
    return np.asarray(pq.read_table(path, columns=[WRENCH])[WRENCH].to_pylist(), dtype=np.float32)


def common_scale_loss(prediction, target, normalizer, scale_m=.01):
    """Equal physical XYZ weights; retain the old 4/7 quaternion L1 weighting."""
    with torch.autocast(prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.float()
        physical = normalizer.unnormalize_actions(prediction)
        normalized_target = normalizer.actions(target)
        position = ((physical[..., :3] - target[..., :3]).abs() / scale_m).mean()
        rotation = (prediction[..., 3:] - normalized_target[..., 3:]).abs().mean()
        loss = (3 / 7) * position + (4 / 7) * rotation
        return loss, {'position_loss': position, 'rotation_l1': rotation,
                      'normalized_l1': (prediction - normalized_target).abs().mean()}


class PACEACT(SFP0ACT):
    def __init__(self, config, stats, *, initialize_backbone=True,cameras=CAMERAS):
        super().__init__(config, stats, initialize_backbone=initialize_backbone,cameras=cameras)
        self.pace_config = config
        core, dim = self.policy.model, config.dim_model
        # Keep all added parameters under policy.model so ACT optimizer groups include them.
        core.pace_phase_encoder = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        core.pace_modulators = nn.ModuleList(nn.Linear(128, 2 * dim) for _ in core.encoder.layers)
        core.pace_phase_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in core.encoder.layers)
        core.pace_event_gate = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                             Float32Linear(64, 7), nn.Sigmoid())
        core.pace_event_residual = nn.Sequential(nn.Linear(config.tactile_history * 6, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), Float32Linear(128, config.correction_horizon * 7))
        # Start without a learned additive correction. All paths learn end to end.
        for modulator in core.pace_modulators:
            nn.init.zeros_(modulator.weight)
            nn.init.zeros_(modulator.bias)
        nn.init.zeros_(core.pace_event_residual[-1].weight)
        nn.init.zeros_(core.pace_event_residual[-1].bias)

    def _predict(self, batch, *, return_hidden=False):
        core = self.policy.model
        prepared = self.normalizer.observations(batch)
        if PHASE not in batch or HISTORY not in batch:
            raise ValueError('PACE requires causal contact state and wrench history')
        state = prepared[STATE]
        size = len(state)
        phase, history = batch[PHASE].float(), batch[HISTORY].float()
        if phase.shape != (size, 6) or history.shape != (size, self.pace_config.tactile_history, 6):
            raise ValueError('Incorrect PACE context shapes')
        if not torch.isfinite(phase).all() or not torch.isfinite(history).all():
            raise ValueError('Nonfinite PACE contact history')
        mean, std = self.normalizer.state_mean[6:], self.normalizer.state_std[6:].clamp_min(1e-6)
        condition = core.pace_phase_encoder((phase - mean) / std)
        normalized_history = (history - mean) / std
        change = normalized_history[:, -1] - normalized_history[:, 0]
        event = torch.cat((change, change.norm(dim=-1, keepdim=True)), dim=-1)
        latent = state.new_zeros((size, self.pace_config.latent_dim))
        tokens = [core.encoder_latent_input_proj(latent).unsqueeze(0),
                  core.encoder_robot_state_input_proj(state).unsqueeze(0)]
        positions = [core.encoder_1d_feature_pos_embed.weight.unsqueeze(1)]
        for key in self.cameras:
            features = core.backbone(prepared[key])['feature_map']
            positions.append(core.encoder_cam_feat_pos_embed(features).to(features.dtype).flatten(2).permute(2, 0, 1))
            tokens.append(core.encoder_img_feat_input_proj(features).flatten(2).permute(2, 0, 1))
        memory, positions = torch.cat(tokens, dim=0), torch.cat(positions, dim=0)
        for layer, modulator, norm in zip(core.encoder.layers, core.pace_modulators, core.pace_phase_norms):
            memory = layer(memory, pos_embed=positions)
            gamma, beta = modulator(condition).chunk(2, dim=-1)
            memory = (1 + gamma.unsqueeze(0)) * norm(memory) + beta.unsqueeze(0)
        memory = core.encoder.norm(memory)
        queries = positions.new_zeros((self.pace_config.chunk_size, size, self.pace_config.dim_model))
        hidden = core.decoder(queries, memory, encoder_pos_embed=positions,
                              decoder_pos_embed=core.decoder_pos_embed.weight.unsqueeze(1)).transpose(0, 1)
        generated = core.action_head(hidden).float()
        gate = core.pace_event_gate(event)
        horizon = self.pace_config.correction_horizon
        residual = core.pace_event_residual(normalized_history.flatten(1)).view(size, horizon, 7)
        correction = gate[:, None] * residual
        corrected = torch.cat((generated[:, :horizon] + correction, generated[:, horizon:]), dim=1)
        correction_m = correction[..., :3] * self.normalizer.action_std[:3]
        metrics = {'event_gate_mean': gate.mean(),
                   'event_position_correction_mm': correction_m.norm(dim=-1).mean() * 1000}
        return (corrected, metrics, hidden) if return_hidden else (corrected, metrics)

    def forward(self, batch):
        if batch.get('action_is_pad') is not None and batch['action_is_pad'].any():
            raise ValueError('PACE training requires complete future action windows')
        prediction, event = self._predict(batch)
        from act_objectives import common_scale_loss as masked_loss
        loss, metrics = masked_loss(prediction, batch['action'], self.normalizer,
            self.pace_config.position_scale_m,batch.get('action_mask',batch.get('action_weights')))
        return loss, {key: value.detach().item() for key, value in {**metrics, **event}.items()}

    @torch.no_grad()
    def normalized_action_chunk(self, batch):
        # No actions, future force readings, or per-batch EMA updates enter generation.
        return self._predict(batch)[0]

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return unit_tcp_quaternions(self.normalizer.unnormalize_actions(self.normalized_action_chunk(batch)))


def load_pace_checkpoint(payload, device):
    config = PACEConfig(**payload['config'])
    config.device = device
    if device == 'cpu':
        config.amp = 'none'
    config.validate()
    model = PACEACT(config, payload['stats'], initialize_backbone=False)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), payload
