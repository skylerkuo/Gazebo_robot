"""Force-conditioned action Flow Matching, Lipman et al. Eq. (22)-(23).

The conditional OT Gaussian path is x_tau=(1-(1-sigma_min)*tau)*eps+tau*a.
Its target velocity is a-(1-sigma_min)*eps. ODE time runs NOISE -> DATA.
The ResNet/DETR architecture is an AIC adaptation, not the paper's image U-Net.
"""
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn

from ftact import CAMERAS, STATE, unit_tcp_quaternions
from ftact_v2 import SFPConfig, SFP0ACT

FORMAT = 'aic-sfp0-flow-matching-v1'
PAPER = 'https://arxiv.org/abs/2210.02747'


@dataclass
class FlowConfig(SFPConfig):
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_flow_v1'
    sigma_min: float = 1e-4
    ode_steps: int = 10
    ode_solver: str = 'euler'
    validation_noise_seed: int = 10042

    def validate(self):
        super().validate()
        if not math.isfinite(self.sigma_min) or not 0 < self.sigma_min < 1:
            raise ValueError('sigma_min must be finite and between zero and one')
        validate_solver(self.ode_steps, self.ode_solver)
        if self.dim_model % 2:
            raise ValueError('Flow time embedding requires an even dim_model')


def validate_solver(steps, solver):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('ODE steps must be a positive integer')
    if solver not in ('euler', 'heun'):
        raise ValueError('ODE solver must be euler or heun')


def ot_path(actions, noise, tau, sigma_min):
    """Original CFM objective: one uniform tau per entire action chunk."""
    if actions.shape != noise.shape or tau.shape != (len(actions),):
        raise ValueError('Expected matching action/noise tensors and B flow times')
    if not 0 < sigma_min < 1 or not torch.isfinite(tau).all() or ((tau < 0) | (tau > 1)).any():
        raise ValueError('Invalid OT path parameters')
    tau = tau.float().view(-1, 1, 1)
    return ((1 - (1 - sigma_min) * tau) * noise.float() + tau * actions.float(),
            actions.float() - (1 - sigma_min) * noise.float())


def integrate_velocity(velocity, initial, steps, solver='euler'):
    """Fixed-step ODE, accumulated in FP32; no projection/clipping inside the path."""
    validate_solver(steps, solver)
    value = initial.float().clone()
    delta = 1. / steps
    for i in range(steps):
        tau = torch.full((len(value),), i / steps, device=value.device, dtype=torch.float32)
        first = velocity(value, tau).float()
        if solver == 'heun':
            second = velocity(value + delta * first, tau + delta).float()
            value = value + delta * (first + second) * .5
        else:
            value = value + delta * first
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite Flow Matching ODE output')
    return value


def anchor_noise(episodes, frames, seed, chunk_size=20):
    """One sample per anchor, stable across batches, ordering and validation calls.

    CPU generation also makes the initial noise identical on CPU and GPU.
    This does not inspect actions, choose the best sample, or consume train RNG.
    """
    values = []
    for episode, frame in zip(episodes, frames):
        key = f'{seed}:{int(episode)}:{int(frame)}'.encode()
        anchor_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63 - 1)
        generator = torch.Generator(device='cpu').manual_seed(anchor_seed)
        values.append(torch.randn(chunk_size, 7, generator=generator))
    if not values:
        raise ValueError('Cannot generate noise for an empty batch')
    return torch.stack(values)


class ForceTorqueFlow(SFP0ACT):
    """Same observation encoder as ACT; noisy action/time conditioned decoder."""
    def __init__(self, config, stats, *, initialize_backbone=True, cameras=CAMERAS):
        super().__init__(config, stats, initialize_backbone=initialize_backbone,cameras=cameras)
        self.flow_config = config
        core = self.policy.model
        core.flow_action_input_proj = nn.Linear(7, config.dim_model)
        core.flow_time_mlp = nn.Sequential(nn.Linear(config.dim_model, config.dim_model), nn.SiLU(),
                                          nn.Linear(config.dim_model, config.dim_model))
        self.sampling_steps, self.sampling_solver = config.ode_steps, config.ode_solver
        self.reset_sampler(config.seed)

    def reset_sampler(self, seed):
        self.noise_generator = torch.Generator(device='cpu').manual_seed(int(seed))

    def configure_sampling(self, steps=None, solver=None, seed=None):
        steps = self.sampling_steps if steps is None else steps
        solver = self.sampling_solver if solver is None else solver
        validate_solver(steps, solver)
        self.sampling_steps, self.sampling_solver = steps, solver
        if seed is not None:
            self.reset_sampler(seed)

    def encode_observations(self, batch):
        """Cache observation-only memory ONCE per chunk, independent of ODE tau."""
        core = self.policy.model
        prepared = self.normalizer.observations(batch)  # Requires all six wrench channels.
        state = prepared[STATE]
        latent = state.new_zeros((len(state), self.flow_config.latent_dim))
        tokens = [core.encoder_latent_input_proj(latent).unsqueeze(0),
                  core.encoder_robot_state_input_proj(state).unsqueeze(0)]
        positions = [core.encoder_1d_feature_pos_embed.weight.unsqueeze(1)]
        for key in self.cameras:
            features = core.backbone(prepared[key])['feature_map']
            position = core.encoder_cam_feat_pos_embed(features).to(features.dtype)
            tokens.append(core.encoder_img_feat_input_proj(features).flatten(2).permute(2, 0, 1))
            positions.append(position.flatten(2).permute(2, 0, 1))
        positions = torch.cat(positions, dim=0)
        memory = core.encoder(torch.cat(tokens, dim=0), pos_embed=positions)
        return memory, positions

    def velocity(self, actions, tau, condition):
        core = self.policy.model
        dimension = self.flow_config.dim_model
        frequency = torch.exp(torch.arange(dimension // 2, device=tau.device, dtype=torch.float32)
                              * (-math.log(10000.) / (dimension // 2 - 1)))
        angles = tau.float()[:, None] * frequency[None] * 1000.
        time_embedding = core.flow_time_mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))
        content = core.flow_action_input_proj(actions) + time_embedding[:, None]
        memory, positions = condition
        decoded = core.decoder(content.transpose(0, 1), memory, encoder_pos_embed=positions,
                               decoder_pos_embed=core.decoder_pos_embed.weight.unsqueeze(1))
        return core.action_head(decoded.transpose(0, 1))

    def forward(self, batch):
        target = self.normalizer.actions(batch['action'])
        if target.shape[1:] != (20, 7) or batch.get('action_is_pad', torch.zeros(1)).any():
            raise ValueError('Flow training requires complete future 20x7 chunks')
        noise = torch.randn_like(target)
        tau = torch.rand(len(target), device=target.device)
        state, velocity_target = ot_path(target, noise, tau, self.flow_config.sigma_min)
        prediction = self.velocity(state, tau, self.encode_observations(batch))
        loss = (prediction.float() - velocity_target).square().mean()
        return loss, {'flow_mse': loss.detach().item()}

    @torch.no_grad()
    def normalized_action_chunk(self, batch, *, noise=None, steps=None, solver=None):
        # Deliberately ignore any action/label keys in batch during generation.
        condition = self.encode_observations(batch)
        shape = (len(batch[STATE]), self.flow_config.chunk_size, 7)
        if noise is None:
            noise = torch.randn(shape, generator=self.noise_generator, device='cpu')
        if tuple(noise.shape) != shape or not torch.isfinite(noise).all():
            raise ValueError('Invalid initial Flow Matching noise')
        noise = noise.to(device=batch[STATE].device, dtype=torch.float32)
        return integrate_velocity(lambda x, tau: self.velocity(x, tau, condition), noise,
            self.sampling_steps if steps is None else steps,
            self.sampling_solver if solver is None else solver)

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        normalized = self.normalized_action_chunk(batch, **kwargs)
        # AIC pose adaptation: normalize the quaternion ONLY after ODE/unnormalization.
        return unit_tcp_quaternions(self.normalizer.unnormalize_actions(normalized))


def load_flow_checkpoint(payload, device):
    if payload.get('format') != FORMAT or payload.get('action_offset') != 1:
        raise ValueError('Not a next-step SFP0 Flow Matching checkpoint')
    config = FlowConfig(**payload['config'])
    config.device = device
    if device == 'cpu':
        config.amp = 'none'
    config.validate()
    model = ForceTorqueFlow(config, payload['stats'], initialize_backbone=False)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), payload
