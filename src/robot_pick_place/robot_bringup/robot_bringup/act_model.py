"""ACT with fine-tuned ResNet18 and DETR attention, scaled for a single arm.

Reference: tonyzhaozh/act, detr/models/{backbone,transformer,detr_vae}.py.
Position/query embeddings enter Q/K at every attention layer, never V.
New checkpoints use format 2; the previous compact architecture is unsupported.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


CHECKPOINT_VERSION = 2


@dataclass
class ACTConfig:
    cameras: tuple = ('scene', 'wrist', 'side')
    state_dim: int = 7
    chunk_size: int = 20
    hidden_dim: int = 256
    latent_dim: int = 32
    nheads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    posterior_layers: int = 4
    feedforward_dim: int = 1024
    dropout: float = .1
    train_backbone: bool = True

    def __post_init__(self):
        if (not self.cameras or len(set(self.cameras)) != len(self.cameras) or
                self.state_dim != 7 or self.chunk_size < 1 or self.nheads < 1 or
                self.hidden_dim % 4 or self.hidden_dim % self.nheads or
                min(self.hidden_dim, self.latent_dim, self.encoder_layers, self.decoder_layers,
                    self.posterior_layers, self.feedforward_dim) < 1 or not 0 <= self.dropout < 1):
            raise ValueError('Invalid ACT model configuration')


class FrozenBatchNorm2d(nn.Module):
    """Official ACT freezes BN affine values AND running statistics, not CNN weights."""
    def __init__(self, channels):
        super().__init__()
        for name, value in (('weight', 1.), ('bias', 0.), ('running_mean', 0.), ('running_var', 1.)):
            self.register_buffer(name, torch.full((channels,), value))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(prefix + 'num_batches_tracked', None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, pixels):
        scale = self.weight * torch.rsqrt(self.running_var + 1e-5)
        bias = self.bias - self.running_mean * scale
        return pixels * scale[None, :, None, None] + bias[None, :, None, None]


def sine_positions(length, dimension, device):
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, dimension, 2, device=device) * (-math.log(10000.) / dimension))
    result = torch.empty(length, dimension, device=device)
    result[:, 0::2], result[:, 1::2] = torch.sin(positions * frequency), torch.cos(positions * frequency)
    return result


def spatial_positions(height, width, dimension, device):
    """DETR's normalized 2-D sine encoding, reset separately for each camera."""
    half = dimension // 2
    frequency = 10000. ** (2 * torch.div(torch.arange(half, device=device), 2, rounding_mode='floor') / half)
    def axis(size):
        coordinate = torch.arange(1, size + 1, device=device, dtype=torch.float32) / (size + 1e-6) * (2 * math.pi)
        angle = coordinate[:, None] / frequency
        return torch.stack((angle[:, 0::2].sin(), angle[:, 1::2].cos()), dim=-1).flatten(1)
    y, x = axis(height), axis(width)
    return torch.cat((y[:, None].expand(-1, width, -1), x[None].expand(height, -1, -1)), dim=-1)


class EncoderLayer(nn.Module):
    """DETR post-norm encoder: positional Q/K, content V, ReLU feedforward."""
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.self_attn = nn.MultiheadAttention(d, config.nheads, dropout=config.dropout, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(d, config.feedforward_dim), nn.Linear(config.feedforward_dim, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1, self.dropout2 = nn.Dropout(config.dropout), nn.Dropout(config.dropout)

    def forward(self, content, positions, padding=None):
        query = content + positions
        attended = self.self_attn(query, query, content, key_padding_mask=padding, need_weights=False)[0]
        content = self.norm1(content + self.dropout1(attended))
        update = self.linear2(self.dropout(F.relu(self.linear1(content))))
        return self.norm2(content + self.dropout2(update))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.self_attn = nn.MultiheadAttention(d, config.nheads, dropout=config.dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d, config.nheads, dropout=config.dropout, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(d, config.feedforward_dim), nn.Linear(config.feedforward_dim, d)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1, self.dropout2, self.dropout3 = (nn.Dropout(config.dropout) for _ in range(3))

    def forward(self, content, memory, positions, queries):
        query = content + queries
        update = self.self_attn(query, query, content, need_weights=False)[0]
        content = self.norm1(content + self.dropout1(update))
        update = self.multihead_attn(content + queries, memory + positions, memory, need_weights=False)[0]
        content = self.norm2(content + self.dropout2(update))
        update = self.linear2(self.dropout(F.relu(self.linear1(content))))
        return self.norm3(content + self.dropout3(update))


class ACT(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        self.config = config
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None,
                            norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.backbone.requires_grad_(config.train_backbone)
        d = config.hidden_dim
        self.visual_projection = nn.Conv2d(512, d, 1)
        self.state_projection = nn.Linear(config.state_dim, d)
        self.latent_projection = nn.Linear(config.latent_dim, d)
        self.special_positions = nn.Embedding(2, d)
        self.queries = nn.Embedding(config.chunk_size, d)
        self.action_head = nn.Linear(d, config.state_dim)
        self.visual_encoder = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.posterior_encoder = nn.ModuleList([EncoderLayer(config) for _ in range(config.posterior_layers)])
        self.posterior_action = nn.Linear(config.state_dim, d)
        self.posterior_state = nn.Linear(config.state_dim, d)
        self.posterior_cls = nn.Embedding(1, d)
        self.posterior_parameters = nn.Linear(d, config.latent_dim * 2)
        self.decoder = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.decoder_norm = nn.LayerNorm(d)
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        for stack in (self.visual_encoder, self.posterior_encoder, self.decoder):
            for parameter in stack.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, qpos, images, actions=None, is_pad=None):
        """B x 7 + B x views x 3 x H x W -> B x K x 7 normalized positions.

        Train/official posterior validation receive actions; deployment uses z=0.
        Labels remain future measured qpos(t+dt ... t+Kdt) for direct teaching.
        """
        batch = qpos.shape[0]
        if qpos.shape != (batch, self.config.state_dim):
            raise ValueError('qpos must be B x 7')
        if images.ndim != 5 or images.shape[:3] != (batch, len(self.config.cameras), 3):
            raise ValueError('images must be B x cameras x 3 x H x W')
        mu = logvar = None
        if actions is not None:
            if actions.shape != (batch, self.config.chunk_size, self.config.state_dim):
                raise ValueError('Unexpected action chunk shape')
            if is_pad is None or is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool:
                raise ValueError('Training needs a boolean B x K padding mask')
            tokens = torch.cat((self.posterior_cls.weight[None].expand(batch, -1, -1),
                                self.posterior_state(qpos)[:, None], self.posterior_action(actions)), dim=1)
            padding = F.pad(is_pad, (2, 0), value=False)
            positions = sine_positions(tokens.shape[1], self.config.hidden_dim, qpos.device)[None]
            for layer in self.posterior_encoder:
                tokens = layer(tokens, positions, padding)
            mu, logvar = self.posterior_parameters(tokens[:, 0]).float().chunk(2, dim=-1)
            latent = mu + torch.exp(logvar * .5) * torch.randn_like(mu)
        else:
            latent = qpos.new_zeros(batch, self.config.latent_dim)

        pixels = (images.reshape(-1, *images.shape[2:]) - self.image_mean) / self.image_std
        with (nullcontext() if self.config.train_backbone else torch.no_grad()):
            features = self.backbone(pixels)
        features = self.visual_projection(features)
        _, dimension, height, width = features.shape
        # Official ACT shares one backbone and concatenates views along width.
        features = features.reshape(batch, len(self.config.cameras), dimension, height, width)
        features = features.permute(0, 2, 3, 1, 4).reshape(batch, dimension, height, -1)
        visual_tokens = features.flatten(2).transpose(1, 2)
        position = spatial_positions(height, width, dimension, features.device).to(features.dtype)
        position = position.repeat(1, len(self.config.cameras), 1).reshape(1, -1, dimension)
        special = torch.stack((self.latent_projection(latent), self.state_projection(qpos)), dim=1)
        memory = torch.cat((special, visual_tokens), dim=1)
        positions = torch.cat((self.special_positions.weight[None], position), dim=1)
        for layer in self.visual_encoder:
            memory = layer(memory, positions)
        queries = self.queries.weight[None].expand(batch, -1, -1)
        decoded = torch.zeros_like(queries)
        for layer in self.decoder:
            decoded = layer(decoded, memory, positions, queries)
        # Use the FINAL layer, so every configured decoder layer receives loss.
        return self.action_head(self.decoder_norm(decoded)), mu, logvar


def masked_l1(prediction, target, is_pad):
    """Official policy.py: zero padded errors, then mean over the whole B*K*D."""
    if is_pad.all():
        raise ValueError('An action batch cannot be entirely padding')
    valid = (~is_pad).unsqueeze(-1)
    return ((prediction.float() - target.float()).abs() * valid).mean()


def act_loss(prediction, target, is_pad, mu, logvar, kl_weight=10.):
    reconstruction = masked_l1(prediction, target, is_pad)
    kl = -.5 * (1. + logvar.float() - mu.float().square() - logvar.float().exp()).sum(-1).mean()
    return reconstruction + kl_weight * kl, reconstruction, kl
