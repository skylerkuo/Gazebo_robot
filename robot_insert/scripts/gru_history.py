"""Causal TCP/F/T tokens and zero-initialized action-to-history attention."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from contact_history import TCP_HISTORY, FT_HISTORY, VALID, TRACKING, TRACKING_VALID, COMMAND_HISTORY
from contact_se3 import residual, rotate


class GRUHistoryEncoder(nn.Module):
    def __init__(self, stats, hidden_dim=512, output_dim=512, tracking=False, raw_pose_pair=False, actual_pose_only=False):
        super().__init__()
        for name in ('ft_mean', 'ft_std', 'tcp_mean', 'tcp_std'):
            self.register_buffer(name, torch.tensor(stats[name], dtype=torch.float32))
        if sum((tracking,raw_pose_pair,actual_pose_only))>1:raise ValueError('Choose one history representation')
        self.tracking = tracking
        self.raw_pose_pair = raw_pose_pair
        self.actual_pose_only = actual_pose_only
        self.input_dim = 14 if actual_pose_only else 21 if raw_pose_pair else 19 if tracking else 13
        if raw_pose_pair:
            for name in ('command_xyz_mean','command_xyz_std'):
                value=torch.tensor(stats[name],dtype=torch.float32)
                if value.shape!=(3,) or not torch.isfinite(value).all():raise ValueError('Invalid command XYZ statistics')
                if name.endswith('std') and (value<=0).any():raise ValueError('Command XYZ std must be positive')
                self.register_buffer(name,value)
        if tracking:
            for name in ('tracking_mean', 'tracking_std'):
                value=torch.tensor(stats[name],dtype=torch.float32)
                if value.shape!=(6,) or not torch.isfinite(value).all():raise ValueError('Invalid tracking statistics')
                if name.endswith('std') and (value<=0).any():raise ValueError('Tracking std must be positive')
                self.register_buffer(name,value)
        self.gru = nn.GRU(self.input_dim, hidden_dim, num_layers=1, batch_first=True,
                          bidirectional=False, dropout=0.)
        self.token_projection = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)
        # Preserve the current absolute TCP features used by the CNN version.
        self.current_tcp_projection = nn.Linear(9, output_dim)

    def forward(self, batch):
        tcp = batch[TCP_HISTORY].float()
        ft = batch[FT_HISTORY].float()
        valid = batch[VALID]
        if (tcp.ndim != 3 or tcp.shape[-1] != 7 or ft.shape != (*tcp.shape[:2], 6)
                or valid.shape != tcp.shape[:2] or valid.dtype != torch.bool):
            raise ValueError('History must be TCP [B,H,7], F/T [B,H,6], boolean valid [B,H]')
        if not torch.isfinite(tcp[valid]).all() or not torch.isfinite(ft[valid]).all():
            raise ValueError('Nonfinite valid TCP/F/T history')
        b, h = valid.shape
        if h < 1:
            raise ValueError('History cannot be empty')
        # Replace invalid values before any quaternion/SE(3) arithmetic. Invalid
        # padding must not affect either recurrent state or attention K/V.
        identity = tcp.new_tensor([0., 0., 0., 0., 0., 0., 1.])
        tcp = torch.where(valid[..., None], tcp, identity)
        ft = torch.where(valid[..., None], ft, 0.)
        time_ids = torch.arange(h, device=tcp.device)[None].expand(b, -1)
        last = time_ids.masked_fill(~valid, -1).max(1).values.clamp_min(0)
        current = tcp[torch.arange(b, device=tcp.device), last]
        if self.actual_pose_only:
            from contact_se3 import canonical
            actual_features=torch.cat(((tcp[...,:3]-self.tcp_mean)/self.tcp_std,canonical(tcp[...,3:])), -1)
            parts=[actual_features,(ft-self.ft_mean)/self.ft_std]
        elif self.raw_pose_pair:
            from contact_se3 import canonical
            command=batch[COMMAND_HISTORY].float()
            if command.shape!=tcp.shape or not torch.isfinite(command[valid]).all():
                raise ValueError('Published target history must be finite [B,H,7] on valid rows')
            command=torch.where(valid[...,None],command,identity)
            actual_features=torch.cat(((tcp[...,:3]-self.tcp_mean)/self.tcp_std,canonical(tcp[...,3:])), -1)
            command_features=torch.cat(((command[...,:3]-self.command_xyz_mean)/self.command_xyz_std,canonical(command[...,3:])), -1)
            parts=[actual_features,command_features,(ft-self.ft_mean)/self.ft_std]
        else:
            delta = residual(current[:, None].expand_as(tcp), tcp) / tcp.new_tensor([.01]*3 + [.1]*3)
            parts=[delta, (ft-self.ft_mean)/self.ft_std]
        if self.tracking:
            error=batch[TRACKING].float()
            tracking_valid=batch.get(TRACKING_VALID,valid)
            if tracking_valid.shape!=valid.shape or tracking_valid.dtype!=torch.bool or (tracking_valid & ~valid).any():
                raise ValueError('Tracking availability must be boolean and a subset of TCP/F/T validity')
            if error.shape!=(*tcp.shape[:2],6) or not torch.isfinite(error[tracking_valid]).all():
                raise ValueError('Tracking history must be finite [B,H,6] on valid rows')
            error=torch.where(tracking_valid[...,None],error,self.tracking_mean)
            parts.append(((error-self.tracking_mean)/self.tracking_std).masked_fill(~tracking_valid[...,None],0.))
        features = torch.cat((*parts,valid[...,None].float()),-1)
        features = features.masked_fill(~valid[..., None], 0.)
        # Compact valid samples in chronological order before the GRU. Merely
        # masking attention afterwards would let left padding alter its state.
        order = (time_ids + (~valid).long()*h).argsort(1)
        compact = features.gather(1, order[..., None].expand(-1, -1, self.input_dim))
        lengths = valid.sum(1).clamp_min(1).cpu()
        packed = pack_padded_sequence(compact, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=h)
        encoded = encoded.gather(1, order.argsort(1)[..., None].expand(-1, -1, encoded.shape[-1]))
        axes = current.new_tensor([[1., 0., 0.], [0., 1., 0.]])[None].expand(b, -1, -1)
        rotation = rotate(current[:, None, 3:].expand(-1, 2, -1), axes).flatten(1)
        pose = torch.cat(((current[:, :3]-self.tcp_mean)/self.tcp_std, rotation), -1)
        tokens = self.token_projection(encoded) + self.current_tcp_projection(pose)[:, None]
        return tokens.masked_fill(~valid[..., None], 0.), valid


class HistoryCrossAttention(nn.Module):
    def __init__(self, width, heads=8):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, query, tokens, valid):
        # An entirely missing history is a data-validity case, not a contact or
        # phase gate. Give softmax one safe dummy key, then discard its update.
        present = valid.any(1)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~present
        memory = self.memory_norm(tokens.masked_fill(~valid[..., None], 0.))
        update, _ = self.attention(self.query_norm(query), memory, memory,
                                   key_padding_mask=~safe_valid, need_weights=False)
        update = torch.where(present[:, None, None], update, 0.)
        return query + update
