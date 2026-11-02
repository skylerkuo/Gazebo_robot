"""Single-task ACT. Three RGBs + state6/wrench6; 20 future absolute TCP poses."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ftact import (TrainConfig, ForceTorqueACT, AICDataset, STATE, WRENCH, CAMERAS,
                   read_data_contract, unit_tcp_quaternions)
import torch
from torch import nn


@dataclass
class SFPConfig(TrainConfig):
    index_path: str = '/mnt/d/robot_datasets/aic_sfp0_v2/index.json'
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_act_v2'
    eval_samples: int = 1200
    num_workers: int = 2
    steps: int = 30000
    grad_clip_norm: float = 1.
    warmup_steps: int = 500
    minimum_lr_ratio: float = .1

    def validate(self):
        super().validate()
        if self.eval_samples != 1200 or self.chunk_size != 20:
            raise ValueError('SFP0 v2 requires exactly 1200 validation anchors and a 20-step chunk')
        if self.warmup_steps < 0 or not 0 < self.minimum_lr_ratio <= 1:
            raise ValueError('Invalid learning-rate schedule')


def read_index(path):
    manifest = json.loads(Path(path).read_text())
    digest = manifest.pop('index_content_sha256')
    actual = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['index_content_sha256'] = digest
    if manifest['format'] != 'aic-sfp0-index-v2' or actual != digest:
        raise ValueError('Unrecognized or changed SFP0 data index')
    if manifest['task']['target_module'] != 'nic_card_mount_0':
        raise ValueError('This trainer is scoped to nic_card_mount_0')
    _, _, contract = read_data_contract(manifest['dataset_root'])
    if contract != manifest['source_contract']:
        raise ValueError('Source metadata changed after preparing the task index')
    seen = {}
    for segment in manifest['segments']:
        source = segment['source_episode_index']
        if source in seen:
            raise ValueError('Repeated source episode in the task index')
        seen[source] = segment['split']
        if not 0 <= segment['start'] < segment['end'] <= segment['episode_length']:
            raise ValueError('Invalid task segment bounds')
    val = manifest['validation_anchors']
    if len(val) != 1200 or len(set(map(tuple, val))) != 1200:
        raise ValueError('VAL must contain 1200 distinct fixed anchors')
    segments = {s['episode_index']: s for s in manifest['segments']}
    for episode, frame in val:
        s = segments[episode]
        if s['split'] != 'validation' or not s['start'] <= frame < s['end'] - 20:
            raise ValueError('VAL anchor outside its held-out task segment')
    return manifest


class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split, *, smoke=False):
        self.segments = sorted([s for s in manifest['segments'] if s['split'] == split],
                               key=lambda s: s['episode_index'])
        # Smoke limits training only. Validation always uses the real fixed 1200 anchors.
        if smoke and split == 'train':
            self.segments = self.segments[:2]
        self.base = AICDataset(manifest['dataset_root'], [s['episode_index'] for s in self.segments], 21)
        offsets, offset = {}, 0
        for s in self.segments:
            offsets[s['episode_index']] = offset
            offset += s['episode_length']
        if offset != len(self.base):
            raise ValueError('Official reader episode lengths differ from the task index')
        self.anchors = (manifest['validation_anchors'] if split == 'validation' else
            [[s['episode_index'], f] for s in self.segments for f in range(s['start'], s['end'] - 20)])
        self.indices = [offsets[e] + f for e, f in self.anchors]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.base[self.indices[index]]
        episode, frame = self.anchors[index]
        if int(sample['episode_index']) != episode or int(sample['frame_index']) != frame:
            raise ValueError('Official reader index does not match the fixed task anchor')
        # Observation at t -> target commands at t+0.05 ... t+1.00 seconds.
        sample['action'] = sample['action'][1:]
        sample['action_is_pad'] = sample['action_is_pad'][1:]
        if sample['action_is_pad'].any():
            raise ValueError('Task windows must be complete and must never cross a scene boundary')
        return sample


class Float32Linear(nn.Linear):
    def forward(self, value):
        with torch.autocast(value.device.type, enabled=False):
            return nn.functional.linear(value.float(), self.weight.float(), self.bias.float())


class SFP0ACT(ForceTorqueACT):
    def __init__(self, config, stats, *, initialize_backbone=True, cameras=CAMERAS):
        super().__init__(config, stats, initialize_backbone=initialize_backbone, use_vae=False, cameras=cameras)
        # The architecture keeps the latent input token at zero in BOTH train and inference.
        core = self.policy.model
        head = core.action_head
        precise_head = Float32Linear(head.in_features, head.out_features)
        precise_head.load_state_dict(head.state_dict())
        core.action_head = precise_head

    def forward(self, batch):
        prepared = self.normalizer.observations(batch)
        # Call the official ACT core without actions; it cannot use the answers as input.
        prepared['observation.images'] = [prepared[key] for key in self.cameras]
        prediction = self.policy.model(prepared)[0]
        target = self.normalizer.actions(batch['action'])
        loss = (prediction - target).abs().mean()
        return loss, {'l1_loss': loss.detach().item()}


def load_sfp_checkpoint(payload, device):
    config = SFPConfig(**payload['config'])
    config.device = device
    if device == 'cpu':
        config.amp = 'none'
    config.validate()
    model = SFP0ACT(config, payload['stats'], initialize_backbone=False)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), payload
