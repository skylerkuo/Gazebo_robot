"""Physical action supervision for the first four receding-horizon commands."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from contact_se3 import rotation_error, rotate

PORT_AXIS = 'label.insertion_axis_base'
NEAR_PORT_DEMO = 'label.near_port_demo'
HORIZON = 4


def insertion_axis(pose):
    q = torch.tensor(pose[3:], dtype=torch.float32)
    return rotate(q, q.new_tensor([0., 0., 1.])).numpy()


def port_action_errors(prediction, target, axis):
    """Return lateral and axial errors in mm, and shortest angle in degrees."""
    delta = (prediction[..., :3].float() - target[..., :3].float()) * 1000
    axis = axis.float()[:, None]
    axial = (delta * axis).sum(dim=-1)
    lateral = (delta - axial[..., None] * axis).norm(dim=-1)
    rotation = rotation_error(prediction[..., 3:], target[..., 3:]) * (180 / math.pi)
    return torch.stack((lateral, axial.abs(), rotation), dim=-1)


def physical_action_loss(prediction, batch):
    errors = port_action_errors(
        prediction, batch['action'][:, :HORIZON], batch[PORT_AXIS])
    scales = errors.new_tensor([1., 2., .25])
    penalties = F.smooth_l1_loss(errors / scales, torch.zeros_like(errors), reduction='none')
    near = batch[NEAR_PORT_DEMO][:, None].float()
    weights = torch.stack((1 + near, torch.ones_like(near), 1 + near), dim=-1)
    step_weights = errors.new_tensor([.4, .3, .2, .1])
    return (((penalties * weights).sum(-1) / weights.sum(-1)) * step_weights).sum(-1).mean()


def direct_probe_index(index, count=3, per_episode=100):
    """Reserve whole direct episodes from this run while retaining legacy VAL anchors."""
    direct = sorted((ep for ep in index['episodes'] if ep['collection_group'] == 'new_direct'
                     and ep['split'] == 'train'), key=lambda ep: ep['episode_index'])
    reserved = [direct[i] for i in np.linspace(0, len(direct) - 1, count + 2, dtype=int)[1:-1]]
    ids = [ep['episode_index'] for ep in reserved]
    if len(set(ids)) != count:
        raise ValueError('Could not reserve distinct direct validation episodes')
    train = {**index, 'episodes': [ep for ep in index['episodes'] if ep['episode_index'] not in ids]}
    probes = []
    for episode in reserved:
        frames = episode['expert_validation_frames']
        if len(frames) < per_episode:
            raise ValueError('Direct validation episode has too few complete action windows')
        probes.extend([episode['episode_index'], frames[i]]
                      for i in np.linspace(0, len(frames) - 1, per_episode, dtype=int))
    validation = {**index, 'episodes': [{**ep, 'split': 'validation'} for ep in reserved],
                  'validation_anchors': probes}
    return train, validation, ids
