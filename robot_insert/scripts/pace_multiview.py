"""Four-view ACT + existing PACE Phase/Event on causal assisted rollouts."""
from dataclasses import dataclass
import math

import numpy as np
import torch

from act_multiview import MultiViewConfig,MultiViewDataset,FOUR_CAMERAS
from ftact import WRENCH
from pace_act import PACEACT,TactileContext,PHASE,HISTORY

FORMAT='aic-sfp0-act-pace-multiview-v1'


@dataclass
class MultiViewPACEConfig(MultiViewConfig):
    policy_type: str = 'pace'
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_pace_expert_v1'
    phase_alpha: float = .05
    tactile_history: int = 4
    correction_horizon: int = 5

    def validate(self,*,legacy_inference=False):
        super().validate(legacy_inference=legacy_inference)
        if self.policy_type not in ('pace','pace_pred'):raise ValueError('PACE config requires a PACE policy type')
        if not math.isfinite(self.phase_alpha) or not 0<self.phase_alpha<=1:raise ValueError('Invalid PACE EMA alpha')
        if isinstance(self.tactile_history,bool) or not isinstance(self.tactile_history,int) or self.tactile_history<2:
            raise ValueError('PACE tactile_history must be an integer >= 2')
        if isinstance(self.correction_horizon,bool) or not isinstance(self.correction_horizon,int) or not 1<=self.correction_horizon<=20:
            raise ValueError('PACE correction_horizon must fit inside the action chunk')


class MultiViewPACEDataset(MultiViewDataset):
    """Same anchors/masks as ACT; context resets per episode, not at handoff."""
    def __init__(self,index,split,*,correction_repeat=1,alpha=.05,history=4):
        super().__init__(index,split,correction_repeat=correction_repeat)
        self.alpha,self.history=alpha,history
        self.contexts={}

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        ep,frame=self.anchors[index]
        if ep not in self.contexts:
            numeric=self.numeric[ep]
            wrench=numeric[WRENCH]
            times=numeric['simulation_time']
            states=np.empty_like(wrench,dtype=np.float32)
            histories=np.empty((len(wrench),self.history,6),dtype=np.float32)
            context=TactileContext(self.alpha,self.history)
            for i,value in enumerate(wrench):
                if i and times[i]-times[i-1]>1.:context.reset()
                features=context.update(value,float(times[i]))
                states[i]=features[PHASE];histories[i]=features[HISTORY]
            self.contexts[ep]=(states,histories)
        states,histories=self.contexts[ep]
        sample[PHASE]=torch.from_numpy(states[frame].copy())
        sample[HISTORY]=torch.from_numpy(histories[frame].copy())
        if not torch.equal(sample[HISTORY][-1],sample[WRENCH]):raise ValueError('PACE history is not causal/aligned')
        return sample


class MultiViewPACE(PACEACT):
    def __init__(self,config,stats,*,initialize_backbone=True):
        super().__init__(config,stats,initialize_backbone=initialize_backbone,cameras=FOUR_CAMERAS)


def initialize_pace_from_act(model,weights):
    expected={k for k in model.state_dict() if '.pace_' in k}
    result=model.load_state_dict(weights,strict=False)
    if result.unexpected_keys or set(result.missing_keys)!=expected:
        raise ValueError('PACE initialization may omit only newly added Phase/Event parameters')


def load_pace_multiview_checkpoint(payload,device):
    config=MultiViewPACEConfig(**payload['config']);config.device=device
    if device=='cpu':config.amp='none'
    config.validate()
    if payload.get('camera_order')!=list(FOUR_CAMERAS):raise ValueError('PACE four-view camera order differs')
    model=MultiViewPACE(config,payload['stats'],initialize_backbone=False)
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),payload
