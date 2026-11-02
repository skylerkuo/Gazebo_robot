"""Frozen ACT + supervised residual FM on complete successful expert windows.

ACT--Flow (arXiv:2610.29117), warmup Eq. (4) and successful-action FM Eq. (7).
No critic, online behavior absorption, candidate weighting or privileged input.
TCP normalization/composition is an explicit adaptation of the paper's actions.
"""
import copy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from act_multiview import (MultiViewACT, MultiViewConfig, MultiViewDataset,
    FOUR_CAMERAS, ROLLOUT_INDEX_FORMAT, read_index)
from flow_multiview import (MultiViewFlow, MultiViewFlowConfig, masked_velocity_mse,
    initialize_flow_encoder_from_act)
from flow_matching import anchor_noise, integrate_velocity
from ftact import STATE, unit_tcp_quaternions

FORMAT = 'aic-sfp0-supervised-residual-flow-multiview-v1'
PAPER = 'https://arxiv.org/abs/2610.29117'
ALTERNATING_FORMAT = 'aic-sfp0-alternating-act-residual-flow-multiview-v1'


@dataclass
class ResidualFlowConfig(MultiViewFlowConfig):
    policy_type: str = 'residual_flow'
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_residual_flow_expert_v1'
    sigma_min: float = 0.
    residual_scale: float = 1.
    base_act_config: dict | None = None
    base_act_sha256: str | None = None
    base_act_step: int | None = None
    new_dataset: bool = False
    flow_encoder_init_from_act: bool = True

    def validate(self, *, legacy_inference=False):
        super().validate(legacy_inference=legacy_inference)
        if self.policy_type!='residual_flow':raise ValueError('Select the residual_flow policy type')
        if not math.isfinite(self.residual_scale) or self.residual_scale<=0:
            raise ValueError('residual_scale must be finite and positive')
        if not self.expert_only or self.retain_model_motion or self.correction_repeat!=1:
            raise ValueError('Residual supervision uses expert-only complete windows, once each')
        if not isinstance(self.flow_encoder_init_from_act,bool):
            raise ValueError('flow_encoder_init_from_act must be a boolean')
        if not self.flow_encoder_init_from_act and self.pretrained_backbone:
            raise ValueError('Random Flow initialization cannot load an ImageNet backbone')


def bind_base_act(config, payload):
    """Bind architecture and provenance once, before constructing the two models."""
    if payload.get('format') in (ALTERNATING_FORMAT,FORMAT):
        if (payload.get('smoke_test') or payload.get('camera_order')!=list(FOUR_CAMERAS)
                or payload.get('action_offset')!=1):
            raise ValueError('Select a real four-view ACT + Flow pair')
        for key in ('dim_model','n_heads','dim_feedforward','n_encoder_layers','n_decoder_layers',
                    'latent_dim','chunk_size','dropout','position_scale_m','residual_scale'):
            if getattr(config,key)!=payload['config'][key]:
                raise ValueError(f'Keep {key} unchanged when preserving a trained ACT + Flow pair')
        config.base_act_config=copy.deepcopy(payload['config']['base_act_config'])
        config.base_act_sha256=payload['_initialization_sha256']
        config.base_act_step=payload['step']
        return
    if payload.get('format')!='aic-sfp0-act-multiview-v1' or payload.get('smoke_test'):
        raise ValueError('Residual base must be a real four-view ACT checkpoint')
    config.base_act_config=copy.deepcopy(payload['config'])
    config.base_act_sha256=payload['_initialization_sha256']
    config.base_act_step=payload['step']


def load_residual_index(config, initial=None, resume=None, *, verify_files=True):
    if config.index_path=='auto':
        if not config.new_dataset or initial is None or resume is not None:
            raise ValueError('Residual Flow requires an existing frozen index_path or explicit new_dataset initialization')
        from prepare_expert_only import prepare_expert_only
        index=prepare_expert_only(config.dataset_root,initial,eval_samples=config.eval_samples,
            seed=config.seed,retain_model_motion=True,progress=True)
        config.index_path=str(Path(index['_directory'])/'index.json')
    else:index=read_index(config.index_path,verify_files=verify_files)
    return validate_residual_index(config,index,initial,resume)


def validate_residual_index(config,index,initial=None,resume=None):
    """Check the run contract of an index already fully validated by read_index."""
    if index['format']!=ROLLOUT_INDEX_FORMAT:
        raise ValueError('Residual Flow requires the retained assisted-rollout snapshot')
    reference=initial if initial is not None else resume
    if reference is not None:
        if reference['stats']!=index['stats']:
            raise ValueError('Frozen ACT and residual training must use the same snapshot and statistics')
        if reference['data_index_sha256']!=index['index_content_sha256']:
            if (not config.new_dataset or resume is not None or initial is None
                    or index.get('normalization_parent_index_sha256')!=initial['data_index_sha256']):
                raise ValueError('Frozen ACT and residual training must use the same snapshot and statistics')
    if index['validation_samples']!=config.eval_samples or Path(index['dataset_root'])!=Path(config.dataset_root).resolve():
        raise ValueError('Residual dataset or fixed VAL settings differ')
    return index


def residual_readiness(config):
    if config.index_path=='auto' and config.new_dataset:
        from prepare_expert_only import inspect_expert_collection
        data=inspect_expert_collection(config.dataset_root,eval_samples=config.eval_samples,
            seed=config.seed,retain_model_motion=True)
        return dict(data['report'],frozen_snapshot=False,base_checkpoint=config.init_from,
            train_expert_windows=sum(len(data['rollout_data'][i][3]) for i in data['train']),
            supervision='complete successful expert-recovery windows; ACT frozen')
    index=load_residual_index(config,verify_files=False)
    train=[e for e in index['episodes'] if e['split']=='train']
    val=[e for e in index['episodes'] if e['split']=='validation']
    count=sum(len(e['expert_validation_frames']) for e in train)
    return {'accepted_episodes':len(index['episodes']),'train_episodes':len(train),
        'validation_episodes':len(val),'train_expert_windows':count,
        'validation_samples':len(index['validation_anchors']),'ready':count>0,
        'base_checkpoint':config.init_from,'data_index_sha256':index['index_content_sha256'],
        'supervision':'only complete expert future-20 windows; ACT frozen'}


class ResidualExpertDataset(MultiViewDataset):
    def __init__(self,index,split,*,correction_repeat=1):
        super().__init__(index,split,correction_repeat=correction_repeat)
        eligible={e['episode_index']:set(e['expert_validation_frames']) for e in self.episodes.values()}
        if split=='train':
            self.anchors=[[ep,f] for ep in self.episodes for f in sorted(eligible[ep])]
        if not self.anchors or any(f not in eligible[ep] for ep,f in self.anchors):
            raise ValueError('Residual dataset needs complete successful expert action windows')

    def __getitem__(self,i):
        sample=super().__getitem__(i)
        ep,frame=self.anchors[i];a=self.numeric[ep];future=slice(frame+1,frame+21)
        if (not np.all(a['action_source'][future]=='expert')
                or not np.all(a['action_segment'][future]==a['action_segment'][frame+1])
                or not np.allclose(np.diff(a['simulation_time'][frame:frame+21]),.05,rtol=0,atol=1e-5)
                or not bool(sample.get('action_mask',torch.ones(20,dtype=torch.bool)).all())):
            raise ValueError('Residual target crosses model/hold/segment/gap or an excluded target')
        return sample


class SupervisedResidualFlow(MultiViewFlow):
    def __init__(self,config,stats,*,initialize_backbone=True):
        super().__init__(config,stats,initialize_backbone=initialize_backbone)
        if config.base_act_config is None:raise ValueError('Missing embedded frozen ACT configuration')
        base_config=MultiViewConfig(**copy.deepcopy(config.base_act_config))
        base_config.device=config.device;base_config.amp=config.amp
        base_config.validate(legacy_inference=True)
        self.base_act=MultiViewACT(base_config,stats,initialize_backbone=False)
        self.base_act.requires_grad_(False).eval()
        self.policy.model.flow_base_action_input_proj=nn.Linear(7,config.dim_model)
        self.residual_enabled=True

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'base_act'):self.base_act.eval()
        return self

    @torch.no_grad()
    def base_normalized_action(self,batch):
        # Same current four views/state/FT; no expert actions or future signals.
        return self.normalizer.actions(self.base_act.predict_action_chunk(batch)).detach()

    def encode_residual_condition(self,batch,base):
        memory,positions=self.encode_observations(batch)
        core=self.policy.model
        tokens=core.flow_base_action_input_proj(base).transpose(0,1)
        base_positions=core.decoder_pos_embed.weight[:,None].expand(-1,positions.shape[1],-1)
        return torch.cat((memory,tokens),dim=0),torch.cat((positions,base_positions),dim=0)

    def residual_target(self,actions,base):
        target=unit_tcp_quaternions(actions)
        base_physical=self.normalizer.unnormalize_actions(base)
        dot=(target[...,3:]*base_physical[...,3:]).sum(-1,keepdim=True)
        target[...,3:]=torch.where(dot<0,-target[...,3:],target[...,3:])
        return (self.normalizer.actions(target)-base)/self.flow_config.residual_scale

    def forward(self,batch):
        target=batch['action']
        mask=batch.get('action_mask',torch.ones(target.shape[:-1],device=target.device,dtype=torch.bool))
        if (target.shape[1:]!=(20,7) or mask.dtype!=torch.bool or mask.shape!=target.shape[:-1]
                or not mask.all() or batch.get('action_is_pad',torch.zeros(1)).any()):
            raise ValueError('Residual SFM requires complete valid expert future-20 actions')
        base=self.base_normalized_action(batch)
        residual=self.residual_target(target,base).detach()
        noise=torch.randn_like(residual);tau=torch.rand(len(residual),device=residual.device)[:,None,None]
        x=(1-(1-self.flow_config.sigma_min)*tau)*noise+tau*residual
        velocity_target=residual-(1-self.flow_config.sigma_min)*noise
        prediction=self.velocity(x,tau[:,0,0],self.encode_residual_condition(batch,base))
        loss=masked_velocity_mse(prediction,velocity_target,mask)
        physical_delta=(residual[...,:3]*self.flow_config.residual_scale*self.normalizer.action_std[:3])
        return loss,{'flow_mse':loss.detach().item(),'flow_valid_steps':mask.sum().item(),
            'teacher_position_correction_mm':physical_delta.norm(dim=-1).mean().item()*1000}

    @torch.no_grad()
    def normalized_action_chunk(self,batch,*,noise=None,steps=None,solver=None):
        base=self.base_normalized_action(batch)
        if not self.residual_enabled:return base
        shape=(len(batch[STATE]),self.flow_config.chunk_size,7)
        if noise is None:noise=torch.randn(shape,generator=self.noise_generator,device='cpu')
        if tuple(noise.shape)!=shape or not torch.isfinite(noise).all():raise ValueError('Invalid residual noise')
        noise=noise.to(device=base.device,dtype=torch.float32)
        condition=self.encode_residual_condition(batch,base)
        residual=integrate_velocity(lambda x,t:self.velocity(x,t,condition),noise,
            self.sampling_steps if steps is None else steps,self.sampling_solver if solver is None else solver)
        return base+self.flow_config.residual_scale*residual

    def validation_metadata(self):
        return {**super().validation_metadata(),'generation':'ACT_plus_supervised_residual_flow',
            'base_act_frozen':True,'base_act_step':self.flow_config.base_act_step,
            'base_act_sha256':self.flow_config.base_act_sha256,'residual_enabled':self.residual_enabled,
            'residual_scale':self.flow_config.residual_scale}


def initialize_residual_from_act(model,payload):
    random_flow=not model.flow_config.flow_encoder_init_from_act
    if random_flow and payload.get('format')!='aic-sfp0-act-multiview-v1':
        raise ValueError('A fresh random Flow must be paired with an ACT-only checkpoint')
    if payload.get('format')==ALTERNATING_FORMAT:
        counters={'act_update_count','flow_update_count'}
        weights={name:value for name,value in payload['model'].items() if name not in counters}
        model.load_state_dict(weights,strict=True)
        model.base_act.requires_grad_(False).eval()
        return {'scope':'full trained ACT + Flow pair retained; ACT frozen; only Flow fine-tuned',
            'source_format':ALTERNATING_FORMAT,'frozen_base_step':payload['step'],
            'flow_weights_preserved':True,'optimizer_restarted':True}
    model.base_act.load_state_dict(payload['model'],strict=True)
    report=({'flow_initialization':'fully_random','encoder_copied_from_act':False,
             'flow_weights_preserved':False}
            if random_flow else initialize_flow_encoder_from_act(model,payload['model']))
    model.base_act.requires_grad_(False).eval()
    return {**report,'frozen_base_step':payload['step'],
        'scope':'full frozen ACT plus independently random Flow' if random_flow else
                'full frozen ACT plus trainable residual observation encoder'}


def load_residual_checkpoint(payload,device):
    config=ResidualFlowConfig(**payload['config']);config.device=device
    if device=='cpu':config.amp='none'
    config.validate()
    if payload.get('camera_order')!=list(FOUR_CAMERAS) or payload.get('action_offset')!=1:
        raise ValueError('Residual camera/action contract differs')
    model=SupervisedResidualFlow(config,payload['stats'],initialize_backbone=False)
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),payload
