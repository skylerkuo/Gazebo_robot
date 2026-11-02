"""Original deterministic SFP0 ACT, four RGB views, common-scale physical XYZ loss."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ftact import CAMERAS, STATE, WRENCH, TrainConfig
from ftact_v2 import SFPConfig, SFP0ACT
from act_objectives import common_scale_loss

FOUR_CAMERAS = (*CAMERAS, 'observation.images.side_camera')
FORMAT = 'aic-sfp0-act-multiview-v1'
INDEX_FORMAT = 'aic-sfp0-act-multiview-index-v2'
CORRECTION_INDEX_FORMAT = 'aic-sfp0-act-corrections-index-v1'
EXPERT_ONLY_INDEX_FORMAT = 'aic-sfp0-act-expert-only-index-v1'
ROLLOUT_INDEX_FORMAT = 'aic-sfp0-act-assisted-rollout-index-v1'
DEFAULT_PREPARED = '/mnt/d/robot_datasets/aic_sfp0_multiview_v1/_training/act_auto_5pct_val1000'
VALIDATION_SAMPLES = 1000
SUPPORTED_VALIDATION_SAMPLES = (1000,1500,2000)


@dataclass
class MultiViewConfig(SFPConfig):
    dataset_root: str = '/mnt/d/robot_datasets/aic_sfp0_multiview_v1'
    index_path: str = 'auto'
    output_dir: str = '/mnt/d/robot_models/aic_sfp0_act_multiview_v1'
    eval_every: int = 500
    eval_samples: int = VALIDATION_SAMPLES
    save_every: int = 500
    learning_rate: float = 3e-5
    backbone_lr: float = 3e-6
    lr_schedule_steps: int | None = 0
    position_scale_m: float = .01
    init_from: str | None = None
    corrections_root: str | None = None
    correction_repeat: int = 1
    refresh_data: bool = False
    expert_only: bool = False
    retain_model_motion: bool = False
    policy_type: str = 'act'
    comparison_index_path: str | None = None

    def validate(self, *, legacy_inference=False):
        # SFPConfig's original three-camera contract fixes VAL at 1200.
        # Keep its training/schedule checks; preserve old 1000-anchor checkpoints too.
        TrainConfig.validate(self)
        if self.policy_type not in ('act','pace','pace_pred','flow','residual_flow','stacked_residual_flow','alternating_residual_flow'):raise ValueError('Unknown comparison policy type')
        if self.chunk_size != 20 or (self.eval_samples not in SUPPORTED_VALIDATION_SAMPLES
                and not (legacy_inference and self.eval_samples == 1200)):
            raise ValueError('Four-view ACT requires 1000, 1500 or 2000 validation anchors and a 20-step chunk')
        if self.warmup_steps < 0 or not 0 < self.minimum_lr_ratio <= 1:
            raise ValueError('Invalid learning-rate schedule')
        if self.lr_schedule_steps is not None and (not isinstance(self.lr_schedule_steps,int) or self.lr_schedule_steps<0):
            raise ValueError('lr_schedule_steps must be a nonnegative integer')
        for key in ('learning_rate','backbone_lr'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<=0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.position_scale_m) or self.position_scale_m <= 0:
            raise ValueError('position_scale_m must be finite and positive')
        if not isinstance(self.correction_repeat,int) or self.correction_repeat<1:
            raise ValueError('correction_repeat must be a positive integer')


def index_digest(value):
    payload = {k:v for k,v in value.items() if k != 'index_content_sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def split_episodes(episodes, seed=42, *, validation_candidates=None):
    if len(episodes) < 2 or len(set(episodes)) != len(episodes):
        raise ValueError('Need distinct source episodes for a grouped split')
    order = np.random.default_rng(seed).permutation(sorted(episodes)).tolist()
    val_count = max(1, math.ceil(len(order)*.05))
    eligible=set(episodes if validation_candidates is None else validation_candidates)
    if not eligible.issubset(episodes):raise ValueError('VAL candidates must belong to the source episodes')
    held_out=[e for e in order if e in eligible][:val_count]
    if len(held_out)<val_count:raise ValueError(f'Need {val_count} expert-eligible VAL episodes for the 5% split; only {len(held_out)} available')
    return sorted(set(episodes)-set(held_out)),sorted(held_out)


def validation_anchors(episodes, count=VALIDATION_SAMPLES, seed=10042):
    rng = np.random.default_rng(seed)
    queues = [(e['episode_index'], rng.permutation(e.get('train_anchor_frames',list(range(e['length']-20)))).tolist()) for e in episodes]
    result = []
    while len(result) < count:
        changed = False
        for episode, frames in queues:
            if frames:
                result.append([episode,int(frames.pop())]);changed = True
                if len(result)==count: break
        if not changed:
            raise ValueError(f'Only {len(result)} complete VAL anchors; need {count}')
    return result


def extended_split(sources,parent,seed):
    """Choose the entire new holdout from episodes absent from the parent snapshot."""
    current={row[0]:row[1:] for row in sources}
    previous=parent['preparation_settings']['sources']
    if any(current.get(row[0])!=row[1:] for row in previous):
        raise ValueError('Parent episodes were removed or changed; cannot safely extend this split')
    old_ids={row[0] for row in previous}
    new_ids=sorted(set(current)-old_ids)
    count=math.ceil(len(current)*.05)
    if count>len(new_ids):raise ValueError(f'Need at least {count} new episodes to form a new 5% VAL; only {len(new_ids)} available')
    held_out=set(np.random.default_rng(seed).permutation(new_ids).tolist()[:count])
    return sorted(set(current)-held_out),sorted(held_out)


def training_statistics(episodes):
    result = {}
    for key,width,floor in ((STATE,6,[.001]*6),(WRENCH,6,[.5]*3+[.05]*3),('action',7,[.01]*7)):
        values = np.concatenate([e[key] for e in episodes]).astype(np.float64)
        if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
            raise ValueError(f'Invalid training values: {key}')
        raw = values.std(0)
        result[key] = {'mean':values.mean(0).tolist(),'std':np.maximum(raw,floor).tolist(),
            'min':values.min(0).tolist(),'max':values.max(0).tolist(),
            'raw_std':raw.tolist(),'std_floor':floor,'count':[len(values)]}
    for key in FOUR_CAMERAS:
        result[key] = {'mean':[[[.485]],[[.456]],[[.406]]], 'std':[[[.229]],[[.224]],[[.225]]]}
    return result


def read_index(path, *, verify_files=True, progress=False):
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get('format') not in (INDEX_FORMAT,CORRECTION_INDEX_FORMAT,EXPERT_ONLY_INDEX_FORMAT,ROLLOUT_INDEX_FORMAT) or value.get('index_content_sha256') != index_digest(value):
        raise ValueError('Changed or unrecognized four-view ACT index')
    if value['cameras'] != list(FOUR_CAMERAS) or value['fps'] != 20 or value['action_offset'] != 1:
        raise ValueError('Wrong camera order, rate, or future-action offset')
    episodes = value['episodes']
    rollout=value['format']==ROLLOUT_INDEX_FORMAT
    expert_only=value['format'] in (EXPERT_ONLY_INDEX_FORMAT,ROLLOUT_INDEX_FORMAT)
    fresh_normalization=expert_only and value.get('normalization_source')=='current training episodes only'
    if fresh_normalization and (value.get('normalization_parent_index_sha256') is not None
            or value.get('preparation_settings',{}).get('normalization_source')!='current training episodes only'):
        raise ValueError('Fresh normalization cannot inherit checkpoint statistics')
    if expert_only and not fresh_normalization and len(value.get('normalization_parent_index_sha256',''))!=64:
        raise ValueError('Expert-only index must identify the parent normalization source')
    if len(episodes)<2:
        raise ValueError('Need at least two complete source episodes')
    ids = [e['episode_index'] for e in episodes]
    if len(set(ids)) != len(ids) or len({e['source_attempt'] for e in episodes}) != len(episodes):
        raise ValueError('Repeated source episode/attempt')
    settings=value['preparation_settings']
    if settings['episodes']!=len(episodes) or settings['sources']!=[
            [e['episode_index'],e['source_attempt'],e['source_sha256']] for e in episodes]:
        raise ValueError('Frozen source membership differs from the preparation snapshot')
    train = [e for e in episodes if e['split']=='train']
    val = [e for e in episodes if e['split']=='validation']
    if len(val) != math.ceil(len(episodes)*.05) or len(train)+len(val) != len(episodes):
        raise ValueError('Split must reserve 5% of whole episodes for VAL (rounded up)')
    parent=None
    if value.get('transfer'):
        transfer=value['transfer']
        parent_path=Path(transfer['parent_index_path']).resolve()
        if parent_path==path.resolve():raise ValueError('Transfer index cannot reference itself')
        parent=read_index(parent_path,verify_files=False)
        if (parent['index_content_sha256']!=transfer['parent_index_sha256'] or
                Path(parent['dataset_root']).resolve()!=Path(value['dataset_root']).resolve()):
            raise ValueError('Transfer parent index/source collection changed')
        expected_train,expected_val=extended_split(settings['sources'],parent,settings['seed'])
        if value['stats']!=parent['stats']:raise ValueError('Transferred weights require frozen parent normalization statistics')
    else:
        candidates=settings.get('validation_candidate_episodes')
        if candidates is not None:
            if not rollout or candidates!=sorted(e['episode_index'] for e in episodes if e['expert_validation_frames']):
                raise ValueError('VAL candidates must be exactly the expert-eligible episodes')
        expected_train,expected_val=split_episodes(ids,settings['seed'],validation_candidates=candidates)
    if sorted(e['episode_index'] for e in train)!=expected_train or sorted(e['episode_index'] for e in val)!=expected_val:
        raise ValueError('Episode split differs from the frozen seed')
    if fresh_normalization and value.get('normalization_training_episode_ids')!=expected_train:
        raise ValueError('Fresh normalization must use exactly the training episodes')
    if value['validation_samples'] not in SUPPORTED_VALIDATION_SAMPLES or value['validation_fraction']!=.05:
        raise ValueError('Expected 5% whole-episode VAL and 1000, 1500 or 2000 fixed anchors')
    if value['input_shape'] != [3,256,288] or value['chunk_size'] != 20:
        raise ValueError('Wrong image shape or prediction horizon')
    lookup = {e['episode_index']:e for e in episodes}
    anchors = value['validation_anchors']
    if len(anchors)!=value['validation_samples'] or len(set(map(tuple,anchors)))!=value['validation_samples']:
        raise ValueError(f"Exactly {value['validation_samples']} fixed unique VAL anchors are required")
    for episode,frame in anchors:
        e = lookup[episode]
        if e['split']!='validation' or not 0<=frame<e['length']-20:
            raise ValueError('VAL anchor leaks into training or crosses an episode boundary')
        if expert_only and frame not in e['train_anchor_frames']:
            raise ValueError('Expert-only VAL contains a model/hold action window')
        if rollout and frame not in e['expert_validation_frames']:raise ValueError('VAL must use expert-only targets')
    for key in (() if parent is not None or (expert_only and not fresh_normalization) else (STATE,WRENCH,'action')):
        if value['stats'][key]['count'] != [sum(e['length'] for e in train)]:
            raise ValueError('Statistics do not come from training episodes only')
    corrections=value.get('correction_episodes',[])
    if corrections and value['format']!=CORRECTION_INDEX_FORMAT:raise ValueError('Unexpected correction episodes')
    correction_ids=[e['episode_index'] for e in corrections]
    if set(correction_ids)&set(ids) or len(set(correction_ids))!=len(correction_ids):
        raise ValueError('Repeated base/correction episode index')
    for e in corrections:
        frames=e['train_anchor_frames']
        if e['split']!='train' or len(frames)!=len(set(frames)) or not frames:
            raise ValueError('Correction windows must be unique training-only anchors')
        if any(not 0<=f<e['length']-20 for f in frames):raise ValueError('Correction anchor crosses episode boundary')
    audited=[*episodes,*corrections]
    if progress:
        print(json.dumps({'type':'cache_verification','completed':0,'total':len(audited),
            'full_checksums':verify_files}),flush=True)
    for number,e in enumerate(audited,1):
        for kind in (('images','numeric','motion') if rollout else ('images','numeric')):
            artifact = path.parent/e[kind]['path']
            if not artifact.is_file() or artifact.stat().st_size != e[kind]['bytes']:
                raise ValueError(f'Missing/truncated {kind} cache for episode {e["episode_index"]}')
            if verify_files and file_digest(artifact) != e[kind]['sha256']:
                raise ValueError(f'Changed {kind} cache for episode {e["episode_index"]}')
        if e in corrections or expert_only:
            if expert_only and e.get('expert_source')!='expert':raise ValueError('Expert-only episode has wrong action provider')
            from human_control import expert_anchor_frames
            with np.load(path.parent/e['numeric']['path'],allow_pickle=False) as arrays:
                if rollout:
                    from rollout_sampling import rollout_windows
                    with np.load(path.parent/e['motion']['path'],allow_pickle=False) as motion:
                        weights,expected,expected_val,_=rollout_windows(arrays['action_source'],arrays['action_segment'],arrays['human_active'],
                            arrays['simulation_time'],arrays['action'],arrays[WRENCH],motion['tcp_pose'],e['rollout_filter'])
                        if e['rollout_filter']['version']>=2:
                            mask=motion['action_mask']
                            correct=mask.dtype==np.bool_ and np.array_equal(weights>0,mask)
                        else:correct=np.array_equal(weights,motion['action_weights'])
                        if not correct or expected_val!=e['expert_validation_frames']:
                            raise ValueError('Rollout action weighting or expert-only validation changed')
                else:expected=expert_anchor_frames(arrays['action_source'],arrays['action_segment'],arrays['human_active'],arrays['simulation_time'],source=e.get('expert_source','human'))
                if not rollout and e.get('idle_filter'):
                    from idle_sampling import filter_idle_anchors
                    expected,_=filter_idle_anchors(expected,arrays['action'],arrays[WRENCH],arrays['simulation_time'],e['idle_filter'])
            if expected!=e['train_anchor_frames']:raise ValueError('Correction labels include model/hold/mixed or discontinuous actions')
        if progress and (number==1 or number%10==0 or number==len(audited)):
            print(json.dumps({'type':'cache_verification','completed':number,'total':len(audited),
                'episode':e['episode_index'],'full_checksums':verify_files}),flush=True)
    value['_directory'] = str(path.parent.resolve())
    return value


class MultiViewDataset(Dataset):
    """Small numeric arrays + bounded file descriptors; no full RGB Arrow cache in RAM."""
    def __init__(self, index, split, *, correction_repeat=1):
        self.root = Path(index['_directory'])
        base=[e for e in index['episodes'] if e['split']==split]
        corrections=index.get('correction_episodes',[]) if split=='train' else []
        self.episodes = {e['episode_index']:e for e in [*base,*corrections]}
        self.anchors = (index['validation_anchors'] if split=='validation' else
            [[e['episode_index'],f] for e in base for f in e.get('train_anchor_frames',range(e['length']-20))]+
            [[e['episode_index'],f] for e in corrections for f in e['train_anchor_frames']]*correction_repeat)
        self.numeric = {}
        self.descriptors = OrderedDict()

    def __len__(self): return len(self.anchors)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['numeric'], state['descriptors'] = {}, OrderedDict()
        return state

    def close(self):
        for fd in self.descriptors.values(): os.close(fd)
        self.descriptors.clear()

    def __del__(self):
        if hasattr(self,'descriptors'): self.close()

    def __getitem__(self, index):
        ep,frame = self.anchors[index]
        e = self.episodes[ep]
        if ep not in self.numeric:
            with np.load(self.root/e['numeric']['path'],allow_pickle=False) as archive:
                self.numeric[ep] = {k:archive[k].copy() for k in archive.files}
            if e.get('motion'):
                with np.load(self.root/e['motion']['path'],allow_pickle=False) as motion:
                    if 'action_mask' in motion.files:self.numeric[ep]['__action_mask']=motion['action_mask'].copy()
                    else:self.numeric[ep]['__action_weights']=motion['action_weights'].copy()
        values = self.numeric[ep]
        if ep not in self.descriptors:
            self.descriptors[ep] = os.open(self.root/e['images']['path'],os.O_RDONLY)
        self.descriptors.move_to_end(ep)
        while len(self.descriptors)>8: os.close(self.descriptors.popitem(last=False)[1])
        result = {STATE:torch.from_numpy(values[STATE][frame].copy()),
            WRENCH:torch.from_numpy(values[WRENCH][frame].copy()),
            'action':torch.from_numpy(values['action'][frame+1:frame+21].copy()),
            'action_is_pad':torch.zeros(20,dtype=torch.bool),
            'episode_index':torch.tensor(ep),'frame_index':torch.tensor(frame)}
        if result['action'].shape!=(20,7): raise ValueError('Future chunk crosses an episode boundary')
        if '__action_weights' in values:
            result['action_weights']=torch.from_numpy(values['__action_weights'][frame+1:frame+21].copy())
        if '__action_mask' in values:
            result['action_mask']=torch.from_numpy(values['__action_mask'][frame+1:frame+21].copy())
        cv2.setNumThreads(1)
        for i,key in enumerate(FOUR_CAMERAS):
            offset,length = values['image_offsets'][frame,i]
            encoded = os.pread(self.descriptors[ep],int(length),int(offset))
            if len(encoded)!=length: raise ValueError('Truncated image cache')
            image = cv2.imdecode(np.frombuffer(encoded,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape!=(256,288,3): raise ValueError('Invalid cached RGB image')
            result[key] = torch.from_numpy(np.ascontiguousarray(image[...,::-1].transpose(2,0,1))).float()/255
        return result


class MultiViewACT(SFP0ACT):
    def __init__(self, config, stats, *, initialize_backbone=True):
        super().__init__(config,stats,initialize_backbone=initialize_backbone,cameras=FOUR_CAMERAS)
        self.multiview_config = config

    def forward(self, batch):
        if batch.get('action_is_pad') is not None and batch['action_is_pad'].any():
            raise ValueError('Training requires complete future action chunks')
        prepared = self.normalizer.observations(batch)
        prepared['observation.images'] = [prepared[key] for key in self.cameras]
        prediction = self.policy.model(prepared)[0].float()
        loss, metrics = common_scale_loss(prediction,batch['action'],self.normalizer,
                                           self.multiview_config.position_scale_m,batch.get('action_mask',batch.get('action_weights')))
        return loss,{key:value.detach().item() for key,value in metrics.items()}


def load_multiview_checkpoint(payload,device):
    config = MultiViewConfig(**payload['config'])
    config.device = device
    if device=='cpu': config.amp='none'
    config.validate(legacy_inference=True)
    if payload.get('camera_order')!=list(FOUR_CAMERAS): raise ValueError('Checkpoint camera order differs')
    model = MultiViewACT(config,payload['stats'],initialize_backbone=False)
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),payload
