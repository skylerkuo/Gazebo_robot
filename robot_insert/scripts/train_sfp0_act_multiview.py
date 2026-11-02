#!/usr/bin/env python3
"""Four-view ACT: all available episodes, 5% whole-episode VAL, common-scale XYZ."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import fcntl
import json
import hashlib
import math
from pathlib import Path
import shutil
import signal
import time

import torch
from torch.utils.data import DataLoader

from ftact import INSERT_ROOT,EpochBatchSampler,json_write,move_batch,load_checkpoint
from act_multiview import (FORMAT,FOUR_CAMERAS,MultiViewConfig,MultiViewACT,MultiViewDataset,
                           file_digest,read_index)
from act_objectives import evaluate_pose_chunks as evaluate
from train_sfp0_act import lr_factor as original_lr_factor
from train_ftact import (TrainingBatches,seed_all,seed_worker,autocast_context,
                         rng_state,restore_rng,append_metrics)


def training_format(config):
    if config.policy_type=='stacked_residual_flow':
        from stacked_residual_flow_multiview import FORMAT as stacked_format
        return stacked_format
    if config.policy_type=='residual_flow':
        from residual_flow_multiview import FORMAT as residual_format
        return residual_format
    if config.policy_type=='flow':
        from flow_multiview import FORMAT as flow_format
        return flow_format
    if config.policy_type=='pace_pred':
        from pace_pred_multiview import FORMAT as pred_format
        return pred_format
    if config.policy_type=='pace':
        from pace_multiview import FORMAT as pace_format
        return pace_format
    return FORMAT


def checkpoint(path,model,optimizer,scaler,config,index,step,batches,best,best_position,
               validation,validation_step,smoke,resumable):
    payload = {'format':training_format(config),'config':asdict(config),'stats':index['stats'],'task':index['task'],
        'camera_order':list(FOUR_CAMERAS),'data_index_sha256':index['index_content_sha256'],
        'step':step,'cursor':batches.state_dict(),'best_validation_loss':best,
        'best_position_l2_m':best_position,'metrics':validation,'validation_step':validation_step,
        'smoke_test':smoke,'model':model.state_dict(),
        'initialization':getattr(model,'initialization',None),
        'optimizer':optimizer.state_dict() if resumable else None,
        'scaler':scaler.state_dict() if resumable else None,'rng':rng_state() if resumable else None,
        'action_offset':1,'execution':{'fps':20,'chunk_size':20,'n_action_steps':1,
            'temporal_ensemble_coeff':.01,'camera_order':list(FOUR_CAMERAS)},
        'loss_definition':'(3/7)*mean(abs(dXYZ)/position_scale_m)+(4/7)*normalized_quaternion_L1'}
    if config.policy_type in ('pace','pace_pred'):
        from pace_act import PAPER
        payload['reference_paper']=PAPER
        payload['execution']['event_before_temporal_ensemble']=True
    if config.policy_type=='pace_pred':
        payload['loss_definition'] += '+pred_loss_weight*masked_normalized_FT_increment_MSE'
        payload['best_selection_metric']='imitation_only_validation.loss'
        payload['execution']['pred_head_at_inference']=False
    if config.policy_type in ('flow','residual_flow','stacked_residual_flow'):
        from flow_matching import PAPER
        payload['reference_paper']=PAPER
        payload['loss_definition']='masked_CFM_velocity_MSE; paper Eq.(22)-(23)'
        payload['best_selection_metric']='generated_action_validation.loss'
        payload['execution'].pop('temporal_ensemble_coeff')
        payload['execution'].update(ode_steps=config.ode_steps,ode_solver=config.ode_solver,
            temporal_ensemble=False,noise_seed=config.seed)
        payload['action_normalization']={'xyz_common_scale_m':config.position_scale_m,
            'mean_and_quaternion':'frozen parent statistics; actual transform stored in model buffers'}
    if config.policy_type=='residual_flow':
        from residual_flow_multiview import PAPER
        payload['reference_paper']=PAPER
        payload['loss_definition']='supervised_residual_CFM_MSE; expert - frozen ACT at same observation'
        payload['base_act_contract']={'sha256':config.base_act_sha256,'step':config.base_act_step,
            'embedded_in_model':True,'frozen':True,'data_index_sha256':index['index_content_sha256']}
        payload['execution']['residual_scale']=config.residual_scale
    if config.policy_type=='stacked_residual_flow':
        payload['loss_definition']='supervised_second_residual_CFM_MSE; expert - frozen ACT + first Flow at same observation'
        payload['method_note']='Experimental stacked supervised residual extension; no critic or ACT absorption'
        payload['base_policy_contract']={'sha256':config.base_policy_sha256,'step':config.base_policy_step,
            'format':'aic-sfp0-supervised-residual-flow-multiview-v1',
            'embedded_in_model':True,'frozen':True,'data_index_sha256':index['index_content_sha256'],
            'anchor_noise_seed':config.base_noise_seed,
            'ode_steps':config.base_policy_config['ode_steps'],
            'ode_solver':config.base_policy_config['ode_solver']}
        payload['execution'].update(residual_layers=2,residual_scale=config.residual_scale,
            base_noise_stream='separate seed-derived stream; offline uses fixed per-anchor noise')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.pt.partial')
    torch.save(payload,temporary)
    temporary.replace(path)


def configuration_class(policy):
    if policy=='stacked_residual_flow':
        from stacked_residual_flow_multiview import StackedResidualFlowConfig
        return StackedResidualFlowConfig
    if policy=='residual_flow':
        from residual_flow_multiview import ResidualFlowConfig
        return ResidualFlowConfig
    if policy=='flow':
        from flow_multiview import MultiViewFlowConfig
        return MultiViewFlowConfig
    if policy=='pace_pred':
        from pace_pred_multiview import MultiViewPACEPredConfig
        return MultiViewPACEPredConfig
    if policy=='pace':
        from pace_multiview import MultiViewPACEConfig
        return MultiViewPACEConfig
    if policy=='act':return MultiViewConfig
    raise ValueError('Unknown training policy')


def parse_args(default_config=None,expected_policy='act'):
    parser = argparse.ArgumentParser(description=f'Four-view {expected_policy.upper()} on assisted rollouts; shared 5% episode VAL and physical XYZ loss.')
    parser.add_argument('--config',type=Path,default=default_config or INSERT_ROOT/'configs/sfp0_act_expert_finetune.json')
    parser.add_argument('--resume',type=Path)
    if expected_policy=='act':
        parser.add_argument('--from-scratch',action='store_true',
            help='Random ACT and ResNet; do not load a checkpoint or ImageNet weights')
    parser.add_argument('--refresh-data',action='store_true',default=None,
        help='With --init-from: use all episodes, select VAL only from new episodes, preserve normalization')
    parser.add_argument('--smoke-test',action='store_true',help='3 updates; still validates the full fixed VAL set')
    parser.add_argument('--check-data',action='store_true',help='Read dataset/VAL readiness only; do not start training')
    for key in ('dataset_root','index_path','output_dir','device','amp','init_from','corrections_root','comparison_index_path'):
        parser.add_argument('--'+key.replace('_','-'))
    for key in ('steps','batch_size','num_workers','gradient_accumulation_steps','correction_repeat','eval_samples','eval_every'):
        parser.add_argument('--'+key.replace('_','-'),type=int)
    parser.add_argument('--learning-rate',type=float,help='Base model LR; explicitly overrides a resumed checkpoint')
    parser.add_argument('--backbone-lr',type=float,help='Base ResNet LR; explicitly overrides a resumed checkpoint')
    parser.add_argument('--position-scale-m',type=float)
    if expected_policy in ('pace','pace_pred'):
        parser.add_argument('--phase-alpha',type=float)
        parser.add_argument('--tactile-history',type=int)
        parser.add_argument('--correction-horizon',type=int)
    if expected_policy=='pace_pred':parser.add_argument('--pred-loss-weight',type=float)
    if expected_policy in ('flow','residual_flow','stacked_residual_flow'):
        parser.add_argument('--sigma-min',type=float)
        parser.add_argument('--ode-steps',type=int)
        parser.add_argument('--ode-solver',choices=('euler','heun'))
        parser.add_argument('--validation-noise-seed',type=int)
    if expected_policy in ('residual_flow','stacked_residual_flow'):parser.add_argument('--residual-scale',type=float)
    if expected_policy=='stacked_residual_flow':parser.add_argument('--base-noise-seed',type=int)
    args = parser.parse_args()
    if getattr(args,'from_scratch',False) and (args.resume or args.init_from):
        parser.error('--from-scratch cannot be combined with --resume or --init-from')
    if args.resume and args.init_from:parser.error('Use --resume for continuation or --init-from for a new fine-tune, not both')
    if args.resume and args.refresh_data:parser.error('Use --init-from with --refresh-data in a new output; resume keeps its frozen snapshot')
    resume = torch.load(args.resume,map_location='cpu',weights_only=True) if args.resume else None
    Config=configuration_class(expected_policy)
    expected_format=training_format(Config())
    if resume and (resume.get('format')!=expected_format or resume.get('optimizer') is None):
        parser.error('Resume requires this four-view trainer\'s last.pt, not an old ACT/PACE checkpoint')
    config = Config(**(resume['config'] if resume else json.loads(args.config.read_text())))
    if config.policy_type!=expected_policy:parser.error('Use the matching ACT or PACE training entrypoint')
    if args.smoke_test and not resume:
        config.steps,config.log_every,config.eval_every,config.save_every,config.warmup_steps = 3,1,3,3,1
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        config.output_dir = f'/mnt/d/robot_models/aic_sfp0_{expected_policy}_multiview_checks/{stamp}'
    for key in asdict(config):
        if hasattr(args,key) and getattr(args,key) is not None: setattr(config,key,getattr(args,key))
    if getattr(args,'from_scratch',False):
        config.init_from=None;config.pretrained_backbone=False
    if config.index_path!='auto': config.index_path = str(Path(config.index_path).resolve())
    config.output_dir = str(Path(config.output_dir).resolve())
    for key in ('init_from','corrections_root'):
        if getattr(config,key) is not None:setattr(config,key,str(Path(getattr(config,key)).resolve()))
    if config.corrections_root and not config.init_from and not resume:
        parser.error('Human correction fine-tuning requires --init-from to preserve parent scales and VAL')
    if config.refresh_data and not config.init_from:parser.error('--refresh-data requires --init-from')
    if config.expert_only and (config.refresh_data or config.corrections_root):parser.error('Expert-only training uses its own dataset; omit refresh-data/corrections-root')
    if config.expert_only and not config.init_from and not resume and config.policy_type!='act':
        parser.error('This expert-only policy requires --init-from')
    if config.retain_model_motion and not config.expert_only:parser.error('retain_model_motion requires the retained assisted dataset')
    if resume and config.steps!=resume['config']['steps']:
        if config.steps<resume['config']['steps']:
            parser.error('--steps can extend the saved training budget, not shorten it')
        # Preserve the original cosine schedule; extending the stopping point must
        # not unexpectedly raise a converged model's learning rate or restart warmup.
        config.lr_schedule_steps=resume['config'].get('lr_schedule_steps',0) or resume['config']['steps']
    try:
        (replace(config,device='cpu',amp='none') if args.check_data else config).validate()
    except ValueError as error:
        parser.error(str(error)+'; use a fresh output when changing the validation contract')
    if not Path(config.output_dir).is_relative_to(Path('/mnt/d')):
        parser.error('Model output must be on D:')
    if resume:
        if config.steps<=resume['step']: parser.error('Total steps must exceed the saved step')
        mutable = {'steps','lr_schedule_steps','batch_size','num_workers','device','amp','learning_rate','backbone_lr'}
        saved_config=asdict(Config(**resume['config']))
        for key,value in asdict(config).items():
            if key not in mutable and value!=saved_config[key]:
                parser.error(f'Keep {key} unchanged on resume; use a fresh output for a changed objective/index')
    return args,config,resume


def lr_factor(step,config):
    schedule_steps=config.lr_schedule_steps or config.steps
    return original_lr_factor(step,replace(config,steps=schedule_steps))


def set_optimizer_learning_rates(optimizer,config,step):
    """Apply both base rates after optimizer restore, retaining the original schedule."""
    factor=lr_factor(step,config)
    for group,base_lr in zip(optimizer.param_groups,(config.learning_rate,config.backbone_lr),strict=True):
        group['lr']=base_lr*factor


def load_training_index(config, resume=None, initial=None, *, allow_test=False):
    if config.policy_type in ('residual_flow','stacked_residual_flow'):
        from residual_flow_multiview import load_residual_index
        index=load_residual_index(config,initial,resume)
    elif initial is not None and config.expert_only and config.comparison_index_path:
        from prepare_act_pace_comparison import prepare_comparison
        index=prepare_comparison(config,initial)
        config.index_path=str(Path(index['_directory'])/'index.json')
    elif initial is not None and config.expert_only:
        from prepare_expert_only import prepare_expert_only
        index=prepare_expert_only(config.dataset_root,initial,eval_samples=config.eval_samples,seed=config.seed,
            retain_model_motion=config.retain_model_motion)
        config.index_path=str(Path(index['_directory'])/'index.json')
    elif config.expert_only and initial is None and resume is None and config.index_path=='auto':
        from prepare_expert_only import prepare_expert_only
        index=prepare_expert_only(config.dataset_root,eval_samples=config.eval_samples,seed=config.seed,
            retain_model_motion=config.retain_model_motion,progress=True)
        config.index_path=str(Path(index['_directory'])/'index.json')
    elif initial is not None:
        parent_path=str(Path(initial['config']['index_path']).resolve())
        if config.index_path not in ('auto',parent_path):raise ValueError('Fine-tuning must preserve the parent frozen split')
        index=read_index(parent_path)
        if initial['data_index_sha256']!=index['index_content_sha256'] or initial['stats']!=index['stats']:
            raise ValueError('Initialization data index/statistics differ from the parent checkpoint')
        if config.refresh_data:
            from prepare_sfp0_act_multiview import prepare
            index=prepare(config.dataset_root,seed=config.seed,parent=index,eval_samples=config.eval_samples)
        if config.corrections_root:
            from prepare_human_corrections import prepare_corrections
            index=prepare_corrections(index,config.corrections_root,allow_test=allow_test)
        config.index_path=str(Path(index['_directory'])/'index.json')
    elif config.index_path=='auto':
        if resume: raise ValueError('Resume must use the frozen index path saved in its checkpoint')
        from prepare_sfp0_act_multiview import prepare
        index=prepare(config.dataset_root,seed=config.seed,eval_samples=config.eval_samples)
        config.index_path=str(Path(index['_directory'])/'index.json')
    else:
        if not Path(config.index_path).is_file():
            raise FileNotFoundError('Frozen training index is missing; use --index-path auto for a new snapshot')
        index=read_index(config.index_path)
    if index.get('test_only') and not allow_test:raise ValueError('Synthetic/test data is restricted to --smoke-test')
    if index['validation_samples']!=config.eval_samples:
        raise ValueError('VAL sample count differs from the frozen index; use a fresh training run')
    if Path(config.dataset_root).resolve()!=Path(index['dataset_root']):
        raise ValueError('Config points to a different source collection')
    if resume and resume['data_index_sha256']!=index['index_content_sha256']:
        raise ValueError('Resume requires the same frozen episodes, statistics and fixed VAL anchors')
    return index


def load_initialization(path,expected_policy='act'):
    with Path(path).open('rb') as stream:
        payload=torch.load(stream,map_location='cpu',weights_only=True)
        stream.seek(0);digest=hashlib.file_digest(stream,'sha256').hexdigest()
    expected_format=('aic-sfp0-supervised-residual-flow-multiview-v1'
                     if expected_policy=='stacked_residual_flow' else FORMAT)
    accepted={expected_format}
    if expected_policy=='residual_flow':
        accepted.add('aic-sfp0-alternating-act-residual-flow-multiview-v1')
    if expected_policy=='alternating_residual_flow':
        accepted.update(('aic-sfp0-alternating-act-residual-flow-multiview-v1',
                         'aic-sfp0-supervised-residual-flow-multiview-v1'))
    if payload.get('format') not in accepted or payload.get('camera_order')!=list(FOUR_CAMERAS):
        raise ValueError('Initialization requires a four-view '+
                         ('ACT + residual Flow checkpoint' if expected_policy=='stacked_residual_flow' else 'ACT checkpoint'))
    payload['_initialization_sha256']=digest
    return payload,{'source':str(Path(path).resolve()),'step':payload['step'],'sha256':digest}


def prepare_backbone_cache():
    old = Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    target = Path('/mnt/d/robot_cache/torch/hub')
    destination = target/'checkpoints/resnet18-f37072fd.pth'
    if not destination.exists():
        if not old.exists(): raise FileNotFoundError('ResNet-18 pretrained weights are not cached locally')
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary = destination.with_suffix('.pth.partial')
        shutil.copyfile(old,temporary)
        if file_digest(old)!=file_digest(temporary): raise ValueError('Backbone cache copy checksum mismatch')
        temporary.replace(destination)
    torch.hub.set_dir(str(target))


def main(default_config=None,expected_policy='act'):
    args,config,resume = parse_args(default_config,expected_policy)
    extension=({'from_steps':resume['config']['steps'],'to_steps':config.steps,
                'lr_schedule_steps':config.lr_schedule_steps or config.steps,
                'warmup_restarted':False}
               if resume and config.steps!=resume['config']['steps'] else None)
    rate_changes={key:{'from':resume['config'][key],'to':getattr(config,key)}
        for key in ('learning_rate','backbone_lr')
        if resume and resume['config'][key]!=getattr(config,key)}
    torch.set_num_threads(config.num_threads)
    seed_all(config.seed)
    if args.check_data:
        from prepare_act_pace_comparison import comparison_readiness
        if config.policy_type=='stacked_residual_flow':
            from stacked_residual_flow_multiview import stacked_readiness
            report=stacked_readiness(config)
        elif config.policy_type=='residual_flow':
            from residual_flow_multiview import residual_readiness
            report=residual_readiness(config)
        else:report=comparison_readiness(config)
        if config.policy_type=='act':
            report.update(from_scratch=config.init_from is None and not config.pretrained_backbone,
                initialization_checkpoint=config.init_from,pretrained_backbone=config.pretrained_backbone)
        print(json.dumps({'policy':config.policy_type,**report},indent=2),flush=True)
        return
    initial=initialization=None
    if config.init_from and not resume:initial,initialization=load_initialization(config.init_from,expected_policy)
    elif not resume and config.policy_type=='act':
        initialization={'mode':'imagenet_backbone' if config.pretrained_backbone else 'from_scratch',
            'source':'ResNet18_Weights.IMAGENET1K_V1' if config.pretrained_backbone else None,
            'checkpoint_weights_loaded':False,'pretrained_backbone':config.pretrained_backbone,
            'normalization':'current training episodes only'}
    if initial is not None and config.policy_type=='stacked_residual_flow':
        from stacked_residual_flow_multiview import bind_base_policy
        bind_base_policy(config,initial)
    if initial is not None and config.policy_type=='residual_flow':
        from residual_flow_multiview import bind_base_act
        bind_base_act(config,initial)
    output = Path(config.output_dir)
    if not resume and output.exists() and any(p.name!='training.lock' for p in output.iterdir()):
        raise FileExistsError('Output is not empty; choose another output or resume this trainer\'s last.pt')
    print(f'Checking/preparing a frozen snapshot: 5% whole-episode VAL, {config.eval_samples} anchors...',flush=True)
    try:
        index = load_training_index(config,resume,initial,allow_test=args.smoke_test or bool(resume and resume['smoke_test']))
    except ValueError as error:
        raise SystemExit(f'Dataset preparation stopped: {error}') from None
    smoke = args.smoke_test or bool(resume and resume['smoke_test'])
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'training.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another trainer is writing this output') from None
    if config.policy_type in ('pace','pace_pred'):
        from pace_multiview import MultiViewPACE,MultiViewPACEDataset,initialize_pace_from_act
        Dataset,Model=MultiViewPACEDataset,MultiViewPACE
        if config.policy_type=='pace_pred':
            from pace_pred_multiview import MultiViewPACEPred,MultiViewPACEPredDataset
            Dataset,Model=MultiViewPACEPredDataset,MultiViewPACEPred
        dataset_options={'alpha':config.phase_alpha,'history':config.tactile_history}
    elif config.policy_type=='stacked_residual_flow':
        from stacked_residual_flow_multiview import StackedResidualFlow,initialize_stacked_from_residual
        from residual_flow_multiview import ResidualExpertDataset
        Dataset,Model,dataset_options=ResidualExpertDataset,StackedResidualFlow,{}
    elif config.policy_type=='residual_flow':
        from residual_flow_multiview import (SupervisedResidualFlow,ResidualExpertDataset,
            initialize_residual_from_act)
        Dataset,Model,dataset_options=ResidualExpertDataset,SupervisedResidualFlow,{}
    elif config.policy_type=='flow':
        from flow_multiview import MultiViewFlow,initialize_flow_encoder_from_act
        Dataset,Model,dataset_options=MultiViewDataset,MultiViewFlow,{}
    else:Dataset,Model,dataset_options=MultiViewDataset,MultiViewACT,{}
    train,val = Dataset(index,'train',correction_repeat=config.correction_repeat,**dataset_options),Dataset(index,'validation',**dataset_options)
    workers = {'num_workers':config.num_workers,'pin_memory':config.device.startswith('cuda'),
               'worker_init_fn':seed_worker}
    if config.num_workers:
        workers.update(multiprocessing_context='spawn',prefetch_factor=1,persistent_workers=True)
    sampler = EpochBatchSampler(len(train),config.batch_size,config.seed)
    train_loader = DataLoader(train,batch_sampler=sampler,
        generator=torch.Generator().manual_seed(config.seed),**workers)
    val_workers = dict(workers)
    if config.num_workers: val_workers['persistent_workers']=False
    val_loader = DataLoader(val,batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(config.seed+1),**val_workers)
    batches = TrainingBatches(train_loader,sampler,resume['cursor'] if resume else None,
                              resume['config']['batch_size'] if resume else None)
    if not resume and not initial and config.pretrained_backbone: prepare_backbone_cache()
    model = Model(config,index['stats'],initialize_backbone=not bool(resume or initial)).to(config.device).train()
    if initial:
        if config.policy_type in ('pace','pace_pred'):initialize_pace_from_act(model,initial['model'])
        elif config.policy_type=='stacked_residual_flow':
            initialization['transfer']=initialize_stacked_from_residual(model,initial)
        elif config.policy_type=='residual_flow':
            initialization['transfer']=initialize_residual_from_act(model,initial)
        elif config.policy_type=='flow':
            initialization['transfer']=initialize_flow_encoder_from_act(model,initial['model'])
        else:model.load_state_dict(initial['model'],strict=True)
        del initial
    model.initialization=initialization if not resume else resume.get('initialization')
    optimizer = torch.optim.AdamW(model.policy.get_optim_params(),lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda',enabled=config.amp=='fp16')
    step,best,best_position,validation,validation_step = 0,float('inf'),float('inf'),None,None
    if resume:
        model.load_state_dict(resume['model'],strict=True)
        optimizer.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler'])
        step,best,best_position = resume['step'],resume['best_validation_loss'],resume['best_position_l2_m']
        validation,validation_step = resume['metrics'],resume['validation_step']
        restore_rng(resume['rng']);del resume
    set_optimizer_learning_rates(optimizer,config,step+1)
    json_write(output/'run_config.json',{'config':asdict(config),'task':index['task'],
        'data_index_sha256':index['index_content_sha256'],'counts':index['counts'],
        'train_anchors':len(train),'validation_samples':len(val),'camera_order':list(FOUR_CAMERAS),
        'architecture':('Frozen ACT + first Flow, trainable second residual Flow on complete expert chunks'
                        if config.policy_type=='stacked_residual_flow' else 'Frozen ACT + supervised conditional residual Flow on complete expert chunks'
                        if config.policy_type=='residual_flow' else 'Conditional OT Gaussian Flow; cached ResNet/Encoder, noisy action/time velocity decoder'
                        if config.policy_type=='flow' else f'Deterministic ACT (z=0), state6+force6, four RGBs; policy={config.policy_type}'),
        'comparison_index_path':config.comparison_index_path,
        'parameters':sum(p.numel() for p in model.parameters()),'smoke_test':smoke,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'initialization':model.initialization,'human_correction_repeat':config.correction_repeat,
        'best_selection_metric':'validation.loss','position_best_selection_metric':'validation.position_l2_m'})
    journal = output/'metrics.jsonl'
    append_metrics(journal,{'type':'resume' if args.resume else 'start','step':step,
        'policy_type':config.policy_type,
        'counts':index['counts'],'train_anchors':len(train),'validation_samples':len(val),
        'data_index_sha256':index['index_content_sha256'],'index_path':config.index_path,
        'eval_every':config.eval_every,'smoke_test':smoke,
        'base_learning_rate':config.learning_rate,'base_backbone_lr':config.backbone_lr,
        'next_learning_rate':optimizer.param_groups[0]['lr'],
        'next_backbone_lr':optimizer.param_groups[1]['lr']})
    if rate_changes:
        append_metrics(journal,{'type':'learning_rate_change','step':step,'changes':rate_changes,
            'next_learning_rate':optimizer.param_groups[0]['lr'],
            'next_backbone_lr':optimizer.param_groups[1]['lr'],'warmup_restarted':False})
    if extension:
        append_metrics(journal,{'type':'training_extended','step':step,**extension,
            'next_learning_rate':optimizer.param_groups[0]['lr'],
            'next_backbone_lr':optimizer.param_groups[1]['lr']})
    stopping=[]
    def request_stop(signum,_):
        if not stopping: print('Stop requested; finishing this update and saving last.pt.',flush=True)
        stopping.append(signum)
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    if config.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    started,last_saved,interval,updates = time.perf_counter(),step if args.resume else -1,{},0
    side_gradient = None
    supervised=index['episodes'] if config.expert_only else index.get('correction_episodes',[])
    correction_ids={e['episode_index'] for e in supervised}
    correction_sources={e.get('expert_source','human') for e in supervised}
    correction_metric='rollout_samples' if config.retain_model_motion else (next(iter(correction_sources)) if len(correction_sources)==1 else 'correction')+'_samples'
    if initialization is not None and initialization.get('checkpoint_weights_loaded',True):
        if config.policy_type in ('residual_flow','stacked_residual_flow'):
            model.residual_enabled=False
            try:base_validation=evaluate(model,val_loader,config)
            finally:model.residual_enabled=True
            baseline_type='base_policy_validation' if config.policy_type=='stacked_residual_flow' else 'base_act_validation'
            append_metrics(journal,{'type':baseline_type,'step':0,**base_validation})
        validation=evaluate(model,val_loader,config);validation_step=0
        best,best_position=validation['loss'],validation['position_l2_m']
        append_metrics(journal,{'type':'validation','step':0,'baseline_before_finetuning':True,**validation})
        for name,metric in (('best.pt','loss'),('best_position.pt','position_l2_m')):
            checkpoint(output/'checkpoints'/name,model,optimizer,scaler,config,index,0,batches,
                       best,best_position,validation,0,smoke,False)
            append_metrics(journal,{'type':'best_checkpoint','step':0,'checkpoint':name,
                'selection_metric':metric,'value':validation[metric],'samples':len(val),'baseline_before_finetuning':True})
    while step<config.steps and not stopping:
        set_optimizer_learning_rates(optimizer,config,step+1)
        optimizer.zero_grad(set_to_none=True)
        current={}
        for _ in range(config.gradient_accumulation_steps):
            batch=move_batch(next(batches),config.device)
            check_side=smoke and step==0
            if check_side: batch[FOUR_CAMERAS[-1]].requires_grad_(True)
            with autocast_context(config): loss,metrics=model(batch)
            if correction_ids:
                metrics[correction_metric]=float(sum(int(e) in correction_ids for e in batch['episode_index'].tolist()))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite four-view ACT loss')
            scaler.scale(loss/config.gradient_accumulation_steps).backward()
            if check_side:
                side_gradient=batch[FOUR_CAMERAS[-1]].grad.float().norm().item()
                if not math.isfinite(side_gradient) or side_gradient<=0:
                    raise AssertionError('Side camera did not receive a finite nonzero gradient')
            for key,value in {'loss':loss.item(),**metrics}.items():
                current[key]=current.get(key,0.)+value/config.gradient_accumulation_steps
        scaler.unscale_(optimizer)
        force_gradient=model.policy.model.encoder_robot_state_input_proj.weight.grad[:,6:].float().norm().item()
        pace_gradients={}
        if config.policy_type in ('pace','pace_pred'):
            from train_sfp0_pace import gradient_metrics
            pace_gradients=gradient_metrics(model)
            if config.policy_type=='pace_pred':
                gradients=[p.grad.float().square().sum() for p in model.policy.model.pace_pred_head.parameters() if p.grad is not None]
                pace_gradients['pred_head_gradient_norm']=torch.stack(gradients).sum().sqrt().item() if gradients else 0.
        if smoke and (not math.isfinite(force_gradient) or force_gradient<=0):
            raise AssertionError('Force input did not receive a finite nonzero gradient')
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip_norm,error_if_nonfinite=True).item()
        scaler.step(optimizer);scaler.update();step+=1
        for key,value in current.items(): interval[key]=interval.get(key,0.)+value
        updates+=1
        if step%config.log_every==0 or step==1 or step==config.steps or stopping:
            append_metrics(journal,{'type':'train','step':step,**{'mean_'+k:v/updates for k,v in interval.items()},
                'averaged_updates':updates,'last_batch_loss':current['loss'],'learning_rate':optimizer.param_groups[0]['lr'],
                'backbone_learning_rate':optimizer.param_groups[1]['lr'],
                'gradient_norm_before_clip':gradient,'force_projection_gradient_norm':force_gradient,
                **pace_gradients,
                'side_image_gradient_norm_smoke':side_gradient,'elapsed_seconds':time.perf_counter()-started,
                **batches.state_dict(),'peak_cuda_memory_gib':torch.cuda.max_memory_allocated()/2**30
                if config.device.startswith('cuda') else None})
            interval,updates={},0
        if not stopping and (step%config.eval_every==0 or step==config.steps):
            validation=evaluate(model,val_loader,config);validation_step=step
            append_metrics(journal,{'type':'validation','step':step,**validation})
            improved_loss,improved_position=validation['loss']<best,validation['position_l2_m']<best_position
            best,best_position=min(best,validation['loss']),min(best_position,validation['position_l2_m'])
            for improved,name,metric in ((improved_loss,'best.pt','loss'),(improved_position,'best_position.pt','position_l2_m')):
                if improved:
                    checkpoint(output/'checkpoints'/name,model,optimizer,scaler,config,index,step,batches,
                        best,best_position,validation,validation_step,smoke,False)
                    append_metrics(journal,{'type':'best_checkpoint','step':step,'checkpoint':name,
                        'selection_metric':metric,'value':validation[metric],'samples':len(val)})
        if step%config.save_every==0 or step==config.steps or stopping:
            checkpoint(output/'checkpoints/last.pt',model,optimizer,scaler,config,index,step,batches,
                best,best_position,validation,validation_step,smoke,True)
            last_saved=step
    if stopping and last_saved!=step:
        checkpoint(output/'checkpoints/last.pt',model,optimizer,scaler,config,index,step,batches,
            best,best_position,validation,validation_step,smoke,True)
    result={'completed_steps':step,'smoke_test':smoke,'stopped_by_signal':bool(stopping),
        'best_validation_loss':best if math.isfinite(best) else None,
        'best_position_l2_m':best_position if math.isfinite(best_position) else None,
        'last_validation':validation,'validation_step':validation_step,
        'last_checkpoint':str(output/'checkpoints/last.pt'),'closed_loop_evaluated':False}
    if smoke and validation:
        model.eval();reference=move_batch(next(iter(val_loader)),config.device)
        with torch.inference_mode(),autocast_context(config): expected=model.predict_action_chunk(reference).cpu()
        del optimizer
        restored,_=load_checkpoint(output/'checkpoints/last.pt',config.device)
        with torch.inference_mode(),autocast_context(config): actual=restored.predict_action_chunk(reference).cpu()
        torch.testing.assert_close(actual,expected)
        result.update(checkpoint_roundtrip_passed=True,prediction_shape=list(actual.shape),
            side_camera_gradient_verified=side_gradient is not None and side_gradient>0,
            force_gradient_verified=True)
    json_write(output/'training_summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
