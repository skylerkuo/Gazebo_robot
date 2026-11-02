#!/usr/bin/env python3
"""Alternate expert ACT BC and residual Flow MSE on one frozen four-view split."""
import argparse
from dataclasses import asdict, replace
import fcntl
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader

from act_multiview import FOUR_CAMERAS
from act_objectives import evaluate_pose_chunks
from alternating_residual_flow_multiview import (
    FORMAT, AlternatingResidualFlowConfig, AlternatingResidualFlow,
    initialize_alternating_from_act, configure_scratch_act,
)
from ftact import INSERT_ROOT, EpochBatchSampler, json_write, move_batch
from residual_flow_multiview import (bind_base_act,load_residual_index,validate_residual_index,
    residual_readiness,ResidualExpertDataset)
from train_ftact import (TrainingBatches, seed_all, seed_worker, autocast_context,
                         rng_state, restore_rng, append_metrics)
from train_sfp0_act import lr_factor
from train_sfp0_act_multiview import load_initialization


def parse_args(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
        default=default_config or INSERT_ROOT/'configs/sfp0_alternating_residual_flow_expert.json')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--from-scratch', action='store_true', default=None,
        help='Random ACT and Flow, including ResNet; no checkpoint or ImageNet weights')
    for name in ('output_dir', 'init_from', 'index_path', 'device', 'amp'):
        parser.add_argument('--'+name.replace('_', '-'))
    for name in ('steps', 'batch_size', 'num_workers', 'gradient_accumulation_steps',
                 'eval_every', 'save_every', 'log_every', 'ode_steps'):
        parser.add_argument('--'+name.replace('_', '-'), type=int)
    for name in ('act_learning_rate', 'act_backbone_lr', 'learning_rate', 'backbone_lr'):
        parser.add_argument('--'+name.replace('_', '-'), type=float)
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error('Use --resume for continuation or --init-from to start a new run from trained weights')
    if args.from_scratch and (args.resume or args.init_from):
        parser.error('--from-scratch cannot be combined with --resume or --init-from')
    resume = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if resume and (resume.get('format') != FORMAT or not resume.get('act_optimizer')
                   or not resume.get('flow_optimizer')):
        parser.error('Resume requires this alternating trainer\'s last.pt with both optimizers')
    config = AlternatingResidualFlowConfig(**(resume['config'] if resume else json.loads(args.config.read_text())))
    for name in asdict(config):
        if hasattr(args, name) and getattr(args, name) is not None:
            setattr(config, name, getattr(args, name))
    if args.from_scratch:
        config.from_scratch, config.init_from, config.pretrained_backbone = True, None, False
        config.base_act_config, config.base_act_sha256, config.base_act_step = None, None, None
        if args.output_dir is None and config.output_dir == '/mnt/d/robot_models/aic_sfp0_alternating_residual_flow_expert_v1':
            config.output_dir = '/mnt/d/robot_models/aic_sfp0_alternating_residual_flow_scratch_v1'
    elif args.init_from:
        config.from_scratch = False
    for name in ('output_dir', 'index_path', 'dataset_root', 'init_from'):
        if getattr(config, name) is not None:
            if name=='index_path' and config.index_path=='auto':continue
            setattr(config, name, str(Path(getattr(config, name)).resolve()))
    try:
        (replace(config, device='cpu', amp='none') if args.check_data else config).validate()
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if not Path(config.output_dir).is_relative_to(Path('/mnt/d')):
        parser.error('Model output must be on D:')
    if config.refresh_data or config.corrections_root:
        parser.error('Alternating training keeps the existing frozen dataset; omit refresh_data/corrections_root')
    if not resume and not config.from_scratch and not config.init_from:
        parser.error('Use --from-scratch or initialize from an existing four-view ACT via --init-from')
    if config.index_path=='auto' and resume:
        parser.error('Resume must keep its saved concrete snapshot; start a new run to repartition data')
    if resume:
        if config.steps <= resume['step']:
            parser.error('Total steps must exceed the saved cycle')
        saved = asdict(AlternatingResidualFlowConfig(**resume['config']))
        if config.steps < saved['steps']:
            parser.error('Resume can extend total steps but cannot shorten the saved training target')
        # Extend the run without stretching the old cosine and raising its LR.
        config.lr_schedule_steps = saved['lr_schedule_steps'] or saved['steps']
        mutable = {'steps', 'lr_schedule_steps', 'batch_size', 'num_workers', 'device', 'amp',
                   'act_learning_rate', 'act_backbone_lr', 'learning_rate', 'backbone_lr'}
        for name, value in asdict(config).items():
            if name not in mutable and value != saved[name]:
                parser.error(f'Keep {name} unchanged on resume; start a fresh output to change the training contract')
    return args, config, resume


def make_optimizers(model, config):
    return {
        'act': torch.optim.AdamW(model.base_act.policy.get_optim_params(),
            lr=config.act_learning_rate, weight_decay=config.weight_decay),
        'flow': torch.optim.AdamW(model.policy.get_optim_params(),
            lr=config.learning_rate, weight_decay=config.weight_decay),
    }


def make_scalers(config):
    return {name: torch.amp.GradScaler('cuda', enabled=config.amp == 'fp16') for name in ('act', 'flow')}


def set_learning_rates(optimizers, config, step):
    schedule = replace(config, steps=config.lr_schedule_steps or config.steps)
    factor = lr_factor(step, schedule)
    rates = {'act': (config.act_learning_rate, config.act_backbone_lr),
             'flow': (config.learning_rate, config.backbone_lr)}
    for name, optimizer in optimizers.items():
        for group, rate in zip(optimizer.param_groups, rates[name], strict=True):
            group['lr'] = rate*factor


def alternating_update(model, batches, optimizers, scalers, config):
    """One complete transaction: ACT step, clear grads, relabel, Flow step.

    Reuse the same observations, but never reuse an ACT prediction/target computed
    before ACT.step(). A signal handler may request stopping; this finishes both
    phases before a resumable checkpoint is committed.
    """
    if not batches or len(batches) != config.gradient_accumulation_steps:
        raise ValueError('Supply the configured number of accumulation microbatches')
    model.train()
    result = {}
    sizes = [len(batch['action']) for batch in batches]
    total = sum(sizes)
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    for name in ('act', 'flow'):
        optimizer, scaler = optimizers[name], scalers[name]
        module = model.base_act if name == 'act' else model.policy
        for batch, size in zip(batches, sizes, strict=True):
            with autocast_context(config):
                loss, metrics = model.act_loss(batch) if name == 'act' else model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {name} loss')
            weight = size/total
            scaler.scale(loss*weight).backward()
            result[name+'_loss'] = result.get(name+'_loss', 0.)+loss.item()*weight
            for key, value in metrics.items():
                metric_name = key if key.startswith(name+'_') else name+'_'+key
                result[metric_name] = result.get(metric_name, 0.)+value*weight
        scaler.unscale_(optimizer)
        gradient = module.policy.model.encoder_robot_state_input_proj.weight.grad if name == 'act' else module.model.encoder_robot_state_input_proj.weight.grad
        result[name+'_force_projection_gradient_norm'] = gradient[:, 6:].float().norm().item()
        result[name+'_gradient_norm_before_clip'] = torch.nn.utils.clip_grad_norm_(
            module.parameters(), config.grad_clip_norm, error_if_nonfinite=True).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if name == 'act':
            model.act_update_count.add_(1)
        else:
            model.flow_update_count.add_(1)
    return result


def evaluate_pair(model, loader, config):
    act = evaluate_pose_chunks(model.base_act, loader, config)
    combined = evaluate_pose_chunks(model, loader, config)
    act.update(generation='ACT_only', act_updates=int(model.act_update_count.item()))
    combined.update(act_only_loss=act['loss'], act_only_position_l2_mm=act['position_l2_mm'],
        residual_position_improvement_mm=act['position_l2_mm']-combined['position_l2_mm'])
    return {'act': act, 'combined': combined}


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(path, model, optimizers, scalers, config, index, step, cursor,
                    best, validation, validation_step, *, resumable):
    if int(model.act_update_count.item()) != step or int(model.flow_update_count.item()) != step:
        raise ValueError('Checkpoint must be at a complete ACT + Flow cycle boundary')
    payload = {'format': FORMAT, 'config': asdict(config), 'stats': index['stats'],
        'task': index['task'], 'camera_order': list(FOUR_CAMERAS), 'action_offset': 1,
        'data_index_sha256': index['index_content_sha256'], 'step': step,
        'act_updates': step, 'flow_updates': step, 'optimizer_updates': step*2,
        'model': model.state_dict(), 'initialization': getattr(model, 'initialization', None),
        'cursor': cursor.state_dict(), 'best': dict(best), 'metrics': validation,
        'validation_step': validation_step, 'smoke_test': False,
        'act_optimizer': optimizers['act'].state_dict() if resumable else None,
        'flow_optimizer': optimizers['flow'].state_dict() if resumable else None,
        'act_scaler': scalers['act'].state_dict() if resumable else None,
        'flow_scaler': scalers['flow'].state_dict() if resumable else None,
        'rng': rng_state() if resumable else None,
        'loss_definition': {'act': 'common-scale physical XYZ + normalized quaternion L1',
                            'flow': 'expert - updated ACT; detached supervised residual velocity MSE'},
        'reference_paper': 'https://arxiv.org/abs/2610.29117',
        'method_note': 'Offline alternating expert BC and successful-action residual FM; no critic, no online collection',
        'execution': {'fps': 20, 'chunk_size': 20, 'n_action_steps': 1,
            'temporal_ensemble': False, 'ode_steps': config.ode_steps, 'ode_solver': config.ode_solver,
            'noise_seed': config.seed, 'residual_scale': config.residual_scale},
        'base_act_contract': {'frozen': False, 'embedded_in_model': True,
            'initialization_sha256': config.base_act_sha256,
            'initialization_step': config.base_act_step, 'current_run_updates': step}}
    atomic_save(path, payload)


def export_act(path, model, config, index, step, metrics):
    """Export the ACT belonging to this evaluated cycle, independently deployable."""
    act_config = asdict(model.base_act.multiview_config)
    for key in ('dataset_root', 'index_path', 'output_dir', 'eval_samples', 'eval_every',
                'save_every', 'steps', 'batch_size', 'num_workers', 'position_scale_m', 'init_from'):
        act_config[key] = getattr(config, key)
    act_config.update(policy_type='act', expert_only=True, retain_model_motion=False,
                      learning_rate=config.act_learning_rate, backbone_lr=config.act_backbone_lr)
    atomic_save(path, {'format': 'aic-sfp0-act-multiview-v1', 'config': act_config,
        'stats': index['stats'], 'task': index['task'], 'camera_order': list(FOUR_CAMERAS),
        'action_offset': 1, 'data_index_sha256': index['index_content_sha256'], 'step': step,
        'model': model.base_act.state_dict(), 'metrics': metrics, 'smoke_test': False,
        'optimizer': None, 'initialization': getattr(model, 'initialization', None),
        'execution': {'fps': 20, 'chunk_size': 20, 'n_action_steps': 1,
                      'temporal_ensemble_coeff': .01},
        'training_note': 'ACT-only export from offline alternating BC + residual Flow training'})


def record_validation(output, journal, model, loader, optimizers, scalers, config, index, step, cursor, best):
    validation = evaluate_pair(model, loader, config)
    append_metrics(journal, {'type': 'act_validation', 'step': step, **validation['act']})
    append_metrics(journal, {'type': 'validation', 'step': step, **validation['combined']})
    candidates = [('combined_loss', 'combined', 'loss', 'best.pt'),
                  ('combined_position', 'combined', 'position_l2_m', 'best_position.pt'),
                  ('act_loss', 'act', 'loss', 'best_act.pt'),
                  ('act_position', 'act', 'position_l2_m', 'best_act_position.pt')]
    improved = []
    for key, branch, metric, filename in candidates:
        value = validation[branch][metric]
        if value < best[key]:
            best[key] = value
            improved.append((branch, metric, filename, value))
    for branch, metric, filename, value in improved:
        path = output/'checkpoints'/filename
        if branch == 'act':
            export_act(path, model, config, index, step, validation['act'])
        else:
            save_checkpoint(path, model, optimizers, scalers, config, index, step,
                            cursor, best, validation, step, resumable=False)
        append_metrics(journal, {'type': 'best_checkpoint', 'step': step, 'checkpoint': filename,
            'selection_metric': branch+'.'+metric, 'value': value, 'samples': config.eval_samples})
    return validation


def prepare_training_index(config,initial=None,resume=None):
    if config.index_path=='auto':
        if resume is not None or (config.from_scratch and initial is not None):
            raise ValueError('A new split requires a new run; from-scratch runs must be without previous weights')
        if not config.from_scratch and initial is None:
            raise ValueError('Fine-tuning on a new split requires the initialization checkpoint')
        from prepare_expert_only import prepare_expert_only
        index=prepare_expert_only(config.dataset_root,initial,eval_samples=config.eval_samples,
            seed=config.seed,retain_model_motion=True,progress=True)
        config.index_path=str(Path(index['_directory'])/'index.json')
        if config.base_act_config is not None:config.base_act_config['index_path']=config.index_path
        # Explicit new experiment: membership changes, but transferred weights must
        # retain their normalization. Ordinary fixed-index initialization/resume
        # still enforces the exact snapshot in load_residual_index below.
        if initial is not None and index['stats']!=initial['stats']:
            raise ValueError('Transferred ACT + Flow must preserve checkpoint normalization')
        # prepare_expert_only just performed the complete checksum/provenance
        # audit. Reuse that result instead of rereading every image pack again.
        return validate_residual_index(config,index)
    return load_residual_index(config,initial,resume)


def main(default_config=None):
    args, config, resume = parse_args() if default_config is None else parse_args(default_config)
    if args.check_data:
        if config.index_path=='auto':
            from prepare_expert_only import inspect_expert_collection
            data=inspect_expert_collection(config.dataset_root,eval_samples=config.eval_samples,
                seed=config.seed,retain_model_motion=True)
            report=dict(data['report'],frozen_snapshot=False,
                train_expert_windows=sum(len(data['rollout_data'][ep][3]) for ep in data['train']))
        else:report = residual_readiness(config)
        report['supervision'] = 'ACT expert BC + detached Flow residuals on the same expert windows'
        report['from_scratch'] = config.from_scratch
        report['pretrained_backbone'] = config.pretrained_backbone
        print(json.dumps(report, indent=2), flush=True)
        return
    torch.set_num_threads(config.num_threads)
    seed_all(config.seed)
    output = Path(config.output_dir)
    if not resume and output.exists() and any(p.name != 'training.lock' for p in output.iterdir()):
        raise FileExistsError('Output is not empty; use this trainer\'s last.pt with --resume or choose a new output')
    initial = provenance = None
    if not resume and config.from_scratch:
        configure_scratch_act(config)
        provenance = {'mode': 'from_scratch', 'source': None, 'step': 0,
            'pretrained_backbone': False, 'flow_encoder_initialization': 'independently random',
            'checkpoint_weights_loaded': False}
    elif not resume:
        initial, provenance = load_initialization(config.init_from,expected_policy='alternating_residual_flow')
        bind_base_act(config, initial)
        provenance.update(mode='warm_start',checkpoint_weights_loaded=True,
            source_format=initial['format'])
    print('Checking frozen expert windows; 5% whole-episode VAL; ACT and composed VAL each use '
          f'{config.eval_samples} fixed anchors...', flush=True)
    index = prepare_training_index(config, initial, resume)
    if index.get('test_only'):
        raise ValueError('Synthetic data cannot start a formal training run')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another trainer is writing this output') from None
        train, val = ResidualExpertDataset(index, 'train'), ResidualExpertDataset(index, 'validation')
        workers = {'num_workers': config.num_workers, 'pin_memory': config.device.startswith('cuda'),
                   'worker_init_fn': seed_worker}
        if config.num_workers:
            workers.update(multiprocessing_context='spawn', prefetch_factor=1, persistent_workers=True)
        sampler = EpochBatchSampler(len(train), config.batch_size, config.seed)
        train_loader = DataLoader(train, batch_sampler=sampler,
            generator=torch.Generator().manual_seed(config.seed), **workers)
        val_workers = dict(workers)
        if config.num_workers:
            val_workers['persistent_workers'] = False
        val_loader = DataLoader(val, batch_size=config.batch_size,
            generator=torch.Generator().manual_seed(config.seed+1), **val_workers)
        cursor = TrainingBatches(train_loader, sampler, resume['cursor'] if resume else None,
                                 resume['config']['batch_size'] if resume else None)
        model = AlternatingResidualFlow(config, index['stats'], initialize_backbone=False).to(config.device)
        if initial:
            provenance['transfer'] = initialize_alternating_from_act(model, initial)
            del initial
        model.initialization = resume.get('initialization') if resume else provenance
        optimizers, scalers = make_optimizers(model, config), make_scalers(config)
        step = resume['step'] if resume else 0
        best = dict(resume['best']) if resume else {name: float('inf') for name in
            ('combined_loss', 'combined_position', 'act_loss', 'act_position')}
        validation = resume['metrics'] if resume else None
        validation_step = resume['validation_step'] if resume else None
        if resume:
            model.load_state_dict(resume['model'], strict=True)
            for name in ('act', 'flow'):
                optimizers[name].load_state_dict(resume[name+'_optimizer'])
                scalers[name].load_state_dict(resume[name+'_scaler'])
            restore_rng(resume['rng'])
            del resume
        if int(model.act_update_count.item()) != step or int(model.flow_update_count.item()) != step:
            raise ValueError('Resume update counters do not match a complete cycle')
        set_learning_rates(optimizers, config, step+1)
        json_write(output/'run_config.json', {'config': asdict(config), 'initialization': model.initialization,
            'task': index['task'], 'data_index_sha256': index['index_content_sha256'],
            'train_anchors': len(train), 'validation_samples': len(val), 'counts': index['counts'],
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'cycle_definition': 'one ACT BC optimizer step then one Flow optimizer step on the same observations',
            'no_model_action_targets': True, 'no_online_collection': True})
        journal = output/'metrics.jsonl'
        append_metrics(journal, {'type': 'resume' if args.resume else 'start', 'step': step,
            'policy_type': config.policy_type, 'train_anchors': len(train), 'validation_samples': len(val),
            'act_initialization_step': config.base_act_step, 'eval_every': config.eval_every,
            'from_scratch': config.from_scratch,
            'act_learning_rate': config.act_learning_rate, 'flow_learning_rate': config.learning_rate,
            'data_index_sha256': index['index_content_sha256'], 'optimizer_updates_per_step': 2})
        stopping = []
        def request_stop(signum, _):
            if not stopping:
                print('Stop requested; completing ACT + Flow cycle, then saving both optimizers.', flush=True)
            stopping.append(signum)
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        if config.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        if not args.resume and not stopping:
            validation = record_validation(output, journal, model, val_loader, optimizers, scalers, config,
                                           index, step, cursor, best)
            validation_step = step
        start = time.perf_counter()
        interval, updates, last_saved = {}, 0, -1
        while step < config.steps and not stopping:
            set_learning_rates(optimizers, config, step+1)
            batches = [move_batch(next(cursor), config.device) for _ in range(config.gradient_accumulation_steps)]
            metrics = alternating_update(model, batches, optimizers, scalers, config)
            step += 1
            for key, value in metrics.items():
                interval[key] = interval.get(key, 0.)+value
            updates += 1
            if step % config.log_every == 0 or step == 1 or step == config.steps or stopping:
                append_metrics(journal, {'type': 'train', 'step': step, 'act_updates': step, 'flow_updates': step,
                    **{'mean_'+k: v/updates for k, v in interval.items()}, 'averaged_cycles': updates,
                    'act_learning_rate': optimizers['act'].param_groups[0]['lr'],
                    'flow_learning_rate': optimizers['flow'].param_groups[0]['lr'],
                    'elapsed_seconds': time.perf_counter()-start, **cursor.state_dict(),
                    'peak_cuda_memory_gib': torch.cuda.max_memory_allocated()/2**30
                    if config.device.startswith('cuda') else None})
                interval, updates = {}, 0
            if not stopping and (step % config.eval_every == 0 or step == config.steps):
                validation = record_validation(output, journal, model, val_loader, optimizers, scalers, config,
                                               index, step, cursor, best)
                validation_step = step
            if step % config.save_every == 0 or step == config.steps or stopping:
                save_checkpoint(output/'checkpoints/last.pt', model, optimizers, scalers, config,
                                index, step, cursor, best, validation, validation_step, resumable=True)
                last_saved = step
        # Includes Ctrl+C during initial VAL or between update cycles.
        if last_saved != step:
            save_checkpoint(output/'checkpoints/last.pt', model, optimizers, scalers, config,
                            index, step, cursor, best, validation, validation_step, resumable=True)
        summary = {'completed_cycles': step, 'act_updates': step, 'flow_updates': step,
            'stopped_by_signal': bool(stopping), 'best': best, 'validation_step': validation_step,
            'last_validation': validation, 'closed_loop_evaluated': False,
            'last_checkpoint': str(output/'checkpoints/last.pt')}
        json_write(output/'training_summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
