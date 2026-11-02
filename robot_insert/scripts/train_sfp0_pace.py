#!/usr/bin/env python3
"""Fresh SFP0 ACT with common-scale XYZ loss and PACE Phase + Event."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader

from ftact import INSERT_ROOT, EpochBatchSampler, json_write, move_batch
from ftact_v2 import read_index
from pace_act import FORMAT, PAPER, PACEConfig, PACEACT, PACETaskDataset, common_scale_loss
from train_sfp0_act import lr_factor
from train_ftact import (TrainingBatches, seed_all, seed_worker, autocast_context,
                         rng_state, restore_rng, append_metrics)


@torch.inference_mode()
def evaluate(model, loader, config):
    previous_mode = model.training
    model.eval()
    total_loss = total_l1 = count = 0
    position = torch.zeros(20, dtype=torch.float64)
    angle = torch.zeros_like(position)
    errors = []
    try:
        for batch in loader:
            batch = move_batch(batch, config.device)
            with autocast_context(config):
                prediction = model.normalized_action_chunk(batch)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite PACE validation output')
            target = batch['action'].float()
            loss, values = common_scale_loss(prediction, target, model.normalizer, config.position_scale_m)
            total_loss += loss.item() * len(target)
            total_l1 += values['normalized_l1'].item() * len(target)
            physical = model.normalizer.unnormalize_actions(prediction)
            error = (physical[..., :3] - target[..., :3]).double().cpu()
            errors.append(error.abs())
            position += error.norm(dim=-1).sum(0)
            # FP64 avoids acos precision loss for the small rotations of this task.
            q = torch.nn.functional.normalize(physical[..., 3:].double(), dim=-1)
            target_q = torch.nn.functional.normalize(target[..., 3:].double(), dim=-1)
            dot = (q * target_q).sum(-1).abs().clamp(0, 1)
            angle += torch.rad2deg(2 * torch.acos(dot)).sum(0).cpu()
            count += len(target)
        if count != 1200:
            raise ValueError(f'Validation used {count} anchors; exactly 1200 required')
        absolute = torch.cat(errors)
        position, angle = position / count, angle / count
        return {'loss': total_loss / count, 'l1': total_l1 / count,
            'position_l2_m': position.mean().item(), 'position_l2_mm': position.mean().item() * 1000,
            'position_xyz_mae_mm': (absolute.mean((0, 1)) * 1000).tolist(),
            'position_xyz_p90_mm': (absolute.flatten(0, 1).quantile(.9, dim=0) * 1000).tolist(),
            'position_xyz_by_horizon_mm': (absolute.mean(0) * 1000).tolist(),
            'rotation_deg': angle.mean().item(), 'first_step_position_m': position[0].item(),
            'last_step_position_m': position[-1].item(), 'position_by_horizon_m': position.tolist(),
            'rotation_by_horizon_deg': angle.tolist(), 'samples': count, 'valid_action_steps': count * 20,
            'latent': 'zero', 'action_offset': 1, 'position_scale_m': config.position_scale_m,
            'best_selection_metric': 'loss', 'xyz_frame': 'base_link'}
    finally:
        model.train(previous_mode)


def checkpoint(path, model, optimizer, scaler, config, index, step, batches,
               best, best_position, validation, validation_step, smoke, resumable):
    payload = {'format': FORMAT, 'config': asdict(config), 'stats': index['stats'],
        'task': index['task'], 'data_index_sha256': index['index_content_sha256'],
        'step': step, 'cursor': batches.state_dict(), 'best_validation_loss': best,
        'best_position_l2_m': best_position, 'metrics': validation, 'validation_step': validation_step,
        'smoke_test': smoke, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if resumable else None,
        'scaler': scaler.state_dict() if resumable else None,
        'rng': rng_state() if resumable else None, 'action_offset': 1,
        'execution': {'fps': 20, 'chunk_size': 20, 'n_action_steps': 1,
                      'event_before_temporal_ensemble': True},
        'reference_paper': PAPER,
        'loss_definition': f'(3/7)*mean(abs(dXYZ)/{config.position_scale_m}m)+(4/7)*normalized_quaternion_L1'}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=INSERT_ROOT / 'configs/sfp0_pace.json')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke-test', action='store_true', help='Three real updates and all 1200 fixed VAL anchors')
    for key in ('index_path', 'output_dir', 'device', 'amp'):
        parser.add_argument('--' + key.replace('_', '-'))
    for key in ('steps', 'batch_size', 'num_workers', 'gradient_accumulation_steps', 'seed',
                'tactile_history', 'correction_horizon'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ('position_scale_m', 'phase_alpha'):
        parser.add_argument('--' + key.replace('_', '-'), type=float)
    args = parser.parse_args()
    resume = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if resume and (resume.get('format') != FORMAT or resume.get('optimizer') is None):
        parser.error('Resume requires a PACE last.pt with optimizer state; old ACT/Flow weights are separate')
    config = PACEConfig(**(resume['config'] if resume else json.loads(args.config.read_text())))
    if args.smoke_test and not resume:
        config.steps, config.batch_size, config.num_workers = 3, 2, 0
        config.log_every, config.eval_every, config.save_every, config.warmup_steps = 1, 3, 3, 1
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        config.output_dir = f'/mnt/d/robot_models/aic_sfp0_pace_checks/{stamp}'
    for key in asdict(config):
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    config.index_path = str(Path(config.index_path).resolve())
    config.output_dir = str(Path(config.output_dir).resolve())
    config.validate()
    if resume:
        if config.steps <= resume['step']:
            parser.error('--steps is the total target and must exceed the saved step')
        mutable = {'batch_size', 'num_workers', 'device', 'amp'}
        for key, value in asdict(config).items():
            if key not in mutable and value != resume['config'][key]:
                parser.error(f'Keep {key} unchanged on resume; use a fresh run for a changed objective/architecture')
    return args, config, resume


def gradient_metrics(model):
    core = model.policy.model
    def norm(parameters):
        values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
        return torch.stack(values).sum().sqrt().item() if values else 0.
    force_grad = core.encoder_robot_state_input_proj.weight.grad
    return {'force_projection_gradient_norm': force_grad[:, 6:].float().norm().item() if force_grad is not None else 0.,
        'phase_encoder_gradient_norm': norm(core.pace_phase_encoder.parameters()),
        'phase_modulation_gradient_norm': norm(core.pace_modulators.parameters()),
        'event_gate_gradient_norm': norm(core.pace_event_gate.parameters()),
        'event_residual_gradient_norm': norm(core.pace_event_residual.parameters())}


def main():
    args, config, resume = parse_args()
    torch.set_num_threads(config.num_threads)
    seed_all(config.seed)
    index = read_index(config.index_path)
    if Path(config.dataset_root).resolve() != Path(index['dataset_root']).resolve():
        raise ValueError('Training config and task index refer to different source datasets')
    if resume and resume['data_index_sha256'] != index['index_content_sha256']:
        raise ValueError('Resume requires the same index, training statistics, and fixed VAL anchors')
    output = Path(config.output_dir)
    if not resume and output.exists() and any(p.name != 'training.lock' for p in output.iterdir()):
        raise FileExistsError('Output is not empty; choose another --output-dir or resume PACE last.pt')
    output.mkdir(parents=True, exist_ok=True)
    training_lock = (output / 'training.lock').open('w')
    try:
        fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another trainer is writing to this output directory') from None
    smoke = args.smoke_test or bool(resume and resume['smoke_test'])
    print('Loading existing D: Arrow data and causal per-segment PACE contact context...', flush=True)
    dataset_options = {'alpha': config.phase_alpha, 'history': config.tactile_history}
    train = PACETaskDataset(index, 'train', smoke=smoke, **dataset_options)
    val = PACETaskDataset(index, 'validation', **dataset_options)
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
        generator=torch.Generator().manual_seed(config.seed + 1), **val_workers)
    batches = TrainingBatches(train_loader, sampler, resume['cursor'] if resume else None,
                             resume['config']['batch_size'] if resume else None)
    model = PACEACT(config, index['stats'], initialize_backbone=not bool(resume)).to(config.device).train()
    optimizer = torch.optim.AdamW(model.policy.get_optim_params(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp == 'fp16')
    step, best, best_position, validation, validation_step = 0, float('inf'), float('inf'), None, None
    if resume:
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        step, best, best_position = resume['step'], resume['best_validation_loss'], resume['best_position_l2_m']
        validation, validation_step = resume['metrics'], resume['validation_step']
        restore_rng(resume['rng'])
        del resume
    json_write(output / 'run_config.json', {'config': asdict(config), 'task': index['task'],
        'data_index_sha256': index['index_content_sha256'], 'train_anchors': len(train),
        'validation_anchors': len(val), 'smoke_test': smoke, 'reference_paper': PAPER,
        'architecture': 'Existing deterministic ACT + PACE Phase/Event; no CVAE or Pred.',
        'contact_context': 'Causal segment-local EMA; first-reading initialization and left padding',
        'parameters': sum(p.numel() for p in model.parameters()),
        'best_selection_metric': 'validation.loss', 'position_best_selection_metric': 'validation.position_l2_m',
        'resumed_from': str(args.resume) if args.resume else None})
    metrics_path = output / 'metrics.jsonl'
    append_metrics(metrics_path, {'type': 'resume' if args.resume else 'start', 'step': step,
        'train_anchors': len(train), 'validation_samples': len(val), 'task': index['task'],
        'smoke_test': smoke, 'data_index_sha256': index['index_content_sha256'],
        'position_scale_m': config.position_scale_m, 'phase_alpha': config.phase_alpha,
        'tactile_history': config.tactile_history, 'correction_horizon': config.correction_horizon})
    stopping = []
    def request_stop(signum, _frame):
        if not stopping:
            print('Stop requested; finish this optimizer update and save PACE last.pt.', flush=True)
        stopping.append(signum)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if config.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    since, last_saved_step = time.perf_counter(), step if args.resume else -1
    interval, updates = {}, 0
    while step < config.steps and not stopping:
        for group, base_lr in zip(optimizer.param_groups, (config.learning_rate, config.backbone_lr)):
            group['lr'] = base_lr * lr_factor(step + 1, config)
        optimizer.zero_grad(set_to_none=True)
        current = {}
        for _ in range(config.gradient_accumulation_steps):
            batch = move_batch(next(batches), config.device)
            with autocast_context(config):
                loss, metrics = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite PACE loss after update {step}')
            scaler.scale(loss / config.gradient_accumulation_steps).backward()
            for key, value in {'loss': loss.item(), **metrics}.items():
                current[key] = current.get(key, 0.) + value / config.gradient_accumulation_steps
        scaler.unscale_(optimizer)
        gradients = gradient_metrics(model)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm, error_if_nonfinite=True).item()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        for key, value in current.items():
            interval[key] = interval.get(key, 0.) + value
        updates += 1
        if step % config.log_every == 0 or step == 1 or step == config.steps or stopping:
            append_metrics(metrics_path, {'type': 'train', 'step': step,
                **{'mean_' + key: value / updates for key, value in interval.items()},
                'averaged_updates': updates, 'last_batch_loss': current['loss'],
                'learning_rate': optimizer.param_groups[0]['lr'], 'gradient_norm_before_clip': grad,
                **gradients, 'elapsed_seconds': time.perf_counter() - since, **batches.state_dict(),
                'peak_cuda_memory_gib': torch.cuda.max_memory_allocated() / 2**30 if config.device.startswith('cuda') else None})
            interval, updates = {}, 0
        if not stopping and (step % config.eval_every == 0 or step == config.steps):
            validation = evaluate(model, val_loader, config)
            validation_step = step
            append_metrics(metrics_path, {'type': 'validation', 'step': step, **validation})
            improved_loss = validation['loss'] < best
            improved_position = validation['position_l2_m'] < best_position
            best = min(best, validation['loss'])
            best_position = min(best_position, validation['position_l2_m'])
            for improved, name, metric in ((improved_loss, 'best.pt', 'loss'),
                                           (improved_position, 'best_position.pt', 'position_l2_m')):
                if improved:
                    checkpoint(output / 'checkpoints' / name, model, optimizer, scaler, config, index,
                               step, batches, best, best_position, validation, validation_step, smoke, False)
                    append_metrics(metrics_path, {'type': 'best_checkpoint', 'step': step, 'checkpoint': name,
                        'selection_metric': metric, 'value': validation[metric], 'samples': 1200})
        if step % config.save_every == 0 or step == config.steps or stopping:
            checkpoint(output / 'checkpoints/last.pt', model, optimizer, scaler, config, index, step,
                       batches, best, best_position, validation, validation_step, smoke, True)
            last_saved_step = step
    if stopping and last_saved_step != step:
        checkpoint(output / 'checkpoints/last.pt', model, optimizer, scaler, config, index, step,
                   batches, best, best_position, validation, validation_step, smoke, True)
    result = {'completed_steps': step, 'stopped_by_signal': bool(stopping), 'smoke_test': smoke,
        'best_validation_loss': best if math.isfinite(best) else None,
        'best_position_l2_m': best_position if math.isfinite(best_position) else None,
        'last_validation': validation, 'validation_step': validation_step,
        'last_checkpoint': str(output / 'checkpoints/last.pt'), 'closed_loop_evaluated': False}
    if smoke and validation:
        from ftact import load_checkpoint
        model.eval()
        reference = move_batch(next(iter(val_loader)), config.device)
        with torch.inference_mode(), autocast_context(config):
            expected = model.predict_action_chunk(reference).cpu()
        restored, _ = load_checkpoint(output / 'checkpoints/last.pt', config.device)
        with torch.inference_mode(), autocast_context(config):
            actual = restored.predict_action_chunk(reference).cpu()
        torch.testing.assert_close(actual, expected)
        if not all(math.isfinite(value) and value > 0 for value in gradients.values()):
            raise AssertionError(f'PACE branch gradient check failed: {gradients}')
        result.update(checkpoint_roundtrip_passed=True, prediction_shape=list(actual.shape),
                      phase_event_force_gradients_verified=True)
    json_write(output / 'training_summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
