#!/usr/bin/env python3
"""Train a fresh, single-task SFP0 ACT with exactly 1200 fixed validation anchors."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import signal
import time

from ftact import INSERT_ROOT, EpochBatchSampler, json_write, move_batch
from ftact_v2 import SFPConfig, SFP0ACT, TaskDataset, read_index
from train_ftact import (TrainingBatches, seed_all, seed_worker, autocast_context,
                         rng_state, restore_rng, append_metrics)
import torch
from torch.utils.data import DataLoader


def lr_factor(step, config):
    if config.warmup_steps and step < config.warmup_steps:
        return max(1, step) / config.warmup_steps
    progress = min(1., max(0., (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)))
    return config.minimum_lr_ratio + (1 - config.minimum_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


@torch.inference_mode()
def evaluate(model, loader, config):
    model.eval()
    l1, position, angle = 0., torch.zeros(20, dtype=torch.float64), torch.zeros(20, dtype=torch.float64)
    count = 0
    for batch in loader:
        batch = move_batch(batch, config.device)
        with autocast_context(config):
            predicted = model.normalized_action_chunk(batch)
        if not torch.isfinite(predicted).all():
            raise FloatingPointError('Nonfinite validation output')
        target = batch['action'].float()
        l1 += (predicted - model.normalizer.actions(target)).abs().sum().item()
        physical = model.normalizer.unnormalize_actions(predicted)
        position += (physical[..., :3] - target[..., :3]).norm(dim=-1).sum(0).double().cpu()
        q = torch.nn.functional.normalize(physical[..., 3:], dim=-1)
        dot = (q * target[..., 3:]).sum(-1).abs().clamp(0, 1)
        angle += torch.rad2deg(2 * torch.acos(dot)).sum(0).double().cpu()
        count += len(target)
    if count != 1200:
        raise ValueError(f'Validation used {count} anchors; exactly 1200 required')
    model.train()
    position, angle = position / count, angle / count
    return {'l1': l1 / (count * 20 * 7), 'position_l2_m': position.mean().item(),
        'rotation_deg': angle.mean().item(), 'first_step_position_m': position[0].item(),
        'last_step_position_m': position[-1].item(), 'position_by_horizon_m': position.tolist(),
        'rotation_by_horizon_deg': angle.tolist(), 'samples': count,
        'valid_action_steps': count * 20, 'latent': 'zero', 'action_offset': 1}


def checkpoint(path, model, optimizer, scaler, config, index, step, batches, best, validation, smoke, resumable):
    payload = {'format': 'aic-sfp0-act-v2', 'config': asdict(config), 'stats': index['stats'],
        'task': index['task'], 'data_index_sha256': index['index_content_sha256'],
        'step': step, 'cursor': batches.state_dict(), 'best_validation_l1': best,
        'metrics': validation, 'smoke_test': smoke, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if resumable else None,
        'scaler': scaler.state_dict() if resumable else None,
        'rng': rng_state() if resumable else None, 'action_offset': 1,
        'execution': {'fps': 20, 'chunk_size': 20, 'n_action_steps': 10}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=INSERT_ROOT / 'configs/sfp0_act_v2.json')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke-test', action='store_true', help='Three real updates; validation still uses all fixed 1200 anchors')
    for key in ('index_path', 'output_dir', 'device', 'amp'):
        parser.add_argument('--' + key.replace('_', '-'))
    for key in ('steps', 'batch_size', 'num_workers', 'gradient_accumulation_steps'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    args = parser.parse_args()
    resume = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if resume and (resume.get('format') != 'aic-sfp0-act-v2' or resume.get('optimizer') is None):
        parser.error('Resume requires a v2 last.pt; old v1 weights are deliberately excluded')
    config = SFPConfig(**(resume['config'] if resume else json.loads(args.config.read_text())))
    if args.smoke_test and not resume:
        config.steps, config.batch_size, config.num_workers = 3, 2, 0
        config.log_every, config.eval_every, config.save_every, config.warmup_steps = 1, 3, 3, 1
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        config.output_dir = f'/mnt/d/robot_models/aic_sfp0_act_v2_checks/{stamp}'
    for key in asdict(config):
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    config.index_path = str(Path(config.index_path).resolve())
    config.output_dir = str(Path(config.output_dir).resolve())
    config.validate()
    if resume and config.steps <= resume['step']:
        parser.error('--steps is the total target and must exceed the saved step')
    if resume and config.steps != resume['config']['steps']:
        parser.error('Keep the original total steps on resume so the cosine schedule remains unchanged')
    return args, config, resume


def main():
    args, config, resume = parse_args()
    torch.set_num_threads(config.num_threads)
    seed_all(config.seed)
    index = read_index(config.index_path)
    if Path(config.dataset_root).resolve() != Path(index['dataset_root']).resolve():
        raise ValueError('Training config and task index refer to different source datasets')
    if resume and resume['data_index_sha256'] != index['index_content_sha256']:
        raise ValueError('Resume must use the same task segments, statistics and fixed VAL anchors')
    output = Path(config.output_dir)
    if not resume and output.exists() and any(p.name != 'training.lock' for p in output.iterdir()):
        raise FileExistsError('Output is not empty; choose another --output-dir or resume v2 last.pt')
    output.mkdir(parents=True, exist_ok=True)
    training_lock = (output / 'training.lock').open('w')
    try:
        fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another trainer is already writing to this output directory') from None
    smoke = args.smoke_test or bool(resume and resume['smoke_test'])
    print('Loading single-task anchors using existing D: Arrow caches...', flush=True)
    train, val = TaskDataset(index, 'train', smoke=smoke), TaskDataset(index, 'validation')
    workers = {'num_workers': config.num_workers, 'pin_memory': config.device.startswith('cuda'),
               'worker_init_fn': seed_worker}
    if config.num_workers:
        workers.update(multiprocessing_context='spawn', prefetch_factor=1, persistent_workers=True)
    sampler = EpochBatchSampler(len(train), config.batch_size, config.seed)
    train_loader = DataLoader(train, batch_sampler=sampler,
        generator=torch.Generator().manual_seed(config.seed), **workers)
    # Validation workers do not persist and consume RAM while training runs.
    val_workers = dict(workers)
    if config.num_workers:
        val_workers['persistent_workers'] = False
    val_loader = DataLoader(val, batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(config.seed + 1), **val_workers)
    batches = TrainingBatches(train_loader, sampler, resume['cursor'] if resume else None,
                             resume['config']['batch_size'] if resume else None)
    model = SFP0ACT(config, index['stats'], initialize_backbone=not bool(resume)).to(config.device).train()
    optimizer = torch.optim.AdamW(model.policy.get_optim_params(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp == 'fp16')
    step, best, validation = 0, float('inf'), None
    if resume:
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        step, best, validation = resume['step'], resume['best_validation_l1'], resume['metrics']
        restore_rng(resume['rng'])
        del resume
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / 'run_config.json', {'config': asdict(config), 'task': index['task'],
        'data_index_sha256': index['index_content_sha256'], 'train_anchors': len(train),
        'validation_anchors': len(val), 'smoke_test': smoke, 'architecture': 'ACT with fixed z=0; no posterior',
        'action_times_seconds': [(i + 1) / 20 for i in range(20)],
        'parameters': sum(p.numel() for p in model.parameters()),
        'resumed_from': str(args.resume) if args.resume else None})
    metrics_path = output / 'metrics.jsonl'
    append_metrics(metrics_path, {'type': 'resume' if args.resume else 'start', 'step': step,
        'train_anchors': len(train), 'validation_samples': len(val), 'task': index['task'],
        'smoke_test': smoke, 'data_index_sha256': index['index_content_sha256']})
    stopping = []
    def request_stop(signum, _frame):
        if not stopping:
            print('Stop requested; finish this optimizer update and save last.pt.', flush=True)
        stopping.append(signum)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if config.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    since = time.perf_counter()
    last_saved_step = step if args.resume else -1
    interval_loss = interval_count = 0
    while step < config.steps and not stopping:
        factor = lr_factor(step + 1, config)
        for group, base_lr in zip(optimizer.param_groups, (config.learning_rate, config.backbone_lr)):
            group['lr'] = base_lr * factor
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        for _ in range(config.gradient_accumulation_steps):
            batch = move_batch(next(batches), config.device)
            with autocast_context(config):
                loss, _ = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss after update {step}')
            scaler.scale(loss / config.gradient_accumulation_steps).backward()
            loss_value += loss.detach().item() / config.gradient_accumulation_steps
        scaler.unscale_(optimizer)
        force_grad = model.policy.model.encoder_robot_state_input_proj.weight.grad[:, 6:].float().norm().item()
        if smoke and not (math.isfinite(force_grad) and force_grad > 0):
            raise AssertionError('Force channels did not receive a finite nonzero gradient')
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm, error_if_nonfinite=True).item()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        interval_loss += loss_value
        interval_count += 1
        if step % config.log_every == 0 or step == 1 or step == config.steps or stopping:
            append_metrics(metrics_path, {'type': 'train', 'step': step,
                'mean_l1': interval_loss / interval_count, 'averaged_updates': interval_count,
                'last_batch_l1': loss_value, 'learning_rate': optimizer.param_groups[0]['lr'],
                'gradient_norm_before_clip': grad, 'force_projection_gradient_norm': force_grad,
                'elapsed_seconds': time.perf_counter() - since, **batches.state_dict(),
                'peak_cuda_memory_gib': torch.cuda.max_memory_allocated() / 2**30 if config.device.startswith('cuda') else None})
            interval_loss = interval_count = 0
        if not stopping and (step % config.eval_every == 0 or step == config.steps):
            validation = evaluate(model, val_loader, config)
            append_metrics(metrics_path, {'type': 'validation', 'step': step, **validation})
            if validation['l1'] < best:
                best = validation['l1']
                checkpoint(output / 'checkpoints/best.pt', model, optimizer, scaler, config,
                           index, step, batches, best, validation, smoke, False)
                append_metrics(metrics_path, {'type': 'best_checkpoint', 'step': step, 'l1': best, 'samples': 1200})
        if step % config.save_every == 0 or step == config.steps or stopping:
            checkpoint(output / 'checkpoints/last.pt', model, optimizer, scaler, config,
                       index, step, batches, best, validation, smoke, True)
            last_saved_step = step
    # SIGINT can arrive between the loop's save condition and its next iteration.
    if stopping and last_saved_step != step:
        checkpoint(output / 'checkpoints/last.pt', model, optimizer, scaler, config,
                   index, step, batches, best, validation, smoke, True)
    result = {'completed_steps': step, 'stopped_by_signal': bool(stopping), 'smoke_test': smoke,
        'best_validation_l1': best if math.isfinite(best) else None, 'last_validation': validation,
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
        result.update(checkpoint_roundtrip_passed=True, prediction_shape=list(actual.shape),
                      force_input_gradient_verified=True)
    json_write(output / 'training_summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
