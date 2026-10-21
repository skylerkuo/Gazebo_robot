"""Train ACT with a fine-tuned ResNet18 from successful bottle RGB episodes.

Run with the existing PyTorch venv: python -m robot_bringup.train_act --help.
This is an offline program: it does not initialize ROS, Gazebo or controllers.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from robot_bringup.act_data import (ACTION_SOURCE, DEFAULT_CAMERAS, discover_episodes,
                                    load_episode, normalization_stats, split_episodes)
from robot_bringup.act_model import ACT, ACTConfig, CHECKPOINT_VERSION, act_loss
from robot_bringup.act_policy import image_tensor
from robot_bringup.demonstration_images import JOINT_NAMES

DEFAULT_VALIDATION_SAMPLES = 1024


class TimeBudgetExpired(Exception):
    """Stop at an optimizer boundary, or abort an incomplete validation."""


class TimeBudget:
    def __init__(self, minutes=0., started=None, clock=None):
        self.clock = clock or time.monotonic
        self.started = self.clock() if started is None else started
        self.seconds = minutes * 60
        # Reserve time for final checkpoint writes and worker shutdown. This
        # is a cooperative deadline, not a process kill during disk/GPU I/O.
        self.reserve = min(60., self.seconds * .1)

    def check(self):
        if self.seconds and self.clock() - self.started >= self.seconds - self.reserve:
            raise TimeBudgetExpired()


def apply_preset(args):
    defaults = dict(batch_size=4, accumulation_steps=2, chunk_size=20,
                    image_height=480, image_width=640, hidden_dim=256,
                    validate_every_steps=0, validation_samples=DEFAULT_VALIDATION_SAMPLES, train_backbone=True)
    if args.preset == 'small':
        defaults.update(batch_size=16, accumulation_steps=2, image_height=192, image_width=256,
                        hidden_dim=128, validate_every_steps=200)
    elif args.preset == 'standard':
        defaults.update(image_height=240, image_width=320)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def model_configuration(options):
    architecture = {}
    if options.get('preset') == 'small':
        architecture.update(nheads=4, encoder_layers=1, decoder_layers=2, posterior_layers=1,
                            feedforward_dim=512, latent_dim=16)
    elif options.get('preset') == 'standard':
        architecture.update(encoder_layers=2, decoder_layers=4, posterior_layers=2)
    return ACTConfig(cameras=tuple(options['cameras']), chunk_size=options['chunk_size'],
                     hidden_dim=options['hidden_dim'], train_backbone=options['train_backbone'], **architecture)


def validation_subset(dataset, limit):
    if not limit or len(dataset) <= limit:
        return dataset
    # Validation always uses frame mode: these indices identify fixed
    # observations spread across the held-out episodes and their timelines.
    indices = np.linspace(0, len(dataset) - 1, limit, dtype=np.int64).tolist()
    return Subset(dataset, indices)


class ChunkDataset(Dataset):
    def __init__(self, episodes, statistics, chunk_size, image_size, sampling='episode', samples_per_episode=1):
        self.episodes = episodes
        self.ends = np.cumsum([len(episode.starts) for episode in episodes])
        self.statistics = {key: np.asarray(value, dtype=np.float32) for key, value in statistics.items()}
        self.chunk_size, self.image_size = chunk_size, image_size
        if sampling not in ('episode', 'frame'):
            raise ValueError('Unknown sampling mode')
        if samples_per_episode < 1:
            raise ValueError('samples_per_episode must be positive')
        if sampling == 'frame' and samples_per_episode != 1:
            raise ValueError('--samples-per-episode requires --sampling episode')
        self.sampling = sampling
        self.samples_per_episode = samples_per_episode

    def __len__(self):
        return len(self.episodes) * self.samples_per_episode if self.sampling == 'episode' else int(self.ends[-1])

    def __getitem__(self, index):
        if self.sampling == 'episode':
            # Repeat episode indices, not observations: every access draws a
            # fresh random start, and each episode keeps equal sampling weight.
            episode = self.episodes[index % len(self.episodes)]
            start = int(np.random.choice(episode.starts))
        else:
            episode_index = int(np.searchsorted(self.ends, index, side='right'))
            offset = int(self.ends[episode_index - 1]) if episode_index else 0
            episode = self.episodes[episode_index]
            start = int(episode.starts[index - offset])
        images = []
        for path in episode.image_paths[start]:
            with Image.open(path) as image:
                images.append(image_tensor(image, self.image_size))
        actions, is_pad = episode.chunk(start, self.chunk_size)
        qpos = (episode.qpos[start] - self.statistics['qpos_mean']) / self.statistics['qpos_std']
        actions = (actions - self.statistics['action_mean']) / self.statistics['action_std']
        return (torch.stack(images), torch.from_numpy(qpos), torch.from_numpy(actions),
                torch.from_numpy(is_pad))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def choose_device(requested):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: auto, cpu, cuda, cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable. Install a compatible GPU PyTorch build or use --device cpu')
        # Tensor.to accepts bare 'cuda', but cuda.set_device requires an
        # explicit index. Resolve both --device cuda and auto before scanning.
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index {index} is unavailable')
        device = torch.device('cuda', index)
    return device


def make_optimizer(model, options):
    groups = [{'params': [p for name, p in model.named_parameters()
                          if not name.startswith('backbone.') and p.requires_grad],
               'lr': options['learning_rate']}]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad] if hasattr(model, 'backbone') else []
    if backbone:
        groups.append({'params': backbone, 'lr': options['backbone_learning_rate']})
    return torch.optim.AdamW(groups, weight_decay=options['weight_decay'])


def validate(model, loader, device, amp_context, action_std, budget=None, kl_weight=10., seed=42):
    """Fixed posterior draws for comparable validation; separate z=0 metrics.

    Forking RNG state also prevents validation frequency from changing the
    subsequent training dropout masks or posterior samples.
    """
    model.eval()
    error_sum, physical_error, count = 0., torch.zeros(7, device=device), 0
    posterior_sum = np.zeros(3, dtype=np.float64)
    batches = 0
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        for batch in loader:
            if budget is not None:
                budget.check()
            images, qpos, actions, is_pad = [item.to(device, non_blocking=True) for item in batch]
            with amp_context():
                reconstructed, mu, logvar = model(qpos, images, actions, is_pad)
                posterior = act_loss(reconstructed, actions, is_pad, mu, logvar, kl_weight)
                # This second forward deliberately has NO future actions.
                prediction, _, _ = model(qpos, images)
            posterior_sum += [value.item() for value in posterior]
            batches += 1
            valid = (~is_pad).unsqueeze(-1)
            error = (prediction.float() - actions.float()).abs() * valid
            error_sum += error.sum().item()
            physical_error += (error * action_std).sum((0, 1))
            count += int(valid.sum().item())
    if not count:
        raise ValueError('No valid validation actions')
    loss, l1, kl = posterior_sum / batches
    return {'val_posterior_loss': float(loss), 'val_posterior_l1': float(l1), 'val_posterior_kl': float(kl),
            'val_prior_l1': error_sum / (count * 7), 'val_joint_mae_rad': (physical_error / count).cpu().tolist()}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/mnt/d/robot_datasets/bottle_pick_place_rgb'))
    parser.add_argument('--output', type=Path, default=Path('runs/act_bottle'), help='Parent of a new timestamped run')
    parser.add_argument('--resume', type=Path, help='Restore last.pt; restores data/model/settings, except options below')
    parser.add_argument('--epochs', type=int, help='Total episode-sampling epochs; default 2000; can extend on resume')
    parser.add_argument('--preset', choices=('medium', 'standard', 'small'), default='medium')
    parser.add_argument('--sampling', choices=('episode', 'frame'), default='episode',
                        help='Training only: episode draws random starts; frame visits all observations. VAL uses fixed frames')
    parser.add_argument('--samples-per-episode', type=int,
                        help='Random training observations per episode per epoch; default 1, can override on resume')
    parser.add_argument('--max-minutes', type=float,
                        help='Wall-time budget including setup; reserves up to 60s to save. 0 disables it')
    parser.add_argument('--device', default='auto', help='auto, cuda, cuda:0 or cpu; also applies on resume')
    parser.add_argument('--workers', type=int, default=2, help='Image loader workers; also applies on resume')
    parser.add_argument('--cameras', nargs='+', choices=DEFAULT_CAMERAS, default=list(DEFAULT_CAMERAS))
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--chunk-size', type=int, help='At 10 FPS, 20 means a two-second action chunk')
    parser.add_argument('--image-height', type=int)
    parser.add_argument('--image-width', type=int)
    parser.add_argument('--hidden-dim', type=int)
    parser.add_argument('--validate-every-steps', type=int, help='Optimizer steps per validation; 0 means epoch-only')
    parser.add_argument('--save-every-epochs', type=int, default=100,
                        help='Full last.pt interval; also saves first/final epoch and at the time budget')
    parser.add_argument('--validation-samples', type=int,
                        help=f'Fixed validation frames spread across held-out episodes; default {DEFAULT_VALIDATION_SAMPLES}, 0 means all')
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--backbone-learning-rate', type=float, default=1e-5)
    parser.add_argument('--kl-weight', type=float, default=10.)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    backbone = parser.add_mutually_exclusive_group()
    backbone.add_argument('--train-backbone', action='store_true', default=None, help='Fine-tune shared ResNet18 (default)')
    backbone.add_argument('--freeze-backbone', dest='train_backbone', action='store_false', help='Optional frozen-CNN ablation')
    parser.add_argument('--random-backbone', action='store_true', help='No download; requires --train-backbone')
    parser.add_argument('--no-amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--grad-clip', type=float, default=0., help='Gradient norm limit; 0 matches official unclipped updates')
    parser.add_argument('--val-fraction', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=float, default=10.)
    parser.add_argument('--timing-tolerance-ms', type=float, default=25.)
    args = parser.parse_args(argv)
    # Preserve None on resume, so an omitted override keeps the saved limit.
    validation_samples_override = args.validation_samples
    args = apply_preset(args)
    args.validation_samples_override = validation_samples_override
    if args.epochs is not None and args.epochs < 1:
        parser.error('epochs must be positive')
    if args.workers < 0:
        parser.error('workers must be nonnegative')
    if args.samples_per_episode is not None and args.samples_per_episode < 1:
        parser.error('samples-per-episode must be positive')
    if args.save_every_epochs < 1:
        parser.error('save-every-epochs must be positive')
    if args.max_minutes is not None and (not math.isfinite(args.max_minutes) or args.max_minutes < 0):
        parser.error('max-minutes must be finite and nonnegative')
    if args.validation_samples < 0:
        parser.error('validation sample limit must be nonnegative')
    if not args.resume:
        if min(args.batch_size, args.accumulation_steps, args.chunk_size, args.image_height, args.image_width) < 1:
            parser.error('batch/chunk/image sizes and accumulation must be positive')
        if min(args.image_height, args.image_width) < 32:
            parser.error('image dimensions must be at least 32 pixels')
        if min(args.validate_every_steps, args.validation_samples) < 0:
            parser.error('validation steps/sample limits must be nonnegative')
        if args.random_backbone and not args.train_backbone:
            parser.error('--random-backbone requires --train-backbone')
        numeric = (args.learning_rate, args.backbone_learning_rate, args.kl_weight, args.weight_decay,
                   args.fps, args.timing_tolerance_ms, args.val_fraction, args.grad_clip)
        if not all(math.isfinite(x) for x in numeric) or min(numeric) < 0:
            parser.error('numeric options must be finite and nonnegative')
        if min(args.learning_rate, args.backbone_learning_rate, args.fps) <= 0:
            parser.error('learning rates and fps must be positive')
    return args


def main(argv=None):
    launched = time.monotonic()
    args = arguments(argv)
    device = choose_device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    checkpoint = None
    if args.resume:
        resume = args.resume.expanduser().resolve()
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != CHECKPOINT_VERSION or checkpoint.get('action_source') != ACTION_SOURCE:
            raise ValueError('Checkpoint format 2 required; start a new --preset medium run')
        if 'optimizer' not in checkpoint:
            raise ValueError('Resume requires last.pt; best.pt contains inference weights only')
        if checkpoint['joint_names'] != JOINT_NAMES:
            raise ValueError('Checkpoint joint order differs from this robot')
        options = dict(checkpoint['run_config'])
        options.update(epochs=args.epochs if args.epochs is not None else options['epochs'], workers=args.workers)
        options['samples_per_episode'] = (args.samples_per_episode if args.samples_per_episode is not None
                                          else options.get('samples_per_episode', 1))
        if args.validation_samples_override is not None:
            options['validation_samples'] = args.validation_samples_override
        elif 'validation_settings' not in checkpoint:
            # Old limits counted episodes; migrate to the fixed-frame default.
            options['validation_samples'] = DEFAULT_VALIDATION_SAMPLES
        if args.max_minutes is not None:
            options['max_minutes'] = args.max_minutes
        run = resume.parent
        config = ACTConfig(**checkpoint['model_config'])
        manifest = checkpoint['manifest']
        train_episodes = [load_episode(path, config.cameras, options['fps'], options['timing_tolerance_ms'])
                          for path in manifest['train']]
        val_episodes = [load_episode(path, config.cameras, options['fps'], options['timing_tolerance_ms'])
                        for path in manifest['validation']]
        statistics = checkpoint['normalization']
    else:
        options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                   if key != 'validation_samples_override'}
        options['epochs'] = args.epochs or 2000
        options['samples_per_episode'] = args.samples_per_episode or 1
        config = model_configuration(options)
        print(f'Reading successful RGB episodes from {args.dataset}; checking image paths...', flush=True)
        episodes, skipped = discover_episodes(args.dataset, config.cameras, args.fps, args.timing_tolerance_ms)
        train_episodes, val_episodes = split_episodes(episodes, args.val_fraction, args.seed)
        statistics = normalization_stats(train_episodes)
        manifest = {'train': [str(episode.directory) for episode in train_episodes],
                    'validation': [str(episode.directory) for episode in val_episodes], 'skipped': skipped}
        run = args.output.expanduser().resolve() / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
        run.mkdir(parents=True, exist_ok=False)

    budget = TimeBudget(options.get('max_minutes') or 0., started=launched)
    random.seed(options['seed'])
    np.random.seed(options['seed'])
    torch.manual_seed(options['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(options['seed'])
    image_size = (options['image_height'], options['image_width'])
    train_data = ChunkDataset(train_episodes, statistics, config.chunk_size, image_size, options['sampling'],
                              options['samples_per_episode'])
    val_data = validation_subset(ChunkDataset(val_episodes, statistics, config.chunk_size, image_size, 'frame'),
                                 options.get('validation_samples', 0))
    validation_settings = dict(version=1, sampling='fixed_frames', seed=options['seed'],
                               sample_limit=options.get('validation_samples', 0), batch_size=options['batch_size'])
    loader_options = dict(batch_size=options['batch_size'], num_workers=options['workers'],
                          pin_memory=device.type == 'cuda', persistent_workers=options['workers'] > 0,
                          timeout=30 if options['workers'] else 0)
    generator = torch.Generator()
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_generator = torch.Generator().manual_seed(options['seed'])
    val_loader = DataLoader(val_data, shuffle=False, generator=val_generator, **loader_options)
    model = ACT(config, pretrained=checkpoint is None and not options['random_backbone']).to(device)
    optimizer = make_optimizer(model, options)
    amp_enabled = device.type == 'cuda' and not options['no_amp']
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and amp_dtype == torch.float16)

    def amp_context():
        return torch.autocast('cuda', dtype=amp_dtype) if amp_enabled else nullcontext()

    first_epoch, global_step, best = 0, 0, float('inf')
    baseline_policy = None
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # CPU/bf16 runs have no scaling state. Switching to an fp16 device on
        # resume should start a new scaler rather than loading an empty state.
        if scaler.is_enabled() and checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])
        first_epoch, global_step, best = checkpoint['epoch'] + 1, checkpoint['global_step'], checkpoint['best_val_loss']
        if checkpoint.get('validation_settings') != validation_settings:
            best = float('inf')
            baseline_policy = checkpoint
        # best.pt can be newer than the periodic full recovery checkpoint.
        # Keep that policy's score so resuming an older last.pt cannot replace
        # it with a worse policy merely because last.pt has an older best score.
        if (run / 'best.pt').is_file():
            retained = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
            if retained.get('format_version') != CHECKPOINT_VERSION or retained['model_config'] != asdict(config):
                raise ValueError('best.pt and last.pt have incompatible configurations')
            if retained.get('validation_settings') == validation_settings:
                best = min(best, retained['best_val_loss'])
                baseline_policy = None
            else:
                baseline_policy = retained
        torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng'])
        numpy_state = checkpoint['numpy_rng']
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        if device.type == 'cuda' and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    if first_epoch >= options['epochs']:
        raise ValueError('Checkpoint already reached --epochs; set a larger total epoch count')
    write_json(run / 'config.json', {'model': asdict(config), 'training': options,
                                   'validation_settings': validation_settings, 'action_source': ACTION_SOURCE,
                                   'joint_names': JOINT_NAMES, 'image_size': image_size})
    write_json(run / 'split.json', manifest)
    write_json(run / 'normalization.json', statistics)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / options['accumulation_steps'])
    sampling_description = options['sampling']
    if options['sampling'] == 'episode':
        sampling_description += f'; {options["samples_per_episode"]} random observations per episode'
    print(f'Run: {run}\nDevice: {device}\nTrain/validation episodes: {len(train_episodes)}/{len(val_episodes)}\n'
          f'Training samples per epoch: {len(train_data)} ({sampling_description}); '
          f'skipped episodes: {len(manifest["skipped"])}\n'
          f'Optimizer steps per epoch: {optimizer_steps_per_epoch}\n'
          f'Validation: {len(val_data)} fixed frames, sequential order, fixed posterior seed {options["seed"]}\n'
          f'Inputs: {list(config.cameras)} RGB + current 7-joint qpos\n'
          f'Output: {config.chunk_size} future measured joint positions at {options["fps"]:g} Hz; no goal pose\n'
          f'Backbone trainable: {config.train_backbone}; AMP: {amp_enabled}; '
          f'batch: {options["batch_size"]} x accumulation {options["accumulation_steps"]}\n'
          f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n'
          f'Time budget: {options.get("max_minutes") or "unlimited"} minutes; '
          f'validation observations: {len(val_data)}', flush=True)
    action_std = torch.tensor(statistics['action_std'], device=device)
    epoch, epoch_complete, epoch_batches = first_epoch, False, 0

    def payload():
        # An incomplete epoch resumes from its beginning using the saved
        # updated weights/optimizer. No unfinished gradients are checkpointed.
        numpy_state = np.random.get_state()
        return {'format_version': CHECKPOINT_VERSION, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(), 'model_config': asdict(config), 'run_config': options,
                'normalization': statistics, 'manifest': manifest, 'image_size': list(image_size),
                'joint_names': JOINT_NAMES, 'action_source': ACTION_SOURCE, 'fps': options['fps'],
                'epoch': epoch if epoch_complete else epoch - 1,
                'partial_epoch': None if epoch_complete else epoch, 'partial_epoch_batches': epoch_batches,
                'global_step': global_step, 'best_val_loss': best, 'selection_metric': 'val_posterior_loss',
                'validation_settings': validation_settings,
                'python_rng': random.getstate(), 'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}

    def evaluate_and_save():
        nonlocal best, baseline_policy
        if baseline_policy is not None:
            # Scores from random episodes/different frame limits cannot be
            # compared. Re-score the retained policy without touching AdamW.
            print('Re-evaluating saved best policy on the fixed validation frames...', flush=True)
            current_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            try:
                model.load_state_dict(baseline_policy['model'])
                baseline = validate(model, val_loader, device, amp_context, action_std, budget,
                                    options['kl_weight'], seed=options['seed'])
            finally:
                model.load_state_dict(current_weights)
            score = baseline['val_posterior_loss']
            if not math.isfinite(score):
                raise RuntimeError('Nonfinite baseline validation loss; saved policy retained')
            retained = {key: value for key, value in baseline_policy.items() if key not in
                        ('optimizer', 'scaler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng')}
            retained.update(best_val_loss=score, validation_settings=validation_settings)
            retained['run_config'] = dict(retained['run_config'], validation_samples=options['validation_samples'])
            save_checkpoint(run / 'best.pt', retained)
            best, baseline_policy = score, None
        validation = validate(model, val_loader, device, amp_context, action_std, budget, options['kl_weight'],
                              seed=options['seed'])
        val_loss = validation['val_posterior_loss']
        if not all(math.isfinite(value) for key, value in validation.items() if key != 'val_joint_mae_rad'):
            raise RuntimeError('Nonfinite validation loss; previous checkpoint retained')
        improved = val_loss < best
        best = min(best, val_loss)
        metrics = {'event': 'epoch' if epoch_complete else 'validation', 'epoch': epoch + 1,
                   'epoch_complete': epoch_complete, 'epoch_batches': epoch_batches, 'global_step': global_step,
                   'train_l1': total_l1 / samples, 'train_kl': total_kl / samples, **validation,
                   'validation_observations': len(val_data),
                   'training_samples_per_epoch': len(train_data),
                   'optimizer_steps_per_epoch': optimizer_steps_per_epoch,
                   'validation_settings': validation_settings,
                   'seconds': time.monotonic() - started}
        with (run / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n')
        save_full = (epoch_complete and (epoch == first_epoch or (epoch + 1) % options['save_every_epochs'] == 0))
        # Explicit step-based validation keeps its corresponding recovery point.
        save_full = save_full or (not epoch_complete and bool(options.get('validate_every_steps')))
        snapshot = payload() if save_full or improved else None
        if save_full:
            save_checkpoint(run / 'last.pt', snapshot)
        if improved:
            policy_snapshot = {key: value for key, value in snapshot.items() if key not in
                               ('optimizer', 'scaler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng')}
            save_checkpoint(run / 'best.pt', policy_snapshot)
        print(f'Epoch {epoch + 1}/{options["epochs"]}, step {global_step}: train L1={metrics["train_l1"]:.5f}, '
              f'KL={metrics["train_kl"]:.5f}, val posterior loss={val_loss:.5f}, '
              f'val prior L1={validation["val_prior_l1"]:.5f}, '
              f'{metrics["seconds"]:.1f}s' + (' [best]' if improved else ''), flush=True)
        model.train()

    stop_reason = 'epochs_completed'
    try:
        for epoch in range(first_epoch, options['epochs']):
            started = time.monotonic()
            epoch_complete, epoch_batches = False, 0
            generator.manual_seed(options['seed'] + epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_l1, total_kl, samples = 0., 0., 0
            report_at = time.monotonic() + 30
            for step, batch in enumerate(train_loader):
                if step % options['accumulation_steps'] == 0:
                    # Finish a whole accumulation group before stopping, so
                    # last.pt never silently drops pending gradients.
                    budget.check()
                images, qpos, actions, is_pad = [item.to(device, non_blocking=True) for item in batch]
                group_start = (step // options['accumulation_steps']) * options['accumulation_steps']
                group_samples = min(options['batch_size'] * options['accumulation_steps'],
                                    len(train_data) - group_start * options['batch_size'])
                size = qpos.shape[0]
                with amp_context():
                    prediction, mu, logvar = model(qpos, images, actions, is_pad)
                    loss, l1, kl = act_loss(prediction, actions, is_pad, mu, logvar, options['kl_weight'])
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite training loss in epoch {epoch + 1}; previous checkpoint retained')
                scaler.scale(loss * size / group_samples).backward()
                total_l1 += l1.item() * size
                total_kl += kl.item() * size
                samples += size
                epoch_batches = step + 1
                boundary = epoch_batches % options['accumulation_steps'] == 0 or epoch_batches == len(train_loader)
                if boundary:
                    scaler.unscale_(optimizer)
                    if options['grad_clip']:
                        nnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), options['grad_clip'])
                    else:
                        nnorm = torch.stack([p.grad.detach().float().norm() for p in model.parameters()
                                            if p.grad is not None]).norm()
                    if not torch.isfinite(nnorm) and not scaler.is_enabled():
                        raise RuntimeError('Nonfinite gradients; previous checkpoint retained')
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    interval = options.get('validate_every_steps', 0)
                    if interval and (global_step == 1 or global_step % interval == 0):
                        evaluate_and_save()
                if time.monotonic() >= report_at:
                    print(f'  Epoch {epoch + 1}: batch {epoch_batches}/{len(train_loader)}, '
                          f'optimizer steps {global_step}, '
                          f'elapsed {(time.monotonic() - launched) / 60:.1f} min', flush=True)
                    report_at = time.monotonic() + 30
            epoch_complete = True
            evaluate_and_save()
    except TimeBudgetExpired:
        stop_reason = 'time_budget_reached'
        print('Time budget reached; saving current weights and optimizer.', flush=True)
        save_checkpoint(run / 'last.pt', payload())
    if stop_reason == 'epochs_completed':
        save_checkpoint(run / 'last.pt', payload())
    write_json(run / 'run_status.json', {'state': stop_reason, 'global_step': global_step,
                                       'elapsed_seconds': time.monotonic() - launched,
                                       'epoch_complete': epoch_complete, 'epoch_batches': epoch_batches})
    if (run / 'best.pt').is_file():
        print(f'Finished ({stop_reason}). Best checkpoint: {run / "best.pt"}; latest: {run / "last.pt"}', flush=True)
    else:
        print(f'Finished ({stop_reason}). Latest checkpoint: {run / "last.pt"}. '
              'No complete validation yet; best.pt has not been selected.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
