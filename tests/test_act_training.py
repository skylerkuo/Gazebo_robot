"""Contract tests for future labels, episode isolation and ACT padding/prior.

These tests are provided for later execution; writing the trainer does not run
them, download pretrained weights or start any training job.
"""
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robot_bringup.act_data import (DEFAULT_CAMERAS, discover_episodes, load_episode,
                                    normalization_stats, split_episodes)
from robot_bringup.demonstration_images import JOINT_NAMES


class ACTDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def episode(self, name='episode_0000', times=(1000, 1100, 1200, 1300, 1400), offset=0.):
        directory = self.root / name
        directory.mkdir(parents=True)
        metadata = {'schema_version': 3, 'robot': 'kinova_gen3_lite', 'status': 'placement_verified',
                    'training_candidate': True, 'execution_success': True,
                    'image_check': {'verified': True}, 'placement_check': {'verified': True},
                    'task_status': {'execution_start_ns': times[0] * 1_000_000,
                                    'execution_end_ns': times[-1] * 1_000_000},
                    'cameras': {name: {'fps': 10.} for name in DEFAULT_CAMERAS}}
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        rows = []
        for index, time_ms in enumerate(times):
            stamp = time_ms * 1_000_000
            images = {}
            for camera in DEFAULT_CAMERAS:
                path = Path('images') / camera / f'{stamp}.jpg'
                (directory / path).parent.mkdir(parents=True, exist_ok=True)
                (directory / path).write_bytes(b'fixture')
                images[camera] = {'stamp_ns': stamp, 'path': str(path)}
            rows.append({'stamp_ns': stamp, 'images': images, 'joint_names': list(JOINT_NAMES),
                         'joint_state': {'stamp_ns': stamp, 'qpos': [offset + index + j * .01 for j in range(7)]}})
        self.write_rows(directory, rows)
        return directory, metadata, rows

    @staticmethod
    def write_rows(directory, rows):
        (directory / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_labels_are_future_states_and_never_current_qpos(self):
        directory, _, _ = self.episode()
        episode = load_episode(directory)
        actions, mask = episode.chunk(0, 3)
        np.testing.assert_allclose(actions, episode.qpos[1:4])
        self.assertFalse(mask.any())
        self.assertFalse(np.array_equal(actions[0], episode.qpos[0]))

    def test_tail_is_masked_and_no_chunk_crosses_a_gap(self):
        directory, _, _ = self.episode(times=(1000, 1100, 1300, 1400))
        episode = load_episode(directory)
        np.testing.assert_array_equal(episode.starts, [0, 2])
        actions, mask = episode.chunk(0, 3)
        np.testing.assert_array_equal(mask, [False, True, True])
        np.testing.assert_allclose(actions[0], episode.qpos[1])
        actions, mask = episode.chunk(2, 3)
        np.testing.assert_array_equal(mask, [False, True, True])
        np.testing.assert_allclose(actions[0], episode.qpos[3])

    def test_execution_bounds_remove_warmup_and_hold(self):
        directory, metadata, _ = self.episode()
        metadata['task_status'].update(execution_start_ns=1_100_000_000, execution_end_ns=1_300_000_000)
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        episode = load_episode(directory)
        np.testing.assert_array_equal(episode.stamps, [1_100_000_000, 1_200_000_000, 1_300_000_000])

    def test_failed_or_incomplete_episode_cannot_become_training_data(self):
        good, _, _ = self.episode()
        bad, metadata, _ = self.episode('episode_0001')
        metadata.update(status='failed', training_candidate=False)
        (bad / 'metadata.json').write_text(json.dumps(metadata))
        episodes, skipped = discover_episodes(self.root)
        self.assertEqual([episode.directory for episode in episodes], [good])
        self.assertEqual(len(skipped), 1)

    def test_split_is_by_whole_episode_and_statistics_use_training_only(self):
        first, _, _ = self.episode(offset=0.)
        second, _, _ = self.episode('episode_0001', offset=100.)
        episodes = [load_episode(first), load_episode(second)]
        train, validation = split_episodes(episodes, .5, 7)
        self.assertNotEqual(train[0].directory, validation[0].directory)
        stats = normalization_stats(train)
        np.testing.assert_allclose(stats['qpos_mean'], train[0].qpos[train[0].starts].mean(0), rtol=1e-5)
        np.testing.assert_allclose(stats['action_mean'], train[0].qpos[train[0].starts + 1].mean(0), rtol=1e-5)
        with self.assertRaises(ValueError):
            split_episodes(train)

    def test_stale_nonfinite_noncausal_and_missing_images_are_rejected(self):
        for kind in ('stale', 'future', 'nan', 'missing', 'escape'):
            with self.subTest(kind=kind):
                directory, _, rows = self.episode('episode_' + kind)
                if kind == 'stale':
                    rows[1]['joint_state']['stamp_ns'] -= 51_000_000
                elif kind == 'future':
                    rows[1]['joint_state']['stamp_ns'] += 1
                elif kind == 'nan':
                    rows[1]['joint_state']['qpos'][0] = float('nan')
                elif kind == 'missing':
                    (directory / rows[1]['images']['scene']['path']).unlink()
                else:
                    rows[1]['images']['scene']['path'] = '../outside.jpg'
                self.write_rows(directory, rows)
                with self.assertRaises(ValueError):
                    load_episode(directory)

    def test_joint_order_is_canonicalized(self):
        directory, _, rows = self.episode()
        for row in rows:
            row['joint_names'].reverse()
            row['joint_state']['qpos'].reverse()
        self.write_rows(directory, rows)
        episode = load_episode(directory)
        np.testing.assert_allclose(episode.qpos[0], np.arange(7) * .01, atol=1e-6)


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional ACT PyTorch environment is not installed')
class ACTModelTest(unittest.TestCase):
    def test_loss_ignores_padding_and_backward_is_finite(self):
        import torch
        from robot_bringup.act_model import act_loss
        prediction = torch.ones(1, 3, 7, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[:, 1:] = 999.
        padding = torch.tensor([[False, True, True]])
        mu, logvar = torch.zeros(1, 2), torch.zeros(1, 2)
        loss, l1, kl = act_loss(prediction, target, padding, mu, logvar)
        self.assertAlmostEqual(l1.item(), 1. / 3.)  # Official mean includes masked padding slots.
        self.assertEqual(kl.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad[:, 1:].abs().sum().item(), 0.)

    def test_prior_forward_and_posterior_padding(self):
        import torch
        from robot_bringup.act_model import ACT, ACTConfig
        config = ACTConfig(cameras=('scene',), chunk_size=3, hidden_dim=32, latent_dim=4,
                           nheads=4, encoder_layers=1, decoder_layers=1, posterior_layers=1,
                           feedforward_dim=64, dropout=0., train_backbone=False)
        model = ACT(config, pretrained=False).eval()
        qpos = torch.zeros(1, 7)
        images = torch.rand(1, 1, 3, 64, 64)
        actions = torch.zeros(1, 3, 7)
        padding = torch.tensor([[False, True, True]])
        with torch.no_grad():
            prediction, mu, logvar = model(qpos, images)
            self.assertEqual(tuple(prediction.shape), (1, 3, 7))
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertIsNone(mu)
            self.assertIsNone(logvar)
            _, mu_first, _ = model(qpos, images, actions, padding)
            actions[:, 1:] = 99.
            _, mu_second, _ = model(qpos, images, actions, padding)
            torch.testing.assert_close(mu_first, mu_second)
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        model.train()
        from robot_bringup.act_model import FrozenBatchNorm2d
        self.assertTrue(any(isinstance(module, FrozenBatchNorm2d) for module in model.backbone.modules()))

    def test_checkpoint_policy_roundtrip_needs_only_current_observation(self):
        import torch
        from PIL import Image
        from dataclasses import asdict
        from robot_bringup.act_data import ACTION_SOURCE
        from robot_bringup.act_model import ACT, ACTConfig, CHECKPOINT_VERSION
        from robot_bringup.act_policy import ACTPolicy
        config = ACTConfig(cameras=('scene',), chunk_size=3, hidden_dim=32, latent_dim=4,
                           nheads=4, encoder_layers=1, decoder_layers=1, posterior_layers=1,
                           feedforward_dim=64, dropout=0.)
        model = ACT(config, pretrained=False).eval()
        payload = {'format_version': CHECKPOINT_VERSION, 'action_source': ACTION_SOURCE, 'joint_names': JOINT_NAMES,
                   'model_config': asdict(config), 'model': model.state_dict(), 'image_size': [64, 64],
                   'fps': 10., 'normalization': {'qpos_mean': [0.] * 7, 'qpos_std': [1.] * 7,
                                               'action_mean': [0.] * 7, 'action_std': [1.] * 7}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'test.pt'
            torch.save(payload, path)
            policy = ACTPolicy(path)
            actions = policy.predict({'scene': Image.new('RGB', (64, 64))}, [0.] * 7)
        self.assertEqual(actions.shape, (3, 7))
        self.assertTrue(np.isfinite(actions).all())
        self.assertEqual(policy.control_dt, .1)


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional ACT PyTorch environment is not installed')
class ACTRuntimeTest(unittest.TestCase):
    def test_cuda_and_auto_resolve_the_current_device_index(self):
        from robot_bringup import train_act
        for requested in ('cuda', 'auto'):
            with self.subTest(requested=requested), \
                    patch.object(train_act.torch.cuda, 'is_available', return_value=True), \
                    patch.object(train_act.torch.cuda, 'current_device', return_value=1), \
                    patch.object(train_act.torch.cuda, 'device_count', return_value=2):
                self.assertEqual(str(train_act.choose_device(requested)), 'cuda:1')

    def test_explicit_cuda_index_is_preserved_and_invalid_index_rejected(self):
        from robot_bringup import train_act
        with patch.object(train_act.torch.cuda, 'is_available', return_value=True), \
                patch.object(train_act.torch.cuda, 'current_device') as current, \
                patch.object(train_act.torch.cuda, 'device_count', return_value=1):
            self.assertEqual(str(train_act.choose_device('cuda:0')), 'cuda:0')
            current.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'index 1 is unavailable'):
                train_act.choose_device('cuda:1')

    def test_cpu_and_auto_without_cuda_do_not_initialize_gpu(self):
        from robot_bringup import train_act
        with patch.object(train_act.torch.cuda, 'is_available', return_value=False), \
                patch.object(train_act.torch.cuda, 'current_device') as current:
            self.assertEqual(str(train_act.choose_device('auto')), 'cpu')
            self.assertEqual(str(train_act.choose_device('cpu')), 'cpu')
            current.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'CUDA unavailable'):
                train_act.choose_device('cuda')

    def test_small_preset_and_explicit_overrides(self):
        from robot_bringup.train_act import arguments, model_configuration
        small = arguments(['--preset', 'small'])
        config = model_configuration(vars(small))
        self.assertEqual((small.image_height, small.image_width, small.batch_size), (192, 256, 16))
        self.assertEqual(small.accumulation_steps, 2)
        self.assertEqual(small.batch_size * small.accumulation_steps, 32)
        self.assertEqual((config.hidden_dim, config.encoder_layers, config.decoder_layers), (128, 1, 2))
        self.assertEqual((small.validation_samples, small.validate_every_steps), (1024, 200))
        custom = arguments(['--preset', 'small', '--batch-size', '2', '--hidden-dim', '64'])
        self.assertEqual((custom.batch_size, custom.hidden_dim), (2, 64))
        self.assertEqual(arguments([]).image_width, 640)

    def test_small_model_has_far_fewer_trainable_parameters(self):
        from robot_bringup.train_act import arguments, model_configuration
        from robot_bringup.act_model import ACT
        counts = []
        for preset in ('standard', 'small'):
            model = ACT(model_configuration(vars(arguments(['--preset', preset]))), pretrained=False)
            counts.append(sum(p.numel() for name, p in model.named_parameters()
                              if p.requires_grad and not name.startswith('backbone.')))
        self.assertLess(counts[1], counts[0] / 3)

    def test_small_model_training_forward_and_gradients(self):
        import torch
        from robot_bringup.train_act import arguments, model_configuration
        from robot_bringup.act_model import ACT, act_loss
        config = model_configuration(vars(arguments(['--preset', 'small'])))
        model = ACT(config, pretrained=False).train()
        actions = torch.zeros(1, config.chunk_size, 7)
        padding = torch.zeros(1, config.chunk_size, dtype=torch.bool)
        prediction, mu, logvar = model(torch.zeros(1, 7), torch.rand(1, 3, 3, 64, 64), actions, padding)
        self.assertEqual(tuple(prediction.shape), (1, 20, 7))
        loss, _, _ = act_loss(prediction, actions, padding, mu, logvar)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))

    def test_budget_reserves_save_time_and_zero_disables_it(self):
        from robot_bringup.train_act import TimeBudget, TimeBudgetExpired
        now = [100.]
        budget = TimeBudget(90, clock=lambda: now[0])
        now[0] = 100 + 89 * 60 - 1
        budget.check()
        now[0] += 1
        with self.assertRaises(TimeBudgetExpired):
            budget.check()
        TimeBudget(0, started=0, clock=lambda: 1e10).check()

    def test_validation_subset_covers_entire_held_out_set(self):
        from robot_bringup.train_act import validation_subset
        selected = validation_subset(list(range(1000)), 32)
        self.assertEqual(len(selected), 32)
        self.assertEqual((selected[0], selected[-1]), (0, 999))
        self.assertEqual(list(selected), list(validation_subset(list(range(1000)), 32)))

    @staticmethod
    def fixture(root):
        from PIL import Image
        for number in range(2):
            directory = root / 'data' / f'episode_{number:04d}'
            directory.mkdir(parents=True)
            metadata = {'schema_version': 3, 'robot': 'kinova_gen3_lite', 'status': 'placement_verified',
                        'training_candidate': True, 'execution_success': True,
                        'image_check': {'verified': True}, 'placement_check': {'verified': True},
                        'task_status': {'execution_start_ns': 1_000_000_000, 'execution_end_ns': 1_300_000_000},
                        'cameras': {name: {'fps': 10.} for name in DEFAULT_CAMERAS}}
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            for camera in DEFAULT_CAMERAS:
                Image.new('RGB', (32, 32)).save(directory / f'{camera}.jpg')
            rows = []
            for index in range(4):
                stamp = 1_000_000_000 + index * 100_000_000
                rows.append({'stamp_ns': stamp, 'joint_names': JOINT_NAMES,
                             'joint_state': {'stamp_ns': stamp, 'qpos': [float(index)] * 7},
                             'images': {name: {'stamp_ns': stamp, 'path': f'{name}.jpg'}
                                        for name in DEFAULT_CAMERAS}})
            (directory / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def run_budget_fixture(self, root, minutes):
        import torch
        from torch import nn
        from robot_bringup import train_act
        self.fixture(root)
        now = [0.]

        class TinyTestModel(nn.Module):
            def __init__(self, config, pretrained=False):
                super().__init__()
                self.config = config
                self.head = nn.Linear(7, 7)

            def forward(self, qpos, images, actions=None, is_pad=None):
                now[0] += 1. if actions is not None else .01
                prediction = self.head(qpos)[:, None].expand(-1, self.config.chunk_size, -1)
                if actions is None:
                    return prediction, None, None
                mu = qpos.new_zeros(qpos.shape[0], self.config.latent_dim)
                return prediction, mu, torch.zeros_like(mu)

        command = ['--dataset', str(root / 'data'), '--output', str(root / 'runs'), '--epochs', '100',
                   '--device', 'cpu', '--workers', '0', '--batch-size', '1', '--accumulation-steps', '2',
                   '--image-height', '32', '--image-width', '32', '--validate-every-steps', '1',
                   '--validation-samples', '2', '--random-backbone', '--train-backbone', '--sampling', 'frame',
                   '--max-minutes', str(minutes)]
        with patch.object(train_act, 'ACT', TinyTestModel), \
                patch.object(train_act.time, 'monotonic', side_effect=lambda: now[0]), redirect_stdout(io.StringIO()):
            self.assertEqual(train_act.main(command), 0)
        run = next((root / 'runs').iterdir())
        return run, TinyTestModel

    def test_budget_saves_mid_epoch_and_checkpoint_can_resume(self):
        import torch
        from robot_bringup import train_act
        with tempfile.TemporaryDirectory() as temporary:
            run, model_class = self.run_budget_fixture(Path(temporary), .1)
            status = json.loads((run / 'run_status.json').read_text())
            self.assertEqual(status['state'], 'time_budget_reached')
            self.assertGreater(status['global_step'], 0)
            checkpoint = torch.load(run / 'last.pt', weights_only=True)
            self.assertIsNotNone(checkpoint['partial_epoch'])
            self.assertGreater(checkpoint['partial_epoch_batches'], 0)
            self.assertTrue((run / 'best.pt').is_file())
            with patch.object(train_act, 'ACT', model_class), redirect_stdout(io.StringIO()):
                self.assertEqual(train_act.main(['--resume', str(run / 'last.pt'), '--max-minutes', '0',
                                                '--epochs', str(checkpoint['epoch'] + 2),
                                                '--workers', '0', '--device', 'cpu']), 0)
            resumed = torch.load(run / 'last.pt', weights_only=True)
            self.assertGreater(resumed['global_step'], checkpoint['global_step'])
            self.assertIsNone(resumed['partial_epoch'])
            self.assertEqual(json.loads((run / 'run_status.json').read_text())['state'], 'epochs_completed')

    def test_interrupted_validation_does_not_publish_a_best_checkpoint(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = self.run_budget_fixture(Path(temporary), .001)
            self.assertTrue((run / 'last.pt').is_file())
            self.assertFalse((run / 'best.pt').exists())
            self.assertFalse((run / 'metrics.jsonl').exists())
            checkpoint = torch.load(run / 'last.pt', weights_only=True)
            self.assertEqual(checkpoint['global_step'], 1)

    def test_resume_can_validate_every_frame_without_changing_episode_training(self):
        import torch
        from robot_bringup import train_act
        with tempfile.TemporaryDirectory() as temporary:
            run, model_class = self.run_budget_fixture(Path(temporary), .1)
            checkpoint = torch.load(run / 'last.pt', weights_only=True)
            checkpoint.update(epoch=0, partial_epoch=None)
            checkpoint['run_config'].update(sampling='episode', validate_every_steps=0)
            torch.save(checkpoint, run / 'last.pt')
            original_validate = train_act.validate
            with patch.object(train_act, 'ACT', model_class), \
                    patch.object(train_act, 'validate', wraps=original_validate) as checked, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(train_act.main(['--resume', str(run / 'last.pt'), '--epochs', '2',
                    '--workers', '0', '--device', 'cpu', '--max-minutes', '0', '--validation-samples', '0']), 0)
            self.assertEqual(checked.call_count, 2)  # Re-score best, then assess current weights.
            for call in checked.call_args_list:
                loader = call.args[1]
                self.assertEqual(loader.dataset.sampling, 'frame')
                self.assertEqual(len(loader.dataset), 3)
                self.assertEqual(type(loader.sampler).__name__, 'SequentialSampler')
            last = torch.load(run / 'last.pt', weights_only=True)
            self.assertEqual(last['partial_epoch_batches'], 1)  # One training observation per episode.
            self.assertEqual(last['validation_settings']['sample_limit'], 0)
            metrics = json.loads((run / 'metrics.jsonl').read_text().splitlines()[-1])
            self.assertEqual(metrics['validation_observations'], 3)

    def test_resume_can_train_more_per_epoch_without_changing_fixed_validation(self):
        import torch
        from robot_bringup import train_act
        with tempfile.TemporaryDirectory() as temporary:
            run, model_class = self.run_budget_fixture(Path(temporary), .1)
            checkpoint = torch.load(run / 'last.pt', weights_only=True)
            checkpoint.update(epoch=0, partial_epoch=None)
            checkpoint['run_config'].update(sampling='episode', validate_every_steps=0)
            checkpoint['run_config'].pop('samples_per_episode', None)  # Checkpoint predates this option.
            torch.save(checkpoint, run / 'last.pt')
            original_validate = train_act.validate
            with patch.object(train_act, 'ACT', model_class), \
                    patch.object(train_act, 'validate', wraps=original_validate) as checked, \
                    redirect_stdout(io.StringIO()):
                train_act.main(['--resume', str(run / 'last.pt'), '--epochs', '2', '--samples-per-episode', '10',
                    '--workers', '0', '--device', 'cpu', '--max-minutes', '0'])
            self.assertEqual(checked.call_count, 1)  # Same VAL protocol; no re-scoring required.
            last = torch.load(run / 'last.pt', weights_only=True)
            self.assertEqual(last['global_step'] - checkpoint['global_step'], 5)  # batch 1, accumulate 2.
            self.assertEqual(last['partial_epoch_batches'], 10)
            self.assertEqual(last['run_config']['samples_per_episode'], 10)
            self.assertEqual(last['validation_settings'], checkpoint['validation_settings'])
            metrics = json.loads((run / 'metrics.jsonl').read_text().splitlines()[-1])
            self.assertEqual(metrics['training_samples_per_epoch'], 10)
            self.assertEqual(metrics['optimizer_steps_per_epoch'], 5)
            self.assertEqual(metrics['validation_observations'], 2)
            # Omitting the option next time retains the saved training length.
            with patch.object(train_act, 'ACT', model_class), redirect_stdout(io.StringIO()):
                train_act.main(['--resume', str(run / 'last.pt'), '--epochs', '3',
                    '--workers', '0', '--device', 'cpu', '--max-minutes', '0'])
            following = torch.load(run / 'last.pt', weights_only=True)
            self.assertEqual(following['global_step'] - last['global_step'], 5)
            self.assertEqual(following['run_config']['samples_per_episode'], 10)

    def test_random_validation_checkpoint_best_is_rescored_before_comparison(self):
        import torch
        from robot_bringup import train_act
        with tempfile.TemporaryDirectory() as temporary:
            run, model_class = self.run_budget_fixture(Path(temporary), .1)
            checkpoint = torch.load(run / 'last.pt', weights_only=True)
            checkpoint.update(epoch=0, partial_epoch=None, best_val_loss=-100.)
            checkpoint['run_config'].update(validate_every_steps=0, sampling='episode')
            checkpoint.pop('validation_settings')
            torch.save(checkpoint, run / 'last.pt')
            best = torch.load(run / 'best.pt', weights_only=True)
            best.pop('validation_settings')
            best['best_val_loss'] = -100.
            torch.save(best, run / 'best.pt')
            def metrics(value):
                return dict(val_posterior_loss=value, val_posterior_l1=value,
                    val_posterior_kl=0., val_prior_l1=value, val_joint_mae_rad=[value]*7)
            seen = []
            def validation(model, *args, **kwargs):
                seen.append({key: value.clone() for key, value in model.state_dict().items()})
                return metrics(.4 if len(seen) == 1 else .6)
            with patch.object(train_act, 'ACT', model_class), patch.object(train_act, 'validate', side_effect=validation), \
                    redirect_stdout(io.StringIO()):
                train_act.main(['--resume', str(run / 'last.pt'), '--epochs', '2', '--max-minutes', '0',
                                '--workers', '0', '--device', 'cpu'])
            retained = torch.load(run / 'best.pt', weights_only=True)
            last = torch.load(run / 'last.pt', weights_only=True)
            self.assertEqual(len(seen), 2)
            self.assertEqual(retained['best_val_loss'], .4)
            self.assertEqual(retained['global_step'], best['global_step'])
            self.assertEqual(retained['validation_settings']['sample_limit'], 1024)
            for name in retained['model']:
                torch.testing.assert_close(retained['model'][name], best['model'][name], rtol=0, atol=0)
                torch.testing.assert_close(last['model'][name], seen[1][name], rtol=0, atol=0)

    def test_best_checkpoint_uses_posterior_loss_even_when_prior_improves(self):
        import torch
        from robot_bringup import train_act
        with tempfile.TemporaryDirectory() as temporary:
            run, model_class = self.run_budget_fixture(Path(temporary), .1)
            checkpoint = torch.load(run / 'last.pt', weights_only=True)
            checkpoint.update(epoch=0, partial_epoch=None, best_val_loss=float('inf'))
            checkpoint['run_config']['validate_every_steps'] = 0
            torch.save(checkpoint, run / 'last.pt')
            (run / 'best.pt').unlink(missing_ok=True)
            def metrics(posterior, prior):
                return dict(val_posterior_loss=posterior, val_posterior_l1=posterior,
                            val_posterior_kl=0., val_prior_l1=prior, val_joint_mae_rad=[prior]*7)
            with patch.object(train_act, 'ACT', model_class), \
                    patch.object(train_act, 'validate', side_effect=[metrics(.5, .8), metrics(.6, .1)]), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(train_act.main(['--resume', str(run / 'last.pt'), '--max-minutes', '0',
                                                '--epochs', '3', '--workers', '0', '--device', 'cpu']), 0)
            best = torch.load(run / 'best.pt', weights_only=True)
            last = torch.load(run / 'last.pt', weights_only=True)
            self.assertEqual(best['selection_metric'], 'val_posterior_loss')
            self.assertEqual(best['best_val_loss'], .5)
            self.assertLess(best['global_step'], last['global_step'])
            self.assertNotIn('optimizer', best)
            self.assertIn('optimizer', last)
            with self.assertRaisesRegex(ValueError, 'Resume requires last.pt'):
                train_act.main(['--resume', str(run / 'best.pt'), '--device', 'cpu'])
            # A later policy improvement may precede the next full checkpoint.
            best['best_val_loss'] = .1
            torch.save(best, run / 'best.pt')
            with patch.object(train_act, 'ACT', model_class), \
                    patch.object(train_act, 'validate', return_value=metrics(.3, .2)), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(train_act.main(['--resume', str(run / 'last.pt'), '--max-minutes', '0',
                                                '--epochs', '4', '--workers', '0', '--device', 'cpu']), 0)
            retained = torch.load(run / 'best.pt', weights_only=True)
            self.assertEqual(retained['global_step'], best['global_step'])
            self.assertEqual(retained['best_val_loss'], .1)


if __name__ == '__main__':
    unittest.main()
