"""Physical-axis weighting, causal contact context, PACE learning and resume."""
from dataclasses import asdict
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import torch

from ftact import CAMERAS, STATE, WRENCH, FTNormalizer, load_checkpoint
from pace_act import (FORMAT, PHASE, HISTORY, PACEConfig, PACEACT, PACETaskDataset,
                      TactileContext, common_scale_loss)
from train_sfp0_pace import evaluate, checkpoint, gradient_metrics, parse_args
from train_ftact import restore_rng


def stats():
    result = {key: {'mean': [0.] * 6, 'std': [1.] * 6} for key in (STATE, WRENCH)}
    result['action'] = {'mean': [.4, -.1, .2, 0., 0., 0., 1.],
                        'std': [.01, .01, .08, .01, .01, .01, .06]}
    result.update({key: {'mean': [[[.485]], [[.456]], [[.406]]],
                        'std': [[[.229]], [[.224]], [[.225]]]} for key in CAMERAS})
    return result


def config():
    return PACEConfig(device='cpu', amp='none', dim_model=32, n_heads=4,
                      dim_feedforward=64, n_encoder_layers=2, n_decoder_layers=1,
                      pretrained_backbone=False, dropout=.1)


def batch(size=2):
    history = torch.randn(size, 4, 6)
    return {STATE: torch.randn(size, 6), WRENCH: history[:, -1].clone(),
            PHASE: history.mean(1), HISTORY: history,
            **{key: torch.rand(size, 3, 256, 288) for key in CAMERAS},
            'action': torch.tensor(stats()['action']['mean']).repeat(size, 20, 1) + torch.randn(size, 20, 7) * .002,
            'action_is_pad': torch.zeros(size, 20, dtype=torch.bool)}


class PACEMathTests(unittest.TestCase):
    def test_resume_rejects_changed_objective_output_and_old_checkpoint(self):
        cfg = config()
        payload = {'format': FORMAT, 'optimizer': {}, 'config': asdict(cfg), 'step': 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            torch.save(payload, path)
            for option in [('--position-scale-m', '.02'), ('--phase-alpha', '.2'),
                           ('--tactile-history', '8'), ('--output-dir', directory)]:
                with patch.object(sys, 'argv', ['train', '--resume', str(path), *option]), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit): parse_args()
            payload['format'] = 'aic-sfp0-act-v2'
            torch.save(payload, path)
            with patch.object(sys, 'argv', ['train', '--resume', str(path)]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): parse_args()

    def test_same_millimetre_error_has_equal_loss_and_physical_gradient_on_xyz(self):
        normalizer = FTNormalizer(stats())
        target = torch.tensor(stats()['action']['mean']).view(1, 1, 7)
        losses, gradients = [], []
        for axis in range(3):
            physical = target.clone()
            physical[..., axis] += .001
            physical.requires_grad_()
            loss, _ = common_scale_loss(normalizer.actions(physical), target, normalizer)
            loss.backward()
            losses.append(loss.item())
            gradients.append(physical.grad[..., axis].item())
        np.testing.assert_allclose(losses, [.1 / 7] * 3, rtol=2e-5)
        np.testing.assert_allclose(gradients, [1 / .07] * 3, rtol=1e-6)
        # New Z position weight is eight times the old one with these synthetic stds.
        physical = target.clone(); physical[..., 2] += .001
        old = (normalizer.actions(physical) - normalizer.actions(target)).abs().mean()
        self.assertAlmostEqual(losses[2] / old.item(), 8., places=4)

    def test_ema_padding_missing_ticks_duplicate_and_task_reset(self):
        stream = TactileContext(alpha=.5, history=4)
        first = stream.update(np.ones(6), 1.)
        np.testing.assert_array_equal(first[HISTORY], np.ones((4, 6)))
        result = stream.update(np.full(6, 3), 1.15)
        np.testing.assert_allclose(result[PHASE], np.full(6, 2))
        np.testing.assert_array_equal(result[HISTORY][:, 0], [1, 1, 1, 3])
        repeated = stream.update(np.full(6, 999), 1.15)
        np.testing.assert_array_equal(result[PHASE], repeated[PHASE])
        with self.assertRaisesRegex(ValueError, 'backwards'): stream.update(np.ones(6), 1.1)
        with self.assertRaisesRegex(ValueError, 'one second'): stream.update(np.ones(6), 3.)
        stream.reset()
        result = stream.update(np.full(6, 10), 0.)
        np.testing.assert_array_equal(result[PHASE], np.full(6, 10))
        np.testing.assert_array_equal(result[HISTORY], np.full((4, 6), 10))

    def test_dataset_context_is_causal_segment_local_and_preserves_future_labels(self):
        values = {episode: np.full((40, 6), 999., dtype=np.float32) for episode in (8, 9)}
        for episode in values:
            values[episode][4:30] = (np.arange(4, 30) + (episode - 8) * 100)[:, None]
        class Reader:
            def __init__(self, root, episodes, chunk): self.episodes = episodes
            def __len__(self): return 40 * len(self.episodes)
            def __getitem__(self, i):
                episode, frame = self.episodes[i // 40], i % 40
                return {'episode_index': torch.tensor(episode), 'frame_index': torch.tensor(frame),
                    WRENCH: torch.from_numpy(values[episode][frame].copy()),
                    'action': torch.arange(frame, frame + 21).float()[:, None].repeat(1, 7),
                    'action_is_pad': torch.zeros(21, dtype=torch.bool)}
        manifest = {'dataset_root': 'unused', 'segments': [
            {'episode_index': e, 'split': 'train', 'episode_length': 40, 'start': 4, 'end': 30} for e in (8, 9)]}
        with patch('ftact_v2.AICDataset', Reader), patch('pace_act.read_episode_wrench', side_effect=lambda root,e: values[e]):
            data = PACETaskDataset(manifest, 'train', alpha=.5)
            item = data[2]  # episode 8, frame 6
            np.testing.assert_allclose(item[PHASE], np.full(6, 5.25))
            np.testing.assert_array_equal(item[HISTORY][:, 0], [4, 4, 5, 6])
            self.assertEqual(item['action'][:, 0].tolist(), list(range(7, 27)))
            values[8][7:] += 1000  # Future observations must not influence frame 6.
            again = PACETaskDataset(manifest, 'train', alpha=.5)[2]
            torch.testing.assert_close(item[PHASE], again[PHASE])
            torch.testing.assert_close(item[HISTORY], again[HISTORY])
            np.testing.assert_array_equal(data[6][PHASE], np.full(6, 104))
            np.testing.assert_array_equal(data[6][HISTORY], np.full((4, 6), 104))

    def test_validation_checks_1200_and_reports_actual_xyz_millimetres(self):
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.normalizer = FTNormalizer(stats())
            def normalized_action_chunk(self, b):
                target = b['action'].clone()
                target[..., :3] += torch.tensor([.001, .002, .010])
                return self.normalizer.actions(target)
        model = Model()
        target = torch.tensor(stats()['action']['mean']).repeat(400, 20, 1)
        result = evaluate(model, [{'action': target}] * 3, config())
        np.testing.assert_allclose(result['position_xyz_mae_mm'], [1, 2, 10], atol=3e-5)
        np.testing.assert_allclose(result['position_xyz_p90_mm'], [1, 2, 10], atol=3e-5)
        self.assertAlmostEqual(result['position_l2_mm'], np.sqrt(105), places=4)
        self.assertAlmostEqual(result['loss'], 1.3 / 7, places=5)
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, 'exactly 1200'):
            evaluate(model, [{'action': target[:1]}], config())


class PACELearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_prediction_ignores_answers_and_event_only_changes_corrected_prefix(self):
        torch.manual_seed(42)
        model = PACEACT(config(), stats(), initialize_backbone=False).eval()
        b = batch()
        with torch.no_grad():
            before = model.normalized_action_chunk(b)
            b['action'] = torch.full_like(b['action'], 1000)
            torch.testing.assert_close(before, model.normalized_action_chunk(b), rtol=0, atol=0)
            model.policy.model.pace_event_residual[-1].bias.fill_(.2)
            after = model.normalized_action_chunk(b)
            torch.testing.assert_close(before[:, 5:], after[:, 5:], rtol=0, atol=0)
            self.assertGreater((before[:, :5] - after[:, :5]).abs().max().item(), .01)
            physical = model.predict_action_chunk(b)
            torch.testing.assert_close(physical[..., 3:].norm(dim=-1), torch.ones(2, 20))
        del b[HISTORY]
        with self.assertRaisesRegex(ValueError, 'requires causal'): model.normalized_action_chunk(b)

    def test_phase_and_event_receive_gradients_and_all_parameters_are_optimized(self):
        torch.manual_seed(43)
        model = PACEACT(config(), stats(), initialize_backbone=False).train()
        groups = model.policy.get_optim_params()
        optimizer_ids = {id(p) for g in groups for p in g['params']}
        self.assertEqual(optimizer_ids, {id(p) for p in model.parameters() if p.requires_grad})
        optimizer = torch.optim.AdamW(groups, lr=1e-4)
        b = batch()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(b)
            loss.backward()
            gradients = gradient_metrics(model)
            optimizer.step()
        self.assertTrue(all(np.isfinite(value) and value > 0 for value in gradients.values()), gradients)
        model.eval()
        # Learned Phase changes predictions while holding images/current state/Event history fixed.
        before = model.normalized_action_chunk(b)
        altered = dict(b); altered[PHASE] = b[PHASE] + 10
        self.assertGreater((before - model.normalized_action_chunk(altered)).abs().max().item(), 1e-7)

    def test_resumable_checkpoint_restores_next_optimizer_update_and_predictor(self):
        torch.manual_seed(44)
        cfg = config()
        model = PACEACT(cfg, stats(), initialize_backbone=False).train()
        optimizer = torch.optim.AdamW(model.policy.get_optim_params(), lr=1e-4)
        b = batch()
        def update(m, opt):
            opt.zero_grad(set_to_none=True)
            loss, _ = m(b); loss.backward(); opt.step()
            return loss.detach()
        update(model, optimizer)
        index = {'stats': stats(), 'task': {'target_module': 'nic_card_mount_0'}, 'index_content_sha256': 'test'}
        cursor = NS(state_dict=lambda: {'epoch': 0, 'samples_consumed': 2, 'batches_consumed': 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            checkpoint(path, model, optimizer, torch.amp.GradScaler('cuda', enabled=False), cfg, index,
                       1, cursor, 1., .01, {'loss': 1.}, 1, True, True)
            reference_loss = update(model, optimizer)
            restored, payload = load_checkpoint(path)
            self.assertEqual(payload['format'], FORMAT)
            self.assertEqual(payload['validation_step'], 1)
            restored.train()
            resumed_optimizer = torch.optim.AdamW(restored.policy.get_optim_params(), lr=1e-4)
            resumed_optimizer.load_state_dict(payload['optimizer'])
            restore_rng(payload['rng'])
            torch.testing.assert_close(update(restored, resumed_optimizer), reference_loss, rtol=0, atol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
