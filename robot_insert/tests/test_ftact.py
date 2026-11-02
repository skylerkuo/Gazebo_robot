"""Force input, train-only normalization, pose validity, and reproducible data resumption."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ftact import CAMERAS, STATE, WRENCH, EpochBatchSampler, FTNormalizer, unit_tcp_quaternions
from train_ftact import TrainingBatches, parse_args, resume_best_state
import torch
from torch.utils.data import DataLoader


def example_stats():
    stats = {STATE: {"mean": [1.] * 6, "std": [2.] * 6},
             WRENCH: {"mean": [10., 20., 30., 40., 50., 60.], "std": [1., 2., 3., 4., 5., 6.]},
             "action": {"mean": [1., 2., 3., 0., 0., 0., 1.], "std": [2.] * 7}}
    stats.update({key: {"mean": [[[.5]]] * 3, "std": [[[.25]]] * 3} for key in CAMERAS})
    return stats


def example_batch():
    batch = {STATE: torch.full((2, 6), 5.),
             WRENCH: torch.tensor([[11., 24., 39., 56., 75., 96.]]).repeat(2, 1)}
    batch.update({key: torch.full((2, 3, 256, 288), .75) for key in CAMERAS})
    return batch


class FTACTTests(unittest.TestCase):
    def test_force_channels_are_concatenated_and_normalized_in_sensor_order(self):
        normalizer = FTNormalizer(example_stats())
        batch = example_batch()
        batch[WRENCH].requires_grad_()
        result = normalizer.observations(batch)
        torch.testing.assert_close(result[STATE][:, :6], torch.full((2, 6), 2.))
        torch.testing.assert_close(result[STATE][:, 6:], torch.arange(1., 7.).repeat(2, 1))
        for key in CAMERAS:
            torch.testing.assert_close(result[key], torch.ones_like(result[key]))
        result[STATE][:, 6:].sum().backward()
        self.assertTrue((batch[WRENCH].grad.abs() > 0).all())
        self.assertNotIn(WRENCH, result)  # A single 12-D token, no duplicate ENV token.

    def test_force_is_mandatory_and_shape_checked(self):
        normalizer = FTNormalizer(example_stats())
        batch = example_batch()
        del batch[WRENCH]
        with self.assertRaisesRegex(ValueError, "requires"):
            normalizer.observations(batch)
        batch[WRENCH] = torch.zeros(2, 5)
        with self.assertRaisesRegex(ValueError, "wrist wrench"):
            normalizer.observations(batch)

    def test_action_roundtrip_and_unit_quaternions(self):
        normalizer = FTNormalizer(example_stats())
        action = torch.tensor([[[.2, -.3, .4, 0., .3, 0., -.4]]])
        restored = normalizer.unnormalize_actions(normalizer.actions(action))
        torch.testing.assert_close(restored, action)
        pose = unit_tcp_quaternions(restored)
        torch.testing.assert_close(pose[..., :3], action[..., :3])
        torch.testing.assert_close(pose[..., 3:].norm(dim=-1), torch.ones(1, 1))
        torch.testing.assert_close(pose[..., 3:], torch.tensor([[[0., .6, 0., -.8]]]))
        with self.assertRaisesRegex(ValueError, "degenerate"):
            unit_tcp_quaternions(torch.zeros(1, 20, 7))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            unit_tcp_quaternions(torch.full((1, 20, 7), float("nan")))

    def test_resume_cursor_does_not_repeat_or_drop_samples(self):
        original = list(EpochBatchSampler(31, 4, seed=42, epoch=3))
        resumed = list(EpochBatchSampler(31, 4, seed=42, epoch=3, start_batch=5))
        self.assertEqual(resumed, original[5:])
        self.assertEqual(sorted(index for batch in original for index in batch), list(range(31)))
        self.assertNotEqual(original, list(EpochBatchSampler(31, 4, seed=42, epoch=4)))
        self.assertEqual(list(EpochBatchSampler(31, 4, seed=42, start_batch=8)), [])

    def test_resume_with_larger_batches_preserves_exact_sample_position(self):
        data = [{"action": torch.tensor([index])} for index in range(31)]
        sampler = EpochBatchSampler(31, 8, seed=42)
        batches = TrainingBatches(DataLoader(data, batch_sampler=sampler), sampler,
            {"epoch": 3, "batches_consumed": 3}, saved_batch_size=4)
        expected = [i for b in EpochBatchSampler(31, 4, seed=42, epoch=3) for i in b][12:]
        first = next(batches)["action"].flatten().tolist()
        self.assertEqual(first, expected[:8])
        cursor = batches.state_dict()
        self.assertEqual(cursor["samples_consumed"], 20)
        # A second migration must use the saved sample position rather than
        # multiplying the mixed-size historical batch counter by the new size.
        smaller = EpochBatchSampler(31, 4, seed=42)
        restored = TrainingBatches(DataLoader(data, batch_sampler=smaller), smaller,
                                  cursor, saved_batch_size=8)
        tail = [i for _ in range(3) for i in next(restored)["action"].flatten().tolist()]
        self.assertEqual(tail, expected[8:])
        self.assertEqual(restored.state_dict()["samples_consumed"], 31)
        next_epoch = next(restored)["action"].flatten().tolist()
        self.assertEqual(next_epoch, next(iter(EpochBatchSampler(31, 4, seed=42, epoch=4))))

    def test_legacy_cursor_at_partial_epoch_end_starts_next_epoch(self):
        data = [{"action": torch.tensor([i])} for i in range(31)]
        sampler = EpochBatchSampler(31, 16, seed=42)
        batches = TrainingBatches(DataLoader(data, batch_sampler=sampler), sampler,
            {"epoch": 0, "batches_consumed": 8}, saved_batch_size=4)
        self.assertEqual(next(batches)["action"].flatten().tolist(),
                         next(iter(EpochBatchSampler(31, 16, seed=42, epoch=1))))

    def test_batch_override_keeps_legacy_validation_frames_and_target_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            for old_eval_batches in (100, 0):
                torch.save({'format':'aic-ftact-v1', 'optimizer':{}, 'step':10000,
                    'config':{'device':'cpu','amp':'none','batch_size':4,
                              'gradient_accumulation_steps':2,'eval_batches':old_eval_batches}}, path)
                with patch.object(sys, 'argv', ['train_ftact.py','--resume',str(path),
                        '--batch-size','32','--gradient-accumulation-steps','1','--num-workers','4']):
                    _, config, _ = parse_args()
                self.assertEqual(config.batch_size, 32)
                self.assertEqual(config.gradient_accumulation_steps, 1)
                self.assertEqual(config.eval_samples, old_eval_batches * 4)
                self.assertEqual(config.steps, 50000)
                with patch.object(sys, 'argv', ['train_ftact.py','--resume',str(path),
                        '--eval-samples','1000','--reset-best']):
                    args, config, _ = parse_args()
                self.assertEqual(config.eval_samples, 1000)
                self.assertEqual(config.eval_every, 1000)
                self.assertTrue(args.reset_best)

    def test_new_validation_baseline_accepts_first_score_even_if_higher_than_old_best(self):
        old = {'step':14000, 'best_validation_l1':.10496,
               'metrics':{'samples':400,'l1':.13084}}
        for explicit_reset in (False, True):
            threshold, metrics, event = resume_best_state(old, 1000, reset=explicit_reset)
            self.assertIsNone(metrics)
            self.assertTrue(event['first_validation_becomes_best'])
            self.assertEqual(event['validation_samples'], 1000)
            first_new_score = .15  # Worse than the old 400-sample score, but the new baseline.
            self.assertLess(first_new_score, threshold)
        # Reset is also supported when starting over on the same validation subset.
        same_subset = dict(old, metrics={'samples':1000,'l1':.13})
        threshold, _, _ = resume_best_state(same_subset, 1000, reset=True)
        self.assertLess(.15, threshold)

    def test_best_reset_state_survives_checkpoint_resume_without_resetting_again(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'
            # An interruption before the first new VAL must not restore the old threshold.
            torch.save({'step':14500,'best_validation_l1':float('inf'),'metrics':None}, path)
            pending = torch.load(path,weights_only=True)
            threshold, metrics, event = resume_best_state(pending,1000)
            self.assertLess(.15,threshold)
            self.assertIsNone(event)
            # Once the first new VAL was saved, ordinary resume keeps that BEST.
            torch.save({'step':15000,'best_validation_l1':.15,
                        'metrics':{'samples':1000,'l1':.15}},path)
            saved = torch.load(path,weights_only=True)
            threshold, metrics, event = resume_best_state(saved,1000)
            self.assertEqual(threshold,.15)
            self.assertFalse(.16 < threshold)
            self.assertTrue(.14 < threshold)
            self.assertIsNone(event)


if __name__ == "__main__":
    unittest.main()
