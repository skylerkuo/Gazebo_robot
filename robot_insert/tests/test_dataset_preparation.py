"""Contracts that matter for trajectory cleaning and normalization."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_dataset import continuous_quaternions, histogram_stats, source_splits, valid_runs
import numpy as np


class PreparationTests(unittest.TestCase):
    def test_quaternion_cleanup_preserves_rotation_and_translation(self):
        actions = np.array([[1, 2, 3, -.99, 0, 0, 0],
                            [4, 5, 6, 1.01, 0, 0, 0],
                            [7, 8, 9, -1., 0, 0, 0]], dtype=np.float32)
        cleaned, flips, adjusted = continuous_quaternions(actions)
        np.testing.assert_array_equal(cleaned[:, :3], actions[:, :3])
        original_q = actions[:, 3:] / np.linalg.norm(actions[:, 3:], axis=1, keepdims=True)
        np.testing.assert_allclose(np.abs((cleaned[:, 3:] * original_q).sum(axis=1)), 1)
        np.testing.assert_allclose(np.linalg.norm(cleaned[:, 3:], axis=1), 1)
        self.assertTrue(np.all((cleaned[1:, 3:] * cleaned[:-1, 3:]).sum(axis=1) >= 0))
        self.assertEqual(flips, 2)
        self.assertEqual(adjusted, 2)

    def test_sampling_cannot_cross_missing_frames_or_time_gaps(self):
        valid = np.array([True] * 6 + [False] + [True] * 4)
        timestamps = np.arange(len(valid)) / 20
        timestamps[3:] += .1
        runs, dropped = valid_runs(valid, timestamps, 20, minimum=3)
        self.assertEqual(runs, [(0, 3), (3, 6), (7, 11)])
        self.assertEqual(dropped, 0)
        runs, dropped = valid_runs(valid, timestamps, 20, minimum=4)
        self.assertEqual(runs, [(7, 11)])
        self.assertEqual(dropped, 6)

    def test_source_episode_splits_are_deterministic_and_disjoint(self):
        splits = source_splits(list(range(274)), 42)
        self.assertEqual(splits, source_splits(list(reversed(range(274))), 42))
        self.assertEqual([len(x) for x in splits.values()], [219, 27, 28])
        self.assertEqual(len(set().union(*(set(x) for x in splits.values()))), 274)
        self.assertFalse(set(splits['train']) & set(splits['validation']))
        self.assertFalse(set(splits['train']) & set(splits['test']))
        self.assertFalse(set(splits['validation']) & set(splits['test']))

    def test_image_variance_includes_differences_between_frames(self):
        histogram = np.zeros((3, 256), dtype=np.int64)
        histogram[:, 0] = 100
        histogram[:, 255] = 100
        stats = histogram_stats(histogram, frames=2)
        np.testing.assert_allclose(stats['mean'], np.full((3, 1, 1), .5))
        np.testing.assert_allclose(stats['std'], np.full((3, 1, 1), .5))
        self.assertEqual(stats['count'], [2])


if __name__ == '__main__':
    unittest.main()
