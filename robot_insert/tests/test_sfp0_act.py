"""Task boundary isolation, future-label timing, precision and chunk execution contracts."""
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ftact import CAMERAS, STATE, WRENCH
from ftact_v2 import Float32Linear, TaskDataset, SFPConfig
from prepare_sfp0 import numeric_boundary, visual_boundary, balanced_anchors
from ftact_control import TimedActionChunk
from train_sfp0_act import lr_factor
import numpy as np
import torch
from PIL import Image


class SFP0Tests(unittest.TestCase):
    def test_joint_reset_precedes_delayed_command_reset(self):
        state = np.zeros((100, 6)); state[60:, 0] = .5
        action = np.zeros((100, 7)); action[:, 3] = 1; action[80:, 2] = .3
        boundary, reasons = numeric_boundary(state, action, np.arange(100) / 20)
        self.assertEqual(boundary, 60)
        self.assertEqual(reasons['joint_reset'], [60])
        self.assertEqual(reasons['command_reset'], [80])

    def test_camera_reset_can_precede_joint_reset_and_black_start_is_ignored(self):
        class Column:
            def __init__(self, colors): self.colors = colors
            def __len__(self): return len(self.colors)
            def __getitem__(self, i):
                stream = io.BytesIO()
                Image.new('RGB', (12, 12), (self.colors[i],) * 3).save(stream, format='PNG')
                class Scalar:
                    def as_py(self): return {'bytes': stream.getvalue()}
                return Scalar()
        start, boundary, difference = visual_boundary(Column([0, 0, 100, 101, 99, 180, 181]), 6)
        self.assertEqual((start, boundary), (2, 5))
        self.assertGreater(difference, 25)

    def test_validation_is_fixed_unique_and_balanced_across_recordings(self):
        segments = [{'episode_index': i, 'start': 3, 'end': 90 + i} for i in range(27)]
        anchors = balanced_anchors(segments, 1200, 42)
        self.assertEqual(anchors, balanced_anchors(segments, 1200, 42))
        self.assertEqual(len(set(map(tuple, anchors))), 1200)
        counts = [sum(e == i for e, f in anchors) for i in range(27)]
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertTrue(all(3 <= f < segments[e]['end'] - 20 for e, f in anchors))
        with self.assertRaises(ValueError): balanced_anchors(segments[:1], 1200, 42)

    def test_last_window_stays_in_task_and_predicts_next_20_commands(self):
        class Reader:
            def __init__(self, *args): pass
            def __len__(self): return 60
            def __getitem__(self, i):
                return {'episode_index': torch.tensor(8), 'frame_index': torch.tensor(i),
                    'action': torch.arange(i, i + 21).float().reshape(21, 1).repeat(1, 7),
                    'action_is_pad': torch.zeros(21, dtype=torch.bool)}
        manifest = {'dataset_root': 'unused', 'segments': [{'episode_index': 8, 'split': 'train',
            'episode_length': 60, 'start': 4, 'end': 35}]}
        with patch('ftact_v2.AICDataset', Reader):
            dataset = TaskDataset(manifest, 'train')
        self.assertEqual(len(dataset), 11)  # Anchors 4..14, last target frame 34.
        self.assertEqual(dataset[-1]['action'][:, 0].tolist(), list(range(15, 35)))
        self.assertFalse(dataset[-1]['action_is_pad'].any())

    def test_output_head_preserves_sub_bfloat16_pose_changes_and_gradients(self):
        layer = Float32Linear(1, 1)
        with torch.no_grad(): layer.weight.fill_(.001); layer.bias.fill_(1.)
        x = torch.tensor([[1.], [2.]], requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16): y = layer(x)
        self.assertEqual(y.dtype, torch.float32)
        self.assertGreater(float((y[1] - y[0]).detach()), .0009)
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_delayed_inference_skips_expired_elements_and_advances_through_chunk(self):
        queue = TimedActionChunk()
        chunk = np.zeros((20, 7)); chunk[:, 0] = np.arange(20); chunk[:, 3] = 1
        self.assertTrue(queue.needs_refill(3.))
        queue.replace(chunk, 3.)
        target, index = queue.target(3.12)
        self.assertEqual(index, 2)
        self.assertEqual(target[0], 2)
        self.assertEqual(queue.target(3.45)[1], 9)
        self.assertTrue(queue.needs_refill(3.5))
        # The unused second half remains valid if preparation crosses a replan tick.
        self.assertEqual(queue.target(3.5)[1], 10)
        with self.assertRaises(ValueError): queue.target(4.2)
        queue.replace(chunk + np.array([100, 0, 0, 0, 0, 0, 0]), 3.5)
        self.assertEqual(queue.target(3.56)[0][0], 101)
        # New task owns a new queue, with no old predictions.
        self.assertTrue(TimedActionChunk().needs_refill(0.))

    def test_warmup_then_cosine_and_exact_validation_count(self):
        config = SFPConfig(device='cpu', amp='none')
        self.assertLess(lr_factor(1, config), lr_factor(250, config))
        self.assertAlmostEqual(lr_factor(500, config), 1.)
        self.assertGreater(lr_factor(5000, config), lr_factor(20000, config))
        self.assertAlmostEqual(lr_factor(30000, config), .1)
        config.eval_samples = 1000
        with self.assertRaisesRegex(ValueError, '1200'): config.validate()


if __name__ == '__main__':
    unittest.main()
