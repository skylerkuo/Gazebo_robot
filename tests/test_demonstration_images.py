"""Three-view synchronization must use fresh causal robot states and reject gaps."""
import json
from pathlib import Path
import tempfile
import unittest

from robot_bringup.demonstration_images import CAMERAS, JOINT_NAMES, MultiViewImages


class ImageSynchronizationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recorder = MultiViewImages(self.root)
        self.addCleanup(self.recorder.close)

    def state(self, stamp):
        self.recorder.add_joint_state(stamp, list(reversed(JOINT_NAMES)), list(range(7)), [0.] * 7)

    def image(self, name, stamp):
        self.recorder.add_image(name, stamp, name + '_camera_link', 640, 480, b'test-jpeg')

    def rows(self):
        return [json.loads(line) for line in (self.root / 'frames.jsonl').read_text().splitlines()]

    def test_three_images_wait_for_state_and_preserve_each_timestamp(self):
        for name, stamp in zip(CAMERAS, (100_000_000, 110_000_000, 105_000_000)):
            self.image(name, stamp)
        self.assertEqual(self.rows(), [])
        self.state(109_000_000)
        self.state(115_000_000)
        row = self.rows()[0]
        self.assertEqual(row['stamp_ns'], 110_000_000)
        self.assertEqual(row['joint_state']['stamp_ns'], 109_000_000)
        self.assertEqual(row['joint_state']['qpos'], list(reversed(range(7))))
        self.assertEqual(row['camera_skew_ns'], 10_000_000)
        self.assertTrue(all((self.root / f['path']).exists() for f in row['images'].values()))

    def test_missing_camera_cannot_produce_training_frames(self):
        self.state(100_000_000)
        self.image('scene', 100_000_000)
        self.image('wrist', 100_000_000)
        self.assertEqual(self.rows(), [])
        self.assertFalse(self.recorder.quality(100_000_000, 200_000_000)['verified'])

    def test_camera_skew_and_stale_joint_states_are_rejected(self):
        self.state(300_000_000)
        self.image('scene', 100_000_000)
        self.image('wrist', 200_000_000)
        self.image('side', 200_000_000)
        self.assertEqual(self.rows(), [])
        self.image('scene', 200_000_000)
        self.assertEqual(self.rows(), [])
        self.assertGreater(self.recorder.dropped['missing_fresh_joint_state'], 0)

    def test_quality_requires_calibration_and_continuous_three_view_coverage(self):
        for stamp in range(1_000_000_000, 2_000_000_001, 100_000_000):
            self.state(stamp)
            for name in CAMERAS:
                self.image(name, stamp)
        self.assertFalse(self.recorder.quality(1_000_000_000, 2_000_000_000)['verified'])
        for name in CAMERAS:
            self.recorder.add_calibration(name, {'width': 640, 'height': 480})
        self.assertTrue(self.recorder.quality(1_000_000_000, 2_000_000_000)['verified'])
        self.assertFalse(self.recorder.quality(1_000_000_000, 3_000_000_000)['verified'])

    def test_duplicate_image_timestamps_do_not_overwrite_frames(self):
        self.image('scene', 100_000_000)
        self.assertFalse(self.recorder.add_image('scene', 100_000_000, 'frame', 1, 1, b'other'))
        self.assertEqual(len(self.recorder.stamps['scene']), 1)


if __name__ == '__main__':
    unittest.main()
