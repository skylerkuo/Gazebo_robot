"""Archiving must preserve local captures until destination publication succeeds."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from robot_bringup.demonstration_storage import archive_episode


class DemonstrationStorageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.capture = self.root / 'capture' / 'episode_0000'
        self.capture.mkdir(parents=True)
        self.destination = self.root / 'destination with spaces' / 'episode_0000'
        (self.capture / 'metadata.json').write_text(json.dumps({
            'status': 'placement_verified', 'training_candidate': True}))
        (self.capture / 'task.yaml').write_text(yaml.safe_dump({'mtc_node': {'ros__parameters': {
            'demonstration_directory': str(self.capture),
            'known_obstacles_file': str(self.capture / 'known_obstacles.cdr'), 'pick_only': False}}}))
        (self.capture / 'known_obstacles.cdr').write_bytes(b'cdr-data')
        (self.capture / 'bag').mkdir()
        (self.capture / 'bag/bag_0.db3').write_bytes(b'closed-bag-data')
        (self.capture / 'images').mkdir()
        (self.capture / 'images/frame.jpg').write_bytes(b'jpeg-data')
        (self.capture / 'frames.jsonl').write_text('{"path":"images/frame.jpg"}\n')

    def test_archive_is_complete_and_rewrites_only_runtime_paths(self):
        metadata = archive_episode(self.capture, self.destination)
        self.assertTrue(self.capture.exists(), 'Archiving must not delete the source')
        self.assertTrue(metadata['training_candidate'])
        self.assertEqual(metadata['storage']['state'], 'archived')
        parameters = yaml.safe_load((self.destination / 'task.yaml').read_text())['mtc_node']['ros__parameters']
        self.assertEqual(parameters['demonstration_directory'], str(self.destination))
        self.assertEqual(parameters['known_obstacles_file'], str(self.destination / 'known_obstacles.cdr'))
        for path in ('bag/bag_0.db3', 'images/frame.jpg', 'frames.jsonl', 'known_obstacles.cdr'):
            self.assertEqual((self.capture / path).read_bytes(), (self.destination / path).read_bytes())
        self.assertFalse(any(self.destination.parent.glob('.*.copying-*')))

    def test_existing_episode_is_never_overwritten(self):
        self.destination.mkdir(parents=True)
        (self.destination / 'existing').write_text('keep')
        with self.assertRaises(FileExistsError):
            archive_episode(self.capture, self.destination)
        self.assertEqual((self.destination / 'existing').read_text(), 'keep')

    def test_copy_failure_retains_capture_and_never_publishes_partial_episode(self):
        with patch('robot_bringup.demonstration_storage.shutil.copyfile', side_effect=OSError('disk full')):
            with self.assertRaises(Exception):
                archive_episode(self.capture, self.destination)
        self.assertTrue((self.capture / 'bag/bag_0.db3').exists())
        self.assertFalse(self.destination.exists())
        self.assertFalse(any(self.destination.parent.glob('.*.copying-*')))

    def test_interrupt_during_copy_retains_capture(self):
        with patch('robot_bringup.demonstration_storage.shutil.copyfile', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                archive_episode(self.capture, self.destination)
        self.assertTrue((self.capture / 'images/frame.jpg').exists())
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
