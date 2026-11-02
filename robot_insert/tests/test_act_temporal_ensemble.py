"""ACT weighting, absolute-time alignment and TCP quaternion adaptation."""
import copy
import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ftact  # Resolve the installed official LeRobot dependencies.
from ftact_control import ACTTemporalPoseEnsembler
from validate_ftact_run import temporal_ensemble_checks
import numpy as np
import torch


def chunk(offset=0.):
    poses = np.zeros((20, 7), dtype=float)
    poses[:, 0] = offset + np.arange(20)*.001
    poses[:, 3] = 1.
    return poses


class TemporalEnsembleTests(unittest.TestCase):
    def test_matches_official_lerobot_act_for_every_regular_step(self):
        # Use the unmodified official class as the oracle without importing
        # unrelated policy factories and their optional AV/GROOT dependencies.
        path = ftact.INSERT_ROOT / 'tools/dataset_python/lerobot/policies/act/modeling_act.py'
        definition = next(node for node in ast.parse(path.read_text()).body
                          if isinstance(node, ast.ClassDef) and node.name == 'ACTTemporalEnsembler')
        namespace = {'torch': torch, 'Tensor': torch.Tensor}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
        ACTTemporalEnsembler = namespace['ACTTemporalEnsembler']
        for coefficient in (.01, 0., .3):
            actual = ACTTemporalPoseEnsembler(coefficient=coefficient)
            official = ACTTemporalEnsembler(coefficient, 20)
            for tick in range(45):
                poses = chunk(tick*.003)
                poses[:, 1] = np.sin(np.arange(20)+tick)*.01
                actual.add(poses, 10.+tick*.05, tick+1)
                pose, details = actual.target(10.+tick*.05)
                expected = official.update(torch.tensor(poses).unsqueeze(0))[0].numpy()
                np.testing.assert_allclose(pose, expected, rtol=2e-7, atol=2e-9)
                self.assertEqual(details['ensemble_count'], min(tick+1, 20))
                self.assertLessEqual(len(actual.history), 20)

    def test_oldest_prediction_has_weight_exp_zero_like_original_act(self):
        ensemble = ACTTemporalPoseEnsembler()
        ensemble.add(chunk(0.), 1., 1)
        ensemble.add(chunk(.1), 1.05, 2)
        pose, details = ensemble.target(1.05)
        weights = np.exp(-.01*np.arange(2)); weights /= weights.sum()
        np.testing.assert_allclose(details['ensemble_weights'], weights, rtol=1e-12)
        self.assertGreater(details['ensemble_weights'][0], details['ensemble_weights'][1])
        self.assertAlmostEqual(pose[0], weights[0]*.001+weights[1]*.1)
        self.assertEqual(details['ensemble_source_indices'], [1, 0])

    def test_inference_latency_and_skipped_camera_frames_use_same_physical_time(self):
        ensemble = ACTTemporalPoseEnsembler()
        ensemble.add(chunk(), 3., 1)
        ensemble.add(chunk(.1), 3.1, 2)  # No prediction from the skipped 3.05 image.
        pose, details = ensemble.target(3.18)
        self.assertAlmostEqual(details['ensemble_target_time'], 3.2)
        self.assertEqual(details['ensemble_source_indices'], [3, 1])
        for stamp, index in zip(details['ensemble_observation_stamps'], details['ensemble_source_indices']):
            self.assertAlmostEqual(stamp+(index+1)/20, 3.2)
        expected = np.average([.003, .101], weights=np.exp(-.01*np.arange(2)))
        self.assertAlmostEqual(pose[0], expected)

    def test_antipodal_quaternions_do_not_cancel_and_positions_can_be_zero(self):
        ensemble = ACTTemporalPoseEnsembler(coefficient=0.)
        a = chunk(); a[:, :3] = 0
        b = a.copy(); b[:, 3:] *= -1
        ensemble.add(a, 0., 1)
        ensemble.add(b, .05, 2)
        pose, details = ensemble.target(.05)
        np.testing.assert_array_equal(pose, a[0])
        self.assertEqual(details['ensemble_count'], 2)
        self.assertAlmostEqual(np.linalg.norm(pose[3:]), 1.)

    def test_expiration_reset_and_invalid_inputs(self):
        ensemble = ACTTemporalPoseEnsembler()
        ensemble.add(chunk(), 0., 1)
        with self.assertRaisesRegex(ValueError, 'new observation'):
            ensemble.add(chunk(), 0., 2)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            ensemble.add(chunk(), .025, 2)
        with self.assertRaisesRegex(ValueError, 'expired'):
            ensemble.target(1.2)
        self.assertEqual(len(ensemble.history), 0)
        ensemble.reset()
        ensemble.add(chunk(.5), 100., 1)
        self.assertEqual(ensemble.target(100.)[1]['ensemble_count'], 1)
        bad = chunk(); bad[0, 3:] = 0
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            ensemble.add(bad, 100.05, 2)
        with self.assertRaises(ValueError): ACTTemporalPoseEnsembler(coefficient=float('nan'))

    def test_journal_checker_rejects_wrong_weights_and_misaligned_actions(self):
        ensemble = ACTTemporalPoseEnsembler()
        ensemble.add(chunk(), 1., 1)
        ensemble.add(chunk(.1), 1.05, 2)
        pose, details = ensemble.target(1.05)
        row = {'policy_instance': 1, 'task_id': 'task_1', 'simulation_time': 1.05,
               'new_chunk': True, 'inference_id': 2, 'raw_target': pose.tolist(), **details}
        self.assertTrue(all(temporal_ensemble_checks([row], .01).values()))
        bad = copy.deepcopy(row); bad['ensemble_weights'].reverse()
        self.assertFalse(temporal_ensemble_checks([bad], .01)['act_exponential_weights'])
        bad = copy.deepcopy(row); bad['ensemble_source_indices'][0] += 1
        self.assertFalse(temporal_ensemble_checks([bad], .01)['ensemble_same_execution_time'])
        bad = copy.deepcopy(row); bad['raw_target'][0] += .01
        self.assertFalse(temporal_ensemble_checks([bad], .01)['ensemble_matches_weighted_mean'])


if __name__ == '__main__':
    unittest.main()
