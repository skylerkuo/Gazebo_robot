"""Checkpoint-to-live-inference contracts; no simulator or controller is started."""
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional ACT torch environment is not installed')
class LiveInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from robot_bringup.act_data import ACTION_SOURCE
        from robot_bringup.act_model import ACT, ACTConfig, CHECKPOINT_VERSION
        from robot_bringup.demonstration_images import JOINT_NAMES
        cls.temporary = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.temporary.name) / 'best.pt'
        cls.config = ACTConfig(cameras=('side', 'scene', 'wrist'), chunk_size=3, hidden_dim=32,
                               nheads=4, latent_dim=4, encoder_layers=1, decoder_layers=2,
                               posterior_layers=1, feedforward_dim=64, dropout=.1)
        model = ACT(cls.config, pretrained=False)
        cls.statistics = dict(qpos_mean=[i * .1 for i in range(7)], qpos_std=[.2 + .05*i for i in range(7)],
                              action_mean=[-.2 + .03*i for i in range(7)], action_std=[.5 + .1*i for i in range(7)])
        cls.payload = dict(format_version=CHECKPOINT_VERSION, model_config=asdict(cls.config),
                           model=model.state_dict(), normalization=cls.statistics, image_size=[64, 96],
                           fps=10., joint_names=JOINT_NAMES, action_source=ACTION_SOURCE,
                           global_step=22, run_config=dict(no_amp=False))
        torch.save(cls.payload, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def images():
        return dict(side=Image.new('RGB', (100, 80), (255, 0, 0)),
                    scene=Image.new('RGB', (100, 80), (0, 255, 0)),
                    wrist=Image.new('RGB', (100, 80), (0, 0, 255)))

    def policy(self):
        from robot_bringup.act_policy import ACTPolicy
        return ACTPolicy(self.checkpoint, device='cpu')

    def test_live_input_uses_saved_view_order_resize_and_training_statistics(self):
        import torch
        policy = self.policy()
        captured = {}
        class Probe(torch.nn.Module):
            def forward(self, qpos, images, *args, **kwargs):
                captured.update(qpos=qpos.clone(), images=images.clone(), args=args, kwargs=kwargs,
                                inference_mode=torch.is_inference_mode_enabled())
                return torch.full((1, 3, 7), .25), None, None
        policy.model = Probe().eval()
        qpos = np.asarray(self.statistics['qpos_mean']) + 1.5 * np.asarray(self.statistics['qpos_std'])
        with patch('robot_bringup.act_data.normalization_stats', side_effect=AssertionError('Never recompute statistics')):
            actions = policy.predict(self.images(), qpos)
        torch.testing.assert_close(captured['qpos'], torch.full((1, 7), 1.5))
        self.assertEqual(tuple(captured['images'].shape), (1, 3, 3, 64, 96))
        for camera in range(3):
            torch.testing.assert_close(captured['images'][0, camera, :, 0, 0], torch.eye(3)[camera])
        self.assertFalse(captured['args'])
        self.assertFalse(captured['kwargs'])
        self.assertTrue(captured['inference_mode'])
        expected = np.asarray(self.statistics['action_mean']) + .25 * np.asarray(self.statistics['action_std'])
        np.testing.assert_allclose(actions, np.repeat(expected[None], 3, axis=0), rtol=1e-6, atol=1e-7)
        self.assertEqual(actions.dtype, np.float32)

    def test_real_policy_never_calls_posterior_and_is_deterministic(self):
        policy = self.policy()
        self.assertFalse(policy.model.training)
        self.assertFalse(any(p.requires_grad for p in policy.model.parameters()))
        with patch.object(policy.model.posterior_encoder[0], 'forward', side_effect=AssertionError('No future actions')):
            first = policy.predict(self.images(), [0.]*7)
            second = policy.predict(self.images(), [0.]*7)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(first.shape, (3, 7))

    def test_warmup_and_summary_reflect_the_loaded_checkpoint(self):
        policy = self.policy()
        self.assertGreaterEqual(policy.warmup(), 0.)
        info = policy.describe()
        self.assertEqual(info['format_version'], 2)
        self.assertEqual(info['model_config']['cameras'], self.config.cameras)
        self.assertEqual(info['model_config']['decoder_layers'], 2)
        self.assertEqual(info['image_size'], [64, 96])
        self.assertEqual(info['training_step'], 22)
        self.assertEqual(info['precision'], 'fp32')
        self.assertEqual(info['latent'], 'zero_prior')
        self.assertEqual(info['decoder_output'], 'last_layer')
        self.assertEqual(info['normalization_source'], 'checkpoint')
        self.assertEqual(info['control_dt'], .1)
        json.dumps(info)

    def test_bad_statistics_fail_before_model_is_built(self):
        from robot_bringup.act_policy import ACTPolicy
        for key, value in (('qpos_mean', [float('nan')]*7), ('qpos_std', [0.]*7),
                           ('action_std', [-1.]*7), ('action_mean', [0.]*8)):
            payload = dict(self.payload, normalization=dict(self.statistics, **{key: value}))
            with self.subTest(key=key), patch('robot_bringup.act_policy.torch.load', return_value=payload), \
                    patch('robot_bringup.act_policy.ACT') as model, self.assertRaises(ValueError):
                ACTPolicy(self.checkpoint)
            model.assert_not_called()

    def test_invalid_fps_size_and_missing_views_have_clear_errors(self):
        from robot_bringup.act_policy import ACTPolicy
        for key, value in (('fps', 0.), ('fps', float('nan')), ('image_size', [32, 0]), ('image_size', [64., 96])):
            with self.subTest(key=key, value=value), \
                    patch('robot_bringup.act_policy.torch.load', return_value=dict(self.payload, **{key: value})), \
                    self.assertRaises(ValueError):
                ACTPolicy(self.checkpoint)
        policy = self.policy()
        with self.assertRaisesRegex(ValueError, 'Missing RGB views: wrist'):
            policy.predict(dict(side=self.images()['side'], scene=self.images()['scene']), [0.]*7)

    def test_auto_precision_respects_cpu_training_amp_setting_and_cuda_support(self):
        import torch
        from robot_bringup.act_policy import inference_precision
        cpu, gpu = torch.device('cpu'), torch.device('cuda:0')
        with patch('torch.cuda.is_bf16_supported', return_value=True):
            self.assertEqual(inference_precision(gpu, 'auto', training_amp=True), 'bf16')
            self.assertEqual(inference_precision(gpu, 'auto', training_amp=False), 'fp32')
            self.assertEqual(inference_precision(gpu, 'fp32'), 'fp32')
            self.assertEqual(inference_precision(cpu, 'auto'), 'fp32')
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            self.assertEqual(inference_precision(gpu, 'auto'), 'fp16')
            with self.assertRaisesRegex(ValueError, 'does not support bf16'):
                inference_precision(gpu, 'bf16')
        with self.assertRaisesRegex(ValueError, 'require CUDA'):
            inference_precision(cpu, 'bf16')

    def test_cuda_device_names_are_resolved_to_explicit_indices(self):
        from robot_bringup.act_policy import inference_device
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.current_device', return_value=1), patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(str(inference_device('cuda')), 'cuda:1')
            self.assertEqual(str(inference_device('auto')), 'cuda:1')
            self.assertEqual(str(inference_device('cuda:0')), 'cuda:0')
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                inference_device('cuda:2')

    def test_check_model_exits_without_creating_a_ros_runtime(self):
        from robot_bringup import act_rollout
        output = io.StringIO()
        with patch.object(act_rollout, 'create_runtime') as runtime, redirect_stdout(output):
            self.assertEqual(act_rollout.main(['--checkpoint', str(self.checkpoint), '--check-model',
                                              '--device', 'cpu', '--precision', 'fp32']), 0)
        runtime.assert_not_called()
        info = json.loads(output.getvalue()[output.getvalue().index('{'):])
        self.assertEqual(info['format_version'], 2)
        self.assertEqual(info['model_config']['cameras'], list(self.config.cameras))
        self.assertEqual(info['training_step'], 22)
        self.assertEqual(info['precision'], 'fp32')


if __name__ == '__main__':
    unittest.main()
