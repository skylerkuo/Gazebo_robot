"""CFM equations, ODE direction, conditioning, reproducibility, and checkpoint contracts."""
from dataclasses import asdict
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
from ftact import CAMERAS, STATE, WRENCH, load_checkpoint
from flow_matching import (FORMAT, FlowConfig, ForceTorqueFlow, anchor_noise, integrate_velocity, ot_path)
from train_sfp0_flow import evaluate
from train_sfp0_flow import checkpoint
from train_ftact import restore_rng


def stats():
    result = {key: {'mean': [0.]*6, 'std': [1.]*6} for key in (STATE, WRENCH)}
    result['action'] = {'mean': [.4, -.1, .2, 0., 0., 0., 1.], 'std': [.01]*7}
    for key in CAMERAS:
        result[key] = {'mean': [[[.485]], [[.456]], [[.406]]],
                       'std': [[[.229]], [[.224]], [[.225]]]}
    return result


def batch(size=2):
    return {STATE: torch.randn(size, 6), WRENCH: torch.randn(size, 6),
            **{key: torch.rand(size, 3, 256, 288) for key in CAMERAS},
            'action': torch.randn(size, 20, 7)*.01 + torch.tensor(stats()['action']['mean']),
            'action_is_pad': torch.zeros(size, 20, dtype=torch.bool)}


class FlowMathTests(unittest.TestCase):
    def test_original_equations_22_23_and_endpoints(self):
        noise = torch.randn(4, 20, 7)
        actions = torch.randn_like(noise)
        sigma = 1e-4
        value, velocity = ot_path(actions, noise, torch.zeros(4), sigma)
        torch.testing.assert_close(value, noise)
        end, _ = ot_path(actions, noise, torch.ones(4), sigma)
        torch.testing.assert_close(end, actions + sigma*noise)
        torch.testing.assert_close(velocity, actions - (1-sigma)*noise)
        tau = torch.tensor([.15, .3, .6, .85])
        eps = 1e-3
        before, _ = ot_path(actions, noise, tau-eps, sigma)
        after, _ = ot_path(actions, noise, tau+eps, sigma)
        torch.testing.assert_close((after-before)/(2*eps), velocity, rtol=1e-3, atol=3e-4)

    def test_ode_oracle_reaches_data_in_noise_to_data_direction(self):
        noise, target = torch.randn(2, 20, 7), torch.randn(2, 20, 7)
        for solver in ('euler', 'heun'):
            for steps in (1, 5, 10):
                result = integrate_velocity(lambda x, t: target - .9999*noise, noise, steps, solver)
                torch.testing.assert_close(result, target + .0001*noise, rtol=2e-5, atol=2e-6)

    def test_heun_accuracy_and_fp32_state(self):
        initial = torch.ones(2, 20, 7, dtype=torch.bfloat16)
        euler = integrate_velocity(lambda x, t: x, initial, 10, 'euler')
        heun = integrate_velocity(lambda x, t: x, initial, 10, 'heun')
        self.assertEqual(heun.dtype, torch.float32)
        self.assertLess(abs(heun.mean().item()-math.e), abs(euler.mean().item()-math.e))
        for steps, solver in [(0, 'euler'), (1, 'invalid')]:
            with self.assertRaises(ValueError): integrate_velocity(lambda x,t: x, initial, steps, solver)

    def test_validation_noise_is_anchor_specific_batch_independent_and_rng_isolated(self):
        state = torch.get_rng_state().clone()
        together = anchor_noise([4, 6], [100, 70], 42)
        separate = torch.cat([anchor_noise([4], [100], 42), anchor_noise([6], [70], 42)])
        self.assertTrue(torch.equal(together, separate))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(torch.equal(together[0], together[1]))
        self.assertFalse(torch.equal(together, anchor_noise([4, 6], [100, 70], 43)))

    def test_validation_runs_1200_generated_samples_and_uses_actual_distance(self):
        class Normalizer:
            def actions(self, x): return x
            def unnormalize_actions(self, x): return x
        class Model:
            normalizer = Normalizer()
            def eval(self): pass
            def train(self): pass
            def normalized_action_chunk(self, b, noise):
                self.noise = noise
                x = torch.zeros(len(noise), 20, 7)
                x[:, :, 0] = .01
                x[:, :, 6] = 1
                return x
        config = FlowConfig(device='cpu', amp='none')
        target = torch.zeros(1200, 20, 7); target[:, :, 6] = 1
        b = {'action': target, 'episode_index': torch.arange(1200), 'frame_index': torch.ones(1200)}
        result = evaluate(Model(), [b], config)
        self.assertAlmostEqual(result['position_l2_m'], .01, places=6)
        self.assertAlmostEqual(result['rotation_deg'], 0.)
        self.assertEqual(result['samples'], 1200)
        with self.assertRaises(ValueError):
            evaluate(Model(), [{k:v[:2] for k,v in b.items()}], config)


class FlowModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # prepare_dataset changes sys.path during discovery. Restore the same
        # dependency order that the standalone train/inference entrypoints use.
        reader_path = str(Path(__file__).resolve().parents[1]/'tools/lerobot_reader_python')
        sys.path.insert(0, reader_path)
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.config = FlowConfig(device='cpu', amp='none', dim_model=64, n_heads=4,
            dim_feedforward=128, n_encoder_layers=1, n_decoder_layers=1, ode_steps=2)
        cls.model = ForceTorqueFlow(cls.config, stats(), initialize_backbone=False)
        cls.inputs = batch()

    def test_training_reaches_force_and_visual_and_time_and_noisy_action_parameters(self):
        model = self.model.train()
        model.zero_grad(set_to_none=True)
        loss, metrics = model(self.inputs)
        self.assertTrue(math.isfinite(loss.item()))
        self.assertIn('flow_mse', metrics)
        loss.backward()
        core = model.policy.model
        parameters = [core.encoder_robot_state_input_proj.weight.grad[:, 6:],
                      core.backbone.conv1.weight.grad, core.flow_action_input_proj.weight.grad,
                      core.flow_time_mlp[0].weight.grad]
        for grad in parameters:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm().item(), 0.)
        with self.assertRaises(ValueError):
            model({k:v for k,v in self.inputs.items() if k != WRENCH})
        self.assertFalse(core.config.use_vae)

    def test_generation_ignores_labels_and_encodes_each_camera_once(self):
        model = self.model.eval()
        noise = torch.randn(2, 20, 7)
        with patch.object(model.policy.model.backbone, 'forward', wraps=model.policy.model.backbone.forward) as encoder:
            first = model.predict_action_chunk(self.inputs, noise=noise, steps=5)
            self.assertEqual(encoder.call_count, 3)
        altered = dict(self.inputs, action=torch.full_like(self.inputs['action'], float('nan')))
        second = model.predict_action_chunk(altered, noise=noise, steps=5)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first[..., 3:].norm(dim=-1), torch.ones(2, 20))
        changed = dict(self.inputs, **{WRENCH: self.inputs[WRENCH]+10})
        self.assertGreater((model.predict_action_chunk(changed, noise=noise)-model.predict_action_chunk(self.inputs, noise=noise)).abs().max().item(), 1e-7)

    def test_sampling_seed_reset_and_checkpoint_roundtrip(self):
        model = self.model.eval()
        model.reset_sampler(42)
        first = model.predict_action_chunk(self.inputs)
        other = model.predict_action_chunk(self.inputs)
        self.assertFalse(torch.equal(first, other))
        model.reset_sampler(42)
        torch.testing.assert_close(first, model.predict_action_chunk(self.inputs))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.pt'
            torch.save(dict(format=FORMAT, action_offset=1, config=asdict(self.config),
                            stats=stats(), model=model.state_dict()), path)
            restored, _ = load_checkpoint(path, 'cpu')
            restored.reset_sampler(42)
            torch.testing.assert_close(first, restored.predict_action_chunk(self.inputs))

    def test_optimizer_and_noise_rng_resume_produce_the_same_next_update(self):
        from types import SimpleNamespace
        model = self.model.train()
        optimizer = torch.optim.AdamW(model.policy.get_optim_params(), lr=1e-4)
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(optimized, {id(p) for p in model.parameters() if p.requires_grad})
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(self.inputs)
            loss.backward()
            optimizer.step()
            return loss.detach()
        update(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'
            checkpoint(path, model, optimizer, scaler, self.config,
                dict(stats=stats(), task={}, index_content_sha256='test'), 1,
                SimpleNamespace(state_dict=lambda:dict(epoch=0,batches_consumed=1,samples_consumed=2)),
                float('inf'), None, True, True)
            expected_loss = update(model, optimizer)
            expected_weight = model.policy.model.action_head.weight.detach().clone()
            restored, payload = load_checkpoint(path, 'cpu')
            resumed_optimizer = torch.optim.AdamW(restored.policy.get_optim_params(), lr=1e-4)
            resumed_optimizer.load_state_dict(payload['optimizer'])
            restore_rng(payload['rng'])
            restored.train()
            actual_loss = update(restored, resumed_optimizer)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            torch.testing.assert_close(restored.policy.model.action_head.weight, expected_weight, rtol=0, atol=0)
            self.assertEqual(payload['cursor']['samples_consumed'], 2)


if __name__ == '__main__':
    unittest.main()
