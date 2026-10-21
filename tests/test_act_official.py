"""Contracts for the format-2 ACT fine-tuning pipeline; synthetic inputs only."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'Optional ACT torch environment is not installed')
class OfficialACTTest(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        from robot_bringup.act_model import ACTConfig
        values = dict(hidden_dim=32, latent_dim=4, nheads=4, encoder_layers=2,
                      decoder_layers=2, posterior_layers=2, feedforward_dim=64, dropout=0.)
        return ACTConfig(**dict(values, **overrides))

    def test_medium_is_default_and_every_new_preset_fine_tunes_resnet(self):
        from robot_bringup.train_act import arguments, model_configuration
        for preset in ('medium', 'standard', 'small'):
            args = arguments(['--preset', preset])
            self.assertTrue(model_configuration(vars(args)).train_backbone)
        args = arguments([])
        model = model_configuration(vars(args))
        self.assertEqual(args.preset, 'medium')
        self.assertEqual((args.image_height, args.image_width), (480, 640))
        self.assertEqual((model.hidden_dim, model.encoder_layers, model.decoder_layers, model.posterior_layers),
                         (256, 4, 4, 4))
        self.assertEqual((args.learning_rate, args.backbone_learning_rate), (1e-5, 1e-5))
        self.assertEqual((args.batch_size, args.accumulation_steps, args.sampling, args.grad_clip), (4, 2, 'episode', 0.))
        self.assertEqual(args.validation_samples, 1024)
        self.assertFalse(arguments(['--freeze-backbone']).train_backbone)

    def test_backbone_updates_but_bn_buffers_are_frozen_and_all_decoder_layers_learn(self):
        import torch
        from robot_bringup.act_model import ACT, FrozenBatchNorm2d, act_loss
        from robot_bringup.train_act import make_optimizer
        torch.manual_seed(9)
        config = self.config()
        model = ACT(config, pretrained=False).train()
        before = model.backbone[0].weight.detach().clone()
        buffers = {name: value.clone() for name, value in model.backbone.named_buffers()}
        self.assertFalse(any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules()))
        self.assertTrue(any(isinstance(layer, FrozenBatchNorm2d) for layer in model.modules()))
        optimizer = make_optimizer(model, dict(learning_rate=1e-4, backbone_learning_rate=1e-5, weight_decay=1e-4))
        self.assertEqual([group['lr'] for group in optimizer.param_groups], [1e-4, 1e-5])
        registered = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(registered), len(set(registered)))
        self.assertEqual(set(registered), {id(p) for p in model.parameters() if p.requires_grad})
        actions = torch.randn(1, config.chunk_size, 7)
        padding = torch.zeros(1, config.chunk_size, dtype=torch.bool)
        result = model(torch.randn(1, 7), torch.rand(1, 3, 3, 64, 64), actions, padding)
        loss, _, _ = act_loss(result[0], actions, padding, *result[1:])
        loss.backward()
        self.assertGreater(model.backbone[0].weight.grad.abs().sum().item(), 0.)
        for stack in (model.visual_encoder, model.posterior_encoder, model.decoder):
            for layer in stack:
                self.assertGreater(layer.linear2.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(model.backbone[0].weight, before))
        for name, buffer in model.backbone.named_buffers():
            torch.testing.assert_close(buffer, buffers[name], rtol=0, atol=0)

    def test_frozen_bn_loads_regular_pretrained_bn_including_tracking_counter(self):
        import torch
        from robot_bringup.act_model import FrozenBatchNorm2d
        original = torch.nn.BatchNorm2d(4).eval()
        with torch.no_grad():
            original.weight.uniform_(.5, 2.)
            original.bias.uniform_(-.3, .3)
            original.running_mean.uniform_(-1., 1.)
            original.running_var.uniform_(.2, 2.)
        frozen = FrozenBatchNorm2d(4).train()
        frozen.load_state_dict(original.state_dict(), strict=True)
        self.assertFalse(list(frozen.parameters()))
        pixels = torch.randn(2, 4, 5, 7)
        torch.testing.assert_close(frozen(pixels), original(pixels))

    def test_queries_are_positions_with_zero_initial_decoder_content(self):
        import torch
        from robot_bringup.act_model import ACT
        model = ACT(self.config(), pretrained=False).eval()
        inputs = []
        handle = model.decoder[0].register_forward_pre_hook(lambda module, args: inputs.append(args))
        with torch.no_grad():
            model(torch.zeros(1, 7), torch.rand(1, 3, 3, 64, 64))
        handle.remove()
        self.assertEqual(inputs[0][0].abs().sum().item(), 0.)
        torch.testing.assert_close(inputs[0][3][0], model.queries.weight)

    def test_posterior_selection_and_prior_reporting_are_separate_forwards(self):
        import torch
        from robot_bringup.train_act import validate
        calls = []
        class Probe(torch.nn.Module):
            def forward(self, qpos, images, actions=None, is_pad=None):
                calls.append(actions is not None)
                if actions is None:
                    return torch.zeros(1, 2, 7), None, None
                return actions.clone(), torch.ones(1, 2), torch.zeros(1, 2)
        batch = (torch.zeros(1, 3, 3, 32, 32), torch.zeros(1, 7), torch.ones(1, 2, 7),
                 torch.zeros(1, 2, dtype=torch.bool))
        metrics = validate(Probe(), [batch], torch.device('cpu'), nullcontext, torch.ones(7))
        self.assertEqual(calls, [True, False])
        self.assertEqual(metrics['val_posterior_l1'], 0.)
        self.assertEqual(metrics['val_posterior_kl'], 1.)
        self.assertEqual(metrics['val_posterior_loss'], 10.)
        self.assertEqual(metrics['val_prior_l1'], 1.)

    def test_episode_sampling_gives_each_episode_equal_weight_and_random_start(self):
        from PIL import Image
        from robot_bringup.train_act import ChunkDataset
        def episode(root, offset, length):
            image = root / f'{offset}.jpg'
            Image.new('RGB', (32, 32)).save(image)
            state = np.array([[offset + i] * 7 for i in range(length)], dtype=np.float32)
            return SimpleNamespace(starts=np.arange(length - 1), qpos=state,
                image_paths=[[image] for _ in range(length)],
                chunk=lambda start, k: (np.repeat(state[start + 1][None], k, axis=0), np.zeros(k, dtype=bool)))
        with tempfile.TemporaryDirectory() as directory:
            episodes = [episode(Path(directory), 0, 4), episode(Path(directory), 10, 20)]
            stats = dict(qpos_mean=[0.]*7, qpos_std=[1.]*7, action_mean=[0.]*7, action_std=[1.]*7)
            data = ChunkDataset(episodes, stats, 2, (32, 32))
            self.assertEqual(len(data), 2)
            with patch('numpy.random.choice', side_effect=[0, 2, 0]):
                self.assertEqual(data[0][1][0].item(), 0.)
                self.assertEqual(data[0][1][0].item(), 2.)
                self.assertEqual(data[1][1][0].item(), 10.)
            expanded = ChunkDataset(episodes, stats, 2, (32, 32), samples_per_episode=3)
            self.assertEqual(len(expanded), 6)
            with patch('numpy.random.choice', side_effect=[0, 0, 1, 5, 2, 18]) as draw:
                self.assertEqual([expanded[i][1][0].item() for i in range(6)], [0., 10., 1., 15., 2., 28.])
            self.assertEqual([len(call.args[0]) for call in draw.call_args_list], [3, 19, 3, 19, 3, 19])
            self.assertEqual(len(ChunkDataset(episodes, stats, 2, (32, 32), 'frame')), 22)
            with self.assertRaisesRegex(ValueError, 'positive'):
                ChunkDataset(episodes, stats, 2, (32, 32), samples_per_episode=0)
            with self.assertRaisesRegex(ValueError, 'sampling episode'):
                ChunkDataset(episodes, stats, 2, (32, 32), 'frame', samples_per_episode=3)

    def test_fixed_validation_visits_every_valid_start_without_random_sampling(self):
        from PIL import Image
        from robot_bringup.train_act import ChunkDataset, validation_subset
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.jpg'
            Image.new('RGB', (32, 32)).save(path)
            # The missing start at index 1 represents a rejected observation.
            episodes = []
            for offset, starts in ((0, [0, 2, 3]), (10, [0, 1])):
                qpos = np.repeat((offset + np.arange(5, dtype=np.float32))[:, None], 7, axis=1)
                episodes.append(SimpleNamespace(starts=np.asarray(starts), qpos=qpos,
                    image_paths=[[path] for _ in range(5)],
                    chunk=lambda start, k, q=qpos: (np.repeat(q[start+1][None], k, axis=0), np.zeros(k, dtype=bool))))
            stats = dict(qpos_mean=[0.]*7, qpos_std=[1.]*7, action_mean=[0.]*7, action_std=[1.]*7)
            data = ChunkDataset(episodes, stats, 2, (32, 32), sampling='frame')
            with patch('numpy.random.choice', side_effect=AssertionError('VAL must not sample starts')):
                for _ in range(2):
                    self.assertEqual([data[i][1][0].item() for i in range(len(data))], [0., 2., 3., 10., 11.])
                subset = validation_subset(data, 3)
                self.assertEqual([sample[1][0].item() for sample in subset], [0., 3., 11.])
                self.assertIs(validation_subset(data, 0), data)

    def test_validation_repeats_posterior_draws_and_preserves_training_rng_and_weights(self):
        import torch
        from robot_bringup.train_act import validate
        class StochasticProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(.2))
                self.dropout = torch.nn.Dropout(.9)

            def forward(self, qpos, images, actions=None, is_pad=None):
                self.asserted_eval = not self.training and torch.is_inference_mode_enabled()
                shape = (qpos.shape[0], 2, 7)
                if actions is None:
                    return self.bias.expand(shape), None, None
                prediction = self.bias + self.dropout(torch.rand(shape))
                return prediction, torch.ones(qpos.shape[0], 2), torch.zeros(qpos.shape[0], 2)

        batch = (torch.zeros(2, 3, 3, 32, 32), torch.zeros(2, 7), torch.zeros(2, 2, 7),
                 torch.zeros(2, 2, dtype=torch.bool))
        model = StochasticProbe().train()
        before = model.bias.detach().clone()
        state = torch.get_rng_state().clone()
        first = validate(model, [batch], torch.device('cpu'), nullcontext, torch.ones(7), seed=19)
        self.assertTrue(model.asserted_eval)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        torch.rand(23)  # Simulate training consuming randomness between validations.
        later_state = torch.get_rng_state().clone()
        second = validate(model, [batch], torch.device('cpu'), nullcontext, torch.ones(7), seed=19)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(torch.get_rng_state(), later_state))
        torch.testing.assert_close(model.bias, before, rtol=0, atol=0)
        self.assertIsNone(model.bias.grad)
        different = validate(model, [batch], torch.device('cpu'), nullcontext, torch.ones(7), seed=20)
        self.assertNotEqual(first['val_posterior_l1'], different['val_posterior_l1'])

    def test_legacy_checkpoint_is_rejected_before_incompatible_weights_are_read(self):
        import torch
        from robot_bringup.act_policy import ACTPolicy
        from robot_bringup.train_act import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            torch.save({'format_version': 1}, path)
            with self.assertRaisesRegex(ValueError, 'format 2'):
                ACTPolicy(path)
            with self.assertRaisesRegex(ValueError, 'format 2'):
                main(['--resume', str(path), '--device', 'cpu'])

    def test_padded_slots_follow_official_denominator(self):
        import torch
        from robot_bringup.act_model import act_loss
        prediction = torch.ones(2, 4, 7, requires_grad=True)
        actions = torch.zeros_like(prediction)
        padding = torch.tensor([[False, True, True, True], [False, False, False, True]])
        mu = torch.zeros(2, 3)
        loss, l1, kl = act_loss(prediction, actions, padding, mu, mu)
        self.assertEqual(l1.item(), .5)
        loss.backward()
        self.assertEqual(prediction.grad[padding].abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
