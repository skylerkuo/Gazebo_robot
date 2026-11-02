"""Frozen composite policy, second residual targets, sampling and portable reload."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from act_multiview import FOUR_CAMERAS
from act_objectives import evaluate_pose_chunks
from flow_matching import anchor_noise
from ftact import load_checkpoint
from stacked_residual_flow_multiview import (
    FORMAT, StackedResidualFlowConfig, StackedResidualFlow,
    bind_base_policy, initialize_stacked_from_residual,
)
from train_sfp0_act_multiview import checkpoint, parse_args, load_initialization
from test_act_multiview import stats
from test_residual_flow import setup as first_stage, residual_batch


@pytest.fixture(scope='module', autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def setup():
    base, base_cfg, _ = first_stage()
    payload = {'format': 'aic-sfp0-supervised-residual-flow-multiview-v1',
        'config': asdict(base_cfg), 'model': base.state_dict(), 'stats': stats(),
        'step': 7, '_initialization_sha256': 'b'*64, 'data_index_sha256': 'data',
        'camera_order': list(FOUR_CAMERAS), 'action_offset': 1, 'smoke_test': False}
    cfg = StackedResidualFlowConfig(device='cpu', amp='none', dim_model=32,
        n_heads=4, dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
        pretrained_backbone=False, dropout=.1, steps=10, ode_steps=2,
        expert_only=True, index_path='/mnt/d/frozen/index.json', residual_scale=.5)
    bind_base_policy(cfg, payload)
    model = StackedResidualFlow(cfg, stats(), initialize_backbone=False)
    initialize_stacked_from_residual(model, payload)
    return model, cfg, payload


def test_full_composite_weights_and_buffers_frozen_only_second_flow_updates():
    model, _, _ = setup()
    model.train()
    assert not model.base_policy.training and not model.base_policy.base_act.training
    before = {k: v.clone() for k, v in model.base_policy.state_dict().items()}
    optimizer = torch.optim.AdamW(model.policy.get_optim_params(), lr=1e-4)
    parameters = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert not any(id(p) in parameters for p in model.base_policy.parameters())
    b = residual_batch()
    b[FOUR_CAMERAS[-1]].requires_grad_(True)
    loss, metrics = model(b)
    loss.backward()
    assert torch.isfinite(loss) and metrics['flow_valid_steps'] == 40
    assert b[FOUR_CAMERAS[-1]].grad.norm() > 0
    core = model.policy.model
    assert core.encoder_robot_state_input_proj.weight.grad[:, 6:].norm() > 0
    assert core.flow_base_action_input_proj.weight.grad.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.base_policy.parameters())
    head_before = core.action_head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(head_before, core.action_head.weight)
    for key, value in model.base_policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_base_is_full_first_stage_not_act_and_second_target_reconstructs_expert():
    model, cfg, _ = setup()
    model.eval()
    b = residual_batch()
    noise = anchor_noise([0, 1], [9, 10], cfg.base_noise_seed)
    expected = model.base_policy.predict_action_chunk(b, noise=noise)
    base = model.base_normalized_action(b)
    torch.testing.assert_close(model.normalizer.unnormalize_actions(base), expected)
    act = model.base_policy.base_act.predict_action_chunk(b)
    assert not torch.allclose(expected, act)
    target = model.residual_target(b['action'], base)
    reconstructed = model.normalizer.unnormalize_actions(base+cfg.residual_scale*target)
    torch.testing.assert_close(reconstructed[..., :3], b['action'][..., :3])
    torch.testing.assert_close((reconstructed[..., 3:]*b['action'][..., 3:]).sum(-1).abs(),
                              torch.ones(2, 20), atol=2e-5, rtol=1e-5)


def test_oracle_second_flow_reproduces_expert_and_zero_correction_restores_first_stage():
    model, cfg, _ = setup()
    model.eval()
    b = residual_batch()
    base = model.base_normalized_action(b)
    target = model.residual_target(b['action'], base)
    noise = torch.randn_like(target)
    with patch.object(model, 'velocity', side_effect=lambda x, t, c: target-noise):
        result = model.predict_action_chunk(b, noise=noise, steps=4)
    torch.testing.assert_close(result[..., :3], b['action'][..., :3], atol=1e-6, rtol=1e-5)
    # Patch only the outer velocity, leaving the frozen inner sampler intact.
    with patch.object(model, 'velocity', side_effect=lambda x, t, c: -noise):
        result = model.predict_action_chunk(b, noise=noise)
    torch.testing.assert_close(result, model.normalizer.unnormalize_actions(base), atol=1e-6, rtol=1e-5)


def test_offline_base_noise_is_order_independent_label_free_and_keeps_rng():
    model, _, _ = setup()
    model.eval()
    b = residual_batch()
    state = torch.get_rng_state().clone()
    first = model.base_normalized_action(b)
    assert torch.equal(state, torch.get_rng_state())
    reversed_batch = {k: v.flip(0) for k, v in b.items()}
    torch.testing.assert_close(model.base_normalized_action(reversed_batch).flip(0), first,
                              atol=2e-5, rtol=2e-5)
    observation = {k: v for k, v in b.items() if k not in ('action', 'action_mask', 'action_is_pad')}
    torch.testing.assert_close(model.base_normalized_action(observation), first, rtol=0, atol=0)
    model.eval()
    noise = torch.zeros(2, 20, 7)
    expected = model.predict_action_chunk(b, noise=noise)
    torch.testing.assert_close(model.predict_action_chunk(observation, noise=noise), expected,
                              rtol=0, atol=0)
    with pytest.raises(ValueError, match='both episode and frame'):
        model.base_normalized_action({k: v for k, v in observation.items() if k != 'frame_index'})


def test_live_two_layer_noise_streams_reset_without_expert_labels():
    model, _, _ = setup()
    model.eval()
    b = {k: v for k, v in residual_batch().items() if k in FOUR_CAMERAS or k in
         ('observation.state', 'observation.environment_state')}
    model.reset_sampler(42)
    first = model.predict_action_chunk(b)
    second = model.predict_action_chunk(b)
    assert not torch.equal(first, second)
    model.reset_sampler(42)
    torch.testing.assert_close(model.predict_action_chunk(b), first, rtol=0, atol=0)
    assert first.shape == (2, 20, 7)
    torch.testing.assert_close(first[..., 3:].norm(dim=-1), torch.ones(2, 20))


def test_validation_compares_fixed_composite_baseline_and_actual_two_layer_generation():
    model, cfg, _ = setup()
    cfg.eval_samples = 2
    b = residual_batch()
    state = torch.get_rng_state().clone()
    model.residual_enabled = False
    baseline = evaluate_pose_chunks(model, [b], cfg)
    model.residual_enabled = True
    first = evaluate_pose_chunks(model, [b], cfg)
    second = evaluate_pose_chunks(model, [b], cfg)
    assert torch.equal(state, torch.get_rng_state())
    assert first == second
    assert first['generation'] == 'ACT_plus_frozen_residual_flow_plus_second_residual_flow'
    assert first['base_policy_step'] == 7 and first['base_policy_frozen']
    assert first['base_ode_steps'] == 2 and first['ode_steps'] == 2
    assert not baseline['residual_enabled'] and first['residual_enabled']
    assert first['position_l2_m'] != baseline['position_l2_m']


def test_portable_checkpoint_preserves_both_layers_and_resume_contract(tmp_path):
    model, cfg, _ = setup()
    model.eval()
    b = residual_batch()
    noise = torch.randn(2, 20, 7)
    expected = model.predict_action_chunk(b, noise=noise)
    optimizer = torch.optim.AdamW(model.policy.get_optim_params())
    index = {'stats': stats(), 'task': {'target_module': 'nic_card_mount_0'},
             'index_content_sha256': 'data'}
    cursor = NS(state_dict=lambda: {'epoch': 0, 'samples_consumed': 2, 'batches_consumed': 1})
    path = tmp_path/'last.pt'
    checkpoint(path, model, optimizer, torch.amp.GradScaler('cuda', enabled=False), cfg,
               index, 1, cursor, 1., .01, {'loss': 1.}, 1, True, True)
    loaded, payload = load_checkpoint(path)
    assert payload['format'] == FORMAT
    assert payload['base_policy_contract']['sha256'] == 'b'*64
    assert payload['base_policy_contract']['embedded_in_model']
    assert payload['execution']['temporal_ensemble'] is False
    assert payload['execution']['residual_layers'] == 2
    torch.testing.assert_close(loaded.predict_action_chunk(b, noise=noise), expected, rtol=0, atol=0)
    assert not any(p.requires_grad for p in loaded.base_policy.parameters())
    with patch.object(sys, 'argv', ['train', '--resume', str(path)]):
        _, resumed, _ = parse_args(expected_policy='stacked_residual_flow')
    assert resumed.base_policy_step == 7 and resumed.base_act_step == 5
    with patch.object(sys, 'argv', ['train', '--resume', str(path), '--base-noise-seed', '1']):
        with pytest.raises(SystemExit):
            parse_args(expected_policy='stacked_residual_flow')
    with patch.object(sys, 'argv', ['train', '--resume', str(path)]):
        with pytest.raises(SystemExit):
            parse_args(expected_policy='residual_flow')


def test_initialization_accepts_only_real_first_stage_and_freezes_metadata(tmp_path):
    model, cfg, payload = setup()
    original = copy.deepcopy(cfg.base_policy_config)
    payload['config']['ode_steps'] = 99
    assert cfg.base_policy_config == original
    wrong = dict(payload, format=FORMAT)
    with pytest.raises(ValueError, match='real four-view'):
        bind_base_policy(cfg, wrong)
    wrong = dict(payload, smoke_test=True)
    with pytest.raises(ValueError, match='real four-view'):
        bind_base_policy(cfg, wrong)
    path = tmp_path/'first.pt'
    torch.save(payload, path)
    loaded, provenance = load_initialization(path, expected_policy='stacked_residual_flow')
    assert len(provenance['sha256']) == 64 and loaded['step'] == 7
    with pytest.raises(ValueError, match='ACT checkpoint'):
        load_initialization(path)


def test_default_configuration_keeps_data_split_and_distinct_output():
    root = Path(__file__).resolve().parents[1]
    first = json.loads((root/'configs/sfp0_residual_flow_expert.json').read_text())
    second_path = root/'configs/sfp0_stacked_residual_flow_expert.json'
    second = json.loads(second_path.read_text())
    for key in ('dataset_root', 'index_path', 'eval_samples', 'eval_every', 'position_scale_m'):
        assert second[key] == first[key]
    assert second['output_dir'] != first['output_dir']
    assert second['base_noise_seed'] == first['validation_noise_seed']
    assert second['validation_noise_seed'] != second['base_noise_seed']
    with patch.object(sys, 'argv', ['train', '--config', str(second_path), '--check-data']):
        _, cfg, _ = parse_args(expected_policy='stacked_residual_flow')
    assert cfg.steps == 10000 and cfg.eval_samples == 1500
    cfg.device, cfg.amp = 'cpu', 'none'
    cfg.sigma_min = .01
    with pytest.raises(ValueError, match='straight path'):
        cfg.validate()
