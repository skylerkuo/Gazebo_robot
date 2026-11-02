"""Gradient isolation, current-base relabeling, paired VAL and dual-optimizer resume."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from act_multiview import MultiViewACT, MultiViewConfig, FOUR_CAMERAS
from alternating_residual_flow_multiview import (
    FORMAT, AlternatingResidualFlowConfig, AlternatingResidualFlow,
    initialize_alternating_from_act,
)
from ftact import load_checkpoint
from residual_flow_multiview import bind_base_act
from train_ftact import restore_rng
import train_sfp0_alternating_residual_flow_multiview as trainer
from test_act_multiview import stats
from test_residual_flow import residual_batch


@pytest.fixture(scope='module', autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def setup():
    common = dict(device='cpu', amp='none', dim_model=32, n_heads=4, dim_feedforward=64,
        n_encoder_layers=1, n_decoder_layers=1, pretrained_backbone=False, dropout=.1,
        batch_size=2, num_workers=0, steps=10, warmup_steps=1)
    base_config = MultiViewConfig(**common)
    base = MultiViewACT(base_config, stats(), initialize_backbone=False)
    payload = {'format': 'aic-sfp0-act-multiview-v1', 'config': asdict(base_config),
        'step': 5, 'model': base.state_dict(), '_initialization_sha256': 'a'*64,
        'stats': stats(), 'data_index_sha256': 'data', 'camera_order': list(FOUR_CAMERAS)}
    cfg = AlternatingResidualFlowConfig(**common, expert_only=True, ode_steps=2,
        act_learning_rate=1e-4, act_backbone_lr=1e-5, index_path='/mnt/d/frozen/index.json')
    bind_base_act(cfg, payload)
    model = AlternatingResidualFlow(cfg, stats(), initialize_backbone=False)
    initialize_alternating_from_act(model, payload)
    return model, cfg, payload


def states(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def assert_states(module, expected):
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def index():
    return {'stats': stats(), 'task': {'target_module': 'nic_card_mount_0'},
            'index_content_sha256': 'data', 'counts': {}}


def best():
    return {key: float('inf') for key in ('combined_loss', 'combined_position', 'act_loss', 'act_position')}


def test_act_phase_updates_act_with_force_and_side_gradients_but_not_flow():
    model, cfg, _ = setup()
    model.train()
    optimizers = trainer.make_optimizers(model, cfg)
    act_ids = {id(p) for g in optimizers['act'].param_groups for p in g['params']}
    flow_ids = {id(p) for g in optimizers['flow'].param_groups for p in g['params']}
    assert not act_ids & flow_ids
    flow_before, act_before = states(model.policy), states(model.base_act)
    b = residual_batch()
    b[FOUR_CAMERAS[-1]].requires_grad_(True)
    loss, _ = model.act_loss(b)
    loss.backward()
    assert b[FOUR_CAMERAS[-1]].grad.norm() > 0
    assert model.base_act.policy.model.encoder_robot_state_input_proj.weight.grad[:, 6:].norm() > 0
    assert all(p.grad is None for p in model.policy.parameters())
    optimizers['act'].step()
    assert_states(model.policy, flow_before)
    assert any(not torch.equal(value, act_before[key]) for key, value in model.base_act.state_dict().items())
    assert not model.base_act.training


def test_flow_phase_cannot_update_act_weights_buffers_or_gradients():
    model, cfg, _ = setup()
    model.train()
    before = states(model.base_act)
    optimizers = trainer.make_optimizers(model, cfg)
    b = residual_batch()
    b[FOUR_CAMERAS[-1]].requires_grad_(True)
    loss, _ = model(b)
    loss.backward()
    assert b[FOUR_CAMERAS[-1]].grad.norm() > 0
    assert model.policy.model.encoder_robot_state_input_proj.weight.grad[:, 6:].norm() > 0
    assert model.policy.model.flow_base_action_input_proj.weight.grad.norm() > 0
    assert all(p.requires_grad and p.grad is None for p in model.base_act.parameters())
    optimizers['flow'].step()
    assert_states(model.base_act, before)


def test_cycle_recomputes_detached_targets_after_act_step_on_same_observations():
    model, cfg, _ = setup()
    b = residual_batch()
    optimizers, scalers = trainer.make_optimizers(model, cfg), trainer.make_scalers(cfg)
    trainer.set_learning_rates(optimizers, cfg, 1)
    old = model.base_normalized_action(b)
    original = model.base_normalized_action
    captured = []
    def current(batch):
        assert batch is b
        assert int(model.act_update_count) == 1
        value = original(batch)
        assert not value.requires_grad
        captured.append(value)
        return value
    with patch.object(model, 'base_normalized_action', side_effect=current):
        result = trainer.alternating_update(model, [b], optimizers, scalers, cfg)
    after = original(b)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], after, rtol=0, atol=0)
    assert not torch.equal(old, after)
    target = model.residual_target(b['action'], after)
    reconstructed = model.normalizer.unnormalize_actions(after+cfg.residual_scale*target)
    torch.testing.assert_close(reconstructed[..., :3], b['action'][..., :3])
    assert int(model.act_update_count) == int(model.flow_update_count) == 1
    assert result['act_loss'] > 0 and result['flow_loss'] > 0
    assert all(p.grad is None for p in model.parameters())


def test_partial_or_invalid_future_targets_cannot_update_either_model():
    model, _, _ = setup()
    b = residual_batch()
    b['action_mask'][:, 5] = False
    with pytest.raises(ValueError, match='complete valid expert'):
        model.act_loss(b)
    with pytest.raises(ValueError, match='complete valid expert'):
        model(b)


def test_paired_validation_is_repeatable_tracks_both_models_and_keeps_weights_rng():
    model, cfg, _ = setup()
    cfg = replace(cfg, eval_samples=2)
    before = states(model)
    b = residual_batch()
    state = torch.get_rng_state().clone()
    one = trainer.evaluate_pair(model, [b], cfg)
    two = trainer.evaluate_pair(model, [b], cfg)
    assert one == two
    assert torch.equal(state, torch.get_rng_state())
    assert_states(model, before)
    combined = one['combined']
    assert not combined['base_act_frozen'] and not combined['flow_gradient_to_act']
    assert combined['act_initialization_step'] == 5 and combined['act_updates'] == 0
    assert combined['act_only_position_l2_mm'] == one['act']['position_l2_mm']
    assert combined['samples'] == one['act']['samples'] == 2


def test_dual_optimizer_checkpoint_resume_matches_uninterrupted_next_cycle(tmp_path):
    model, cfg, _ = setup()
    b = residual_batch()
    optimizers, scalers = trainer.make_optimizers(model, cfg), trainer.make_scalers(cfg)
    trainer.set_learning_rates(optimizers, cfg, 1)
    trainer.alternating_update(model, [b], optimizers, scalers, cfg)
    cursor = NS(state_dict=lambda: {'epoch': 0, 'batches_consumed': 1, 'samples_consumed': 2})
    path = tmp_path/'last.pt'
    trainer.save_checkpoint(path, model, optimizers, scalers, cfg, index(), 1, cursor,
                            best(), None, None, resumable=True)
    trainer.set_learning_rates(optimizers, cfg, 2)
    expected = trainer.alternating_update(model, [b], optimizers, scalers, cfg)
    expected_state = states(model)
    loaded, payload = load_checkpoint(path)
    assert payload['format'] == FORMAT and payload['act_optimizer']['state'] and payload['flow_optimizer']['state']
    assert payload['optimizer_updates'] == 2 and not payload['base_act_contract']['frozen']
    restored_opts, restored_scalers = trainer.make_optimizers(loaded, cfg), trainer.make_scalers(cfg)
    for name in ('act', 'flow'):
        restored_opts[name].load_state_dict(payload[name+'_optimizer'])
        restored_scalers[name].load_state_dict(payload[name+'_scaler'])
    restore_rng(payload['rng'])
    trainer.set_learning_rates(restored_opts, cfg, 2)
    actual = trainer.alternating_update(loaded, [b], restored_opts, restored_scalers, cfg)
    assert expected == actual
    assert_states(loaded, expected_state)
    assert int(loaded.act_update_count) == int(loaded.flow_update_count) == 2
    with patch.object(sys, 'argv', ['train', '--resume', str(path)]):
        _, restored_config, _ = trainer.parse_args()
    assert restored_config.base_act_step == 5
    with patch.object(sys, 'argv', ['train', '--resume', str(path), '--ode-steps', '3']):
        with pytest.raises(SystemExit):
            trainer.parse_args()


def test_standalone_act_export_and_combined_inference_never_use_future_answers(tmp_path):
    model, cfg, _ = setup()
    model.eval()
    b = residual_batch()
    noise = torch.randn(2, 20, 7)
    expected = model.predict_action_chunk(b, noise=noise)
    observation = {k: v for k, v in b.items() if k not in ('action', 'action_mask', 'action_is_pad')}
    torch.testing.assert_close(model.predict_action_chunk(observation, noise=noise), expected, rtol=0, atol=0)
    path = tmp_path/'best_act.pt'
    trainer.export_act(path, model, cfg, index(), 0, {})
    loaded, payload = load_checkpoint(path)
    assert payload['format'] == 'aic-sfp0-act-multiview-v1'
    assert not payload['config']['retain_model_motion']
    torch.testing.assert_close(loaded.predict_action_chunk(observation),
                              model.base_act.predict_action_chunk(observation), rtol=0, atol=0)
    with patch.object(sys, 'argv', ['train', '--resume', str(path)]):
        with pytest.raises(SystemExit):
            trainer.parse_args()


def test_checkpoint_refuses_partial_cycle_and_reapplies_distinct_learning_rates(tmp_path):
    model, cfg, _ = setup()
    optimizers, scalers = trainer.make_optimizers(model, cfg), trainer.make_scalers(cfg)
    trainer.set_learning_rates(optimizers, cfg, 1)
    assert optimizers['act'].param_groups[0]['lr'] == cfg.act_learning_rate
    assert optimizers['flow'].param_groups[0]['lr'] == cfg.learning_rate
    assert optimizers['act'].param_groups[1]['lr'] == cfg.act_backbone_lr
    model.act_update_count.add_(1)
    with pytest.raises(ValueError, match='complete ACT'):
        trainer.save_checkpoint(tmp_path/'bad.pt', model, optimizers, scalers, cfg, index(),
            1, NS(state_dict=lambda: {}), best(), None, None, resumable=True)


@pytest.mark.parametrize('scratch', [False, True])
def test_main_stop_finishes_cycle_and_saves_two_optimizers_with_cursor(tmp_path, monkeypatch, scratch):
    _, cfg, payload = setup()
    cfg = replace(cfg, output_dir=str(tmp_path/'output'), eval_samples=1500, eval_every=1)
    if scratch:
        cfg = replace(cfg, from_scratch=True, init_from=None, base_act_config=None,
                      base_act_sha256=None, base_act_step=None)
    b = residual_batch()
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 2
        def __getitem__(self, i): return {k: v[i] for k, v in b.items()}
    monkeypatch.setattr(trainer, 'parse_args', lambda: (NS(resume=None, check_data=False), cfg, None))
    def initialization(_,expected_policy='act'):
        assert not scratch, 'A scratch run must never open the previous checkpoint'
        return payload, {'step': 5}
    monkeypatch.setattr(trainer, 'load_initialization', initialization)
    monkeypatch.setattr(trainer, 'load_residual_index', lambda *args: index())
    monkeypatch.setattr(trainer, 'ResidualExpertDataset', Dataset)
    original_evaluate = trainer.evaluate_pair
    monkeypatch.setattr(trainer, 'evaluate_pair',
        lambda model, loader, config: original_evaluate(model, loader, replace(config, eval_samples=2)))
    handlers = {}
    monkeypatch.setattr(trainer.signal, 'signal', lambda number, handler: handlers.update({number: handler}))
    saved = {}
    def capture(path, value):
        saved[Path(path).name] = {k: value.get(k) for k in ('format', 'step', 'act_updates', 'flow_updates', 'cursor')}
        if Path(path).name == 'last.pt':
            assert value['act_optimizer']['state'] and value['flow_optimizer']['state']
    monkeypatch.setattr(trainer, 'atomic_save', capture)
    original = trainer.alternating_update
    def stop_after_cycle(*args):
        metrics = original(*args)
        handlers[signal.SIGINT](signal.SIGINT, None)
        return metrics
    monkeypatch.setattr(trainer, 'alternating_update', stop_after_cycle)
    trainer.main()
    assert saved['last.pt']['step'] == saved['last.pt']['act_updates'] == saved['last.pt']['flow_updates'] == 1
    assert saved['last.pt']['cursor']['samples_consumed'] == 2
    summary = json.loads((Path(cfg.output_dir)/'training_summary.json').read_text())
    assert summary['stopped_by_signal'] and summary['validation_step'] == 0
    run = json.loads((Path(cfg.output_dir)/'run_config.json').read_text())
    if scratch:
        assert run['initialization']['mode'] == 'from_scratch'
        assert run['initialization']['checkpoint_weights_loaded'] is False
    rows = [json.loads(s) for s in (Path(cfg.output_dir)/'metrics.jsonl').read_text().splitlines()]
    assert any(r['type'] == 'act_validation' for r in rows)
    assert any(r['type'] == 'validation' for r in rows)


def test_config_keeps_same_dataset_and_expert_only_supervision():
    root = Path(__file__).resolve().parents[1]
    fresh = json.loads((root/'configs/sfp0_alternating_residual_flow_expert.json').read_text())
    old = json.loads((root/'configs/sfp0_residual_flow_expert.json').read_text())
    for key in ('dataset_root', 'index_path', 'eval_every', 'eval_samples', 'position_scale_m'):
        assert fresh[key] == old[key]
    assert fresh['output_dir'] != old['output_dir']
    assert fresh['from_scratch'] and fresh['init_from'] is None and not fresh['pretrained_backbone']
    assert fresh['act_learning_rate'] < fresh['learning_rate']
    cfg = AlternatingResidualFlowConfig(**fresh)
    cfg.device, cfg.amp = 'cpu', 'none'
    cfg.validate()
    cfg.retain_model_motion = True
    with pytest.raises(ValueError, match='expert-only'):
        cfg.validate()


def scratch_config():
    return AlternatingResidualFlowConfig(device='cpu', amp='none', dim_model=32, n_heads=4,
        dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1, pretrained_backbone=False,
        dropout=.1, batch_size=2, num_workers=0, steps=10, warmup_steps=1, ode_steps=2,
        expert_only=True, from_scratch=True, init_from=None, index_path='/mnt/d/frozen/index.json',
        output_dir='/mnt/d/robot_models/aic_sfp0_alternating_residual_flow_scratch_v1')


def test_scratch_constructs_independent_models_without_reading_or_transferring_weights():
    cfg = scratch_config()
    with patch('torch.load', side_effect=AssertionError('No checkpoint reads')), \
         patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('No pretrained download')), \
         patch('alternating_residual_flow_multiview.initialize_residual_from_act',
               side_effect=AssertionError('No encoder transfer')):
        model = AlternatingResidualFlow(cfg, stats())
    assert cfg.base_act_step == 0 and cfg.base_act_sha256 is None
    assert not cfg.base_act_config['pretrained_backbone'] and cfg.base_act_config['init_from'] is None
    act_weight = next(model.base_act.policy.model.backbone.parameters())
    flow_weight = next(model.policy.model.backbone.parameters())
    assert act_weight.data_ptr() != flow_weight.data_ptr() and not torch.equal(act_weight, flow_weight)
    assert int(model.act_update_count) == int(model.flow_update_count) == 0
    opts, scalers = trainer.make_optimizers(model, cfg), trainer.make_scalers(cfg)
    trainer.set_learning_rates(opts, cfg, 1)
    metrics = trainer.alternating_update(model, [residual_batch()], opts, scalers, cfg)
    assert metrics['act_loss'] > 0 and metrics['flow_loss'] > 0
    assert int(model.act_update_count) == int(model.flow_update_count) == 1


def test_scratch_checkpoint_roundtrip_and_resume_keep_own_weights(tmp_path):
    cfg = scratch_config()
    model = AlternatingResidualFlow(cfg, stats(), initialize_backbone=False)
    opts, scalers = trainer.make_optimizers(model, cfg), trainer.make_scalers(cfg)
    b = residual_batch()
    trainer.set_learning_rates(opts, cfg, 1)
    trainer.alternating_update(model, [b], opts, scalers, cfg)
    path = tmp_path/'last.pt'
    trainer.save_checkpoint(path, model, opts, scalers, cfg, index(), 1,
        NS(state_dict=lambda: {'epoch': 0, 'batches_consumed': 1, 'samples_consumed': 2}),
        best(), None, None, resumable=True)
    loaded, payload = load_checkpoint(path)
    assert_states(loaded, states(model))
    assert payload['config']['from_scratch'] and payload['config']['base_act_step'] == 0
    with patch.object(sys, 'argv', ['train', '--resume', str(path)]):
        _, resumed, _ = trainer.parse_args()
    assert resumed.from_scratch and resumed.init_from is None and not resumed.pretrained_backbone
    with patch.object(sys, 'argv', ['train', '--resume', str(path), '--steps', '15']):
        _, extended, _ = trainer.parse_args()
    assert extended.steps == 15 and extended.lr_schedule_steps == cfg.steps == 10
    trainer.set_learning_rates(opts, extended, 11)
    assert opts['act'].param_groups[0]['lr'] == pytest.approx(cfg.act_learning_rate*cfg.minimum_lr_ratio)
    assert opts['flow'].param_groups[0]['lr'] == pytest.approx(cfg.learning_rate*cfg.minimum_lr_ratio)
    trainer.set_learning_rates(opts, extended, 15)
    assert opts['flow'].param_groups[0]['lr'] == pytest.approx(cfg.learning_rate*cfg.minimum_lr_ratio)
    with patch.object(sys, 'argv', ['train', '--resume', str(path), '--from-scratch']):
        with pytest.raises(SystemExit):
            trainer.parse_args()


def test_scratch_default_and_cli_cannot_silently_import_previous_weights():
    with patch.object(sys, 'argv', ['train', '--check-data']):
        _, cfg, _ = trainer.parse_args()
    assert cfg.from_scratch and cfg.init_from is None and not cfg.pretrained_backbone
    assert cfg.output_dir.endswith('alternating_residual_flow_scratch_v1')
    with patch.object(sys, 'argv', ['train', '--from-scratch', '--init-from', '/mnt/d/old.pt']):
        with pytest.raises(SystemExit):
            trainer.parse_args()
    cfg = scratch_config()
    cfg.pretrained_backbone = True
    with pytest.raises(ValueError, match='cannot load'):
        cfg.validate()
