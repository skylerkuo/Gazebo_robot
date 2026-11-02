"""Pred label alignment, no future-input leakage, auxiliary gradients and reload."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from act_multiview import MultiViewACT, MultiViewConfig
from act_objectives import evaluate_pose_chunks
from ftact import WRENCH, load_checkpoint
from pace_act import PHASE, HISTORY
from pace_multiview import MultiViewPACEDataset, initialize_pace_from_act
from pace_pred_multiview import (MultiViewPACEPredConfig, MultiViewPACEPredDataset,
    MultiViewPACEPred, FORMAT, PRED_TARGET, PRED_MASK, future_wrench_targets,
    masked_prediction_error)
from train_sfp0_act_multiview import checkpoint, parse_args
from test_act_multiview import batch, stats


@pytest.fixture(scope='module', autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def config():
    return MultiViewPACEPredConfig(device='cpu', amp='none', dim_model=32,
        n_heads=4, dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
        pretrained_backbone=False, dropout=0., steps=10)


def pred_batch():
    b = batch()
    h = torch.randn(2, 4, 6)
    b[WRENCH] = h[:, -1].clone(); b[PHASE] = h.mean(1); b[HISTORY] = h
    b['action_mask'] = torch.ones(2, 20, dtype=torch.bool)
    b[PRED_TARGET] = torch.randn(2, 20, 6)*.1
    b[PRED_MASK] = torch.ones(2, 20, dtype=torch.bool)
    return b


def numeric():
    n = 30
    return {WRENCH: np.arange(n*6, dtype=np.float32).reshape(n, 6),
        'simulation_time': np.arange(n)*.05,
        'action_source': np.full(n, 'expert', dtype='U16'),
        'action_segment': np.zeros(n, dtype=np.int64),
        '__action_mask': np.ones(n, dtype=bool)}


def test_future_targets_are_t_plus_one_increments_not_absolute_force():
    a = numeric(); delta, mask = future_wrench_targets(a, 3)
    np.testing.assert_array_equal(delta, np.full((20, 6), 6.))
    assert mask.all()
    a[WRENCH][:4] -= 500
    changed, _ = future_wrench_targets(a, 3)
    np.testing.assert_array_equal(changed[0], delta[0]+500)
    np.testing.assert_array_equal(changed[1:], delta[1:])


def test_handoff_gaps_hold_and_stationary_pairs_are_masked():
    a = numeric()
    a['action_source'][:5] = 'model'; a['action_segment'][5:] = 1
    a['simulation_time'][9:] += .05
    a['action_source'][12] = 'hold'
    a['__action_mask'][16] = False
    delta, mask = future_wrench_targets(a, 0)
    expected = np.ones(20, dtype=bool)
    expected[[4, 8, 11, 12, 15, 16]] = False
    np.testing.assert_array_equal(mask, expected)
    assert not delta[~mask].any()
    with pytest.raises(ValueError):future_wrench_targets(a, 20)


def test_future_force_labels_do_not_change_observation_or_causal_history(monkeypatch):
    a = numeric()
    def base(self, i):
        self.numeric[0] = a
        return {WRENCH: torch.from_numpy(a[WRENCH][3].copy()),
            PHASE: torch.ones(6), HISTORY: torch.ones(4, 6)}
    monkeypatch.setattr(MultiViewPACEDataset, '__getitem__', base)
    index = {'_directory': 'unused', 'episodes': [{'episode_index': 0,
        'split': 'train', 'length': 30, 'train_anchor_frames': [3]}]}
    ds = MultiViewPACEPredDataset(index, 'train')
    first = ds[0]
    a[WRENCH][4:] += np.arange(26)[:, None]*10
    changed = ds[0]
    for key in [WRENCH, PHASE, HISTORY]:torch.testing.assert_close(first[key], changed[key])
    assert not torch.equal(first[PRED_TARGET], changed[PRED_TARGET])


def test_masked_mse_normalizes_by_std_and_zero_valid_labels_have_zero_gradient():
    p = torch.zeros(1, 20, 6, requires_grad=True)
    target = torch.ones_like(p)*2
    mask = torch.zeros(1, 20, dtype=torch.bool); mask[:, :2] = True
    target[:, 2:] = float('nan')
    loss, _, steps = masked_prediction_error(p, target, mask, torch.ones(6)*2)
    assert loss.item() == 1 and steps.item() == 2
    loss.backward()
    assert p.grad[:, :2].abs().sum() > 0 and not p.grad[:, 2:].any()
    p.grad = None; mask[:] = False
    loss, _, _ = masked_prediction_error(p, target, mask, torch.ones(6))
    loss.backward()
    assert loss.item() == 0 and not p.grad.any()


def test_prediction_loss_reaches_head_decoder_and_preserves_action_generation(tmp_path):
    torch.manual_seed(3); cfg = config()
    model = MultiViewPACEPred(cfg, stats(), initialize_backbone=False).eval()
    b = pred_batch()
    expected = model.predict_action_chunk(b)
    loss, metrics = model(b)
    assert metrics['pred_mse'] > 0
    assert loss.item() == pytest.approx(metrics['imitation_loss']+.05*metrics['pred_mse'])
    # Auxiliary loss alone must change both the new head and action representations.
    _, _, hidden = model._predict(b, return_hidden=True)
    pred_loss, _, _ = model.prediction_objective(b, hidden)
    pred_loss.backward()
    assert model.policy.model.pace_pred_head[-1].weight.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.policy.model.decoder.parameters())
    changed = dict(b); changed[PRED_TARGET] = b[PRED_TARGET]+100
    changed[PRED_MASK] = ~b[PRED_MASK]
    torch.testing.assert_close(model.predict_action_chunk(changed), expected, rtol=0, atol=0)
    # Inference requires neither action targets nor future wrench fields.
    observations = {k: v for k, v in b.items() if k not in ['action', 'action_mask', PRED_TARGET, PRED_MASK]}
    torch.testing.assert_close(model.predict_action_chunk(observations), expected, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.policy.get_optim_params())
    ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in model.policy.model.pace_pred_head.parameters())
    path = tmp_path/'last.pt'
    cursor = SimpleNamespace(state_dict=lambda: {'epoch': 0, 'samples_consumed': 2, 'batches_consumed': 1})
    index = {'stats': stats(), 'task': {'target_module': 'nic_card_mount_0'}, 'index_content_sha256': 'test'}
    checkpoint(path, model, optimizer, torch.amp.GradScaler('cuda', enabled=False),
        cfg, index, 1, cursor, 1., .01, {'loss': 1.}, 1, True, True)
    loaded, payload = load_checkpoint(path)
    assert payload['format'] == FORMAT and not hasattr(loaded.policy.model, 'pace_pred_head')
    assert payload['execution']['event_before_temporal_ensemble']
    assert payload['execution']['pred_head_at_inference'] is False
    assert payload['reference_paper'].endswith('2610.16308')
    torch.testing.assert_close(loaded.predict_action_chunk(observations), expected, rtol=0, atol=0)
    with patch.object(sys, 'argv', ['train', '--resume', str(path)]):
        _, resumed, _ = parse_args(expected_policy='pace_pred')
    assert resumed.pred_loss_weight == .05
    with patch.object(sys, 'argv', ['train', '--resume', str(path), '--pred-loss-weight', '.1']):
        with pytest.raises(SystemExit):parse_args(expected_policy='pace_pred')


def test_validation_keeps_imitation_selection_separate_from_auxiliary_loss():
    model = MultiViewPACEPred(config(), stats(), initialize_backbone=False).eval()
    b = pred_batch(); cfg = config(); cfg.eval_samples = 2
    first = evaluate_pose_chunks(model, [b], cfg)
    b[PRED_TARGET] = b[PRED_TARGET]+100
    second = evaluate_pose_chunks(model, [b], cfg)
    assert first['loss'] == second['loss'] and first['position_l2_m'] == second['position_l2_m']
    assert second['pred_mse'] > first['pred_mse']
    assert second['total_loss'] > first['total_loss'] and first['pred_valid_steps'] == 40


def test_shared_initialization_and_config_preserve_comparison():
    cfg = config()
    act_cfg = MultiViewConfig(**{k: v for k, v in asdict(cfg).items() if k not in
        ['phase_alpha', 'tactile_history', 'correction_horizon', 'pred_loss_weight']})
    act_cfg.policy_type = 'act'
    act = MultiViewACT(act_cfg, stats(), initialize_backbone=False)
    model = MultiViewPACEPred(cfg, stats(), initialize_backbone=False)
    initialize_pace_from_act(model, act.state_dict())
    for key, value in act.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root/'configs/sfp0_pace_expert_finetune.json').read_text())
    new = json.loads((root/'configs/sfp0_pace_pred_expert_finetune.json').read_text())
    for key, value in old.items():
        if key not in ['policy_type', 'output_dir']:assert new[key] == value
    with patch.object(sys, 'argv', ['train', '--config', str(root/'configs/sfp0_pace_pred_expert_finetune.json'), '--check-data']):
        _, parsed, _ = parse_args(expected_policy='pace_pred')
    assert parsed.pred_loss_weight == .05 and parsed.eval_samples == 1500
