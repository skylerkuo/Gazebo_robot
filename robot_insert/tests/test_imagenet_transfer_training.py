"""New ImageNet ACT must seed only its own trainable recovery Flow encoder."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from train_sfp0_act_multiview import parse_args


def test_two_stage_defaults_use_new_act_only_and_match_evaluation_contract():
    configs = Path(__file__).resolve().parents[1]/'configs'
    with patch.object(sys, 'argv', ['train', '--check-data']):
        _, act, resume = parse_args(configs/'sfp0_act_direct100_imagenet.json', 'act')
    assert resume is None and act.init_from is None and act.pretrained_backbone
    assert act.dataset_root.endswith('aic_sfp0_direct_success_100_v1')
    assert act.output_dir.endswith('aic_sfp0_act_direct100_imagenet_v1')
    assert act.steps == 10000 and act.backbone_lr > 0
    assert act.expert_only and not act.retain_model_motion

    # --check-data remains usable before this experiment has produced ACT BEST.
    with patch.object(sys, 'argv', ['train', '--check-data']):
        _, flow, resume = parse_args(configs/'sfp0_flow_recovery_direct100_transfer.json', 'residual_flow')
    assert resume is None and flow.init_from == str(Path(act.output_dir)/'checkpoints/best.pt')
    assert flow.dataset_root.endswith('aic_sfp0_recovery_only_v1')
    assert flow.output_dir.endswith('aic_sfp0_flow_recovery_direct100_transfer_v1')
    assert flow.flow_encoder_init_from_act and not flow.pretrained_backbone
    assert flow.new_dataset and flow.expert_only and not flow.retain_model_motion
    assert flow.steps == 20000 and flow.learning_rate == 3e-5 and flow.backbone_lr == 3e-6
    assert act.seed == flow.seed == 43
    assert act.eval_samples == flow.eval_samples == 1500
    assert act.eval_every == act.save_every == flow.eval_every == flow.save_every == 500
    assert act.position_scale_m == flow.position_scale_m == .01
