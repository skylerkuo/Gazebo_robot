#!/usr/bin/env python3
"""Separate PACE Phase/Event/Pred experiment; preserve the existing PACE run."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main

if __name__ == '__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_pace_pred_expert_finetune.json',
         expected_policy='pace_pred')
