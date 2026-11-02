#!/usr/bin/env python3
"""Four-view conditional Flow Matching, paper Eq. (22)-(23)."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main

if __name__ == '__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_flow_expert_finetune.json',expected_policy='flow')
