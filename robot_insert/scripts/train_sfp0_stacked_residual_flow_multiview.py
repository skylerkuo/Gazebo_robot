#!/usr/bin/env python3
"""Freeze ACT + first residual Flow and train a second residual Flow."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main

if __name__ == '__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_stacked_residual_flow_expert.json',
         expected_policy='stacked_residual_flow')
