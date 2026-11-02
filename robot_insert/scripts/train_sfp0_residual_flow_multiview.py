#!/usr/bin/env python3
"""Freeze the current ACT BEST and learn expert residual corrections."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main

if __name__=='__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_residual_flow_expert.json',
         expected_policy='residual_flow')
