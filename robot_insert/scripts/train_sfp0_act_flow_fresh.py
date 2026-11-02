#!/usr/bin/env python3
"""Fresh ACT + residual Flow on all currently accepted data, with a new 5% split."""
from ftact import INSERT_ROOT
from train_sfp0_alternating_residual_flow_multiview import main

if __name__=='__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_act_flow_fresh_v2.json')
