#!/usr/bin/env python3
"""Keep a trained ACT + Flow pair, freeze ACT and fine-tune only its existing Flow."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main

if __name__=='__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_flow_only_after_scratch.json',
         expected_policy='residual_flow')
