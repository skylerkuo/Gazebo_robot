#!/usr/bin/env python3
"""Train random ACT and ResNet only on successful full-controller demonstrations."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main

if __name__=='__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_act_direct_success.json',expected_policy='act')
