#!/usr/bin/env python3
"""Load the existing ACT + Flow pair; repartition all data and fine-tune both."""
from ftact import INSERT_ROOT
from train_sfp0_alternating_residual_flow_multiview import main

if __name__=='__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_act_flow_finetune_v2.json')
