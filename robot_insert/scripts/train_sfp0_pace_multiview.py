#!/usr/bin/env python3
"""Four-view PACE Phase/Event; same assisted snapshot and trainer as ACT."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main


if __name__=='__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_pace_expert_finetune.json',expected_policy='pace')
