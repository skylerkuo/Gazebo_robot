#!/usr/bin/env python3
"""Train a new ACT on 100 direct demonstrations, fine-tuning ImageNet ResNet-18."""
from ftact import INSERT_ROOT
from train_sfp0_act_multiview import main

if __name__ == '__main__':
    main(default_config=INSERT_ROOT/'configs/sfp0_act_direct100_imagenet.json',
         expected_policy='act')
