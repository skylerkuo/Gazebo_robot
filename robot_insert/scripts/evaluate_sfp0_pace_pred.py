#!/usr/bin/env python3
"""Evaluate PACE trained with Pred on the shared 100-scene protocol."""
from evaluate_sfp0_pace import main

if __name__ == '__main__':
    raise SystemExit(main(policy_type='pace_pred'))
