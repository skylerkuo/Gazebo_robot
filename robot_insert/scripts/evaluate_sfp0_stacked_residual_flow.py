#!/usr/bin/env python3
"""Resumable autonomous evaluation of the two residual Flow layers."""
from evaluate_sfp0_pace import main

if __name__ == '__main__':
    raise SystemExit(main(policy_type='stacked_residual_flow'))
