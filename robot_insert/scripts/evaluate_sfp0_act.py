#!/usr/bin/env python3
"""Evaluate four-view ACT using the same scene and scoring protocol as PACE."""
from evaluate_sfp0_pace import main


if __name__ == '__main__':
    raise SystemExit(main(policy_type='act'))
