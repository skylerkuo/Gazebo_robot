# AIC cable insertion code

The collection, training and Gazebo inference code is in [robot_insert](robot_insert/README.md).
It includes ACT, supervised Residual Flow, PACE experiments, tests and the dataset publishing tools.

## SFP0 data and reusable model components

[Public configuration templates and workflow](robot_insert/docs/PUBLIC_WORKFLOW.md)

- [SFP0 V2 dataset](https://huggingface.co/datasets/Kuoskyler/aic-cable-insertion-gazebo-sfp0-v2)
- [SE(3) pose operations](robot_insert/scripts/contact_se3.py)
- [Causal TCP and force/torque history](robot_insert/scripts/contact_history.py)
- [GRU history encoder and cross-attention](robot_insert/scripts/gru_history.py)
- [Physical action-error supervision](robot_insert/scripts/port_action_supervision.py)
