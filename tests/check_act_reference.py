"""Numerically compare ACT layers with a downloaded, pinned official checkout.

Run: python tests/check_act_reference.py --reference-dir /tmp/act_official_reference
The directory must contain transformer.py, backbone.py and position_encoding.py.
This script does not download code, train on demonstrations, or start ROS.
"""
import argparse
import ast
import json
import math
from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from robot_bringup.act_model import (ACTConfig, DecoderLayer, EncoderLayer,
                                     FrozenBatchNorm2d, spatial_positions)


def reference_definitions(path, names):
    # Select the relevant class/function definitions, avoiding the original
    # repository's optional IPython and util.misc imports. Execute only code
    # from the explicit official source directory provided by the caller.
    parsed = ast.parse(path.read_text())
    selected = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = dict(torch=torch, nn=nn, F=F, Optional=Optional, Tensor=Tensor, math=math)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    args = parser.parse_args()
    root = args.reference_dir
    reference = reference_definitions(root / 'transformer.py',
        {'TransformerEncoderLayer', 'TransformerDecoderLayer', '_get_activation_fn'})
    torch.manual_seed(37)
    torch.set_num_threads(2)
    config = ACTConfig(hidden_dim=32, nheads=4, feedforward_dim=64, dropout=0.)
    results = {}

    def compare(name, current, expected):
        torch.testing.assert_close(current, expected, atol=2e-6, rtol=1e-5)
        results[name] = (current - expected).abs().max().item()

    original = reference['TransformerEncoderLayer'](32, 4, 64, dropout=0.)
    current = EncoderLayer(config)
    current.load_state_dict(original.state_dict())
    source, position = torch.randn(2, 5, 32), torch.randn(1, 5, 32)
    padding = torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
    compare('encoder_max_abs_error', current(source, position, padding),
            original(source.transpose(0, 1), src_key_padding_mask=padding,
                     pos=position.transpose(0, 1)).transpose(0, 1))

    original = reference['TransformerDecoderLayer'](32, 4, 64, dropout=0.)
    current = DecoderLayer(config)
    current.load_state_dict(original.state_dict())
    content, queries = torch.randn(2, 3, 32), torch.randn(1, 3, 32)
    compare('decoder_max_abs_error', current(content, source, position, queries),
            original(content.transpose(0, 1), source.transpose(0, 1),
                     pos=position.transpose(0, 1), query_pos=queries.transpose(0, 1)).transpose(0, 1))

    original = reference_definitions(root / 'backbone.py', {'FrozenBatchNorm2d'})['FrozenBatchNorm2d'](4)
    current = FrozenBatchNorm2d(4)
    original.running_mean.uniform_(-1., 1.)
    original.running_var.uniform_(.1, 2.)
    current.load_state_dict(original.state_dict())
    pixels = torch.rand(2, 4, 3, 5)
    compare('frozen_bn_max_abs_error', current(pixels), original(pixels))

    original = reference_definitions(root / 'position_encoding.py', {'PositionEmbeddingSine'})['PositionEmbeddingSine']
    expected = original(16, normalize=True)(torch.zeros(2, 32, 3, 5))[0].permute(1, 2, 0)
    compare('spatial_positions_max_abs_error', spatial_positions(3, 5, 32, torch.device('cpu')), expected)
    print(json.dumps({'official_commit': (root / 'commit.txt').read_text().strip(), 'passed': True, **results}, indent=2))


if __name__ == '__main__':
    main()
