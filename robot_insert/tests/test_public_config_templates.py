"""Public templates render local settings without overwriting existing files."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from render_sfp0_public_config import main, render_config, TEMPLATES


def test_all_templates_have_expected_slots(tmp_path):
    slots = {
        'collection': set(),
        'act-train': {'ACT_DATASET', 'ACT_OUTPUT'},
        'flow-train': {'RECOVERY_DATASET', 'ACT_CHECKPOINT', 'FLOW_OUTPUT'},
        'act-gazebo': set(),
        'flow-gazebo': set(),
    }
    assert {path.stem for path in TEMPLATES.glob('*.json')} == slots.keys()
    for name, keys in slots.items():
        template = json.loads((TEMPLATES / f'{name}.json').read_text())
        config = render_config(template, {name: tmp_path / name for name in keys})
        assert '{{' not in json.dumps(config)
        if name.endswith('-train'):
            assert Path(config['output_dir']).parent == tmp_path
            assert config['index_path'] == 'auto'


def test_invalid_slots_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='Missing path values'):
        render_config({'path': '{{DATA}}'}, {})
    with pytest.raises(ValueError, match='Unused path values'):
        render_config({'seed': 42}, {'DATA': tmp_path})


def test_cli_preserves_generated_config(tmp_path):
    output = tmp_path / 'act.json'
    argv = ['act-train', '--set', f'ACT_DATASET={tmp_path / "records with spaces"}',
            '--set', f'ACT_OUTPUT={tmp_path / "output"}', '--output', str(output)]
    assert main(argv) == 0
    original = output.read_bytes()
    assert json.loads(original)['dataset_root'] == str(tmp_path / 'records with spaces')
    with pytest.raises(SystemExit) as result:
        main(argv)
    assert result.value.code == 2
    assert output.read_bytes() == original
