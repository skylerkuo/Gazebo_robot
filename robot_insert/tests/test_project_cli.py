"""Current workflow routing, safe help and local documentation dependencies."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import sys
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sfp_project_cli',ROOT/'robot.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def test_all_routed_scripts_exist_and_current_models_are_selected():
    for name in cli.COMMANDS:
        command=cli.build_command(name,[])
        assert command[0]==sys.executable and Path(command[2]).is_file()
    assert 'aic_sfp0_act_direct_scratch_v1' in ' '.join(cli.build_command('run-act',[]))
    assert 'aic_sfp0_flow_recovery_after_direct_act_v1' in ' '.join(cli.build_command('run-flow',[]))
    for name in ('evaluate-act','evaluate-flow'):
        command=cli.build_command(name,['--resume','--gui','true'])
        assert '--output-dir' in command and command[-3:]==['--resume','--gui','true']


def test_routing_replaces_process_and_preserves_arguments(monkeypatch):
    execute=Mock();monkeypatch.setattr(cli.os,'execv',execute)
    assert cli.main(['train-act','--from-scratch','--batch-size','16'])==0
    executable,args=execute.call_args.args
    assert executable==sys.executable
    assert Path(args[2]).name=='train_sfp0_act_direct.py'
    assert args[-3:]==['--from-scratch','--batch-size','16']


def test_preview_and_help_never_execute_a_child(monkeypatch,capsys):
    execute=Mock(side_effect=AssertionError('Read-only CLI operation'))
    monkeypatch.setattr(cli.os,'execv',execute)
    assert cli.main(['--print-command','run-flow','--gui','false'])==0
    assert '--gui false' in capsys.readouterr().out
    assert cli.main([])==0
    assert 'train-act' in capsys.readouterr().out
    with pytest.raises(SystemExit) as result:cli.main(['--help'])
    assert result.value.code==0 and not execute.called


@pytest.mark.parametrize('command',['prepare-recovery','prepare-env'])
def test_help_does_not_collect_data_or_prepare_environment(command):
    import subprocess
    result=subprocess.run([sys.executable,str(ROOT/'robot.py'),command,'--help'],
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and 'usage:' in result.stdout
    assert 'curate_recovery_episode' not in result.stdout


def test_new_dataset_override_is_explicit_and_keeps_scratch_initialization():
    sys.path.insert(0,str(ROOT/'scripts'))
    from train_sfp0_act_multiview import parse_args
    from unittest.mock import patch
    args=['train','--config',str(ROOT/'configs/sfp0_act_direct_success.json'),'--from-scratch',
        '--check-data','--dataset-root','/mnt/d/robot_datasets/example_v2']
    with patch.object(sys,'argv',args):_,cfg,_=parse_args(expected_policy='act')
    assert cfg.dataset_root.endswith('example_v2') and cfg.init_from is None and not cfg.pretrained_backbone


def test_markdown_links_are_resolved_locally():
    documents=[ROOT/'README.md',*sorted((ROOT/'docs').glob('*.md'))]
    for path in documents:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            assert (path.parent/target.split('#',1)[0]).exists(),(path.name,target)


def test_retained_python_does_not_import_removed_modules():
    removed={'training_status_report','sync_training_status_github','run_ftact_gazebo',
        'predict_ftact','predict_sfp0_flow','benchmark_ftact_batch','audit_dataset_variation',
        'download_dataset','validate_dataset','check_lerobot_dataset','check_sideview_reader'}
    for folder in ('scripts','policies','tests'):
        for path in (ROOT/folder).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):names=[node.module or '']
                elif isinstance(node,ast.Import):names=[a.name for a in node.names]
                else:continue
                assert not {name.split('.')[0] for name in names}&removed,path.name
