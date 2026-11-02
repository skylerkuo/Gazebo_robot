"""Fault injection for temporary D: reads and fail-closed session teardown."""
import errno
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import sideview_engine_done as monitor


class FakeClock:
    def __init__(self): self.value=0.
    def now(self): return self.value
    def sleep(self, seconds): self.value+=seconds


@pytest.fixture
def ready(tmp_path):
    (tmp_path/'scoring').mkdir()
    (tmp_path/'eval.log').write_text('[INFO] [aic_engine-7]: process has finished cleanly [pid 104]\n')
    (tmp_path/'settings.json').write_text(json.dumps({'trial_ids':['episode_000000']}))
    (tmp_path/'scoring/scoring.yaml').write_text('total: 75\nepisode_000000: {}\n')
    return tmp_path


def wait(output, **kwargs):
    clock=FakeClock()
    result=monitor.wait_for_completion(output,timeout=2.,read_error_timeout=.5,
        clock=clock.now,sleep=clock.sleep,**kwargs)
    return result,clock


def test_single_enodata_read_retries_and_preserves_strict_success_proof(ready,monkeypatch):
    original=Path.read_text
    failed=[]
    def read(path,*args,**kwargs):
        if path.name=='eval.log' and not failed:
            failed.append(True)
            raise OSError(errno.ENODATA,'No data available')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    report,clock=wait(ready)
    assert report['engine_exit_zero'] and report['all_trials_scored']
    assert report['transient_read_retries']==1 and clock.value==.25
    proof=json.loads((ready/'official-engine-complete.json').read_text())
    assert proof==report
    assert not (ready/'official-engine-complete.json.partial').exists()


def test_persistent_enodata_has_bounded_retry_and_never_writes_success(ready,monkeypatch):
    def failure(_): raise OSError(errno.ENODATA,'No data available')
    monkeypatch.setattr(monitor,'completed',failure)
    with pytest.raises(TimeoutError,match='Cannot read collection status'):
        wait(ready)
    assert not (ready/'official-engine-complete.json').exists()


@pytest.mark.parametrize('code',[errno.EIO,errno.EACCES,errno.ENOSPC])
def test_real_storage_or_permission_errors_are_not_hidden(ready,monkeypatch,code):
    def failure(_): raise OSError(code,'fatal storage error')
    monkeypatch.setattr(monitor,'completed',failure)
    with pytest.raises(OSError) as error: wait(ready)
    assert error.value.errno==code
    assert not (ready/'official-engine-complete.json').exists()


def test_partial_settings_json_is_retried(ready,monkeypatch):
    original=Path.read_text
    failed=[]
    def read(path,*args,**kwargs):
        if path.name=='settings.json' and not failed:
            failed.append(True)
            return '{'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    report,_=wait(ready)
    assert report['transient_read_retries']==1


def test_incomplete_scoring_does_not_allow_teardown(ready):
    (ready/'scoring/scoring.yaml').write_text('total: 75\nepisode_000000:\n')
    assert not monitor.completed(ready)
    with pytest.raises(TimeoutError,match='Official engine did not finish'):wait(ready)
    assert not (ready/'official-engine-complete.json').exists()


def test_engine_error_remains_fatal(ready):
    (ready/'eval.log').write_text('[ERROR] [aic_engine-7]: process has died [pid 104, exit code -6]\n')
    with pytest.raises(RuntimeError,match='engine exited with an error'):wait(ready)
    assert not (ready/'official-engine-complete.json').exists()


@pytest.mark.parametrize('dead_name',['evaluator','model'])
def test_dead_child_fails_without_waiting_for_full_timeout(ready,monkeypatch,dead_name):
    (ready/'eval.log').write_text('[INFO] still initializing\n')
    dead_pid=101 if dead_name=='evaluator' else 102
    monkeypatch.setattr(monitor,'process_alive',lambda pid:pid!=dead_pid)
    with pytest.raises(RuntimeError,match=dead_name+' process'):
        wait(ready,evaluator_pid=101,model_pid=102)
    assert not (ready/'official-engine-complete.json').exists()


def test_transient_proof_write_error_retries_atomic_write(ready,monkeypatch):
    original=Path.write_text
    failed=[]
    def write(path,*args,**kwargs):
        if path.name=='official-engine-complete.json.partial' and not failed:
            failed.append(True)
            raise OSError(errno.ENODATA,'temporarily unavailable')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',write)
    report,_=wait(ready)
    assert report['transient_read_retries']==1
    assert json.loads((ready/'official-engine-complete.json').read_text())==report


def test_process_liveness_reads_linux_proc_and_handles_missing_pid(monkeypatch):
    monkeypatch.setattr(Path,'read_text',lambda *_:'100 (worker) Z 1 2 3')
    assert not monitor.process_alive(100)
    monkeypatch.setattr(Path,'read_text',lambda *_:'100 (worker) S 1 2 3')
    assert monitor.process_alive(100)
    def missing(*_):raise FileNotFoundError()
    monkeypatch.setattr(Path,'read_text',missing)
    assert not monitor.process_alive(100)
    assert monitor.process_alive(None)
