"""Disk-full diagnosis must protect source data and not hide real I/O failures."""
import errno
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import storage_checks as storage


def test_disk_full_stops_before_writing_episode_cache(tmp_path,monkeypatch):
    from test_human_dataset import recorded_run,profile
    from human_dataset import import_human_run
    from prepare_sfp0_act_multiview import prepare_episode
    from act_multiview import file_digest
    monkeypatch.setattr('human_dataset.DATA_DRIVE_ROOT',tmp_path)
    root=tmp_path/'source';run=recorded_run(root,expert=True)
    entry=import_human_run(root,run,profile(),tmp_path/'parent',status=0,expert_source='expert')
    source=root/entry['parquet'];checksum=file_digest(source)
    output=tmp_path/'training';output.mkdir()
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:NS(free=0))
    with pytest.raises(OSError,match='Not enough disk space') as error:
        prepare_episode(root,output,entry,2,command_provenance=True)
    assert error.value.errno==errno.ENOSPC
    assert not list(output.rglob('*.partial')) and file_digest(source)==checksum


def test_drvfs_eio_on_full_drive_is_reported_as_space_error(tmp_path,monkeypatch):
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:NS(free=0))
    with pytest.raises(OSError,match='Drive filled') as error:
        with storage.storage_write_guard(tmp_path):raise OSError(errno.EIO,'Input/output error')
    assert error.value.errno==errno.ENOSPC
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:NS(free=10*storage.GIB))
    with pytest.raises(OSError) as error:
        with storage.storage_write_guard(tmp_path):raise OSError(errno.EIO,'real filesystem error')
    assert error.value.errno==errno.EIO
