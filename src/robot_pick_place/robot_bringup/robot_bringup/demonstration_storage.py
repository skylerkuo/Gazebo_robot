"""Publish a completed local capture to its final filesystem without exposing partial data."""
import json
from pathlib import Path
import shutil
import uuid

import yaml


def archive_episode(capture, destination):
    """Copy after capture stops; publish atomically and retain the source on failure."""
    capture, destination = Path(capture), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name('.' + destination.name + '.copying-' + uuid.uuid4().hex[:8])
    try:
        # copyfile avoids per-file chmod/utime/xattr operations on WSL Windows
        # mounts. Sensor timestamps are stored in filenames, messages and JSON.
        shutil.copytree(capture, pending, copy_function=shutil.copyfile)
        task_file = pending / 'task.yaml'
        if task_file.exists():
            task = yaml.safe_load(task_file.read_text())
            parameters = task['mtc_node']['ros__parameters']
            parameters['demonstration_directory'] = str(destination)
            parameters['known_obstacles_file'] = str(destination / 'known_obstacles.cdr')
            task_file.write_text(yaml.safe_dump(task))
        metadata_file = pending / 'metadata.json'
        metadata = json.loads(metadata_file.read_text())
        metadata['storage'] = {'state': 'archived', 'capture_directory': str(capture),
                               'destination_directory': str(destination)}
        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite {destination}')
        pending.rename(destination)
        return metadata
    except BaseException:
        # Only remove our unpublished copy. The complete local capture remains
        # available for recovery if the destination fills up or copying stops.
        shutil.rmtree(pending, ignore_errors=True)
        raise
