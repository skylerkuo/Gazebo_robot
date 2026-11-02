"""Fail clearly before cache writes when D: is full (DrvFS may report EIO)."""
from contextlib import contextmanager
import errno
from pathlib import Path
import shutil

GIB=1024**3


def free_bytes(path):
    path=Path(path).resolve()
    while not path.exists():path=path.parent
    return shutil.disk_usage(path).free


def ensure_free_space(path,required_bytes,operation='preparing training data'):
    free=free_bytes(path)
    if free<required_bytes:
        raise OSError(errno.ENOSPC,
            f'Not enough disk space for {operation}: {free/GIB:.2f} GiB free; '
            f'need at least {required_bytes/GIB:.2f} GiB. Free space on this drive and rerun; '
            'completed episode caches are reused.',str(path))


@contextmanager
def storage_write_guard(path):
    try:yield
    except OSError as error:
        if error.errno in (errno.EIO,errno.ENOSPC) and free_bytes(path)<GIB:
            raise OSError(errno.ENOSPC,
                'Drive filled while writing training cache. Free space and rerun; source recordings are unchanged.',
                str(path)) from error
        raise
