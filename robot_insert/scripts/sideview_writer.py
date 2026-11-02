"""ROS-independent bounded PNG-pack writer for four-camera demonstrations."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import sqlite3
import threading

import cv2
import numpy as np

from sideview_data import CAMERAS


def rgb_image(message, settings):
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(message.encoding)
    if channels is None or message.step < message.width*channels:
        raise ValueError('Invalid image format or stride')
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if len(raw) != message.height*message.step:
        raise ValueError('Image data length disagrees with its dimensions')
    image = raw.reshape(message.height, message.step)[:, :message.width*channels]
    image = image.reshape(message.height, message.width, channels)[..., :3]
    if message.encoding.startswith('bgr'):
        image = image[..., ::-1]
    return cv2.resize(image, (settings['record_width'], settings['record_height']), interpolation=cv2.INTER_AREA)


def encode_image(message, settings):
    image = rgb_image(message,settings)
    if image.std() < 1:
        raise ValueError('Camera image is blank')
    ok,png = cv2.imencode('.png',image[...,::-1],[cv2.IMWRITE_PNG_COMPRESSION,1])
    if not ok: raise ValueError('PNG encoding failed')
    return png


class EpisodeWriter:
    """Bounded frame queue; incremental commits survive interruptions."""
    def __init__(self, output, name, settings):
        self.output, self.name, self.settings = Path(output), name, settings
        self.queue = queue.Queue(maxsize=64)
        self.failure = None
        self.frames = 0
        self.commands = 0
        self.thread = threading.Thread(target=self.run, name='sideview-writer', daemon=True)
        self.thread.start()

    def put(self, item):
        if self.failure:
            raise RuntimeError(self.failure)
        self.queue.put(item, timeout=2)

    def run(self):
        db = None
        images = None
        encoders = None
        try:
            encoders = ThreadPoolExecutor(max_workers=4,thread_name_prefix='sideview-png')
            # DrvFS has high per-write latency. Buffer PNGs into one sequential
            # file; SQLite stores only small metadata/offset rows, never blobs.
            images = (self.output / f'{self.name}.images').open('wb',buffering=4*1024*1024)
            db = sqlite3.connect(self.output / f'{self.name}.sqlite3')
            # Large WAL pages avoid thousands of small synchronous DrvFS writes.
            db.execute('PRAGMA page_size=65536')
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=NORMAL')
            db.execute('PRAGMA cache_size=-32768')
            # Checkpoint when this episode closes, never during sensor callbacks.
            db.execute('PRAGMA wal_autocheckpoint=0')
            db.execute('CREATE TABLE frames (stamp REAL PRIMARY KEY, metadata TEXT NOT NULL, '
                       'left_png BLOB, center_png BLOB, right_png BLOB, side_png BLOB, image_offsets TEXT)')
            db.execute('CREATE TABLE commands (id INTEGER PRIMARY KEY, stamp REAL, metadata TEXT NOT NULL)')
            while True:
                item = self.queue.get()
                if item is None:
                    break
                kind, record = item[:2]
                if kind == 'command':
                    db.execute('INSERT INTO commands(stamp, metadata) VALUES (?, ?)',
                               (record['simulation_time'], json.dumps(record, allow_nan=False)))
                    self.commands += 1
                else:
                    offsets = []
                    pngs = encoders.map(lambda message:encode_image(message,self.settings),item[2])
                    for camera, png in zip(CAMERAS, pngs):
                        offsets.append([images.tell(),len(png)])
                        images.write(png)
                        if self.frames in (0, 200, 400, 600):
                            (self.output / f'{self.name}_{self.frames:04d}_{camera}.png').write_bytes(png)
                    db.execute('INSERT INTO frames(stamp,metadata,image_offsets) VALUES (?, ?, ?)',
                               (record['simulation_time'], json.dumps(record, allow_nan=False),json.dumps(offsets)))
                    self.frames += 1
                if (self.frames + self.commands) % 32 == 0:
                    images.flush()
                    db.commit()
            images.flush()
            db.commit()
        except Exception as error:
            self.failure = f'{type(error).__name__}: {error}'
        finally:
            if encoders is not None:
                encoders.shutdown(wait=True)
            if images is not None:
                images.close()
            if db is not None:
                db.close()

    def finish(self):
        if self.thread.is_alive():
            self.queue.put(None, timeout=10)
            self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Recorder did not finish within 30 seconds')
        if self.failure:
            raise RuntimeError(self.failure)

