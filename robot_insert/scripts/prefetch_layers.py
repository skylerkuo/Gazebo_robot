"""Resume and verify public GHCR layers before crane assembles the local image."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prefetch(manifest, runtime):
    repository = "intrinsic-dev/aic/aic_eval"
    cache = runtime / "layers"
    cache.mkdir(exist_ok=True)
    token_url = "https://ghcr.io/token?service=ghcr.io&scope=repository:" + repository + ":pull"
    for layer in sorted(manifest["layers"], key=lambda item: item["size"], reverse=True):
        digest, size = layer["digest"], layer["size"]
        target = cache / digest
        if target.is_file() and target.stat().st_size == size and sha256(target) == digest.split(":")[1]:
            print("Verified cached layer", digest[:22], flush=True)
            continue
        with urllib.request.urlopen(token_url, timeout=30) as response:
            token = json.load(response)["token"]
        request = urllib.request.Request(
            "https://ghcr.io/v2/" + repository + "/blobs/" + digest,
            headers={"Authorization": "Bearer " + token, "Range": "bytes=0-0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            url = response.geturl()
            response.read(1)
        chunks = runtime / "chunks" / digest.split(":")[1]
        chunks.mkdir(parents=True, exist_ok=True)
        chunk_size = 16 * 1024 * 1024
        count = (size + chunk_size - 1) // chunk_size
        print("Downloading layer", digest[:22], size, "bytes in", count, "resumable chunks", flush=True)
        def download(index):
            start = index * chunk_size
            end = min(size, start + chunk_size) - 1
            path = chunks / str(index)
            if path.exists() and path.stat().st_size == end - start + 1:
                return path
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=90) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise RuntimeError("Registry did not honor the requested byte range")
                        data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError("Incomplete chunk")
                    partial = path.with_suffix(".partial")
                    partial.write_bytes(data)
                    partial.replace(path)
                    return path
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(attempt + 1)
        with ThreadPoolExecutor(max_workers=8) as pool:
            paths = []
            for index, path in enumerate(pool.map(download, range(count)), 1):
                paths.append(path)
                if index % 16 == 0 or index == count:
                    print("Layer chunks ready:", index, "/", count, flush=True)
        partial = target.with_suffix(".partial")
        with partial.open("wb") as destination:
            for path in paths:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, destination)
        if sha256(partial) != digest.split(":")[1]:
            raise RuntimeError("Layer checksum mismatch: " + digest)
        partial.replace(target)
        shutil.rmtree(chunks)
        print("Layer verified", digest[:22], flush=True)
