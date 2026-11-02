#!/usr/bin/env python3
"""Download a pinned official AIC image without installing a system container engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request

from prefetch_layers import prefetch

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "ghcr.io/intrinsic-dev/aic/aic_eval@sha256:253b99a4aa6e0e4fc0cf8c1cfd0269e64aa86411c2bd585c95a1732d4a56a165"
CRANE_VERSION = "v0.22.1"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    runtime, tools = ROOT / "runtime", ROOT / "tools"
    runtime.mkdir(exist_ok=True)
    tools.mkdir(exist_ok=True)
    rootfs, marker = runtime / "rootfs", runtime / "rootfs.ready"
    if marker.exists():
        required = (rootfs / "entrypoint.sh", rootfs / "ws_aic/install/setup.bash")
        if marker.read_text().strip() != IMAGE or not all(path.is_file() for path in required):
            raise RuntimeError("Prepared image marker or rootfs is inconsistent; inspect before downloading again")
        # Download archives/layers may be removed after a successful extraction.
        # Re-running setup must not fetch them again for an already usable image.
        print("Official AIC root filesystem already ready:", rootfs, flush=True)
        return
    crane = tools / "crane"
    if not crane.exists():
        name = "go-containerregistry_Linux_x86_64.tar.gz"
        base = f"https://github.com/google/go-containerregistry/releases/download/{CRANE_VERSION}/"
        with urllib.request.urlopen(base + "checksums.txt", timeout=60) as response:
            checksums = response.read().decode()
        expected = next(line.split()[0] for line in checksums.splitlines() if line.split()[-1] == name)
        archive = tools / name
        print("Downloading crane", CRANE_VERSION, flush=True)
        urllib.request.urlretrieve(base + name, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise RuntimeError("crane archive checksum mismatch")
        with tarfile.open(archive) as tar:
            member = tar.getmember("crane")
            with tar.extractfile(member) as source, crane.open("wb") as destination:
                destination.write(source.read())
        crane.chmod(0o755)
    config = json.loads(subprocess.check_output([str(crane), "config", IMAGE]))
    (runtime / "image-config.json").write_text(json.dumps(config, indent=2) + "\n")
    (runtime / "image-reference.txt").write_text(IMAGE + "\n")
    archive = runtime / "aic_eval.tar"
    if not archive.exists():
        image_archive = runtime / "aic_image.tar"
        environment = dict(os.environ, GODEBUG="http2client=0")
        if not image_archive.exists():
            image_partial = runtime / "aic_image.tar.partial"
            cache = runtime / "layers"
            cache.mkdir(exist_ok=True)
            manifest = json.loads(subprocess.check_output([str(crane), "manifest", IMAGE], env=environment))
            (runtime / "image-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            prefetch(manifest, runtime)
            for attempt in range(3):
                print("Pulling official image with layer cache, attempt", attempt + 1, flush=True)
                status = subprocess.run([
                    str(crane), "pull", "--cache_path", str(cache), IMAGE, str(image_partial),
                ], env=environment).returncode
                if status == 0:
                    image_partial.rename(image_archive)
                    break
                if attempt == 2:
                    raise RuntimeError("Official image download failed; complete layers remain cached")
                time.sleep(2)
        partial = archive.with_suffix(".tar.partial")
        print("Exporting official image from local archive", flush=True)
        with image_archive.open("rb") as source:
            subprocess.run([str(crane), "export", "-", str(partial)], stdin=source, env=environment, check=True)
        partial.rename(archive)
    if not marker.exists():
        rootfs.mkdir(exist_ok=True)
        print("Extracting official root filesystem", flush=True)
        subprocess.run([
            "tar", "--extract", "--file", str(archive), "--directory", str(rootfs),
            "--no-same-owner", "--no-same-permissions", "--exclude=dev/*",
        ], check=True)
        marker.write_text(IMAGE + "\n")
    print("Official AIC root filesystem ready:", rootfs, flush=True)


if __name__ == "__main__":
    main()
