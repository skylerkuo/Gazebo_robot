#!/usr/bin/env python3
"""Audit the pinned AIC source and build a cleaned LeRobot v3 dataset on D:."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

INSERT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSERT_ROOT / "tools/dataset_python"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

NUMERIC = {"observation.state": 6, "observation.environment_state": 6, "action": 7,
           "timestamp": 1, "frame_index": 1, "episode_index": 1, "index": 1, "task_index": 1}
CAMERAS = [f"observation.images.{name}_camera" for name in ("left", "center", "right")]
QUANTILES = {"q01": .01, "q10": .1, "q50": .5, "q90": .9, "q99": .99}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def matrix(column, width):
    column = column.combine_chunks()
    if pa.types.is_fixed_size_list(column.type):
        return column.values.to_numpy(zero_copy_only=False).reshape(-1, width)
    return column.to_numpy(zero_copy_only=False).reshape(-1, width)


def numeric_stats(values):
    values = np.asarray(values, dtype=np.float64)
    result = {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
              "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
              "count": [len(values)]}
    result.update({name: np.quantile(values, q, axis=0).tolist() for name, q in QUANTILES.items()})
    return result


def histogram_stats(histogram, frames):
    grid = np.arange(256, dtype=np.float64) / 255
    count = histogram.sum(axis=1)
    mean = (histogram * grid).sum(axis=1) / count
    variance = (histogram * grid ** 2).sum(axis=1) / count - mean ** 2
    def shape(value):
        return np.asarray(value).reshape(3, 1, 1).tolist()
    result = {"min": shape(np.argmax(histogram > 0, axis=1) / 255),
              "max": shape((255 - np.argmax(histogram[:, ::-1] > 0, axis=1)) / 255),
              "mean": shape(mean), "std": shape(np.sqrt(np.maximum(variance, 0))), "count": [frames]}
    cdf = histogram.cumsum(axis=1)
    for name, quantile in QUANTILES.items():
        result[name] = shape([np.searchsorted(cdf[c], quantile * (count[c] - 1), side="right") / 255 for c in range(3)])
    return result


def continuous_quaternions(actions):
    """Keep physical rotations unchanged while normalizing and removing q/-q jumps."""
    result = np.asarray(actions, dtype=np.float32).copy()
    quaternion = result[:, 3:].astype(np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    dots = np.sum(quaternion[1:] * quaternion[:-1], axis=1)
    signs = np.r_[1., np.cumprod(np.where(dots < 0, -1., 1.))]
    if quaternion[0, np.argmax(np.abs(quaternion[0]))] < 0:
        signs *= -1
    result[:, 3:] = (quaternion * signs[:, None]).astype(np.float32)
    return result, int((dots < 0).sum()), int((signs < 0).sum())


def valid_runs(valid, timestamps, fps, minimum):
    positions = np.flatnonzero(valid)
    if not len(positions):
        return [], 0
    boundaries = np.flatnonzero((np.diff(positions) != 1) |
        (np.abs(np.diff(timestamps[positions]) - 1 / fps) > 1e-4)) + 1
    runs, discarded = [], 0
    for run in np.split(positions, boundaries):
        if len(run) < minimum:
            discarded += len(run)
        else:
            runs.append((int(run[0]), int(run[-1]) + 1))
    return runs, discarded


def source_splits(episodes, seed):
    shuffled = np.random.default_rng(seed).permutation(sorted(episodes)).tolist()
    train_end, val_end = int(.8 * len(shuffled)), int(.9 * len(shuffled))
    return {"train": shuffled[:train_end], "validation": shuffled[train_end:val_end], "test": shuffled[val_end:]}


def verify_checksum(root, entry):
    path = root / entry["path"]
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise RuntimeError("Missing or wrong-size source: " + str(path))
    lfs = entry.get("lfs")
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    expected = lfs["oid"].removeprefix("sha256:") if lfs else entry["oid"]
    if digest.hexdigest() != expected:
        raise RuntimeError("Source checksum mismatch: " + str(path))


def audit_source(source, manifest, info, preparation):
    count = info["total_frames"]
    values = {name: np.empty((count, width), dtype=np.float32 if name.startswith("observation.") or name in ("action", "timestamp") else np.int64)
              for name, width in NUMERIC.items()}
    image_valid = np.ones(count, dtype=bool)
    image_errors, image_count, file_ranges = [], 0, []
    offset = 0
    entries = sorted(manifest["files"], key=lambda item: item["path"])
    data = [item for item in entries if item["path"].startswith("data/")]
    for entry in entries:
        if not entry["path"].startswith("data/"):
            verify_checksum(source, entry)
    for number, entry in enumerate(data, 1):
        verify_checksum(source, entry)
        table = pq.read_table(source / entry["path"])
        end = offset + len(table)
        if end > count:
            raise RuntimeError("Source contains more rows than meta/info.json")
        for name, width in NUMERIC.items():
            values[name][offset:end] = matrix(table[name], width)
        if not np.array_equal(values["index"][offset:end, 0], np.arange(offset, end)):
            raise RuntimeError("Noncontiguous source global index: " + entry["path"])
        for name in CAMERAS:
            for local, image in enumerate(table[name].to_pylist()):
                image_count += 1
                try:
                    if not image["bytes"]:
                        raise ValueError("Missing embedded image bytes")
                    with Image.open(io.BytesIO(image["bytes"])) as png:
                        if png.format != "PNG" or png.mode != "RGB" or png.size != (288, 256):
                            raise ValueError(f"Unexpected image {png.format} {png.mode} {png.size}")
                        png.verify()
                except Exception as error:
                    image_valid[offset + local] = False
                    if len(image_errors) < 30:
                        image_errors.append({"index": offset + local, "camera": name, "error": str(error)})
        file_ranges.append({"path": entry["path"], "start": offset, "end": end})
        offset = end
        if number % 5 == 0 or number == len(data):
            print(f"Source audit: {number}/{len(data)} shards, {offset:,} frames, {image_count:,} PNG payloads verified", flush=True)
    if offset != count:
        raise RuntimeError("Source row count disagrees with metadata")
    finite = np.ones(count, dtype=bool)
    for name in ("observation.state", "observation.environment_state", "action", "timestamp"):
        finite &= np.isfinite(values[name]).all(axis=1)
    norms = np.linalg.norm(values["action"][:, 3:].astype(np.float64), axis=1)
    unit = np.abs(norms - 1) <= .01
    zero_state = (values["observation.state"] == 0).all(axis=1)
    valid = finite & unit & ~zero_state & image_valid
    report = {
        "source_frames": count, "source_files_verified": len(entries),
        "source_data_shards": len(data), "png_headers_and_crcs_verified": image_count,
        "invalid_image_frames": int((~image_valid).sum()), "image_error_examples": image_errors,
        "nonfinite_numeric_frames": int((~finite).sum()),
        "zero_action_quaternion_frames": int((norms < 1e-6).sum()),
        "nonunit_action_quaternion_frames": int((~unit).sum()),
        "all_zero_state_frames": int(zero_state.sum()), "initial_valid_frames": int(valid.sum()),
    }
    write_json(preparation / "source_audit.json", report)
    return values, valid, file_ranges, report


def image_statistics(table):
    indices = np.unique(np.linspace(0, len(table) - 1, min(32, len(table)), dtype=int))
    selected = table.take(pa.array(indices))
    result, histograms = {}, {}
    for name in CAMERAS:
        histogram = np.zeros((3, 256), dtype=np.int64)
        for image in selected[name].to_pylist():
            with Image.open(io.BytesIO(image["bytes"])) as png:
                pixels = np.asarray(png.convert("RGB"))[::4, ::4].reshape(-1, 3)
                for channel in range(3):
                    histogram[channel] += np.bincount(pixels[:, channel], minlength=256)
        result[name] = histogram_stats(histogram, len(indices))
        histograms[name] = histogram
    return result, histograms, len(indices)


def replace_column(table, name, values):
    field = table.schema.field(name)
    if pa.types.is_fixed_size_list(field.type):
        array = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=field.type.value_type), field.type.list_size)
    else:
        array = pa.array(np.asarray(values).reshape(-1), type=field.type)
    return table.set_column(table.schema.get_field_index(name), field, array)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/d/robot_datasets/aic_cable_insertion_gazebo_v3"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/d/robot_datasets/aic_cable_insertion_gazebo_v3_act"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-segment-frames", type=int, default=20)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and cleaned dataset must be separate directories")
    manifest = json.loads((source / "_preparation/source_manifest.json").read_text())
    if not (source / "_preparation/download_complete.json").is_file():
        raise RuntimeError("Wait for download_complete.json before preparing the full dataset")
    preparation = output / "_preparation"
    preparation.mkdir(parents=True, exist_ok=True)
    settings = {"source_repo": manifest["repo_id"], "source_revision": manifest["revision"],
                "seed": args.seed, "minimum_segment_frames": args.minimum_segment_frames, "format": 1}
    settings_path = preparation / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError("Output preparation settings differ; choose another output directory")
    if (preparation / "complete.json").exists():
        print("Dataset is already prepared:", output)
        return
    write_json(settings_path, settings)
    info = json.loads((source / "meta/info.json").read_text())
    episodes = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet")
    original_rows = episodes.to_pylist()
    if len(original_rows) != info["total_episodes"]:
        raise RuntimeError("Source episode count disagrees with info.json")
    expected_from = 0
    for ep_index, row in enumerate(original_rows):
        if row["episode_index"] != ep_index or row["dataset_from_index"] != expected_from:
            raise RuntimeError("Source episode metadata overlaps or has gaps")
        expected_from = row["dataset_to_index"]
    if expected_from != info["total_frames"]:
        raise RuntimeError("Source episode boundaries do not cover the complete dataset")
    values, valid, file_ranges, report = audit_source(source, manifest, info, preparation)
    segments_by_source, short_frames = {}, 0
    for episode in episodes.to_pylist():
        ep = episode["episode_index"]
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        if end - start != episode["length"] or not np.all(values["episode_index"][start:end, 0] == ep):
            raise RuntimeError("Episode metadata disagrees with source rows")
        if not np.array_equal(values["frame_index"][start:end, 0], np.arange(end - start)):
            raise RuntimeError("Episode frame index is not contiguous")
        if not np.all(values["task_index"][start:end] == 0):
            raise RuntimeError("Unexpected task index")
        runs, dropped = valid_runs(valid[start:end], values["timestamp"][start:end, 0], info["fps"], args.minimum_segment_frames)
        short_frames += dropped
        if runs:
            segments_by_source[ep] = [{"source_episode_index": ep, "source_start": start + a,
                "source_end": start + b, "source_frame_start": a, "source_frame_end": b,
                "source_time_start": float(values["timestamp"][start + a, 0]),
                "source_time_end": float(values["timestamp"][start + b - 1, 0])} for a, b in runs]
    split_sources = source_splits(segments_by_source, args.seed)
    segments, splits, offset = [], {}, 0
    for split, original_ids in split_sources.items():
        first_episode = len(segments)
        for ep in original_ids:
            for segment in segments_by_source[ep]:
                segment.update(episode_index=len(segments), split=split, dataset_from_index=offset)
                offset += segment["source_end"] - segment["source_start"]
                segment["dataset_to_index"] = offset
                segments.append(segment)
        splits[split] = {"episode_ids": list(range(first_episode, len(segments))), "source_episode_ids": original_ids,
                         "frame_count": sum(s["source_end"] - s["source_start"] for s in segments if s["split"] == split)}
    if not segments or not all(splits[key]["episode_ids"] for key in splits):
        raise RuntimeError("Not enough valid episodes for three splits")
    write_json(preparation / "splits.json", {"seed": args.seed, "grouped_by": "source_episode_index", "splits": splits})
    write_json(preparation / "episode_mapping.json", segments)
    final_values = {name: np.empty((offset, width), dtype=values[name].dtype) for name, width in NUMERIC.items()}
    histograms = {split: {name: np.zeros((3, 256), dtype=np.int64) for name in CAMERAS} for split in splits}
    image_frames = {split: 0 for split in splits}
    metadata, flip_count, adjusted_count = [], 0, 0
    cache = OrderedDict()
    def source_table(start, end):
        pieces = []
        for entry in file_ranges:
            if entry["end"] <= start or entry["start"] >= end:
                continue
            path = entry["path"]
            if path not in cache:
                cache[path] = pq.read_table(source / path)
            cache.move_to_end(path)
            pieces.append(cache[path].slice(max(start, entry["start"]) - entry["start"], min(end, entry["end"]) - max(start, entry["start"])))
            while len(cache) > 2:
                cache.popitem(last=False)
        return pa.concat_tables(pieces)
    data_dir = output / "data/chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    for number, segment in enumerate(sorted(segments, key=lambda s: s["source_start"]), 1):
        start, end, ep = segment["source_start"], segment["source_end"], segment["episode_index"]
        n = end - start
        table = source_table(start, end)
        actions, flips, adjusted = continuous_quaternions(values["action"][start:end])
        flip_count += flips
        adjusted_count += adjusted
        replacements = {"action": actions, "episode_index": np.full(n, ep), "frame_index": np.arange(n),
                        "timestamp": (np.arange(n) / info["fps"]).astype(np.float32),
                        "index": np.arange(segment["dataset_from_index"], segment["dataset_to_index"])}
        for name, replacement in replacements.items():
            table = replace_column(table, name, replacement)
        target_slice = slice(segment["dataset_from_index"], segment["dataset_to_index"])
        stats = {}
        for name, width in NUMERIC.items():
            array = matrix(table[name], width)
            final_values[name][target_slice] = array
            stats[name] = numeric_stats(array)
        image_stats, image_histograms, sampled_frames = image_statistics(table)
        stats.update(image_stats)
        split = segment["split"]
        image_frames[split] += sampled_frames
        for name, histogram in image_histograms.items():
            histograms[split][name] += histogram
        destination = data_dir / f"file-{ep:03d}.parquet"
        temporary = destination.with_suffix(".parquet.partial")
        pq.write_table(table, temporary, compression="snappy")
        temporary.replace(destination)
        row = {"episode_index": ep, "tasks": ["Insert the cable connector into the correct target port"],
               "length": n, "data/chunk_index": 0, "data/file_index": ep,
               "dataset_from_index": segment["dataset_from_index"], "dataset_to_index": segment["dataset_to_index"],
               "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
        row.update({f"stats/{feature}/{key}": value for feature, fields in stats.items() for key, value in fields.items()})
        metadata.append(row)
        if number % 10 == 0 or number == len(segments):
            print(f"Cleaned episodes: {number}/{len(segments)}", flush=True)
    meta_dir = output / "meta/episodes/chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata.sort(key=lambda row: row["episode_index"])
    pq.write_table(pa.Table.from_pylist(metadata, schema=episodes.schema), meta_dir / "file-000.parquet")
    pq.write_table(pq.read_table(source / "meta/tasks.parquet"), output / "meta/tasks.parquet")
    aggregate = {name: numeric_stats(array) for name, array in final_values.items()}
    train_indices = np.concatenate([np.arange(s["dataset_from_index"], s["dataset_to_index"]) for s in segments if s["split"] == "train"])
    train_stats = {name: numeric_stats(array[train_indices]) for name, array in final_values.items()}
    for name in CAMERAS:
        aggregate[name] = histogram_stats(sum(histograms[split][name] for split in splits), sum(image_frames.values()))
        train_stats[name] = histogram_stats(histograms["train"][name], image_frames["train"])
    info.update(total_frames=offset, total_episodes=len(segments), splits={
        split: f"{part['episode_ids'][0]}:{part['episode_ids'][-1] + 1}" for split, part in splits.items()})
    write_json(output / "meta/info.json", info)
    write_json(preparation / "full_dataset_stats.json", aggregate)
    # The loader exposes meta/stats.json directly to policy normalization.
    # Keep held-out data out of that file, including when a caller forgets an override.
    write_json(output / "meta/stats.json", train_stats)
    write_json(preparation / "train_stats.json", train_stats)
    preview_dir = preparation / "previews"
    preview_dir.mkdir(exist_ok=True)
    for split, part in splits.items():
        ep = part["episode_ids"][0]
        table = pq.read_table(data_dir / f"file-{ep:03d}.parquet")
        for stage, index in [("start", 0), ("middle", len(table) // 2), ("end", len(table) - 1)]:
            for name in CAMERAS:
                payload = table[name][index].as_py()["bytes"]
                (preview_dir / f"{split}_{stage}_{name.split('.')[-1]}.png").write_bytes(payload)
    report.update(cleaned_frames=offset, cleaned_episodes=len(segments),
        removed_frames=len(valid) - offset,
        short_segment_frames_removed=short_frames, quaternion_antipodal_jumps_corrected=flip_count,
        quaternion_sign_adjusted_frames=adjusted_count, seed=args.seed,
        split_counts={split: {"episodes": len(part["episode_ids"]), "source_episodes": len(part["source_episode_ids"]), "frames": part["frame_count"]} for split, part in splits.items()},
        image_statistics_method="32 uniformly spaced frames per cleaned episode; every fourth pixel; pooled 256-bin RGB histograms",
        state_semantics_warning="Source has no joint names. State[1] resembles a gripper position; do not assume six arm joints without verifying the recorder mapping.",
        normalization_scope="train",
        normalization_warning="meta/stats.json and _preparation/train_stats.json contain training-only statistics; full-dataset statistics are separate.",
        source_repo=manifest["repo_id"], source_revision=manifest["revision"])
    write_json(preparation / "report.json", report)
    write_json(preparation / "complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "source_revision": manifest["revision"]})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
