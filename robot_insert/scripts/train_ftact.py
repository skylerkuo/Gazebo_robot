#!/usr/bin/env python3
"""Train force-conditioned ACT on the prepared AIC dataset; no force-free variant."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from ftact import (AICDataset, EpochBatchSampler, ForceTorqueACT, INSERT_ROOT, TrainConfig,
                   json_write, move_batch, read_data_contract)
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=INSERT_ROOT / "configs/ftact_aic.json")
    parser.add_argument("--resume", type=Path, help="Resume an optimizer checkpoint, normally checkpoints/last.pt")
    parser.add_argument("--reset-best", action="store_true",
                        help="Use the first validation of this run as a fresh BEST baseline")
    parser.add_argument("--smoke-test", action="store_true", help="3 updates on two train/validation episodes, same full model")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"))
    for key in ("steps", "batch_size", "gradient_accumulation_steps", "num_workers", "eval_every",
                "eval_batches", "eval_samples", "save_every", "log_every"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    args = parser.parse_args()
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume.get("format") != "aic-ftact-v1" or resume.get("optimizer") is None:
            parser.error("--resume requires this trainer's optimizer checkpoint (last.pt)")
        config = TrainConfig(**resume["config"])
        # Legacy checkpoints specified batches; preserve the same validation
        # frames when increasing the physical training batch size.
        if config.eval_samples is None:
            config.eval_samples = config.eval_batches * config.batch_size
    else:
        config = TrainConfig(**json.loads(args.config.read_text()))
    if args.smoke_test:
        config.steps, config.batch_size, config.gradient_accumulation_steps = 3, 2, 1
        config.num_workers, config.log_every = 0, 1
        config.eval_every, config.eval_batches, config.save_every = 3, 2, 3
        config.eval_samples = 4
        if not args.resume:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            config.output_dir = str(INSERT_ROOT / "results" / f"ftact_smoke_{stamp}")
    for key in asdict(config):
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    if args.eval_batches is not None and args.eval_samples is None:
        config.eval_samples = None
    config.dataset_root = str(Path(config.dataset_root).resolve())
    config.output_dir = str(Path(config.output_dir).resolve())
    config.validate()
    if resume and config.steps <= resume["step"]:
        parser.error(f"--steps is the total target and must exceed saved step {resume['step']}")
    return args, config, resume


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def autocast_context(config):
    if config.amp == "none":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if config.amp == "bf16" else torch.float16)


class TrainingBatches:
    def __init__(self, loader, sampler, cursor=None, saved_batch_size=None):
        self.loader, self.sampler = loader, sampler
        self.epoch = cursor["epoch"] if cursor else 0
        self.batches = cursor["batches_consumed"] if cursor else 0
        legacy_batch = saved_batch_size if saved_batch_size is not None else sampler.batch_size
        self.samples = cursor.get("samples_consumed", min(self.batches * legacy_batch, sampler.size)) if cursor else 0
        if not 0 <= self.samples <= sampler.size or self.epoch < 0 or self.batches < 0:
            raise ValueError("Invalid saved training data cursor")
        self.iterator = None

    def __next__(self):
        if self.iterator is None:
            self.sampler.epoch, self.sampler.start_sample = self.epoch, self.samples
            self.iterator = iter(self.loader)
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.batches = 0
            self.samples = 0
            self.sampler.epoch, self.sampler.start_sample = self.epoch, 0
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        self.batches += 1
        self.samples += len(batch["action"])
        return batch

    def state_dict(self):
        return {"epoch": self.epoch, "batches_consumed": self.batches,
                "samples_consumed": self.samples}


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scaler, config, stats, provenance, episodes,
                    step, batches, best_l1, metrics, smoke_test, *, resumable):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "aic-ftact-v1", "config": asdict(config), "stats": stats,
        "provenance": provenance, "selected_episodes": episodes, "step": step,
        "cursor": batches.state_dict(), "best_validation_l1": best_l1, "metrics": metrics,
        "smoke_test": smoke_test, "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if resumable else None,
        "scaler": scaler.state_dict() if resumable else None,
        "rng": rng_state() if resumable else None}
    temporary = path.with_suffix(".pt.partial")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, loader, config):
    model.eval()
    l1_sum = position_sum = angle_sum = 0.
    valid_steps = samples = 0
    for batch in loader:
        batch = move_batch(batch, config.device)
        with autocast_context(config):
            prediction = model.normalized_action_chunk(batch)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite validation prediction")
        valid = ~batch["action_is_pad"]
        target = model.normalizer.actions(batch["action"])
        l1_sum += ((prediction - target).abs() * valid.unsqueeze(-1)).sum().item()
        physical = model.normalizer.unnormalize_actions(prediction)
        position_sum += ((physical[..., :3] - batch["action"][..., :3]).norm(dim=-1) * valid).sum().item()
        quaternion = physical[..., 3:]
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dot = (quaternion * batch["action"][..., 3:]).sum(-1).abs().clamp(0, 1)
        angle_sum += (torch.acos(dot) * 2 * valid).sum().item()
        valid_steps += valid.sum().item()
        samples += len(prediction)
    if not valid_steps:
        raise ValueError("No valid validation actions")
    model.train()
    return {"l1": l1_sum / (valid_steps * 7),
            "position_l2_m": position_sum / valid_steps,
            "rotation_deg": math.degrees(angle_sum / valid_steps),
            "samples": samples, "valid_action_steps": valid_steps, "latent": "zero"}


def append_metrics(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False), flush=True)


def resume_best_state(checkpoint, sample_count, *, reset=False):
    """Start a fresh BEST comparison when requested or when validation size changes."""
    best_l1, metrics = checkpoint["best_validation_l1"], checkpoint["metrics"]
    if reset or (metrics is not None and metrics["samples"] != sample_count):
        event = {"type": "best_reset", "step": checkpoint["step"],
            "reason": "requested" if reset else "validation_sample_count_changed",
            "validation_samples": sample_count,
            "previous_best_l1": best_l1 if math.isfinite(best_l1) else None,
            "first_validation_becomes_best": True}
        return float("inf"), None, event
    return best_l1, metrics, None


def main():
    args, config, resume = parse_args()
    torch.set_num_threads(config.num_threads)
    seed_all(config.seed)
    output = Path(config.output_dir)
    if not args.resume and (output / "run_config.json").exists():
        raise FileExistsError("Output already contains a run; use --resume or a new --output-dir")
    if output == Path(config.dataset_root) or Path(config.dataset_root) in output.parents:
        raise ValueError("Store training outputs outside the dataset directory")
    stats, splits, provenance = read_data_contract(config.dataset_root)
    smoke = args.smoke_test or bool(resume and resume["smoke_test"])
    episodes = {key: splits[key]["episode_ids"][:2] if smoke else splits[key]["episode_ids"]
                for key in ("train", "validation")}
    if resume:
        if provenance != resume["provenance"] or episodes != resume["selected_episodes"]:
            raise ValueError("Resume dataset metadata, statistics, or selected episodes differ")
    print("Loading train/validation episodes; first load builds Arrow caches on D:.", flush=True)
    train_data = AICDataset(config.dataset_root, episodes["train"], config.chunk_size)
    print(f"Loaded train: {len(episodes['train'])} episodes, {len(train_data):,} frames.", flush=True)
    validation_data = AICDataset(config.dataset_root, episodes["validation"], config.chunk_size)
    print(f"Loaded validation: {len(episodes['validation'])} episodes, {len(validation_data):,} frames.", flush=True)
    eval_limit = config.eval_samples if config.eval_samples is not None else config.eval_batches * config.batch_size
    eval_count = len(validation_data) if eval_limit == 0 else min(len(validation_data), eval_limit)
    indices = torch.randperm(len(validation_data), generator=torch.Generator().manual_seed(config.seed + 10000))[:eval_count].tolist()
    workers = {"num_workers": config.num_workers, "pin_memory": config.device.startswith("cuda"),
               "worker_init_fn": seed_worker, "persistent_workers": config.num_workers > 0}
    if config.num_workers:
        # Spawn avoids forking a process after CUDA initialization.
        workers.update(multiprocessing_context="spawn", prefetch_factor=2)
    sampler = EpochBatchSampler(len(train_data), config.batch_size, config.seed)
    train_loader = DataLoader(train_data, batch_sampler=sampler,
        generator=torch.Generator().manual_seed(config.seed), **workers)
    validation_loader = DataLoader(Subset(validation_data, indices), batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(config.seed + 1), **workers)
    batches = TrainingBatches(train_loader, sampler, resume["cursor"] if resume else None,
                             resume["config"]["batch_size"] if resume else None)
    model = ForceTorqueACT(config, stats, initialize_backbone=not bool(resume)).to(config.device).train()
    optimizer = torch.optim.AdamW(model.policy.get_optim_params(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp == "fp16")
    step, best_l1 = 0, float("inf")
    last_validation = None
    baseline_event = None
    if resume:
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        step = resume["step"]
        best_l1, last_validation, baseline_event = resume_best_state(
            resume, eval_count, reset=args.reset_best)
        restore_rng(resume["rng"])
        del resume
    versions = {key: importlib.metadata.version(key) for key in ("lerobot", "datasets", "numpy", "torchvision")}
    versions["torch"] = str(torch.__version__)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": asdict(config), "provenance": provenance, "versions": versions,
        "selected_episodes": episodes, "train_frames": len(train_data),
        "validation_frames": len(validation_data), "validation_sampled_frames": eval_count,
        "smoke_test": smoke, "parameters": sum(p.numel() for p in model.parameters()),
        "input_state_order": [f"source_state[{i}]" for i in range(6)] + ["fx", "fy", "fz", "tx", "ty", "tz"],
        "normalization_scope": "full training split only", "resumed_from": str(args.resume) if args.resume else None}
    json_write(output / "run_config.json", manifest)
    json_write(output / "normalization.json", stats)
    print(json.dumps({"output": str(output), "parameters": manifest["parameters"],
        "train_frames": len(train_data), "validation_frames": len(validation_data),
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "starting_step": step, "target_step": config.steps, "smoke_test": smoke}), flush=True)
    metrics_path = output / "metrics.jsonl"
    if args.resume:
        append_metrics(metrics_path, {"type": "resume", "step": step, "checkpoint": str(args.resume),
            "batch_size": config.batch_size, "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
            "validation_sampled_frames": eval_count, "cursor": batches.state_dict()})
    if baseline_event is not None:
        append_metrics(metrics_path, baseline_event)
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    while step < config.steps:
        optimizer.zero_grad(set_to_none=True)
        values = {"loss": 0., "l1_loss": 0., "kld_loss": 0.}
        for _ in range(config.gradient_accumulation_steps):
            batch = move_batch(next(batches), config.device)
            with autocast_context(config):
                loss, details = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss after step {step}")
            scaler.scale(loss / config.gradient_accumulation_steps).backward()
            values["loss"] += loss.item() / config.gradient_accumulation_steps
            for key in ("l1_loss", "kld_loss"):
                values[key] += details[key] / config.gradient_accumulation_steps
        scaler.unscale_(optimizer)
        force_gradient = model.policy.model.encoder_robot_state_input_proj.weight.grad[:, 6:].float().norm().item()
        if smoke and (not math.isfinite(force_gradient) or force_gradient <= 0):
            raise AssertionError("Force channels did not receive a finite nonzero training gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm,
                                                   error_if_nonfinite=True).item()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % config.log_every == 0 or step == 1 or step == config.steps:
            record = {"type": "train", "step": step, **values, "gradient_norm": grad_norm,
                "force_projection_gradient_norm": force_gradient,
                "elapsed_seconds": time.perf_counter() - start, **batches.state_dict()}
            if config.device.startswith("cuda"):
                record["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
                record["peak_cuda_reserved_memory_gib"] = torch.cuda.max_memory_reserved() / 2 ** 30
            append_metrics(metrics_path, record)
        if step % config.eval_every == 0 or step == config.steps:
            last_validation = evaluate(model, validation_loader, config)
            append_metrics(metrics_path, {"type": "validation", "step": step, **last_validation})
            if last_validation["l1"] < best_l1:
                best_l1 = last_validation["l1"]
                save_checkpoint(output / "checkpoints/best.pt", model, optimizer, scaler, config,
                    stats, provenance, episodes, step, batches, best_l1, last_validation, smoke, resumable=False)
                append_metrics(metrics_path, {"type": "best_checkpoint", "step": step,
                    "validation_l1": best_l1, "samples": last_validation["samples"],
                    "checkpoint": str(output / "checkpoints/best.pt")})
        if step % config.save_every == 0 or step == config.steps:
            save_checkpoint(output / "checkpoints/last.pt", model, optimizer, scaler, config,
                stats, provenance, episodes, step, batches, best_l1, last_validation, smoke, resumable=True)
    result = {"completed_steps": step, "smoke_test": smoke, "best_validation_l1": best_l1,
        "last_validation": last_validation, "elapsed_seconds": time.perf_counter() - start,
        "last_checkpoint": str(output / "checkpoints/last.pt"),
        "best_checkpoint": str(output / "checkpoints/best.pt"),
        "closed_loop_evaluated": False}
    if smoke:
        # Round-trip the real checkpoint through the public loader and compare deterministic inference.
        from ftact import load_checkpoint
        model.eval()
        reference_batch = move_batch(next(iter(validation_loader)), config.device)
        with torch.inference_mode():
            expected = model.predict_action_chunk(reference_batch).cpu()
        restored, _ = load_checkpoint(output / "checkpoints/last.pt", config.device)
        with torch.inference_mode():
            actual = restored.predict_action_chunk(reference_batch).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        result.update(checkpoint_roundtrip_passed=True, prediction_shape=list(actual.shape),
                      force_input_gradient_verified=True)
    json_write(output / "training_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
