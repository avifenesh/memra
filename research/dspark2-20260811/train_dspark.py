#!/usr/bin/env python3
"""Detached single-L40S trainer for the frozen 2K DSpark trajectory probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dspark_data import DSparkCorpus, collate_records, load_d2t
from dspark_eval import evaluate_model
from dspark_loss import build_target_to_draft, compute_dspark_loss
from dspark_model import DSparkConfig, DSparkModel, expected_trainable_parameters, load_shared_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("/home/ubuntu/dspark2/corpus/chunks"))
    parser.add_argument(
        "--shared", type=Path, default=Path("/home/ubuntu/dspark2/artifacts/pilot-02000/shared")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-end", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def verify_manifest(root: Path) -> str:
    manifest = root / "sha256.txt"
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"shared artifact hash mismatch: {path}")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_checkpoint(
    path: Path,
    model: DSparkModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    *,
    epoch: int,
    batch_in_epoch: int,
    step: int,
) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "format": "memra-dspark-pilot-checkpoint-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "step": step,
        },
        partial,
    )
    os.replace(partial, path)


def learning_rate_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DSpark pilot training requires the provisioned L40S")
    device = torch.device("cuda", 0)
    gpu_name = torch.cuda.get_device_name(device)
    if "L40S" not in gpu_name:
        raise RuntimeError(f"refusing non-L40S training device: {gpu_name}")
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "train-config.json"
    if config_path.exists() and not args.resume:
        raise FileExistsError(f"{config_path} exists; pass --resume deliberately")

    shared_manifest_hash = verify_manifest(args.shared)
    model_config = DSparkConfig()
    train_data = DSparkCorpus(args.chunks, "train", pair_end=args.pair_end)
    heldout_data = DSparkCorpus(args.chunks, "heldout", pair_end=args.pair_end)
    if train_data.fingerprint() != heldout_data.fingerprint():
        raise ValueError("train/heldout readers disagree on corpus fingerprint")
    d2t = load_d2t(args.shared / "d2t.u32", model_config.draft_vocab)
    target_to_draft = build_target_to_draft(d2t, model_config.target_vocab).to(device)
    embedding, head = load_shared_artifact(args.shared, model_config)
    model = DSparkModel(model_config, embedding, head).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != expected_trainable_parameters(model_config):
        raise ValueError(f"trainable parameter mismatch: {trainable}")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.0,
        fused=True,
    )
    batches_per_epoch = math.ceil(len(train_data) / args.batch_size)
    total_steps = batches_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, round(total_steps * args.warmup_ratio))
    scheduler = LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, total_steps, warmup_steps),
    )
    heldout_loader = DataLoader(
        heldout_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_records,
    )
    receipt = {
        "format": "memra-dspark-pilot-train-v1",
        "code_commit": args.code_commit,
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": args.seed,
        "model": model_config.to_dict(),
        "trainable_parameters": trainable,
        "train_records": len(train_data),
        "heldout_records": len(heldout_data),
        "pair_range": [0, args.pair_end],
        "corpus_fingerprint": train_data.fingerprint(),
        "shared_manifest_sha256": shared_manifest_hash,
        "shared_path": str(args.shared),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "total_steps": total_steps,
        "objective": "0.1 CE + 0.9 TVD + 1.0 confidence BCE; exp(-position/5)",
        "teacher_temperature": 0.7,
        "acceptance_identity": "1 - TVD = 1 - 0.5*L1",
    }
    if not config_path.exists():
        atomic_json(config_path, receipt)

    start_epoch = 0
    start_batch = 0
    step = 0
    checkpoint_path = args.output / "checkpoint-latest.pt"
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["batch_in_epoch"])
        step = int(checkpoint["step"])

    trajectory_path = args.output / "trajectory.jsonl"
    metrics_path = args.output / "train-metrics.jsonl"
    started = time.monotonic()
    last_checkpoint = started
    if step == 0:
        initial = evaluate_model(model, heldout_loader, target_to_draft, device=device, seed=args.seed)
        append_jsonl(
            trajectory_path,
            {"step": 0, "epoch": 0.0, "elapsed_seconds": 0.0, "eval": initial},
        )

    stop = False
    for epoch in range(start_epoch, args.epochs):
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            collate_fn=collate_records,
        )
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    batch["hidden"], batch["tokens"], batch["anchor_position"], target_to_draft
                )
            loss = compute_dspark_loss(
                output.logits,
                output.confidence_logits,
                batch["tokens"][:, 1:],
                batch["top_ids"],
                batch["top_probs"],
                batch["tail_probs"],
                target_to_draft,
            )
            if not torch.isfinite(loss.total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.total.backward()
            grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            elapsed = time.monotonic() - started
            if step % args.log_every == 0 or step == 1:
                row = {
                    "step": step,
                    "epoch": epoch + (batch_index + 1) / batches_per_epoch,
                    "elapsed_seconds": elapsed,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "gpu_memory_allocated": torch.cuda.max_memory_allocated(device),
                    **loss.metrics(),
                }
                append_jsonl(metrics_path, row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if step % args.eval_every == 0:
                evaluation = evaluate_model(
                    model, heldout_loader, target_to_draft, device=device, seed=args.seed
                )
                row = {
                    "step": step,
                    "epoch": epoch + (batch_index + 1) / batches_per_epoch,
                    "elapsed_seconds": elapsed,
                    "eval": evaluation,
                }
                append_jsonl(trajectory_path, row)
                print(json.dumps({"trajectory": row}, sort_keys=True), flush=True)
            now = time.monotonic()
            if now - last_checkpoint >= args.checkpoint_seconds:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    batch_in_epoch=batch_index + 1,
                    step=step,
                )
                last_checkpoint = now
            if step >= total_steps:
                stop = True
                break
        start_batch = 0
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch=epoch + 1,
            batch_in_epoch=0,
            step=step,
        )
        if stop:
            break

    final_evaluation = evaluate_model(
        model, heldout_loader, target_to_draft, device=device, seed=args.seed
    )
    atomic_json(
        args.output / "final-eval.json",
        {
            "step": step,
            "elapsed_seconds": time.monotonic() - started,
            "eval": final_evaluation,
        },
    )
    final_path = args.output / "checkpoint-final.pt"
    os.replace(checkpoint_path, final_path)
    print(json.dumps({"final": final_evaluation, "step": step}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
