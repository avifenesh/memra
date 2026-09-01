#!/usr/bin/env python3
"""Real rank-16 PyTorch training consumer for the memra darklane runner.

The frozen BF16 bank models the resident base weights of a LoRA finetune. A small stack of
frozen dense projections plus trainable low-rank adapters executes real forward, backward,
and AdamW optimizer steps. The bank size is explicit so the runner's launch-time VRAM budget
can be tested without downloading another model checkpoint onto the serving box.

SIGUSR1 writes an atomic adapter+optimizer+step checkpoint and exits 75, matching the v1
darklane checkpoint protocol. SIGTERM exits cleanly. Every allocator observation is JSONL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any


START = time.monotonic()
CHECKPOINT_REQUESTED_NS: int | None = None
TERM_REQUESTED = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {
            "ts": utc_now(),
            "mono_s": round(time.monotonic() - START, 6),
            "event": event,
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            **fields,
        }
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
        return row


def nvidia_process_mib(pid: int) -> int | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def on_usr1(_signum: int, _frame: Any) -> None:
    global CHECKPOINT_REQUESTED_NS
    if CHECKPOINT_REQUESTED_NS is None:
        CHECKPOINT_REQUESTED_NS = time.monotonic_ns()


def on_term(_signum: int, _frame: Any) -> None:
    global TERM_REQUESTED
    TERM_REQUESTED = True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--marker", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--reserve-mb", type=int, default=16 * 1024)
    ap.add_argument("--reserve-chunk-mb", type=int, default=512)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--scratch-mb", type=int, default=192)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--enforce-memory-fraction", action="store_true")
    return ap.parse_args()


def tree_to_cpu(obj: Any, torch: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: tree_to_cpu(value, torch) for key, value in obj.items()}
    if isinstance(obj, list):
        return [tree_to_cpu(value, torch) for value in obj]
    if isinstance(obj, tuple):
        return tuple(tree_to_cpu(value, torch) for value in obj)
    return obj


def move_optimizer_state(optimizer: Any, device: Any, torch: Any) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(path: Path, step: int, model: Any, optimizer: Any, torch: Any,
                    seed: int) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    adapters = {name: value.detach().cpu() for name, value in model.named_parameters()}
    payload = {
        "format": "darktrain2-lora-seam-v1",
        "step": step,
        "seed": seed,
        "adapters": adapters,
        "optimizer": tree_to_cpu(optimizer.state_dict(), torch),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return file_sha256(path), path.stat().st_size


def allocator_fields(torch: Any, device: Any, budget_mib: int) -> dict[str, Any]:
    mib = 2**20
    stats = torch.cuda.memory_stats(device)
    allocated = torch.cuda.memory_allocated(device) / mib
    reserved = torch.cuda.memory_reserved(device) / mib
    process_mib = nvidia_process_mib(os.getpid())
    observed = max(reserved, float(process_mib or 0))
    return {
        "allocated_mib": round(allocated, 3),
        "reserved_mib": round(reserved, 3),
        "max_allocated_mib": round(torch.cuda.max_memory_allocated(device) / mib, 3),
        "max_reserved_mib": round(torch.cuda.max_memory_reserved(device) / mib, 3),
        "inactive_split_mib": round(
            stats.get("inactive_split_bytes.all.current", 0) / mib, 3
        ),
        "nvidia_process_mib": process_mib,
        "budget_mib": budget_mib,
        "budget_ok": budget_mib <= 0 or observed <= budget_mib,
    }


def main() -> int:
    args = parse_args()
    events = EventLog(args.events)
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(f"pid={os.getpid()} ts={utc_now()}\n", encoding="utf-8")
    signal.signal(signal.SIGUSR1, on_usr1)
    signal.signal(signal.SIGTERM, on_term)

    budget_mib = int(os.environ.get("MEMRA_BG_VRAM_MB", "0"))
    events.emit(
        "process_start",
        argv=sys.argv,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        pytorch_alloc_conf=os.environ.get("PYTORCH_ALLOC_CONF"),
        pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        budget_mib=budget_mib,
        nvidia_process_mib=nvidia_process_mib(os.getpid()),
    )

    import_started = time.monotonic()
    import torch
    import torch.nn.functional as functional

    events.emit(
        "torch_imported",
        import_s=round(time.monotonic() - import_started, 6),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_initialized=torch.cuda.is_initialized(),
        nvidia_process_mib=nvidia_process_mib(os.getpid()),
    )
    available = torch.cuda.is_available()
    events.emit(
        "cuda_probed",
        available=available,
        cuda_initialized=torch.cuda.is_initialized(),
        nvidia_process_mib=nvidia_process_mib(os.getpid()),
    )
    if not available:
        events.emit("fatal", reason="CUDA unavailable")
        return 71

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    init_tensor = torch.empty(1, device=device)
    init_tensor.zero_()
    torch.cuda.synchronize(device)
    events.emit(
        "cuda_initialized",
        allocator_backend=torch.cuda.memory.get_allocator_backend(),
        device_name=torch.cuda.get_device_name(device),
        total_mib=round(torch.cuda.get_device_properties(device).total_memory / 2**20, 3),
        **allocator_fields(torch, device, budget_mib),
    )

    if args.enforce_memory_fraction and budget_mib > 0:
        total_mib = torch.cuda.get_device_properties(device).total_memory / 2**20
        fraction = min(1.0, budget_mib / total_mib)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        events.emit("memory_fraction_set", fraction=round(fraction, 8))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    class LoRALinear(torch.nn.Module):
        def __init__(self, dim: int, rank: int):
            super().__init__()
            self.weight = torch.randn(dim, dim, device=device, dtype=torch.bfloat16)
            self.weight.mul_(dim ** -0.5)
            self.weight.requires_grad_(False)
            self.lora_a = torch.nn.Parameter(
                torch.randn(rank, dim, device=device, dtype=torch.bfloat16) * 0.01
            )
            self.lora_b = torch.nn.Parameter(
                torch.zeros(dim, rank, device=device, dtype=torch.bfloat16)
            )
            self.scale = 1.0

        def forward(self, x: Any) -> Any:
            base = functional.linear(x, self.weight)
            update = functional.linear(functional.linear(x, self.lora_a), self.lora_b)
            return functional.gelu(base + update * self.scale)

    class LoRAStack(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                LoRALinear(args.dim, args.rank) for _ in range(args.layers)
            )

        def forward(self, x: Any) -> Any:
            for layer in self.layers:
                x = layer(x)
            return x

    model = LoRAStack()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    x = torch.randn(args.tokens, args.dim, device=device, dtype=torch.bfloat16)
    target = torch.randn_like(x)

    bank: list[Any] = []
    remaining = args.reserve_mb
    while remaining > 0:
        chunk_mib = min(remaining, args.reserve_chunk_mb)
        tensor = torch.empty(chunk_mib * 2**19, device=device, dtype=torch.bfloat16)
        tensor.zero_()
        bank.append(tensor)
        remaining -= chunk_mib
    torch.cuda.synchronize(device)
    setup_fields = allocator_fields(torch, device, budget_mib)
    events.emit(
        "setup_complete",
        reserve_requested_mib=args.reserve_mb,
        reserve_chunks=len(bank),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        frozen_parameters=sum(layer.weight.numel() for layer in model.layers),
        **setup_fields,
    )
    if not setup_fields["budget_ok"]:
        events.emit("budget_violation", phase="setup", **setup_fields)
        return 72

    step = 0
    if args.checkpoint.exists():
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != "darktrain2-lora-seam-v1":
            raise RuntimeError(f"bad checkpoint format: {payload.get('format')!r}")
        named = dict(model.named_parameters())
        for name, value in payload["adapters"].items():
            named[name].data.copy_(value.to(device))
        optimizer.load_state_dict(payload["optimizer"])
        move_optimizer_state(optimizer, device, torch)
        step = int(payload["step"])
        torch.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state(payload["cuda_rng"], device)
        events.emit(
            "checkpoint_loaded",
            checkpoint=str(args.checkpoint),
            checkpoint_sha256=file_sha256(args.checkpoint),
            resumed_step=step,
            **allocator_fields(torch, device, budget_mib),
        )

    recent_step_ms: list[float] = []
    loss_value = float("nan")
    while step < args.max_steps:
        if TERM_REQUESTED:
            events.emit("term_exit", step=step)
            return 0
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = (out.float() - target.float()).square().mean()
        loss.backward()
        optimizer.step()
        scratch_mib = args.scratch_mb + (step % 4) * 32
        scratch = torch.empty(scratch_mib * 2**19, device=device, dtype=torch.bfloat16)
        scratch[0] = loss.detach().to(torch.bfloat16)
        del scratch
        torch.cuda.synchronize(device)
        step += 1
        loss_value = float(loss.detach().cpu())
        recent_step_ms.append((time.monotonic() - t0) * 1000)
        recent_step_ms = recent_step_ms[-max(args.log_every, 20):]

        if CHECKPOINT_REQUESTED_NS is not None:
            save_started = time.monotonic()
            ck_sha, ck_bytes = save_checkpoint(
                args.checkpoint, step, model, optimizer, torch, args.seed
            )
            events.emit(
                "checkpoint_saved",
                step=step,
                loss=loss_value,
                signal_to_checkpoint_ms=round(
                    (time.monotonic_ns() - CHECKPOINT_REQUESTED_NS) / 1e6, 3
                ),
                save_ms=round((time.monotonic() - save_started) * 1000, 3),
                checkpoint=str(args.checkpoint),
                checkpoint_sha256=ck_sha,
                checkpoint_bytes=ck_bytes,
                **allocator_fields(torch, device, budget_mib),
            )
            return 75

        if step == 1 or step % args.log_every == 0:
            fields = allocator_fields(torch, device, budget_mib)
            row = events.emit(
                "optimizer_step",
                step=step,
                loss=round(loss_value, 8),
                step_ms=round(recent_step_ms[-1], 3),
                step_p50_ms=round(statistics.median(recent_step_ms), 3),
                **fields,
            )
            atomic_json(args.state, row)
            if not fields["budget_ok"]:
                events.emit("budget_violation", phase="optimizer_step", step=step, **fields)
                return 72

    ck_sha, ck_bytes = save_checkpoint(args.checkpoint, step, model, optimizer, torch, args.seed)
    events.emit(
        "complete",
        step=step,
        loss=loss_value,
        checkpoint_sha256=ck_sha,
        checkpoint_bytes=ck_bytes,
        **allocator_fields(torch, device, budget_mib),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
