#!/usr/bin/env python3
"""Drive a cold reference or a deliberately lapped plain-affinity resume."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request


CHECKPOINT_POS = 1024
LONG_TOKENS = 9216
REWRITE_TOKENS = 2048
MAX_CTX = 16384
MAX_TOKENS = 16


def metrics(base: str) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as response:
        return json.load(response)


def complete(base: str, prompt_ids: list[int], session_id: str, label: str, out: Path) -> dict:
    body = {
        "model": "step",
        "prompt_ids": prompt_ids,
        "max_ctx": MAX_CTX,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 3407,
        "stream": False,
        "cache_salt": "ringval-lap",
        "session_id": session_id,
    }
    started = time.monotonic()
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        raw = response.read()
        status = response.status
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}-response.json").write_bytes(raw)
    payload = json.loads(raw)
    choice = payload["choices"][0]
    text = choice["text"].encode()
    (out / f"{label}-text.bin").write_bytes(text)
    row = {
        "label": label,
        "http_status": status,
        "prompt_tokens_requested": len(prompt_ids),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
        "prompt_tokens_reported": payload.get("usage", {}).get("prompt_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "text_bytes": len(text),
        "text_sha256": hashlib.sha256(text).hexdigest(),
        "text_utf8_b64": base64.b64encode(text).decode(),
        "wall_s": round(time.monotonic() - started, 6),
    }
    (out / f"{label}-row.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert status == 200
    assert row["prompt_tokens_reported"] == len(prompt_ids)
    assert isinstance(row["completion_tokens"], int) and row["completion_tokens"] > 0
    assert row["finish_reason"] in ("stop", "length")
    assert text
    return row


def prompts(control_id: int) -> tuple[list[int], list[int]]:
    # The only control token is at index 1024, so plain_checkpoint_boundary() snapshots at
    # exactly 1024. The remaining 8,191 rows grow the ring far beyond that checkpoint.
    long_prompt = [55] * CHECKPOINT_POS + [control_id]
    long_prompt.extend([56] * (LONG_TOKENS - len(long_prompt)))
    # Same exact bytes through the checkpoint, then a rewritten suffix. This is not an exact
    # extension of the parked fed stream, so explicit affinity nominates the rewind candidate.
    rewritten = [55] * CHECKPOINT_POS + [57] * (REWRITE_TOKENS - CHECKPOINT_POS)
    assert len(long_prompt) == LONG_TOKENS
    assert len(rewritten) == REWRITE_TOKENS
    assert long_prompt[:CHECKPOINT_POS] == rewritten[:CHECKPOINT_POS]
    assert long_prompt[CHECKPOINT_POS:] != rewritten[CHECKPOINT_POS:]
    return long_prompt, rewritten


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("cold", "lap"))
    parser.add_argument("--base", required=True)
    parser.add_argument("--control-id", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    long_prompt, rewritten = prompts(args.control_id)
    args.out.mkdir(parents=True, exist_ok=True)
    run = {
        "phase": args.phase,
        "n": 1,
        "control_id": args.control_id,
        "checkpoint_pos": CHECKPOINT_POS,
        "ring_rows": 4639,
        "long_prompt_tokens": LONG_TOKENS,
        "rewritten_prompt_tokens": REWRITE_TOKENS,
        "max_ctx": MAX_CTX,
        "max_tokens": MAX_TOKENS,
    }
    (args.out / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.phase == "cold":
        before = metrics(args.base)
        row = complete(args.base, rewritten, "ringval-cold", "cold", args.out)
        after = metrics(args.base)
        summary = {"run": run, "row": row, "metrics_before": before, "metrics_after": after}
    else:
        before = metrics(args.base)
        seed = complete(args.base, long_prompt, "ringval-lap", "seed-long", args.out)
        after_seed = metrics(args.base)
        resumed = complete(args.base, rewritten, "ringval-lap", "resume-declined", args.out)
        after_resume = metrics(args.base)
        assert after_resume["cached_tokens_in"] - after_seed["cached_tokens_in"] == 0
        assert after_resume["computed_tokens_in"] - after_seed["computed_tokens_in"] == REWRITE_TOKENS
        assert after_resume["plain_affinity_rewinds"] - after_seed["plain_affinity_rewinds"] == 0
        assert after_resume["continuation_pool_hits"] - after_seed["continuation_pool_hits"] == 0
        summary = {
            "run": run,
            "seed_row": seed,
            "resume_row": resumed,
            "metrics_before": before,
            "metrics_after_seed": after_seed,
            "metrics_after_resume": after_resume,
            "resume_computed_tokens_delta": (
                after_resume["computed_tokens_in"] - after_seed["computed_tokens_in"]
            ),
            "resume_cached_tokens_delta": (
                after_resume["cached_tokens_in"] - after_seed["cached_tokens_in"]
            ),
        }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
