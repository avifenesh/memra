#!/usr/bin/env python3
"""Q35 cold-prefill regression gate: one frozen mixed90 c=4 sellgate cell.

A FAIL names its class (memra#777), and each class is a different finding:
  token_regression  a request did not return exactly 60 tokens with finish_reason length
  response_failure  a request failed transport (status, stream end, request id)
  usage_mismatch    usage disagrees with the closed form: a hit's cached_tokens is the entry the
                    prompt-end seed published on the GDN prime grid (memra#602, 4832 for the
                    4860-token prompt on grid 32), not the prompt length
  golden_mismatch   a byte-gated golden differs
  cell_accounting   a /metrics counter disagrees with the per-request usage
  seed_failure      the hot-set seeding failed or drifted
  shape             the cell did not carry 20 requests, 18 hits and 2 cold misses

--replay LOG re-judges a banked gate log offline through the same judging code (CPU only): the
requests, the seeds and the cell's /metrics deltas come from the log, the expectation from the
grid law. The live-only /metrics checks (entry retention after seeding) are not replayed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any


def load_sellgate(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("memra_sellgate_replay", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import sellgate harness from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def assert_identity(base: str, model: str, timeout: float) -> None:
    """Refuse to measure whatever happens to answer on this port.

    GATE-INTEGRITY-20260819 section 10. `--base` defaults to 127.0.0.1:8177, which is
    tools/serve-smoke.sh's port. This gate is a client, not a binder, so tools/port-guard.sh does
    not apply to it — but the failure mode it was written for is worse for a client: an occupied
    port makes a BINDER fail to bind and die loudly, while a client cheerfully measures the
    stranger and prints numbers. tools/accept-gate.sh:143 records the live incident: the rig's
    idle llama-server held the port and "had that foreign process instead answered 200 with a
    plausible body, the gate would have measured SOMEONE ELSE'S MODEL and pinned it."

    So the first request this gate makes is an identity probe, and a port that cannot name the
    model under test is a refusal — not a warning, and not a retry against a different port.
    """
    url = f"{base}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=min(timeout, 30.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"identity probe failed: {url} did not answer a model list ({error}). "
            "Boot the server under test, or pass --base."
        ) from error
    served = [
        entry.get("id")
        for entry in (payload.get("data") or [])
        if isinstance(entry, dict)
    ]
    if not served:
        raise ValueError(
            f"identity probe failed: {url} answered with no model ids ({payload!r}). "
            "An empty list is not agreement — something is listening that is not the server "
            "under test."
        )
    if model not in served:
        raise ValueError(
            f"identity probe FAILED: {base} serves {served} and this gate measures {model!r}. "
            "Refusing to run: every number below would come from a different program. Free the "
            "port, boot the right model, or pass --base/--model deliberately."
        )
    print(
        json.dumps(
            {"identity_probe": "ok", "base": base, "model": model, "served": served},
            sort_keys=True,
        ),
        flush=True,
    )


TOKEN_REGRESSION = "token_regression"
SEED_FAILURE = "seed_failure"
SHAPE = "shape"
# Order of the classes in the summary and in serve-smoke's FAIL lines.
CLASS_ORDER = (
    TOKEN_REGRESSION,
    "response_failure",
    "golden_mismatch",
    "usage_mismatch",
    "cell_accounting",
    SEED_FAILURE,
    SHAPE,
)
# serve-smoke section 12 prints one FAIL line per class, in CLASS_ORDER.
FAIL_LINES = {
    TOKEN_REGRESSION: "Q35 mixed c=4 exact-token regression",
    "response_failure": "Q35 mixed c=4 response failure (status, stream end or request id)",
    "golden_mismatch": "Q35 mixed c=4 golden mismatch (byte-gated text differs)",
    "usage_mismatch": (
        "Q35 mixed c=4 usage mismatch (prompt_tokens or cached_tokens against the "
        "grid-aligned entry; not a token regression)"
    ),
    "cell_accounting": "Q35 mixed c=4 /metrics accounting drift",
    SEED_FAILURE: "Q35 mixed c=4 hot-set seeding failed",
    SHAPE: "Q35 mixed c=4 cell shape moved (requests, hits, misses)",
}
EXPECTED_SHAPE = {
    "prompt_tokens": 4860,
    "completion_tokens": 60,
    "hit_requests_per_cycle": 9,
    "miss_requests_per_cycle": 1,
    "minimum_requests_per_cell": 20,
}


def load_frozen_workload(harness: ModuleType, repo: Path) -> dict[str, Any]:
    workload = harness.load_workload(repo / "research/sellgate-20260812/workload.lock.json")
    actual_shape = {key: workload.get(key) for key in EXPECTED_SHAPE}
    if actual_shape != EXPECTED_SHAPE:
        raise ValueError(
            f"frozen workload shape moved: expected {EXPECTED_SHAPE}, got {actual_shape}"
        )
    return workload


def gate_summary(
    harness: ModuleType,
    workload: dict[str, Any],
    grid: int,
    seed_failures: list[str],
    requests: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    hits = [row for row in requests if row.get("cache_role") == "hit"]
    misses = [row for row in requests if row.get("cache_role") == "miss"]
    completion_n = int(workload["completion_tokens"])
    short = [
        {
            "index": row.get("index"),
            "cache_role": row.get("cache_role"),
            "completion_tokens": row.get("completion_tokens"),
            "finish_reason": row.get("finish_reason"),
            "request_id": row.get("request_id"),
            "text_sha256": row.get("text_sha256"),
        }
        for row in requests
        if row.get("completion_tokens") != completion_n or row.get("finish_reason") != "length"
    ]
    shape_ok = len(requests) == 20 and len(hits) == 18 and len(misses) == 2
    cell_ok = len(cells) == 1 and bool(cells[0].get("clean"))
    classes: set[str] = set()
    if short:
        classes.add(TOKEN_REGRESSION)
    for cell in cells:
        classes.update(cell.get("failure_classes") or [])
    if seed_failures:
        classes.add(SEED_FAILURE)
        if any(harness.FAIL_USAGE in line for line in seed_failures):
            classes.add(harness.FAIL_USAGE)
    if not shape_ok or len(cells) != 1:
        classes.add(SHAPE)
    verdict = "PASS" if shape_ok and cell_ok and not seed_failures and not short else "FAIL"
    ordered = [name for name in CLASS_ORDER if name in classes] + sorted(
        classes - set(CLASS_ORDER)
    )
    if verdict == "FAIL" and not ordered:
        ordered = ["unclassified"]
    return {
        "kind": "q35_cold_mixed_gate",
        "schema": "memra.q35-cold-mixed-gate.v2",
        "concurrency": 4,
        "requests": len(requests),
        "hit_requests": len(hits),
        "cold_misses": len(misses),
        "expected_completion_tokens": completion_n,
        "gdn_grid": grid,
        "expected_hit_cached_tokens": harness.expected_hit_cached(
            int(workload["prompt_tokens"]), grid
        ),
        "short_or_non_length": short,
        "seed_failures": seed_failures,
        "cell_clean": cell_ok,
        "cell_integrity_failures": [
            line for cell in cells for line in (cell.get("integrity_failures") or [])
        ],
        "failure_classes": ordered,
        "fail_lines": [
            FAIL_LINES.get(name, f"Q35 mixed c=4 {name}") for name in ordered
        ],
        "verdict": verdict,
    }


def replay(
    harness: ModuleType, workload: dict[str, Any], grid: int, log: Path
) -> dict[str, Any]:
    """Re-judge a banked gate log: same judging code, expectation from the grid law."""
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rows.append(json.loads(line))
    seeds = [row for row in rows if row.get("kind") == "seed"]
    banked_requests = [row for row in rows if row.get("kind") == "request"]
    banked_cells = [row for row in rows if row.get("kind") == "cell"]
    banked_summary = next(
        (row for row in reversed(rows) if row.get("kind") == "q35_cold_mixed_gate"), None
    )
    if not seeds or not banked_requests or len(banked_cells) != 1:
        raise ValueError(
            f"{log}: not a complete gate log ({len(seeds)} seeds, "
            f"{len(banked_requests)} requests, {len(banked_cells)} cells)"
        )
    prompt_n = int(workload["prompt_tokens"])
    goldens: dict[tuple[str, int], str] = {}
    seed_failures: list[str] = []
    for seed in seeds:
        seed_failures.extend(
            harness.judge_seed(
                str(seed["target"]), int(seed["template"]), seed, prompt_n, grid, goldens
            )
        )
    requests = []
    for banked in banked_requests:
        row = dict(banked)
        expected = harness.expected_hit_cached(prompt_n, grid) if row["cache_role"] == "hit" else 0
        golden = (
            goldens.get((str(row["target"]), int(row["template"])))
            if row.get("template") is not None
            else None
        )
        row.update(
            harness.judge_request(row, expected, golden, int(row["concurrency"]), workload)
        )
        requests.append(row)
    cell = dict(banked_cells[0])
    cell["integrity_failures"] = harness.cell_integrity_failures(
        requests, cell["counter_deltas"]
    )
    cell["failure_classes"] = harness.failure_classes(cell["integrity_failures"])
    cell["clean"] = not cell["integrity_failures"]
    summary = gate_summary(harness, workload, grid, seed_failures, requests, [cell])
    summary["mode"] = "replay"
    summary["replayed_log"] = str(log)
    summary["banked_verdict"] = banked_summary.get("verdict") if banked_summary else None
    summary["banked_cell_integrity_failures"] = banked_cells[0].get("integrity_failures")
    summary["banked_expected_hit_cached_tokens"] = sorted(
        {row.get("expected_cached_tokens") for row in banked_requests if row["cache_role"] == "hit"}
    )
    summary["reported_hit_cached_tokens"] = sorted(
        {row.get("cached_tokens") for row in banked_requests if row["cache_role"] == "hit"}
    )
    return summary


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8177")
    parser.add_argument("--model", default="q35-coldfix")
    parser.add_argument("--namespace", default="serve-smoke-q35-coldfix")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--gdn-grid",
        type=int,
        default=None,
        help="the server's GDN prime grid (default: Engine::gdn_chunk_size() from this "
        "process's MEMRA_GDN_CHUNK, the environment serve-smoke boots the server with)",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="re-judge a banked gate log offline instead of measuring a server",
    )
    args = parser.parse_args()

    harness = load_sellgate(repo / "research/sellgate-20260812/sellgate_replay.py")
    grid = harness.gdn_grid(os.environ) if args.gdn_grid is None else args.gdn_grid
    if grid <= 0 or grid % harness.PRIME_MIN_T:
        raise ValueError(f"--gdn-grid {grid} is not a positive multiple of PRIME_MIN_T")

    if args.replay is not None:
        workload = load_frozen_workload(harness, repo)
        summary = replay(harness, workload, grid, args.replay)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if summary["verdict"] == "PASS" else 1

    # BEFORE anything is measured: prove the port belongs to the model under test.
    assert_identity(args.base.rstrip("/"), args.model, args.timeout)
    workload = load_frozen_workload(harness, repo)

    endpoint = harness.Endpoint(label="q35", base=args.base.rstrip("/"), model=args.model)
    goldens: dict[tuple[str, int], str] = {}
    seed_rows, seed_failures = harness.seed_hot_set(
        [endpoint], workload, args.namespace, args.timeout, goldens, grid
    )
    for row in seed_rows:
        print(json.dumps(public(row), sort_keys=True), flush=True)

    requests, samples, cells = harness.run_cell(
        [endpoint], workload, args.namespace, "mixed90", 1, 4, args.timeout, goldens, grid
    )
    for row in [*samples, *requests, *cells]:
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = gate_summary(harness, workload, grid, seed_failures, requests, cells)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
