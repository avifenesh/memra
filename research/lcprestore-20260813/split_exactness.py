#!/usr/bin/env python3
"""HIRADIX-EXACT-ISO split-boundary and sequential request-2 gate.

Two already-running servers must use the same binary/model/config. The control server has
MEMRA_PREFIX_PARTIAL_RESTORE=0; the candidate uses the default-on partial restore. Requests are
alternated between arms inside one GPU lock by the enclosing runner. Every scored request uses a
fresh namespace, so request 2 can only reuse request 1's longer entry at the requested LCP.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SELLGATE = Path(__file__).resolve().parents[1] / "sellgate-20260812"
sys.path.insert(0, str(SELLGATE))

from sellgate_replay import Endpoint, fixed_prompt_ids, load_workload, request  # noqa: E402


def parse_endpoint(raw: str) -> Endpoint:
    parts = raw.split(",", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("endpoint must be LABEL,BASE_URL,MODEL")
    return Endpoint(parts[0], parts[1].rstrip("/"), parts[2])


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50_ms": statistics.median(ordered) if ordered else None,
        "min_ms": ordered[0] if ordered else None,
        "max_ms": ordered[-1] if ordered else None,
    }


def build_prompts(total: int, split: int) -> tuple[list[int], list[int]]:
    if split < 64 or split >= total:
        raise ValueError(f"split {split} outside [64,{total})")
    prefix = fixed_prompt_ids(split, 370)
    suffix = total - split
    prompt_a = prefix + fixed_prompt_ids(suffix, 407)
    prompt_b = prefix + fixed_prompt_ids(suffix, 444)
    if prompt_a[:split] != prompt_b[:split] or prompt_a[split] == prompt_b[split]:
        raise AssertionError("prompt constructor did not produce the requested exact LCP")
    return prompt_a, prompt_b


def run_chain(
    endpoint: Endpoint,
    arm: str,
    split: int,
    rep: int,
    namespace: str,
    workload: dict[str, Any],
    timeout: float,
    physical_gpu: int,
    gpu_uuid: str,
) -> dict[str, dict[str, Any]]:
    prompt_a, prompt_b = build_prompts(int(workload["prompt_tokens"]), split)
    salt = f"{namespace}-{split}-r{rep}"
    rows: dict[str, dict[str, Any]] = {}
    for case, prompt in (("request1", prompt_a), ("request2", prompt_b), ("request3", prompt_b)):
        row = request(endpoint, prompt, salt, workload, timeout)
        row.update(
            {
                "kind": "request",
                "arm": arm,
                "case": case,
                "split": split,
                "rep": rep,
                "namespace": salt,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "physical_gpu_index": physical_gpu,
                "physical_gpu_uuid": gpu_uuid,
            }
        )
        rows[case] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, type=parse_endpoint)
    parser.add_argument("--candidate", required=True, type=parse_endpoint)
    parser.add_argument("--workload-lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--splits", default="64,512,2048,4374")
    parser.add_argument("--main-split", type=int, default=4374)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--require-timing-n", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()

    workload = load_workload(args.workload_lock)
    splits = [int(value) for value in args.splits.split(",")]
    if args.main_split not in splits:
        raise ValueError("main split must be one of --splits")
    if args.repetitions < 1:
        raise ValueError("at least one repetition is required")
    if args.require_timing_n and args.repetitions < 5:
        raise ValueError("request-2 timing requires at least five interleaved repetitions")
    if args.control.model != args.candidate.model:
        raise ValueError("control and candidate must name the same model")

    cells: list[tuple[int, int]] = [(1, split) for split in splits]
    cells.extend((rep, args.main_split) for rep in range(2, args.repetitions + 1))
    all_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    request2_ttft: dict[str, list[float]] = {"control": [], "candidate": []}

    for cell_index, (rep, split) in enumerate(cells):
        order = ("control", "candidate") if cell_index % 2 == 0 else ("candidate", "control")
        chains: dict[str, dict[str, dict[str, Any]]] = {}
        for arm in order:
            endpoint = args.control if arm == "control" else args.candidate
            chain = run_chain(
                endpoint,
                arm,
                split,
                rep,
                args.namespace,
                workload,
                args.timeout,
                args.physical_gpu,
                args.gpu_uuid,
            )
            chains[arm] = chain
            all_rows.extend(chain.values())

        control = chains["control"]
        candidate = chains["candidate"]
        _, prompt_b = build_prompts(int(workload["prompt_tokens"]), split)
        cold_oracle = request(
            args.candidate,
            prompt_b,
            f"{args.namespace}-genuinely-cold-{split}-r{rep}",
            workload,
            args.timeout,
        )
        cold_oracle.update(
            {
                "kind": "request",
                "arm": "candidate-cold-oracle",
                "case": "request2-genuinely-cold",
                "split": split,
                "rep": rep,
                "namespace": f"{args.namespace}-genuinely-cold-{split}-r{rep}",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "physical_gpu_index": args.physical_gpu,
                "physical_gpu_uuid": args.gpu_uuid,
            }
        )
        all_rows.append(cold_oracle)
        if not cold_oracle.get("ok"):
            failures.append(
                f"split={split} rep={rep} cold-oracle: {cold_oracle.get('error')}"
            )
        if int(cold_oracle.get("cached_tokens") or 0) != 0:
            failures.append(f"split={split} rep={rep}: cold oracle was not genuinely cold")
        for arm, chain in chains.items():
            for case, row in chain.items():
                if not row.get("ok"):
                    failures.append(
                        f"split={split} rep={rep} arm={arm} case={case}: {row.get('error')}"
                    )
            expected_request2 = 0 if arm == "control" else split
            if int(chain["request1"].get("cached_tokens") or 0) != 0:
                failures.append(f"split={split} rep={rep} {arm}: request1 was not cold")
            if int(chain["request2"].get("cached_tokens") or 0) != expected_request2:
                failures.append(
                    f"split={split} rep={rep} {arm}: request2 cached_tokens "
                    f"{chain['request2'].get('cached_tokens')} != {expected_request2}"
                )
            if int(chain["request3"].get("cached_tokens") or 0) != split:
                failures.append(
                    f"split={split} rep={rep} {arm}: request3 cached_tokens "
                    f"{chain['request3'].get('cached_tokens')} != {split}"
                )

        # The candidate must stand on its own: both partial-restored requests are compared
        # directly with a request that had no prefix entry at all. The feature-off chain remains
        # the timing control, but its whole-entry request 3 is not an exactness oracle for this arm.
        hashes = {
            "candidate_request2": candidate["request2"].get("text_sha256"),
            "candidate_request3": candidate["request3"].get("text_sha256"),
            "genuinely_cold_request2": cold_oracle.get("text_sha256"),
        }
        if len(set(hashes.values())) != 1:
            failures.append(
                f"BYTE MISMATCH split={split} rep={rep}: "
                + json.dumps(hashes, sort_keys=True)
            )
        if split == args.main_split:
            for arm, chain in chains.items():
                ttft = chain["request2"].get("ttft_ms")
                if ttft is None:
                    failures.append(f"split={split} rep={rep} {arm}: request2 has no TTFT")
                else:
                    request2_ttft[arm].append(float(ttft))

    # HIRADIX-EXACT-ISO item 2 on the live path: one exact prefix hits only inside its own
    # namespace. The same bytes must remain cold in a sibling and in the default namespace.
    iso_prompt = fixed_prompt_ids(int(workload["prompt_tokens"]), 811)
    iso_rows = []
    for case, salt in (
        ("namespace-a-cold", f"{args.namespace}-iso-a"),
        ("namespace-a-hit", f"{args.namespace}-iso-a"),
        ("namespace-b-cold", f"{args.namespace}-iso-b"),
        ("default-cold", ""),
    ):
        row = request(args.candidate, iso_prompt, salt, workload, args.timeout)
        row.update(
            {
                "kind": "namespace-isolation",
                "arm": "candidate",
                "case": case,
                "split": None,
                "rep": 1,
                "namespace": salt,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "physical_gpu_index": args.physical_gpu,
                "physical_gpu_uuid": args.gpu_uuid,
            }
        )
        iso_rows.append(row)
        all_rows.append(row)
    iso_expected = [0, len(iso_prompt), 0, 0]
    iso_cached = [int(row.get("cached_tokens") or 0) for row in iso_rows]
    if iso_cached != iso_expected:
        failures.append(f"namespace isolation cached_tokens {iso_cached} != {iso_expected}")
    if not all(row.get("ok") for row in iso_rows):
        failures.append("namespace isolation request failed")
    if len({row.get("text_sha256") for row in iso_rows}) != 1:
        failures.append("BYTE MISMATCH across namespace-isolation controls")

    control_p50 = (
        statistics.median(request2_ttft["control"])
        if request2_ttft["control"] else None
    )
    candidate_p50 = (
        statistics.median(request2_ttft["candidate"])
        if request2_ttft["candidate"] else None
    )
    summary = {
        "kind": "summary",
        "schema": "memra.lcprestore.split-exactness.v1",
        "model": args.control.model,
        "split_boundaries": splits,
        "main_split": args.main_split,
        "main_split_repetitions": args.repetitions,
        "timing_n_required": args.require_timing_n,
        "arm_order": "alternated by cell in one enclosing GPU lock",
        "physical_gpu_index": args.physical_gpu,
        "physical_gpu_uuid": args.gpu_uuid,
        "namespace_isolation_cached_tokens": iso_cached,
        "request2_ttft": {
            "control": distribution(request2_ttft["control"]),
            "candidate": distribution(request2_ttft["candidate"]),
            "candidate_minus_control_p50_ms": (
                candidate_p50 - control_p50
                if candidate_p50 is not None and control_p50 is not None else None
            ),
            "candidate_vs_control_p50_pct":
                (candidate_p50 / control_p50 - 1.0) * 100.0
                if candidate_p50 is not None and control_p50 else None,
        },
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for row in all_rows:
            output.write(json.dumps(public(row), sort_keys=True) + "\n")
        output.write(json.dumps(summary, sort_keys=True) + "\n")
    for row in all_rows:
        printable = public(row)
        printable.pop("text_utf8_b64", None)
        print(json.dumps(printable, sort_keys=True), flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
