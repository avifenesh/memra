#!/usr/bin/env python3
"""Join a Q27 width-flip client run to diagnostics-only sampler receipts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SAMPLE_PREFIX = "[eosclass-sample] "
HISTORICAL_EOS_SHA256 = (
    "ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73"
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: malformed JSON: {error}") from error
        row["_line"] = line_number
        rows.append(row)
    return rows


def tagged_jsonl(path: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        marker = line.find(prefix)
        if marker < 0:
            continue
        try:
            row = json.loads(line[marker + len(prefix) :])
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: malformed receipt: {error}") from error
        row["_line"] = line_number
        rows.append(row)
    return rows


def width_runs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda row: int(row["generated_index"])):
        width = sample.get("batch_width")
        index = int(sample["generated_index"])
        if runs and runs[-1]["width"] == width:
            runs[-1]["last_generated_index"] = index
            runs[-1]["samples"] += 1
            continue
        runs.append(
            {
                "width": width,
                "first_generated_index": index,
                "last_generated_index": index,
                "samples": 1,
            }
        )
    return runs


def selected_eos(row: dict[str, Any]) -> dict[str, Any] | None:
    selected = int(row["selected"])
    return next(
        (entry for entry in row["logits"]["eos"] if int(entry["id"]) == selected),
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--expect", choices=("observe", "early-eos"), default="observe")
    args = parser.parse_args()

    clients = jsonl(args.run / "client.jsonl")
    samples = tagged_jsonl(args.run / "eosclass-trace.jsonl", SAMPLE_PREFIX)
    targets = [
        row
        for row in clients
        if row.get("kind") == "width_flip_request" and row.get("role") == "target"
    ]
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        trace_id = sample.get("trace_id")
        if trace_id is not None:
            by_trace[str(trace_id)].append(sample)

    errors: list[str] = []
    target_receipts: list[dict[str, Any]] = []
    early_receipts: list[dict[str, Any]] = []
    for target in targets:
        trace_id = str(target.get("trace_id") or "")
        trace = sorted(by_trace.get(trace_id, []), key=lambda row: int(row["generated_index"]))
        completion_tokens = int(target.get("completion_tokens") or 0)
        if not trace_id:
            errors.append(f"client line {target['_line']} has no trace_id")
        if len(trace) != completion_tokens:
            errors.append(
                f"{trace_id}: {len(trace)} sampler receipts != {completion_tokens} completion tokens"
            )
        if any(row.get("sample_source") != "host" for row in trace):
            errors.append(f"{trace_id}: non-host sample present despite host-sampling canary")

        eos = [row for row in trace if row.get("selected_is_eos")]
        early = completion_tokens < 60 and target.get("finish_reason") == "stop"
        if early and (not eos or int(eos[-1]["generated_index"]) != completion_tokens - 1):
            errors.append(f"{trace_id}: early stop lacks a terminal selected-EOS receipt")
        if not early and eos:
            errors.append(f"{trace_id}: selected EOS without an early-stop client result")

        receipt = {
            "trace_id": trace_id,
            "delay_ms": target.get("delay_ms"),
            "completion_tokens": completion_tokens,
            "finish_reason": target.get("finish_reason"),
            "text_sha256": target.get("text_sha256"),
            "sample_count": len(trace),
            "width_runs": width_runs(trace),
        }
        target_receipts.append(receipt)
        for sample in eos:
            eos_logit = selected_eos(sample)
            top1 = sample["logits"]["top1"]
            proof = {
                **receipt,
                "generated_index": sample.get("generated_index"),
                "selected": sample.get("selected"),
                "sample_source": sample.get("sample_source"),
                "logits_len": sample["logits"].get("len"),
                "top1_id": top1.get("id"),
                "top1_logit": top1.get("logit"),
                "top2_id": sample["logits"]["top2"].get("id"),
                "margin": sample["logits"].get("margin"),
                "selected_eos_rank": eos_logit.get("rank") if eos_logit else None,
                "selected_eos_logit": eos_logit.get("logit") if eos_logit else None,
                "host_logits_favor_eos": bool(
                    sample.get("sample_source") == "host"
                    and eos_logit
                    and eos_logit.get("rank") == 1
                    and top1.get("id") == sample.get("selected")
                ),
                "historical_hash_match": target.get("text_sha256")
                == HISTORICAL_EOS_SHA256,
                "trace_line": sample["_line"],
            }
            early_receipts.append(proof)

    summary_rows = [row for row in clients if row.get("kind") == "summary"]
    if len(summary_rows) != 1:
        errors.append(f"expected one client summary, found {len(summary_rows)}")
    if args.expect == "early-eos" and not early_receipts:
        errors.append("expected at least one early-EOS target, found none")
    if any(not row["host_logits_favor_eos"] for row in early_receipts):
        errors.append("one or more selected EOS receipts were not rank-1 in full host logits")

    output = {
        "schema": "memra.eosclass.width-flip-trace-verification.v1",
        "run": str(args.run),
        "targets": len(targets),
        "samples": len(samples),
        "target_text_classes": sorted({str(row.get("text_sha256")) for row in targets}),
        "early_eos_count": len(early_receipts),
        "historical_eos_count": sum(row["historical_hash_match"] for row in early_receipts),
        "all_early_eos_rank1_host": bool(early_receipts)
        and all(row["host_logits_favor_eos"] for row in early_receipts),
        "early_eos": early_receipts,
        "target_receipts": target_receipts,
        "errors": errors,
        "verdict": "PASS" if not errors else "FAIL",
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
