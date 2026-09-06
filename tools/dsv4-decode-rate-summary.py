#!/usr/bin/env python3
"""Audit sampled decode-only ABBA receipts. No HTTP or concurrency extrapolation."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import statistics
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    assert values
    rank = (len(values) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def distribution(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "p95": percentile(values, .95), "p99": percentile(values, .99)}


def looped(tokens):
    for width in range(1, 33):
        length = width * max(4, (32 + width - 1) // width)
        for start in range(len(tokens) - length + 1):
            block = tokens[start:start + width]
            if all(tokens[i:i + width] == block for i in range(start, start + length, width)):
                return True
    return False


def timestamp_ms(value):
    return dt.datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f").replace(
        tzinfo=dt.timezone.utc).timestamp() * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("controller", type=Path)
    parser.add_argument("processes", type=Path)
    parser.add_argument("telemetry", type=Path)
    args = parser.parse_args()
    model = args.model.read_text()
    controller = args.controller.read_text()
    processes = args.processes.read_text()
    assert re.search(r"^EXIT utc=\S+ status=0$", controller, re.M), "controller incomplete/failed"
    assert "PASS sampled plain/DSpark decode timing and output identity; HTTP and concurrency remain separate" in model
    assert "REFUSED" not in controller and "REFUSED" not in processes
    assert "sample_T=1 top_p=1 top_k=0 seed=20260906 EOS=respected loops=excluded" in model
    assert "source_sha256=f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded" in model
    pids = set(re.findall(r"^(\d+), ", processes, re.M))
    assert len(pids) == 1, f"GPU process identities: {pids}"
    owners = set(re.findall(r"owned=(\d+)$", processes, re.M))
    if not owners:
        owners = set(re.findall(r"^OWNED gpu_pid=(\d+) via_timeout=\d+$", processes, re.M))
    assert pids == owners, "GPU process not the owned child"
    rows = [json.loads(line.removeprefix("MEASURE ")) for line in model.splitlines()
            if line.startswith("MEASURE ")]
    sampler_ab = "sampler_ab=true" in model
    grid_ab = "grid_ab=true" in model
    recent_ab = "recent_ab=true" in model
    assert sum([sampler_ab, grid_ab, recent_ab]) <= 1
    paired = sampler_ab or grid_ab or recent_ab
    dimension = "recent_rows" if recent_ab else "grouped_grid" if grid_ab else "sampler_order"
    control, tuned = (0, 512) if recent_ab else ("full", "bounded") if grid_ab else ("comparison", "radix")
    expected_rows = 56 if paired else 28
    assert len(rows) == expected_rows, f"expected {expected_rows} rows, saw {len(rows)}"
    with args.telemetry.open(newline="") as source:
        telemetry = list(csv.DictReader(source, skipinitialspace=True))
    assert telemetry
    for sample in telemetry:
        sample["ms"] = timestamp_ms(sample["timestamp"])
    gpu_ids = {sample["index"] for sample in telemetry}
    assert gpu_ids == {"0", "1"}, gpu_ids
    result = {"scope": "sampled engine decode-only, warm restored C4; not HTTP or concurrency",
              "gpu_pids": sorted(pids), "telemetry_rows": len(telemetry), "prompts": {}}
    for count in [256, 8192]:
        selected = [row for row in rows if row["prompt"] == count]
        for row in selected:
            row.setdefault("sampler_order", "comparison")
            row.setdefault("grouped_grid", "full")
        if grid_ab:
            assert {r["sampler_order"] for r in selected} == {"radix"}
        elif recent_ab:
            assert {r["sampler_order"] for r in selected} == {"radix"}
            assert {r["grouped_grid"] for r in selected} == {"full"}
        elif sampler_ab:
            assert len({r["grouped_grid"] for r in selected}) == 1
        if paired:
            warm = [(control, "plain"), (control, "dspark"), (tuned, "plain"), (tuned, "dspark")]
            measured = [(control, "plain"), (tuned, "plain"), (tuned, "dspark"), (control, "dspark"),
                        (control, "dspark"), (tuned, "dspark"), (tuned, "plain"), (control, "plain")] * 3
        else:
            warm = [(selected[0]["sampler_order"], arm) for arm in ["plain", "dspark"]]
            measured = [(selected[0]["sampler_order"], arm) for arm in ["plain", "dspark", "dspark", "plain"]] * 3
        expected_schedule = warm + measured
        assert len(selected) == len(expected_schedule)
        assert [r["ordinal"] for r in selected] == list(range(len(expected_schedule)))
        assert [(r[dimension], r["arm"]) for r in selected] == expected_schedule
        assert [r["warmup"] for r in selected] == [True] * len(warm) + [False] * len(measured)
        hashes = set()
        for row in selected:
            tokens, commits = row["tokens"], row["commit_ns"]
            assert len(tokens) == row["output_tokens"] == len(commits)
            assert row["eos"] or len(tokens) == 256
            assert commits == sorted(commits) and commits and commits[0] > 0
            assert commits[-1] <= row["decode_wall_ns"]
            assert row["start_unix_ms"] <= row["end_unix_ms"]
            digest = hashlib.sha256(b"".join(t.to_bytes(4, "little") for t in tokens)).hexdigest()
            assert digest == row["token_sha256"]
            hashes.add(digest)
            assert row["looped"] == looped(tokens)
            assert row["eligible"] == (not row["warmup"] and not row["looped"] and len(tokens) >= 32)
            assert row["ep_calls"] > 0 and row["sink_calls"] > 0
            if grid_ab:
                assert (row["grid_calls"] > 0) == (row["grouped_grid"] == "bounded")
            if recent_ab:
                assert (row["recent_gathers"] > 0) == (row["recent_rows"] > 0)
                assert (sum(row["recent_gpu_bytes"]) > 0) == (row["recent_rows"] > 0)
            assert (row["rounds"] > 0) == (row["arm"] == "dspark")
            row["decode_wall_tps"] = len(tokens) * 1e9 / row["decode_wall_ns"]
            row["post_first_tps"] = ((len(tokens) - 1) * 1e9 / (commits[-1] - commits[0])
                                     if len(tokens) > 1 and commits[-1] > commits[0] else None)
            observed = [s for s in telemetry if row["start_unix_ms"] - 300 <= s["ms"] <= row["end_unix_ms"] + 300]
            assert {s["index"] for s in observed} == gpu_ids, "missing per-phase GPU telemetry"
            max_gap = 0
            for dev in gpu_ids:
                stamps = sorted(s["ms"] for s in observed if s["index"] == dev)
                assert stamps[0] <= row["start_unix_ms"] + 1000
                assert stamps[-1] >= row["end_unix_ms"] - 1000
                max_gap = max(max_gap, max((b - a for a, b in zip(stamps, stamps[1:])), default=0))
            assert max_gap <= 1000, f"telemetry gap {max_gap}ms"
            row["telemetry_max_gap_ms"] = max_gap
        assert len(hashes) == 1, "plain/DSpark or repeated-output identity failed"
        summary = {"token_sha256": hashes.pop(), "arms": {}}
        for order, arm in warm:
            all_rows = [r for r in selected if r["arm"] == arm and r[dimension] == order and not r["warmup"]]
            assert len(all_rows) == 6
            valid = [r for r in all_rows if r["eligible"]]
            report = {"measured": len(all_rows), "eligible": len(valid),
                      "excluded_looped": sum(r["looped"] for r in all_rows),
                      "excluded_short": sum(r["output_tokens"] < 32 for r in all_rows)}
            if len(valid) >= 5:
                report["decode_wall_output_tps"] = distribution([r["decode_wall_tps"] for r in valid])
                report["post_first_output_tps"] = distribution([r["post_first_tps"] for r in valid])
                report["restore_ms_not_in_decode"] = distribution([r["restore_ns"] / 1e6 for r in valid])
                report["first_engine_commit_ms"] = distribution([r["commit_ns"][0] / 1e6 for r in valid])
                gaps = [(b - a) / 1e6 for r in valid for a, b in zip(r["commit_ns"], r["commit_ns"][1:])]
                report["engine_commit_itl_ms_including_spec_bursts"] = distribution(gaps)
                report["rounds"] = [r["rounds"] for r in valid]
                report["accepted"] = [r["accepted"] for r in valid]
                report["drafted"] = [r["drafted"] for r in valid]
                report["telemetry_max_gap_ms"] = max(r["telemetry_max_gap_ms"] for r in valid)
            summary["arms"][f"{order}/{arm}" if paired else arm] = report
        result["prompts"][count] = summary
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
