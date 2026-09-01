#!/usr/bin/env python3
"""Join client, worker-tick, TTFT, and nvidia-smi receipts by burst."""

import argparse
import csv
import datetime
import json
import math
import pathlib
import shlex
import statistics


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def median(values):
    return statistics.median(values) if values else None


def parse_fields(line, prefix):
    if not line.startswith(prefix):
        return None
    fields = {}
    for token in shlex.split(line[len(prefix) :]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if value == "na":
            fields[key] = None
            continue
        try:
            fields[key] = float(value) if "." in value else int(value)
        except ValueError:
            fields[key] = value
    return fields


def read_client(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    summaries = sorted(
        (row for row in rows if row["kind"] == "summary"), key=lambda row: row["burst"]
    )
    requests = {
        row["id"]: row for row in rows if row["kind"] == "request" and row.get("id")
    }
    return summaries, requests


def read_server(path):
    ticks = []
    ttft = {}
    for line in path.read_text(errors="replace").splitlines():
        tick = parse_fields(line, "[tick] ")
        if tick is not None:
            tokens = tick.get("prefill_single_tokens", 0) + tick.get(
                "prefill_batch_tokens", 0
            )
            if tokens:
                ticks.append(tick)
            continue
        trace = parse_fields(line, "[ttft] ")
        if trace is not None and trace.get("id"):
            ttft[str(trace["id"])] = trace
    return ticks, ttft


def number(value):
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_gpu(path):
    lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
    header = next(line for line in lines if line.startswith("#Date,"))
    names = [part.strip().lstrip("#") for part in next(csv.reader([header]))]
    samples = []
    for line in lines:
        if line.startswith("#"):
            continue
        values = [part.strip() for part in next(csv.reader([line]))]
        if len(values) != len(names):
            continue
        row = dict(zip(names, values))
        stamp = datetime.datetime.strptime(
            f"{row['Date']} {row['Time']}", "%Y%m%d %H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc)
        samples.append(
            {
                "time": stamp,
                **{
                    key: number(row.get(key))
                    for key in ("pwr", "gtemp", "sm", "mem", "mclk", "pclk", "fb", "smutil", "dram")
                },
            }
        )
    return samples


def consume_ticks(summaries, ticks):
    cursor = 0
    by_burst = {}
    for summary in summaries:
        target = summary["aggregate_prompt_tokens"]
        total = 0
        group = []
        while total < target:
            if cursor >= len(ticks):
                raise RuntimeError(
                    f"server tick log ended in burst {summary['burst']}: {total}/{target} tokens"
                )
            tick = ticks[cursor]
            cursor += 1
            tick_tokens = tick.get("prefill_single_tokens", 0) + tick.get(
                "prefill_batch_tokens", 0
            )
            total += tick_tokens
            group.append(tick)
        if total != target:
            raise RuntimeError(
                f"burst {summary['burst']} crossed token target: {total} != {target}"
            )
        by_burst[summary["burst"]] = group
    if cursor != len(ticks):
        raise RuntimeError(f"{len(ticks) - cursor} unassigned prefill ticks remain")
    return by_burst


def fmt(value, digits=1):
    return "na" if value is None else f"{value:.{digits}f}"


def analyze_arm(label, client_path, server_path, gpu_path):
    summaries, requests = read_client(client_path)
    ticks, ttft = read_server(server_path)
    gpu = read_gpu(gpu_path)
    tick_groups = consume_ticks(summaries, ticks)
    burst_rows = []
    for summary in summaries:
        group = tick_groups[summary["burst"]]
        request_rows = [requests[request_id] for request_id in summary["request_ids"]]
        traces = [ttft[request_id] for request_id in summary["request_ids"]]
        prefill_ms = sum(float(tick["prefill_ms"]) for tick in group)
        start = datetime.datetime.fromisoformat(summary["burst_start_utc"])
        end = datetime.datetime.fromisoformat(summary["last_first_token_utc"])
        # dmon was sampled at 1 Hz. Associate each printed sample with its
        # trailing one-second interval, and retain it iff that interval overlaps
        # the client-observed burst. This avoids pulling in the preceding
        # cooldown while still giving sub-second N=1 bursts one interval sample.
        sample_period = datetime.timedelta(seconds=1)
        active_gpu = [
            sample
            for sample in gpu
            if sample["time"] >= start and sample["time"] - sample_period <= end
        ]
        burst_rows.append(
            {
                "arm": label,
                "burst": summary["burst"],
                "excluded": summary["excluded"],
                "concurrency": summary["concurrency"],
                "repeat": summary["repeat"],
                "prompt_tokens_each_min": summary["prompt_tokens_each_min"],
                "prompt_tokens_each_max": summary["prompt_tokens_each_max"],
                "prefill_ticks": len(group),
                "single_calls": sum(tick.get("prefill_single_calls", 0) for tick in group),
                "batch_calls": sum(tick.get("prefill_batch_calls", 0) for tick in group),
                "server_prefill_ms": prefill_ms,
                "server_aggregate_prefill_tps": summary["aggregate_prompt_tokens"]
                / (prefill_ms / 1000.0),
                "client_aggregate_prompt_tps": summary["client_aggregate_prompt_tps"],
                "prime_wall_ms": [float(trace["prime_ms"]) for trace in traces],
                "queue_wait_ms": [float(trace["queue_wait_ms"]) for trace in traces],
                "prime_wait_ms": [float(trace["prime_wait_ms"]) for trace in traces],
                "client_ttft_s": [row["client_ttft_s"] for row in request_rows],
                "gpu_samples": active_gpu,
            }
        )
    return burst_rows


def cell_rows(bursts):
    groups = {}
    for burst in bursts:
        if burst["excluded"]:
            continue
        groups.setdefault((burst["arm"], burst["concurrency"]), []).append(burst)
    rows = []
    for (arm, concurrency), group in sorted(groups.items()):
        prime_wall = [value for burst in group for value in burst["prime_wall_ms"]]
        queue_wait = [value for burst in group for value in burst["queue_wait_ms"]]
        prime_wait = [value for burst in group for value in burst["prime_wait_ms"]]
        client_ttft = [value for burst in group for value in burst["client_ttft_s"]]
        samples = [sample for burst in group for sample in burst["gpu_samples"]]
        metric = lambda key: [sample[key] for sample in samples if sample.get(key) is not None]
        server_rates = [burst["server_aggregate_prefill_tps"] for burst in group]
        rows.append(
            {
                "arm": arm,
                "c": concurrency,
                "repeats": len(group),
                "requests": len(prime_wall),
                "ticks_med": median([burst["prefill_ticks"] for burst in group]),
                "single_calls_med": median([burst["single_calls"] for burst in group]),
                "batch_calls": sum(burst["batch_calls"] for burst in group),
                "agg_tps_med": median(server_rates),
                "agg_tps_min": min(server_rates),
                "agg_tps_max": max(server_rates),
                "client_tps_med": median(
                    [burst["client_aggregate_prompt_tps"] for burst in group]
                ),
                "prime_wall_p50_ms": median(prime_wall),
                "prime_wall_p95_ms": percentile(prime_wall, 95),
                "queue_wait_p50_ms": median(queue_wait),
                "queue_wait_p95_ms": percentile(queue_wait, 95),
                "prime_wait_p50_ms": median(prime_wait),
                "prime_wait_p95_ms": percentile(prime_wait, 95),
                "ttft_p50_s": median(client_ttft),
                "ttft_p95_s": percentile(client_ttft, 95),
                "gpu_n": len(samples),
                "sm_p50": median(metric("sm")),
                "sm_p95": percentile(metric("sm"), 95),
                "mem_p50": median(metric("mem")),
                "mem_p95": percentile(metric("mem"), 95),
                "gpm_sm_p50": median(metric("smutil")),
                "gpm_sm_p95": percentile(metric("smutil"), 95),
                "dram_p50": median(metric("dram")),
                "dram_p95": percentile(metric("dram"), 95),
                "temp_min": min(metric("gtemp")) if metric("gtemp") else None,
                "temp_max": max(metric("gtemp")) if metric("gtemp") else None,
                "power_p50": median(metric("pwr")),
            }
        )
    return rows


def write_tables(bursts, cells, out_prefix):
    burst_json = out_prefix.with_name(out_prefix.name + "-bursts.jsonl")
    burst_json.write_text(
        "".join(
            json.dumps(
                {**burst, "gpu_samples": len(burst["gpu_samples"])}, sort_keys=True
            )
            + "\n"
            for burst in bursts
        )
    )
    header = [
        "arm", "c", "repeats", "requests", "ticks_med", "single_calls_med",
        "batch_calls", "agg_tps_med", "agg_tps_min", "agg_tps_max",
        "client_tps_med", "prime_wall_p50_ms", "prime_wall_p95_ms",
        "queue_wait_p50_ms", "queue_wait_p95_ms", "prime_wait_p50_ms",
        "prime_wait_p95_ms",
        "ttft_p50_s", "ttft_p95_s", "gpu_n", "sm_p50", "sm_p95",
        "mem_p50", "mem_p95", "gpm_sm_p50", "gpm_sm_p95", "dram_p50",
        "dram_p95", "temp_min", "temp_max", "power_p50",
    ]
    lines = ["\t".join(header)]
    for row in cells:
        values = []
        for key in header:
            value = row[key]
            digits = 3 if key in ("ttft_p50_s", "ttft_p95_s") else 1
            values.append(
                str(value) if isinstance(value, (int, str)) else fmt(value, digits)
            )
        lines.append("\t".join(values))
    table = "\n".join(lines) + "\n"
    out_prefix.with_name(out_prefix.name + "-cells.tsv").write_text(table)
    print(table, end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", nargs=4, action="append", required=True, metavar=("LABEL", "CLIENT", "SERVER", "GPU")
    )
    parser.add_argument("--out-prefix", type=pathlib.Path, required=True)
    args = parser.parse_args()
    bursts = []
    for label, client, server, gpu in args.arm:
        bursts.extend(
            analyze_arm(label, pathlib.Path(client), pathlib.Path(server), pathlib.Path(gpu))
        )
    write_tables(bursts, cell_rows(bursts), args.out_prefix)


if __name__ == "__main__":
    main()
