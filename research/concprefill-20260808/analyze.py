#!/usr/bin/env python3
"""Summarize concurrent-prefill client rows and server tick anatomy."""

import argparse
import json
import math
import pathlib
import shlex
import statistics


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def median(values):
    return statistics.median(values) if values else None


def parse_tick(line):
    if not line.startswith("[tick] "):
        return None
    fields = {}
    for token in shlex.split(line[len("[tick] "):]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = float(value) if "." in value else int(value)
    return fields


def client_table(paths):
    summaries = []
    requests = []
    for path in paths:
        for line in pathlib.Path(path).read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") == "summary":
                summaries.append(row)
            elif row.get("kind") == "request":
                requests.append(row)

    groups = {}
    for row in summaries:
        key = (row["label"], row["concurrency"])
        groups.setdefault(key, []).append(row)
    request_groups = {}
    for row in requests:
        key = (row["label"], row["concurrency"])
        request_groups.setdefault(key, []).append(row)

    lines = [
        "label\tc\trepeats\trequests\tagg_prefill_tps_median\tagg_prefill_tps_min"
        "\tagg_prefill_tps_max\tttft_p95_s\tbackground_tps"
        "\tbackground_itl_p95_ms"
    ]
    for (label, concurrency), rows in sorted(groups.items()):
        request_rows = request_groups[(label, concurrency)]
        rates = [row["aggregate_prefill_tps"] for row in rows]
        ttfts = [row["client_ttft_s"] for row in request_rows]
        bg_rates = [row["background_visible_tps"] for row in rows]
        bg_gaps = [row["background_itl_p95_ms"] for row in rows
                   if row["background_itl_p95_ms"] is not None]
        lines.append(
            f"{label}\t{concurrency}\t{len(rows)}\t{len(request_rows)}"
            f"\t{median(rates):.1f}"
            f"\t{min(rates):.1f}\t{max(rates):.1f}\t{percentile(ttfts, 95):.3f}"
            f"\t{median(bg_rates):.1f}\t"
            + (f"{median(bg_gaps):.1f}" if bg_gaps else "na")
        )
    return "\n".join(lines) + "\n"


def tick_table(specs):
    lines = [
        "label\tticks\tprefill_ticks\tserial_calls\tserial_tokens\tbatch_calls"
        "\tbatch_tokens\tprefill_ms_median\tprefill_ms_p95\tdecode_ms_p95"
    ]
    for spec in specs:
        label, path = spec.split("=", 1)
        ticks = [
            tick for line in pathlib.Path(path).read_text(errors="replace").splitlines()
            if (tick := parse_tick(line)) is not None
        ]
        prefill = [
            tick for tick in ticks
            if tick.get("prefill_single_calls", 0) or tick.get("prefill_batch_calls", 0)
        ]
        lines.append(
            f"{label}\t{len(ticks)}\t{len(prefill)}"
            f"\t{sum(tick.get('prefill_single_calls', 0) for tick in prefill)}"
            f"\t{sum(tick.get('prefill_single_tokens', 0) for tick in prefill)}"
            f"\t{sum(tick.get('prefill_batch_calls', 0) for tick in prefill)}"
            f"\t{sum(tick.get('prefill_batch_tokens', 0) for tick in prefill)}"
            f"\t{median([tick['prefill_ms'] for tick in prefill]):.1f}"
            f"\t{percentile([tick['prefill_ms'] for tick in prefill], 95):.1f}"
            f"\t{percentile([tick['decode_ms'] for tick in ticks], 95):.1f}"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", action="append", required=True)
    parser.add_argument("--server", action="append", required=True,
                        metavar="LABEL=PATH")
    parser.add_argument("--client-table", required=True)
    parser.add_argument("--tick-table", required=True)
    args = parser.parse_args()

    clients = client_table(args.client)
    ticks = tick_table(args.server)
    pathlib.Path(args.client_table).write_text(clients)
    pathlib.Path(args.tick_table).write_text(ticks)
    print(clients, end="")
    print(ticks, end="")


if __name__ == "__main__":
    main()
