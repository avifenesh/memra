#!/usr/bin/env python3
"""Release one concurrent requested-128k burst for the box1 capacity row."""

from __future__ import annotations

import argparse
import pathlib
import threading

from run_admission_workload import append, metrics, request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("rows", type=pathlib.Path)
    parser.add_argument("--concurrency", type=int, default=24)
    args = parser.parse_args()
    if args.rows.exists():
        raise SystemExit(f"refusing to append a second run to {args.rows}")

    append(
        args.rows,
        {
            "kind": "run",
            "n": 1,
            "server_ctx": 262144,
            "requested_max_ctx": 131072,
            "concurrency": args.concurrency,
        },
    )
    append(args.rows, {"kind": "metrics", "phase": "before", "value": metrics(args.base)})
    release = threading.Barrier(args.concurrency)
    rows: list[dict] = []
    lock = threading.Lock()

    def one(index: int) -> None:
        row = request(
            args.base,
            "capacity",
            "burst128k",
            index,
            131072,
            64,
            release,
        )
        with lock:
            rows.append(row)
            append(args.rows, row)

    threads = [threading.Thread(target=one, args=(index,)) for index in range(args.concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    append(args.rows, {"kind": "metrics", "phase": "after", "value": metrics(args.base)})
    completed = sum(bool(row.get("ok")) for row in rows)
    append(
        args.rows,
        {
            "kind": "summary",
            "requests_ok": completed,
            "requests_n": len(rows),
        },
    )
    return 0 if completed == len(rows) == args.concurrency else 1


if __name__ == "__main__":
    raise SystemExit(main())
