#!/usr/bin/env python3
"""Run the frozen serving-shape identity gate with the predecessor harness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from measure_interleaved import run_arm


DEFAULT_OUTPUT = Path(__file__).with_name("raw") / "candidate-v1" / "actual-shape-gate"
EXPECTED_TEXT_SHA256 = {
    "q27": "200ec271e8c0eb57fb6b7d42d3ed53e4590c5e72f0303b5ef3c74d363eab88e7",
    "q35": "b723be26c76590659d44165c5feabc0ad705653a81df16846bbf3aa248ec7be1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=18_839)
    args = parser.parse_args()
    if os.environ.get("MEMRA_5090_LOCK_HELD") != "1":
        raise SystemExit(
            "refusing actual-shape gate: acquire flock /tmp/memra-5090.lock and set "
            "MEMRA_5090_LOCK_HELD=1"
        )

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite actual-shape output: {output}")
    output.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    sequence = 0
    orders = {"q27": ("baseline", "candidate"), "q35": ("candidate", "baseline")}
    results_path = output / "results.jsonl"
    for model_name in ("q27", "q35"):
        for arm in orders[model_name]:
            sequence += 1
            print(
                f"ACTUAL_SHAPE_START sequence={sequence} model={model_name} arm={arm}",
                flush=True,
            )
            result = run_arm(output, sequence, model_name, arm, 1, args.port)
            rows.append(result)
            with results_path.open("a") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                f"ACTUAL_SHAPE_DONE sequence={sequence} model={model_name} arm={arm} "
                f"text_sha256={result['text_sha256']}",
                flush=True,
            )

        hashes = {row["text_sha256"] for row in rows if row["model"] == model_name}
        if hashes != {EXPECTED_TEXT_SHA256[model_name]}:
            raise RuntimeError(
                f"STOP: actual-shape greedy output differs for {model_name}: "
                f"observed={sorted(hashes)} expected={EXPECTED_TEXT_SHA256[model_name]}"
            )

    summary = {
        "schema": "memra.shmconflict.actual-shape-gate.v1",
        "result": "PASS",
        "rows": len(rows),
        "text_sha256": EXPECTED_TEXT_SHA256,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("ACTUAL_SHAPE_IDENTITY PASS rows=4", flush=True)


if __name__ == "__main__":
    main()
