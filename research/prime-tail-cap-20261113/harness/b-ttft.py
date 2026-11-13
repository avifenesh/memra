#!/usr/bin/env python3
"""memra#1258: TTFT and the prime's chunk walls for cold raw-text prompts on an idle card, one boot.
Each prompt file is sent as a raw /v1/completions prompt (no chat template; the server tokenizes it
with the GGUF tokenizer, as run-gen does), streamed, --reps times in turn, greedy, 8 new tokens.
usage: b-ttft.py --bin BIN --model GGUF --draft DIR --out DIR --prompts F1,F2,... [--reps 3]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bcost", HERE / "b-cost.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
g = bc.g
RE_YIELD = re.compile(r"^\[prime-yield\] .*wall_ms=([0-9.]+)")
RE_COST = re.compile(r"\[admission\] request cost: .* ctx=(\d+) path=(\w+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--port", type=int, default=18184)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "serve.log"
    server = g.Server(args.bin, args.model, args.draft, args.port, 32768, log)
    texts = [(Path(p).name, Path(p).read_text()) for p in args.prompts.split(",")]
    try:
        server.boot()
        for rep in range(args.reps):
            for name, text in texts:
                start = len(log.read_text(errors="replace").splitlines())
                r = bc.stream_ttft(server, text, 8)
                time.sleep(0.5)
                lines = g.log_lines(log, start)
                walls = [float(m.group(1)) for l in lines if (m := RE_YIELD.search(l))]
                cost = next((m.groups() for l in lines if (m := RE_COST.search(l))), None)
                row = {"rep": rep, "prompt": name, "status": r["status"], "ttft_s": r["ttft_s"],
                       "chunk_walls_ms": walls, "prime_ms": round(sum(walls), 3),
                       "ctx": cost and int(cost[0]), "path": cost and cost[1],
                       "oom_lines": sum(1 for l in lines if "out of memory" in l)}
                with (args.out / "rows.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"rep {rep} {name}: status {r['status']} ttft {r['ttft_s']} prime {row['prime_ms']} ms "
                      f"chunks {walls} path {row['path']}", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
