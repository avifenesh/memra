#!/usr/bin/env python3
"""memra#1255 review round 2: a cell that reaches the door-off partial reclaim while a cold DFlash
session owns no route state yet (RTX 5090 Laptop, DFlash2). Not a gate: it prints what each binary
did, and the reviewer's question is answered by the reclaim lines.

Shape: a derived prefix budget above its two-entry floor (no MEMRA_PREFIX_CACHE_MB, MEMRA_CTX 8,192,
4 interactive sessions, judge and harvest lanes capped at 1), so `reclaim_partial_applies`; the cache filled with prefix entries by cold prompts;
a decoding keeper (counting run) so the pair meets in one pass and `/metrics` is fresh; a ballast
placing effective free between A's need and B's: A (a ~2,000-token cold prompt) fits and is
admitted with no route state, B (a ~7,400-token cold prompt) fits neither the booked reading nor
the reading without the booking, so it goes to the reclaim at once and the partial reclaim sizes
its need (`[admit-oom] reclaim-on-defer (partial): ... for NMB (MMB of it owed ...)`). A is fired
20 ms before B so it is first in the pass.

Read: B's partial-reclaim line(s) (need, owed, freed, covered), whether B was admitted in A's pass,
out-of-memory lines, both answers. On the head the need counts A's whole charge once (the unbuilt
booking); base books no unbuilt charge.

usage: reclaim-cell.py --bin BIN --model GGUF --draft DIR --out DIR [--fill 6]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import threading
import time
from pathlib import Path

GATE = Path.home() / ".cache/memra-1255/wt/tools/same-pass-admission-gate.py"
spec = importlib.util.spec_from_file_location("gate", GATE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
MB = g.MB
RE_BUDGET = re.compile(r"\[prefix-cache\] on: budget (\d+)MB \((\d+) B, derived: 2 x (\d+) B max entry")
RE_PARTIAL = re.compile(r"\[admit-oom\] reclaim-on-defer \(partial\): evicted (\d+) prefix entries \(([0-9]+)MB, "
                        r"oldest first\) for ([0-9]+)MB \(([0-9]+)MB of it owed by (\d+) priming sessions")
RE_UNBUILT_SETTLE = re.compile(r"\[admission\] settle wait:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fill", type=int, default=2)
    ap.add_argument("--port", type=int, default=18179)
    ap.add_argument("--keeper-tokens", type=int, default=6000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "serve.log"
    server = g.Server(args.bin, args.model, args.draft, args.port, 8192, log)
    _popen = g.subprocess.Popen

    def popen(cmd, *a, **kw):
        env = kw.get("env")
        if env is not None and cmd == [args.bin]:
            env.pop("MEMRA_PREFIX_CACHE_MB", None)  # a derived budget
            # Fewer resident sessions in the budget's session peak (4 interactive + 1 + 1), so the
            # derived budget clears its two-entry floor on the 24 GB card.
            env["MEMRA_LANE_MAX_JUDGE"] = "1"
            env["MEMRA_LANE_MAX_HARVEST"] = "1"
        return _popen(cmd, *a, **kw)

    g.subprocess.Popen = popen
    row: dict = {"bin": args.bin}
    ballast = None
    try:
        server.boot()
        g.subprocess.Popen = _popen
        text = log.read_text(errors="replace")
        m = RE_BUDGET.search(text)
        if m is None:
            g.refuse("no derived prefix budget line")
        budget, entry = int(m.group(2)), int(m.group(3))
        row.update({"budget_mb": budget // MB, "entry_mb": entry // MB})
        if budget <= 2 * entry:
            g.refuse(f"derived budget {budget // MB}MB is on its two-entry floor; no partial reclaim")
        floor = int(g.RE_FLOOR.search(text).group(1)) * MB
        # Learn A's and B's charges cold (each also publishes an entry), then fill the cache.
        prompts = {"A": g.prose(31, 1250), "B": g.prose(37, 4600)}
        learn = {}
        for name, p in prompts.items():
            start = len(log.read_text(errors="replace").splitlines())
            r = server.complete(p, 16)
            time.sleep(1.0)
            charge = g.charge_of(g.log_lines(log, start))
            if r["status"] != 200 or charge is None:
                g.refuse(f"learn {name}: status {r['status']}, charge {charge}")
            learn[name] = {"charge": charge, "prompt_tokens": r["prompt_tokens"], "text": r["text"]}
        for i in range(args.fill):
            server.complete(g.prose(200 + i, 2400), 8)
        time.sleep(1.0)
        inserts = sum(1 for l in log.read_text(errors="replace").splitlines()
                      if l.startswith("[prefix-cache] insert"))
        row.update({"learn": {k: {kk: v[kk] for kk in ("charge", "prompt_tokens")} for k, v in learn.items()},
                    "floor_mb": floor // MB, "prefix_inserts": inserts})
        # The keeper, then the placement: A fits, B fits neither reading.
        keeper: dict = {}
        kt = threading.Thread(target=lambda: keeper.update(server.complete(g.keeper_prompt(), args.keeper_tokens)))
        kt.start()
        time.sleep(6.0)
        busy = g.effective_free(server.fresh_metrics())
        reserve_est = floor + 77 * MB  # the verify-graph debt measured in the gate runs (1,653 - 1,576 MB)
        lo = learn["A"]["charge"] + reserve_est
        hi = learn["B"]["charge"] + reserve_est
        target = (lo + hi) // 2
        row.update({"busy_free_mb": busy // MB, "lo_mb": lo // MB, "hi_mb": hi // MB, "target_mb": target // MB})
        if busy < target:
            g.refuse(f"effective free {busy // MB}MB beside the keeper is under the target {target // MB}MB")
        ballast = g.Ballast(args.out / "ballast.log")
        here = busy
        for _ in range(2):
            if not ballast.set(max(0, ballast.total + here - target)):
                g.refuse("the driver could not give the ballast")
            here = g.effective_free(server.fresh_metrics())
        row["free_before_pair_mb"] = here // MB
        print(f"placed: {here // MB}MB in [{lo // MB}, {hi // MB}) (A {learn['A']['charge'] // MB}MB, "
              f"B {learn['B']['charge'] // MB}MB, reserve est {reserve_est // MB}MB, budget {budget // MB}MB, "
              f"{inserts} prefix inserts)", flush=True)
        if not kt.is_alive():
            g.refuse("the keeper finished before the pair")
        # A, then B 20 ms later: fresh text of the learned lengths, so neither hits the prefix
        # cache (a hit restores the session in `admit`, which then owns its route state).
        pair = {"A": g.prose(41, 1250), "B": g.prose(43, 4600)}
        start = len(log.read_text(errors="replace").splitlines())
        res: dict = {}

        def fire(name: str, delay: float) -> None:
            time.sleep(delay)
            res[name] = server.complete(pair[name], 16)

        ts = [threading.Thread(target=fire, args=("A", 0.0)), threading.Thread(target=fire, args=("B", 0.02))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        kt.join()
        time.sleep(1.0)
        lines = g.log_lines(log, start)
        partial = [m.groups() for l in lines if (m := RE_PARTIAL.search(l))]
        shape = g.pair_shape(lines, {r["prompt_tokens"] for r in res.values() if r["prompt_tokens"]})
        row.update({
            "partial_reclaims": [{"evicted": int(p[0]), "freed_mb": int(p[1]), "need_mb": int(p[2]),
                                  "owed_mb": int(p[3]), "priming": int(p[4])} for p in partial],
            "partial_lines": [l for l in lines if "reclaim-on-defer (partial)" in l or "reclaim booking" in l][:6],
            "flush_lines": sum(1 for l in lines if "reclaim-on-defer: evicted" in l),
            "settle_lines": sum(1 for l in lines if RE_UNBUILT_SETTLE.search(l)),
            "pair_admit_lines": shape["pair_admit_lines"],
            "routes": shape["pair_routes"],
            "both_in_one_pass": shape["both_admitted_in_one_pass"],
            "oom_lines": shape["oom_lines"],
            "answers": {n: {"status": r["status"], "prompt_tokens": r["prompt_tokens"],
                            "completion_tokens": r["completion_tokens"]} for n, r in res.items()},
            "cached_lines": [l[:200] for l in lines if l.startswith("[spec-k] model") and " cached=0 " not in l],
            "keeper": {k: keeper.get(k) for k in ("status", "completion_tokens")},
            "alive": server.proc is not None and server.proc.poll() is None,
        })
        print(json.dumps({k: row[k] for k in ("partial_reclaims", "flush_lines", "settle_lines", "pair_admit_lines",
                                              "routes", "both_in_one_pass", "oom_lines", "answers", "keeper")}),
              flush=True)
    finally:
        g.subprocess.Popen = _popen
        if ballast is not None:
            ballast.stop()
        server.stop()
        (args.out / "summary.json").write_text(json.dumps(row, indent=1) + "\n")


if __name__ == "__main__":
    main()
