#!/usr/bin/env python3
"""agentic_tab.py CAMPAIGN_DIR: memra#1221 part 2's agentic and fleet replay boots, one row each.

Per boot (NN-ARM directories): turns ok, tokens, TTFT median, aggregate tok/s (agent_turns.py's
cells.jsonl) or the fleet replay's summary (replay.json); the prefix cache's loan reconcile lines
(`[prefix-loan] reconcile`, MEMRA_TICK_TRACE=1): how many, and whether every one reads charged ==
real; adoptions and compactions at the end; refused rewinds; the leak check (`[probe-trim]`: the
boot baseline, and after the trim the pool's used bytes and the cache's charged bytes). Then the
per-request texts across boots: agent_turns rows by request index (text_sha), fleet ok lines by
request number (text_sha)."""
import gzip
import json
import pathlib
import re
import statistics
import sys

d = pathlib.Path(sys.argv[1])
rec = re.compile(r"\[prefix-loan\] reconcile why=(\S+) books=(\d+) lent_live=(\d+) adopted=([\d.]+)MB "
                 r"charged=([\d.]+)MB real=([\d.]+)MB adoptions=(\d+) compactions=(\d+)")
base = re.compile(r"\[probe-trim\] baseline pool_reserved=(\d+) pool_used=(\d+) prefix_total_bytes=(\d+)")
after = re.compile(r"\[probe-trim\] after trim pool_reserved=(\d+) pool_used=(\d+) prefix_total_bytes=(\d+) prefix_entries=(\d+)")
fleet_ok = re.compile(r"\[fleet-replay\] ok request=(\d+) .* cached=(\d+) .* text_sha=([0-9a-f]+)")
texts = {}
print("| boot | turns ok | tokens | TTFT median s | agg tok/s | reconcile lines (charged == real) | "
      "adoptions / compactions at the end | max books | refused rewinds | pool used: baseline -> after trim (MiB) | "
      "charged after trim | out of memory |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name)):
    log = c / "server.log.gz"
    text = gzip.open(log, "rt", errors="replace").read() if log.exists() else (c / "server.log").read_text(errors="replace")
    recs = [m.groups() for m in rec.finditer(text)]
    eq = sum(1 for r in recs if r[4] == r[5])
    last = recs[-1] if recs else None
    b = base.search(text)
    a = after.search(text)
    oom = text.count("out of memory")
    refused = text.count("rewind refused below a lent prefix row")
    # Review round 1: rewinds that moved the session into a fresh cache (the floor kept, the
    # session kept) and breaks of the cache's total_bytes invariant (must be 0). Review round 3:
    # the line's prefix, which every form of it starts with (worker.rs LEND_REWIND_FRESH_LINE:
    # "... : the session tries a fresh cache (SITE; model M)"; R2 printed "(SITE): ... moves to").
    fresh = text.count("[spec] rewind below a lent prefix row")
    breaks = text.count("[prefix-loan] INVARIANT BROKEN")
    compacts = text.count("[prefix-loan] compact:")
    cell = None
    if (c / "cells.jsonl").exists():
        cell = json.loads((c / "cells.jsonl").read_text().splitlines()[-1])
        rows = []
        for f in c.glob("*.requests.jsonl"):
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        texts[c.name] = {r["k"]: r["text_sha"] for r in rows if r.get("ok")}
        ok, toks, ttft, tps = f'{cell["ok"]}/{cell["turns"]}', cell["tokens"], cell["ttft_median_s"], cell["aggregate_gen_tps"]
    else:
        oks = [m.groups() for m in fleet_ok.finditer((c / "replay.log").read_text(errors="replace"))] if (c / "replay.log").exists() else []
        texts[c.name] = {int(n): sha for n, _, sha in oks}
        rj = {}
        try:
            rj = json.loads((c / "replay.json").read_text())
        except Exception:  # noqa: BLE001
            pass
        ok, toks, ttft, tps = f'{rj.get("requests_ok")}/{rj.get("requests_attempted")}', rj.get("completion_tokens"), "-", "-"
    mib = lambda x: int(x) >> 20  # noqa: E731
    print(f"| {c.name} | {ok} | {toks} | {ttft} | {tps} | {len(recs)} ({eq}) | "
          f"{(last[6] + ' / ' + last[7]) if last else '-'} | {max((int(r[1]) for r in recs), default=0)} | {refused} (fresh {fresh}, invariant breaks {breaks}, compact lines {compacts}) | "
          f"{(str(mib(b.group(2))) if b else '?')} -> {(str(mib(a.group(2))) if a else '?')} | "
          f"{(a.group(3) if a else '?')} | {oom} |")
names = sorted(texts)
if len(names) > 1:
    print()
    ref = names[0]
    for n in names[1:]:
        common = set(texts[ref]) & set(texts[n])
        same = sum(1 for k in common if texts[ref][k] == texts[n][k])
        diff = sorted(k for k in common if texts[ref][k] != texts[n][k])
        print(f"texts {n} vs {ref}: {same}/{len(common)} requests equal" + (f"; differ at {diff[:12]}" if diff else ""))
