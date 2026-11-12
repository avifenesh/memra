#!/usr/bin/env python3
"""memra#1211: where a cell's ticks spend their time, from the tick trace (MEMRA_TICK_TRACE=1).
Splits a server log into cells at the long request's admission (a `[spec-k]` line with a prompt
over 40 tokens) and, for the cell numbered CELL (0-based), sums phase times over its ticks that
did any work: spec (phase a), prefill (phase b), decode (phase c), formation waits; and the
ticks' count and mean wall share. Only the ticks inside the cell's arrival window can be told
apart from the lead and tail by their content, so it reports ticks that primed or admitted.
usage: tickparts.py SERVER_LOG CELL"""
import re
import statistics
import sys

def read_log(path):
    """A server log, plain or gzip-compressed (the receipts keep them as .log.gz)."""
    import gzip
    import os
    path = str(path)
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace").read()
    return open(path, errors="replace").read()


SPECK = re.compile(r"\[spec-k\] model=\"q\" .* prompt=(\d+)")
lines = read_log(sys.argv[1]).splitlines()
cells, cur = [], None
for line in lines:
    m = SPECK.search(line)
    if m and int(m.group(1)) > 40:
        cur = []
        cells.append(cur)
        continue
    if cur is not None:
        cur.append(line)
cell = cells[int(sys.argv[2])]
F = {k: re.compile(rf"\b{k}=([0-9.]+)") for k in ("spec_ms", "prefill_ms", "decode_ms", "form_wait_ms", "act", "prefill_single_calls", "prefill_batch_calls")}
rows = []
for line in cell:
    if not line.startswith("[tick]"):
        continue
    v = {k: float(r.search(line).group(1)) if r.search(line) else 0.0 for k, r in F.items()}
    rows.append(v)
prime = [r for r in rows if r["prefill_single_calls"] + r["prefill_batch_calls"] > 0]
busy = [r for r in rows if r["act"] >= 2]
def summ(rs, name):
    if not rs:
        print(f"{name}: none")
        return
    tot = [r["spec_ms"] + r["prefill_ms"] + r["decode_ms"] + r["form_wait_ms"] for r in rs]
    print(f"{name}: {len(rs)} ticks; mean ms spec {statistics.fmean(r['spec_ms'] for r in rs):.1f}, "
          f"prefill {statistics.fmean(r['prefill_ms'] for r in rs):.1f}, decode {statistics.fmean(r['decode_ms'] for r in rs):.1f}, "
          f"form wait {statistics.fmean(r['form_wait_ms'] for r in rs):.2f}; sum mean {statistics.fmean(tot):.1f}, p50 {statistics.median(tot):.1f}")
summ(rows, "all ticks")
summ(busy, "ticks with 2+ live")
summ(prime, "ticks that primed")
