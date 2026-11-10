#!/usr/bin/env python3
"""Markdown rows from one or more stop-string gate DIRs (tools/spec-stop-string-gate.py): per
speculative boot and sampler, the stop rows' completion tokens against plain decode's (or, on a
route judged on its own stream, the rows that stop on their completing token), and the close
(last visible chunk to end of response) of the stop-ended and non-stop rows.
usage: summary.py LABEL=DIR [LABEL=DIR ...]"""
import json
import pathlib
import statistics
import sys

STOP = ("stop", "span", "two", "budget", "turn1")
NONSTOP = ("free", "eos", "probe", "reject", "turnfree", "turn2")
print("| run | boot | sampler | stop rows equal to the reference | completion tokens (stop rows) | reference | "
      "close ms, stop rows median / max | close ms, non-stop median / max |")
print("|---|---|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    d = pathlib.Path(d)
    rows = [json.loads(line) for line in (d / "runs.jsonl").read_text().splitlines() if line.strip()]
    boots = [json.loads(line) for line in (d / "boots.jsonl").read_text().splitlines() if line.strip()]
    ref = {r["id"]: r for r in rows if r["case"] == "ref"}
    for b in boots:
        if b["case"] == "ref":
            continue
        for regime in ("t1", "greedy"):
            rr = [r for r in rows if r["case"] == b["case"] and r["regime"] == regime]
            stops = [r for r in rr if r["kind"] in STOP]
            own = regime in b.get("self_ref", [])
            if own:
                by = {r["id"]: r for r in rr}
                txt = lambda rid: (d / "texts" / f"{b['case']}-{rid}.txt").read_text()  # noqa: E731
                eq = 0
                plain = lambda x: bool(x) and (x.get("plain_by_admission") or (  # noqa: E731
                    x.get("cached") and x.get("acceptance") is None
                    and (x.get("error") or "").startswith("the request did not speculate")))
                for r in list(stops):
                    n = r["completion_tokens"]
                    if r["finish"] == "stop":
                        at, lo = by.get(f"{r['id']}-at-{n}"), by.get(f"{r['id']}-at-{n - 1}")
                        if plain(r) or plain(at) or plain(lo):
                            stops.remove(r)  # its reference rows ran on plain decode: not judged
                            continue
                        if at and (lo or n == 1):
                            full = txt(at["id"])
                            ok = any(x in full for x in r["stop"]) and not (lo and any(x in txt(lo["id"]) for x in r["stop"]))
                            eq += ok
                    else:
                        eq += r["finish"] == "length" and n == r["max_tokens"]
                want = "own stream"
            else:
                eq = sum((r["sha"], r["finish"], r["completion_tokens"]) == (ref[r["id"]]["sha"], ref[r["id"]]["finish"],
                         ref[r["id"]]["completion_tokens"]) for r in stops if r["id"] in ref)
                want = ", ".join(str(ref.get(r["id"], {}).get("completion_tokens")) for r in stops)
            sc = [r["close_ms"] for r in stops if r["finish"] == "stop" and r["close_ms"] is not None]
            nc = [r["close_ms"] for r in rr if r["kind"] in NONSTOP and r["close_ms"] is not None]
            f = lambda xs: f"{statistics.median(xs):.0f} / {max(xs):.0f}" if xs else "-"  # noqa: E731
            print(f"| {label} | {b['case']} ({b['route']}) | {regime} | {eq}/{len(stops)} | "
                  f"{', '.join(str(r['completion_tokens']) for r in stops)} | {want} | {f(sc)} | {f(nc)} |")
