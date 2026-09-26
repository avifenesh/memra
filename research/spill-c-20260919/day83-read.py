#!/usr/bin/env python3
"""Day 83 reader for cell `split20` (research/spill-c-20260919/DAY83.md section 1, registered before this script): the
door's CPU work per generated token, split by term from its two log-only clocks, per arm (medians over the arm's runs,
each run's phase deltas: generate minus gate, window minus warm), then the largest leaf of the door-only CPU at I20 and
each leaf's change from I15 to I20 on this host. Also reads earlier cells' receipts for any arm named on the command line
(a leaf needing a clock the arm did not carry prints `-`).

usage: day83-read.py <ev-dir> --arms i15s,i20s,i20c [--rig <rig>] [--check]
  --check: the cell's integrity as registered (every run exit 0 and MATCH, one tape, one host demand sequence).
"""
import hashlib
import re
import statistics
import sys
from pathlib import Path

DISPATCH = re.compile(r"\[moe-cache\] dispatch-clock phase=(\w+) (.*)")
STAGES = re.compile(r"\[experts-via-tier\] stages phase=(\w+) (.*)")
SLOT = re.compile(r" slot=\d+")
PAIR = re.compile(r"(\w+)=(\d+)")
TOKENS = 32


def sections(text):
    """The stage line's sections, keyed engine / owner / fill / bank / proc (a key repeats across sections)."""
    out = {}
    for n, part in enumerate(text.split(" | ")):
        head = part.split(" ", 1)[0]
        name = "engine" if n == 0 else head if head in ("owner", "bank", "proc") else "fill"
        out[name] = {k: int(v) for k, v in PAIR.findall(part)}
    return out


def parse(log):
    text = log.read_text(errors="replace")
    lines = [line.partition("\t")[2] if "\t" in line else line for line in text.splitlines()]
    dispatch, stages = {}, {}
    for line in lines:
        m = DISPATCH.search(line)
        if m:
            dispatch[m.group(1)] = {k: int(v) for k, v in PAIR.findall(m.group(2))}
        m = STAGES.search(line)
        if m:
            stages[m.group(1)] = sections(m.group(2))
    tape = next((line for line in lines if line.startswith("tokens: ")), "")
    trace = [SLOT.sub("", line) for line in lines if "[expert-host-slru] key=" in line]
    match = any("MATCH" in line and "argmax" in line for line in lines)
    return {"dispatch": dispatch, "stages": stages, "tape": tape, "match": match,
            "trace": hashlib.sha256("\n".join(trace).encode()).hexdigest()[:16]}


def delta(phases, a, b, key, section=None):
    try:
        if section is None:
            return phases[b][key] - phases[a][key]
        return phases[b][section][key] - phases[a][section][key]
    except KeyError:
        return None


def terms(run, a, b):
    """Every term in ns over the phase, and the counts; None where the run's clocks lack it."""
    d, s = run["dispatch"], run["stages"]
    t = {}
    for k in ("pf_demand_ns", "pf_resident_ns", "pf_retire_ns", "pf_stage_ns", "pf_reserve_ns", "prefetch_ns",
              "dispatch_ns", "prefetch_issued"):
        t[k] = delta(d, a, b, k)
    for k in ("demand_ns", "finish_ns", "retire_ns", "wait_ns", "validate_ns", "enqueue_ns", "prefetches", "gpu_misses"):
        t["eng_" + k] = delta(s, a, b, k, "engine")
    for k in ("inner_demand_ns", "trace_ns"):
        t["own_" + k] = delta(s, a, b, k, "owner")
    for k in ("stages", "stage_ns", "stage_lookup_ns", "stage_cache_ns", "stage_charge_ns", "publish_ns",
              "publish_output_ns", "publish_policy_ns", "retire_ns", "host_use_ns", "retire_only_ns", "ack_ns",
              "ack_release_ns", "collect_ns"):
        t["bank_" + k] = delta(s, a, b, k, "bank")

    def sub(x, *ys):
        return None if x is None or any(y is None for y in ys) else x - sum(ys)

    t["stage_rest_ns"] = sub(t["bank_stage_ns"], t["bank_stage_lookup_ns"], t["bank_stage_cache_ns"],
                             t["bank_stage_charge_ns"])
    t["publish_rest_ns"] = sub(t["bank_publish_ns"], t["bank_publish_output_ns"], t["bank_publish_policy_ns"])
    # The dispatch adapter's own part of the owner's demand (validated ids and their clones, the batch, the
    # progress loop), outside the bank's stage and publish.
    t["dispatch_inner_ns"] = sub(t["own_inner_demand_ns"], t["bank_stage_ns"], t["bank_publish_ns"])
    # Everything the engine brackets as a demand (the prefetch groups' and the dispatch-time singles') outside the
    # owner's inner demand and trace: the proxy's registry entry, identity checks and pending insert, and the traced
    # adapter's pre-demand lookups.
    both = None if t["pf_demand_ns"] is None or t["eng_demand_ns"] is None else t["pf_demand_ns"] + t["eng_demand_ns"]
    t["outer_ns"] = sub(both, t["own_inner_demand_ns"], t["own_trace_ns"])
    # The engine's retire (every `retire_banked` call, the prefetch path's and the dispatch path's, the scope of the
    # bank's retire counters) outside the bank's finish and any in-flight-bound wait: the in-flight walk, the event
    # queries, the proxy's entry.
    t["retire_outer_ns"] = sub(t["eng_retire_ns"], t["eng_wait_ns"], t["bank_retire_ns"], t["bank_collect_ns"])
    return t


# The door-only CPU's leaves (REF runs none of these; pf_stage, the copy enqueue both programs run, is beside them).
LEAVES = ["outer_ns", "dispatch_inner_ns", "own_trace_ns", "bank_stage_lookup_ns", "bank_stage_cache_ns",
          "bank_stage_charge_ns", "stage_rest_ns", "bank_publish_output_ns", "bank_publish_policy_ns",
          "publish_rest_ns", "pf_resident_ns", "bank_host_use_ns", "bank_retire_only_ns", "bank_ack_ns",
          "bank_collect_ns", "retire_outer_ns"]
BESIDE = ["pf_demand_ns", "pf_resident_ns", "pf_retire_ns", "pf_stage_ns", "own_inner_demand_ns", "bank_stage_ns",
          "bank_publish_ns", "bank_retire_ns", "eng_demand_ns"]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def iqr(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 4:
        return None
    q = statistics.quantiles(xs, n=4)
    return q[2] - q[0]


def us_tok(ns):
    return "-" if ns is None else f"{ns / TOKENS / 1000:.1f}"


def main():
    ev = Path(sys.argv[1])
    arms = sys.argv[sys.argv.index("--arms") + 1].split(",")
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    runs = {arm: [] for arm in arms}
    fails = []
    for arm in arms:
        for log in sorted(ev.glob(f"o[12]-{arm}-r*.log")):
            run = parse(log)
            exit_file = log.with_suffix(".exit")
            run["exit"] = exit_file.read_text().strip() if exit_file.exists() else "?"
            run["label"] = log.stem
            runs[arm].append(run)
    if "--check" in sys.argv:
        every = [r for arm in arms for r in runs[arm]]
        for r in every:
            if r["exit"] != "0" or not r["match"] or not r["tape"]:
                fails.append(f"{r['label']}: exit={r['exit']} match={r['match']} tape={bool(r['tape'])}")
        if len({r["tape"] for r in every}) != 1:
            fails.append("tapes differ")
        if len({r["trace"] for r in every}) != 1:
            fails.append("host demand sequences differ: " + ",".join(sorted({r["trace"] for r in every})))
        counts = {arm: len(runs[arm]) for arm in arms}
        if any(n != 10 for n in counts.values()):
            fails.append(f"runs per arm {counts}")
        print(f"DAY83 SPLIT CHECKS rig={rig} runs={len(every)} integrity={'ok' if not fails else 'FAIL'}"
              + ("" if not fails else " " + "; ".join(fails)))
    table = {}
    for phase, a, b in (("generate", "gate", "generate"), ("window", "warm", "window")):
        for arm in arms:
            per = [terms(r, a, b) for r in runs[arm]]
            if not per:
                continue
            m = {k: med([p[k] for p in per]) for k in per[0]}
            q = {k: iqr([p[k] for p in per]) for k in per[0]}
            table[(arm, phase)] = (m, q)
            tickets, blocks = m["bank_stages"], m["prefetch_issued"] or m["eng_prefetches"]
            print(f"DAY83 SPLIT {arm} {phase} N={len(per)} tickets={tickets} prefetched_blocks={blocks} "
                  f"dispatch_misses={None if m['eng_gpu_misses'] is None or m['eng_prefetches'] is None else m['eng_gpu_misses'] - m['eng_prefetches']}"
                  " (us per generated token, medians)")
            print(f"DAY83 SPLIT {arm} {phase} leaves: " + " ".join(f"{k[:-3]}={us_tok(m[k])}" for k in LEAVES))
            print(f"DAY83 SPLIT {arm} {phase} beside: " + " ".join(f"{k[:-3]}={us_tok(m[k])}" for k in BESIDE))
    if "--check" in sys.argv and not fails and ("i20s", "generate") in table:
        m, q = table[("i20s", "generate")]
        ranked = sorted(((m[k], k) for k in LEAVES if m[k] is not None), reverse=True)
        (v1, k1), (v2, k2) = ranked[0], ranked[1]
        overlap = v1 - (q[k1] or 0) <= v2 + (q[k2] or 0)
        total = sum(v for v, _ in ranked)
        print(f"DAY83 LARGEST rig={rig} i20 generate: {k1[:-3]}={us_tok(v1)} us per token ({v1 / total:.0%} of the "
              f"door-only leaves' {us_tok(total)}); second {k2[:-3]}={us_tok(v2)}"
              + (" -> both named (their IQRs overlap)" if overlap else " -> named"))
        if ("i20c", "generate") in table:
            c = table[("i20c", "generate")][0]
            print(f"DAY83 CLOCK COST rig={rig} generate pf_demand i20s={us_tok(m['pf_demand_ns'])} "
                  f"i20c={us_tok(c['pf_demand_ns'])} us per token (the stage clock's own brackets, deciding nothing)")
        if ("i15s", "generate") in table:
            o = table[("i15s", "generate")][0]
            print(f"DAY83 I15_TO_I20 rig={rig} generate (i20s minus i15s, us per token, deciding nothing): "
                  + " ".join(f"{k[:-3]}={'-' if m[k] is None or o[k] is None else f'{(m[k] - o[k]) / TOKENS / 1000:+.1f}'}"
                             for k in LEAVES))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
