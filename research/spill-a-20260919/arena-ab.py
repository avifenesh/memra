#!/usr/bin/env python3
"""Day-15 arena reserve A/B replay (memra#385, research/spill-a-20260919/DAY15.md): recompute the
pre-registered rule from the harness receipt and check it against the line the harness printed.

Reads `<cell-dir>/receipt.json` (tools/pinned-host-reserve-bench.py --cell), prints the per-pair table,
per-order and pooled medians with N, the correctness pass per arm (byte exactness, driver flags, the THP
backing fraction), the regime from the collector's 250 ms sampler (`command.gpu.csv` in the cell dir or
its highest `-retryN` sibling), the memory state before every allocation, then the rule clauses and the
verdict. Exit 0 when the replayed verdict equals the harness's `rule_line` verdict AND the collector's
mirrored `command.log` carries the same line; 1 otherwise. Integrity and arithmetic only: one card, one
window, executed-not-qualified.
usage: arena-ab.py <cell-dir> [--floor 1.10] [--hugepage-min-fraction 0.9]
"""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

WC = 0x04
PORTABLE = 0x01


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def fmt(xs):
    return f"median {med(xs):.3f} ms (N={len(xs)}, min {min(xs):.3f}, max {max(xs):.3f})" if xs else "none"


def captured_dir(root):
    base = root.parent / re.sub(r"-retry\d+$", "", root.name)
    cands = [d for d in root.parent.glob(f"{base.name}*") if d.is_dir()
             and re.fullmatch(rf"{re.escape(base.name)}(-retry\d+)?", d.name) and (d / "CELL.jsonl").exists()]
    if not cands:
        return root
    return sorted(cands, key=lambda d: int(d.name.rsplit("retry", 1)[1]) if "retry" in d.name else -1)[-1]


def regime(cell):
    p = cell / "command.gpu.csv"
    if not p.exists():
        return "regime: no collector sampler in the cell dir"
    rows = list(csv.DictReader(p.open()))
    if not rows:
        return "regime: sampler empty"
    keys = list(rows[0].keys())
    def col(name):
        k = next((k for k in keys if name in k), None)
        vals = []
        for r in rows:
            try:
                vals.append(float(str(r[k]).split()[0]))
            except (TypeError, ValueError, AttributeError):
                pass
        return vals
    temp, power, limit, sm = col("temperature"), col("power.draw"), col("power.limit"), col("clocks")
    parts = [f"regime: {len(rows)} samples at 250 ms"]
    if temp:
        parts.append(f"temp {min(temp):.0f} to {max(temp):.0f} C")
    if power:
        parts.append(f"power draw {min(power):.1f} to {max(power):.1f} W")
    if limit:
        parts.append(f"limit {max(limit):.0f} W")
    if sm:
        parts.append(f"SM {min(sm):.0f} to {max(sm):.0f} MHz")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell", type=Path)
    ap.add_argument("--floor", type=float, default=1.10)
    ap.add_argument("--hugepage-min-fraction", type=float, default=0.9)
    args = ap.parse_args()
    root = args.cell.resolve()
    receipt = json.loads((root / "receipt.json").read_text())
    a, b = receipt["arms"]
    rows = receipt["rows"]
    print(f"cell {receipt.get('cell')}: arms {a} against {b}, bytes {receipt['bytes']} ({receipt['bytes'] / (1 << 30):.2f} GiB), "
          f"basis {receipt['basis']}, pairs per order {receipt['pairs_per_order']}, chunks {receipt.get('chunks')}")
    print(f"host: cpus {receipt.get('cpus')}, MemTotal {receipt['mem_total'] / (1 << 30):.1f} GiB, at start MemFree "
          f"{receipt['mem_free_at_start'] / (1 << 30):.1f} GiB, MemAvailable {receipt['mem_available_at_start'] / (1 << 30):.1f} GiB, "
          f"HugePages_Total {receipt.get('hugepages_total')}, THP {receipt.get('thp_mode')}, AnonHugePages at start "
          f"{receipt.get('anon_huge_at_start', 0) / (1 << 20):.0f} MiB, loadavg {receipt['loadavg_at_start']} -> {receipt.get('loadavg_at_end')}")
    print(f"gpu at start: {receipt['gpu_at_start']}; at end: {receipt.get('gpu_at_end')}")
    print(regime(captured_dir(root)))
    print("\ncorrectness pass (per arm, full size, before the timed pairs):")
    rt = {x["arm"]: x for x in receipt.get("roundtrips", [])}
    for arm in (a, b):
        x = rt.get(arm)
        if not x:
            print(f"  {arm}: NO ROUNDTRIP")
            continue
        flags = x.get("driver_flags", [])
        print(f"  {arm}: byte_exact={x.get('byte_exact')} blocks={x.get('blocks')} mismatched={x.get('mismatched_blocks')} "
              f"driver_flags={sorted({(f['rc'], f['flags']) for f in flags})} wc_bit_absent={x.get('wc_bit_absent')} "
              f"reserve_ms={x['reserve']['alloc_ms']:.1f} (not pooled) fill {x.get('fill_ms', 0):.0f} ms, H2D {x.get('h2d_ms', 0):.0f} ms "
              f"({x.get('h2d_gib_per_s', 0):.1f} GiB/s), wipe {x.get('wipe_ms', 0):.0f} ms, D2H {x.get('d2h_ms', 0):.0f} ms "
              f"({x.get('d2h_gib_per_s', 0):.1f} GiB/s), compare {x.get('compare_ms', 0):.0f} ms"
              + (f", anon_huge_fraction={x['reserve'].get('anon_huge_fraction')}" if arm == "thp" else ""))
    print("\nper pair (reserve ms):")
    pairs = {}
    for r in rows:
        pairs.setdefault((r["order"], r["pair"]), {})[r["arm"]] = r
    below = 0
    complete = []
    for k in sorted(pairs):
        p = pairs[k]
        if a in p and b in p and p[a]["ok"] and p[b]["ok"]:
            complete.append(p)
            lt = p[b]["alloc_ms"] < p[a]["alloc_ms"]
            below += lt
            extra = ""
            if b == "thp":
                extra = f"  register {p[b]['register_ms']:.3f}  huge {p[b].get('anon_huge_fraction'):.4f}"
            if b == "chunked":
                extra = f"  chunk completions {p[b]['chunk_alloc_ms']}"
            print(f"  {k[0]} pair {k[1] + 1}: {a} {p[a]['alloc_ms']:.3f}  {b} {p[b]['alloc_ms']:.3f}  "
                  f"{'cand<single' if lt else 'cand>=single'}  MemFree before {p[a]['mem_free_before'] / (1 << 30):.1f}/{p[b]['mem_free_before'] / (1 << 30):.1f} GiB{extra}")
        else:
            print(f"  {k[0]} pair {k[1] + 1}: INCOMPLETE {sorted(p)}")
    n = len(complete)
    medians = {}
    print("\nper order and pooled:")
    for arm in (a, b):
        for order in ("AB", "BA"):
            xs = [r["alloc_ms"] for r in rows if r["arm"] == arm and r["order"] == order and r["ok"]]
            medians[(arm, order)] = med(xs)
            print(f"  {arm} {order}: {fmt(xs)}")
        xs = [r["alloc_ms"] for r in rows if r["arm"] == arm and r["ok"]]
        medians[(arm, "pooled")] = med(xs)
        gib = [(receipt["bytes"] / (1 << 30)) / (x / 1e3) for x in xs]
        print(f"  {arm} pooled: {fmt(xs)}; {med(gib):.2f} GiB/s; free median {med([r['free_ms'] for r in rows if r['arm'] == arm and r['ok']]):.1f} ms")
    frees = [r["mem_free_before"] for r in rows]
    avails = [r["mem_available_before"] for r in rows]
    if rows:
        print(f"  MemFree before allocations: {min(frees) / (1 << 30):.1f} to {max(frees) / (1 << 30):.1f} GiB; MemAvailable "
              f"{min(avails) / (1 << 30):.1f} to {max(avails) / (1 << 30):.1f} GiB; at end MemFree {receipt.get('mem_free_at_end', 0) / (1 << 30):.1f} GiB")
    alloc_ok = all(r["ok"] for r in rows) and n == receipt["pairs_per_order"] * 2
    exact = {arm: bool(rt.get(arm, {}).get("byte_exact", False)) for arm in (a, b)}
    wc_absent = all(rt.get(arm, {}).get("wc_bit_absent", False) for arm in (a, b))
    flags_ok = all(rt.get(arm, {}).get("flags_read_ok", False) for arm in (a, b))
    portable_ok = all(rt.get(arm, {}).get("portable_bit_set", True) for arm in (a, b))
    huge_fracs = [r["anon_huge_fraction"] for r in rows if r["arm"] == "thp" and r.get("anon_huge_fraction") is not None]
    if "thp" in rt and rt["thp"].get("reserve", {}).get("anon_huge_fraction") is not None:
        huge_fracs.append(rt["thp"]["reserve"]["anon_huge_fraction"])
    huge_min_seen = min(huge_fracs) if huge_fracs else None
    huge_ok = (huge_min_seen is not None and huge_min_seen >= args.hugepage_min_fraction) if b == "thp" else None
    medians_both = all(medians[(b, o)] < medians[(a, o)] for o in ("AB", "BA"))
    ratio = medians[(a, "pooled")] / medians[(b, "pooled")] if medians[(b, "pooled")] else float("nan")
    integrity = alloc_ok and exact[a] and exact[b] and wc_absent and flags_ok and portable_ok and (huge_ok is not False)
    speed = n > 0 and below == n and medians_both
    material = ratio == ratio and ratio >= args.floor
    verdict = "wins-on-this-card" if integrity and speed and material else ("inconclusive" if integrity else "void")
    if b == "thp" and huge_ok is False:
        verdict = "void (hugepage-requested-not-granted)"
    checks = [
        (f"rule 1a: every allocation ok and freed, {receipt['pairs_per_order'] * 2} complete pairs ({n} seen)", alloc_ok),
        (f"rule 1b: roundtrip byte exact, {a}", exact[a]),
        (f"rule 1c: roundtrip byte exact, {b}", exact[b]),
        ("rule 1d: write-combined bit absent in every read-back, flags readable, PORTABLE set on cuMemHostAlloc arms", wc_absent and flags_ok and portable_ok),
    ]
    if b == "thp":
        checks.append((f"rule 1e: thp backed by huge pages at >= {args.hugepage_min_fraction} (min seen {huge_min_seen})", bool(huge_ok)))
    checks += [
        (f"rule 2a: {b} below {a} at every pair: {below}/{n}", n > 0 and below == n),
        (f"rule 2b: {b} below {a} in both orders' medians (AB {medians[(b, 'AB')]:.3f} vs {medians[(a, 'AB')]:.3f}; BA {medians[(b, 'BA')]:.3f} vs {medians[(a, 'BA')]:.3f})", medians_both),
        (f"rule 3: pooled ratio {a}/{b} = {ratio:.4f} >= floor {args.floor:.2f}", material),
    ]
    print("\nrule (pre-registered, DAY15.md):")
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"candidate {b} on this card: {verdict.upper()}")
    harness_line = receipt.get("rule_line", "")
    m = re.search(r"candidate_arm=(.+)$", harness_line)
    harness_verdict = m.group(1).strip() if m else None
    agree = harness_verdict == verdict
    print(f"{'ok  ' if agree else 'FAIL'} replay agrees with the harness's verdict: {harness_verdict}")
    log = captured_dir(root) / "command.log"
    in_log = log.exists() and harness_line and harness_line in log.read_text(errors="replace")
    print(f"{'ok  ' if in_log else 'FAIL'} the collector's mirrored command.log carries the harness's rule line")
    print(f"harness line, verbatim: {harness_line}")
    fails = [n_ for n_, ok in checks if not ok]
    status = "PASS" if agree and in_log else "FAIL"
    print(f"ARENA AB REPLAY: {status} (agreement {'ok' if agree else 'FAIL'}, log {'ok' if in_log else 'FAIL'}; rule clauses failed: {len(fails)} of {len(checks)}, which is the verdict restated, not an integrity failure)")
    sys.exit(0 if agree and in_log else 1)


if __name__ == "__main__":
    main()
