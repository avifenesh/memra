#!/usr/bin/env python3
"""Day-13 pinned-ab replay (research/spill-a-20260919/DAY13.md pre-registration): recompute the
per-arm per-order medians and the pre-registered rule from the mirrored collector capture of a
`tier-transfer-gate pinned-ab` cell, independently of the binary's own `PINNED-AB rule` and `RESULT`
lines, and print the telemetry regime from the collector's 250 ms sampler. Integrity and arithmetic
only: one card, one window, executed-not-qualified; the target-card cell of the decide-by review,
not a default change.
usage: wc-ab.py <cell-dir> [--markdown]
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(
    r"^PINNED-AB (?P<tag>warmup|timed) order=(?P<order>\d+) pair=(?P<pair>\d+) arm=(?P<arm>[a-z-]+) "
    r"bytes=(?P<bytes>\d+) driver_flags=(?P<flags>\d+) alloc_ms=(?P<alloc>[\d.]+) d2h_ms=(?P<d2h>[\d.]+) "
    r"engine_hash_ms=(?P<engine>[\d.]+) bind_hash_ms=(?P<bind>[\d.]+) compare_ms=(?P<compare>[\d.]+) "
    r"h2d_ms=(?P<h2d>[\d.]+) source_hash_ms=(?P<source>[\d.]+) byte_exact=(?P<exact>true|false)$"
)
PHASES = ["alloc", "d2h", "engine", "bind", "compare", "h2d", "source"]
FLAGS = {"write-combined": 4, "cached": 0}
WC, CACHED = "write-combined", "cached"


def parse(log):
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        m = LINE.match(line.strip())
        if m:
            d = m.groupdict()
            rows.append({
                "tag": d["tag"], "order": int(d["order"]), "pair": int(d["pair"]), "arm": d["arm"],
                "bytes": int(d["bytes"]), "flags": int(d["flags"]), "exact": d["exact"] == "true",
                **{p: float(d[p]) for p in PHASES},
            })
    return rows


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def regime(cell):
    sampler = cell / "command.gpu.csv"
    if not sampler.exists():
        return "telemetry: command.gpu.csv absent"
    rows = list(csv.DictReader(sampler.open(errors="replace"), skipinitialspace=True))

    def col(name):
        out = []
        for r in rows:
            v = r.get(name)
            if v is None:
                continue
            try:
                out.append(float(v.split()[0]))
            except ValueError:
                pass
        return out

    temps, power, sm = col("temperature.gpu"), col("power.draw [W]"), col("clocks.current.sm [MHz]")
    limit = sorted({r.get("power.limit [W]", "").strip() for r in rows})
    return (f"telemetry: {len(rows)} samples at 250 ms; temperature {min(temps):.0f} to {max(temps):.0f} C; "
            f"power draw {min(power):.0f} to {max(power):.0f} W (limit {', '.join(limit)}); "
            f"SM clock {min(sm):.0f} to {max(sm):.0f} MHz")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    markdown = "--markdown" in sys.argv
    cell = Path(args[0]) if args else Path(__file__).resolve().parent / "pro-single-day13" / "pinned-ab-160m"
    log = cell / "command.log"
    rows = parse(log)
    timed = [r for r in rows if r["tag"] == "timed"]
    warm = [r for r in rows if r["tag"] == "warmup"]
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")

    orders = sorted({r["order"] for r in timed})
    pairs = sorted({r["pair"] for r in timed})
    check("two orders", orders == [1, 2], str(orders))
    check("warm-up: one per arm", sorted(r["arm"] for r in warm) == sorted(FLAGS), str([r["arm"] for r in warm]))
    n = len(pairs)
    check("N >= 5 pairs per order", n >= 5 and all(
        sum(1 for r in timed if r["order"] == o and r["arm"] == a) == n for o in orders for a in FLAGS), f"N={n}")
    check("byte_exact in every roundtrip (warm-ups included)", all(r["exact"] for r in rows), f"{sum(r['exact'] for r in rows)}/{len(rows)}")
    # The driver reports DEVICEMAP (2) on every pinned allocation of a UVA platform: 6 and 2 on the target card.
    check("driver flags carry the arm's write-combined bit in every roundtrip", all(r["flags"] & 4 == FLAGS[r["arm"]] for r in rows),
          str(sorted({(r["arm"], r["flags"]) for r in rows})))
    check("one buffer size", len({r["bytes"] for r in rows}) == 1, str({r["bytes"] for r in rows}))

    def at(order, arm, pair, phase):
        return next(r[phase] for r in timed if r["order"] == order and r["arm"] == arm and r["pair"] == pair)

    def per_pair(phase, cmp):
        hits = total = 0
        for o in orders:
            for p in pairs:
                total += 1
                hits += cmp(at(o, CACHED, p, phase), at(o, WC, p, phase))
        return hits, total

    bind = per_pair("bind", lambda c, w: c < w)
    engine = per_pair("engine", lambda c, w: c < w)
    d2h_nr = per_pair("d2h", lambda c, w: c <= w)
    d2h_strict = per_pair("d2h", lambda c, w: c < w)
    h2d_nr = per_pair("h2d", lambda c, w: c <= w)
    medians = {}
    for phase in PHASES:
        for arm in FLAGS:
            per_order = {o: [r[phase] for r in timed if r["order"] == o and r["arm"] == arm] for o in orders}
            pooled = [x for o in orders for x in per_order[o]]
            medians[(phase, arm)] = {"o1": med(per_order.get(1, [])), "o2": med(per_order.get(2, [])),
                                     "pooled": med(pooled), "n": len(pooled), "min": min(pooled), "max": max(pooled)}
    medians_win = all(
        medians[("bind", CACHED)][k] < medians[("bind", WC)][k] and medians[("d2h", CACHED)][k] <= medians[("d2h", WC)][k]
        for k in ("o1", "o2"))
    check("rule 3: cached bind hash below write-combined at every pair", bind[0] == bind[1], f"{bind[0]}/{bind[1]}")
    check("rule 4: cached D2H not above write-combined at every pair", d2h_nr[0] == d2h_nr[1], f"{d2h_nr[0]}/{d2h_nr[1]}")
    check("rule 5: medians in both orders (bind below, D2H not above)", medians_win)
    exact_all = all(r["exact"] for r in rows)
    flags_ok = all(r["flags"] & 4 == FLAGS[r["arm"]] for r in rows)
    wins = exact_all and flags_ok and bind[0] == bind[1] and d2h_nr[0] == d2h_nr[1] and medians_win
    strict = wins and d2h_strict[0] == d2h_strict[1]
    print(f"per pair: bind {bind[0]}/{bind[1]} cached<wc; engine {engine[0]}/{engine[1]} cached<wc; "
          f"d2h {d2h_nr[0]}/{d2h_nr[1]} cached<=wc, {d2h_strict[0]}/{d2h_strict[1]} cached<wc; h2d {h2d_nr[0]}/{h2d_nr[1]} cached<=wc")
    print(f"cached arm on this card: {'WINS' if wins else 'INCONCLUSIVE'} (pre-registered rule); "
          f"strict D2H reading: {'WINS' if strict else 'INCONCLUSIVE'}")
    # Agreement with the binary's own rule line and RESULT.
    text = log.read_text(errors="replace")
    rule_line = next((l for l in text.splitlines() if l.startswith("PINNED-AB rule ")), "")
    check("binary rule line present", bool(rule_line))
    if rule_line:
        binary_verdict = re.search(r" cached_arm=(\S+)", rule_line).group(1)
        check("replay agrees with the binary's verdict", (binary_verdict == "wins-on-this-card") == wins, binary_verdict)
    result_lines = [l[7:] for l in text.splitlines() if l.startswith("RESULT ")]
    check("exactly one RESULT line", len(result_lines) == 1, str(len(result_lines)))
    if len(result_lines) == 1:
        result = json.loads(result_lines[0])
        check("RESULT medians agree with the replay (bind, cached, pooled)",
              abs(result["medians_ms"][f"bind_hash_ms.{CACHED}"]["pooled"] - medians[("bind", CACHED)]["pooled"]) < 0.0015)
        check("RESULT sample count", len(result["samples"]) == len(timed), f"{len(result['samples'])} vs {len(timed)}")
    capture = cell / "command.capture.json"
    if capture.exists():
        cap = json.loads(capture.read_text())
        check("collector status executed-not-qualified, exit 0", cap["status"] == "executed-not-qualified" and cap["exit_code"] == 0,
              f"{cap['status']} exit {cap['exit_code']}")
    print(regime(cell))
    if markdown:
        names = {"alloc": "alloc", "d2h": "D2H", "engine": "engine hash", "bind": "bind hash (the host-read)",
                 "compare": "byte compare", "h2d": "H2D", "source": "H2D source hash"}
        print()
        print("| Phase (ms) | WC order 1 (N=%d) | WC order 2 (N=%d) | cached order 1 (N=%d) | cached order 2 (N=%d) | pooled WC (N=%d) | pooled cached (N=%d) |" % (n, n, n, n, 2 * n, 2 * n))
        print("|---|---|---|---|---|---|---|")
        for phase in PHASES:
            w, c = medians[(phase, WC)], medians[(phase, CACHED)]
            print(f"| {names[phase]} | {w['o1']:.2f} | {w['o2']:.2f} | {c['o1']:.2f} | {c['o2']:.2f} | **{w['pooled']:.2f}** ({w['min']:.2f} to {w['max']:.2f}) | **{c['pooled']:.2f}** ({c['min']:.2f} to {c['max']:.2f}) |")
        print()
        print("| Order | Pair | WC bind hash | cached bind hash | WC D2H | cached D2H | WC engine hash | cached engine hash | WC H2D | cached H2D |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for o in orders:
            for p in pairs:
                print(f"| {o} | {p} | {at(o, WC, p, 'bind'):.2f} | {at(o, CACHED, p, 'bind'):.2f} | {at(o, WC, p, 'd2h'):.3f} | {at(o, CACHED, p, 'd2h'):.3f} | {at(o, WC, p, 'engine'):.2f} | {at(o, CACHED, p, 'engine'):.2f} | {at(o, WC, p, 'h2d'):.3f} | {at(o, CACHED, p, 'h2d'):.3f} |")
    failed = [c for c in checks if not c[1]]
    print(f"WC AB REPLAY: {'PASS' if not failed else 'FAIL'} ({len(checks)} checks, {len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
