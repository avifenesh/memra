#!/usr/bin/env python3
"""Day 33 reading: the hash micro-cell at the pair cell's entry size and at 160 MiB, the two-step arm, and
the day-31 pair receipts' demote lines re-read. Applies the clauses pre-registered in DAY33.md, verbatim,
and prints one `DAY33 HASH-WC VERDICT:` line. Nothing here tunes; a failed integrity check makes the cell
inadmissible and says so.

usage: day33-reading.py <day33-root> [<day31-root>]     (roots: rtx5090-day33, rtx5090-day31)
       day33-reading.py --reread <day31-root>            (the day-31 table only; no day-33 receipts needed)
"""
import csv
import json
import re
import sys
from pathlib import Path

OK = []
FAIL = []


def check(ok, text):
    (OK if ok else FAIL).append(text)
    print(("ok:   " if ok else "FAIL: ") + text)


def med(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def iqr(xs):
    s = sorted(xs)
    n = len(s)
    if n < 4:
        return float("nan")
    return med(s[(n + 1) // 2:]) - med(s[: n // 2])


def stats(xs):
    return f"median {med(xs):.1f} (N={len(xs)}, min {min(xs):.1f}, max {max(xs):.1f}, IQR {iqr(xs):.1f})"


def strip(line):
    # the cell's stamper: "HH:MM:SS.mmm\t<line>"
    return line.split("\t", 1)[1] if "\t" in line else line


def parse_micro(path, kinds, rule_prefix):
    """Return (rule_kv, passes{kind: [ms]}, extra{kind: {field: [..]}})."""
    lines = [strip(l.rstrip("\n")) for l in path.read_text(errors="replace").splitlines()]
    rules = [l for l in lines if l.startswith(rule_prefix)]
    passes = {k: [] for k in kinds}
    extra = {k: {} for k in kinds}
    for l in lines:
        m = re.match(r"pass order=(\d) pass=(\d+) kind=(\w+) ms=([\d.]+)(.*)", l)
        if m and m.group(3) in passes:
            passes[m.group(3)].append(float(m.group(4)))
            for f, v in re.findall(r"(\w+_ms)=([\d.]+)", m.group(5)):
                extra[m.group(3)].setdefault(f, []).append(float(v))
    kv = dict(re.findall(r"(\w+)=(\"[^\"]*\"|\S+)", rules[0])) if len(rules) == 1 else {}
    return rules, kv, passes, extra


def regime(cell):
    sampler = cell / "command.gpu.csv"
    if not sampler.exists():
        return "telemetry: command.gpu.csv absent", {}
    rows = list(csv.DictReader(sampler.open(errors="replace"), skipinitialspace=True))

    def col(name):
        out = []
        for r in rows:
            v = r.get(name)
            if v is None:
                continue
            try:
                out.append(float(v.split()[0]))
            except (ValueError, IndexError):
                pass
        return out

    temps, power, sm, mem = col("temperature.gpu"), col("power.draw [W]"), col("clocks.current.sm [MHz]"), col("memory.used [MiB]")
    limit = {r.get("power.limit [W]", "").strip() for r in rows}
    d = {"samples": len(rows), "temp_lo": min(temps), "temp_hi": max(temps), "power_lo": min(power), "power_max": max(power),
         "limit": ", ".join(sorted(limit)), "sm_lo": min(sm), "sm_hi": max(sm), "mem_lo": min(mem), "mem_hi": max(mem)}
    return (f"telemetry (collector, 250 ms): {len(rows)} samples; temperature {d['temp_lo']:.0f}..{d['temp_hi']:.0f} C; "
            f"power draw {d['power_lo']:.1f}..{d['power_max']:.1f} W (limit {d['limit']}); SM clock {d['sm_lo']:.0f}..{d['sm_hi']:.0f} MHz; "
            f"memory used {d['mem_lo']:.0f}..{d['mem_hi']:.0f} MiB"), d


def reread_day31(day31):
    """The day-31 pair receipts' demote lines per ON boot: `in`, `from submission to completion`, the D2H receipt
    line (present or absent; it carries no timing by format), and `in` minus completion, per demote (N=6 per
    boot, N=12 pooled). The OFF boots' `in` beside them."""
    ev = day31 / "pair" / "wc-pair" / "ev"
    print("== day-31 pair receipts re-read (rtx5090-day31/pair/wc-pair/ev/*-server.log; the logs carry no per-line timestamps)")
    on_in, on_comp, on_delta, off_in = [], [], [], []
    print("| boot | seq | `demote submitted` | `D2H receipt` (items, require) | `from submission to completion` ms | `in` ms | `in` minus completion ms |")
    print("|---|---|---|---|---|---|---|")
    for boot in ("o1-off", "o1-on", "o2-on", "o2-off"):
        log = ev / f"{boot}-server.log"
        if not log.exists():
            check(False, f"{boot}-server.log present")
            continue
        seq = None
        submitted = receipt = None
        comp = None
        for l in log.read_text(errors="replace").splitlines():
            m = re.search(r"demote submitted off the tick: (\d+) tokens, ([\d.]+)MB, ticket seq=(\d+), (\d+) items", l)
            if m:
                seq, submitted = m.group(3), f"{m.group(1)} tok, {m.group(2)} MB, {m.group(4)} items"
                receipt = comp = None
                continue
            m = re.search(r"contracts door D2H receipt: ticket issuer=\d+ seq=(\d+) epochs=\S+ items=(\d+) \(([^)]*)\) complete=(\d+) require=(\w+)", l)
            if m:
                receipt = f"seq={m.group(1)} items={m.group(2)} ({m.group(3)}) complete={m.group(4)} require={m.group(5)}"
                continue
            m = re.search(r"demote published off the tick: ticket seq=(\d+) complete after (\d+) poll\(s\), ([\d.]+)ms from submission to completion \(([^)]*)\)", l)
            if m:
                comp = float(m.group(3))
                continue
            m = re.search(r"\[prefix-host\] demote: (\d+) tokens, ([\d.]+)MB in ([\d.]+)ms", l)
            if m:
                ms = float(m.group(3))
                if boot.endswith("-on"):
                    on_in.append(ms)
                    if comp is not None:
                        on_comp.append(comp)
                        on_delta.append(ms - comp)
                    print(f"| {boot} | {seq} | {submitted} | {receipt or 'ABSENT'} | {comp if comp is not None else 'ABSENT'} | {ms} | {ms - comp:.1f} |" if comp is not None else f"| {boot} | {seq} | {submitted} | {receipt or 'ABSENT'} | ABSENT | {ms} | n/a |")
                else:
                    off_in.append(ms)
                    print(f"| {boot} | (door OFF) | n/a | n/a | n/a | {ms} | n/a |")
    check(len(on_in) == 12 and len(on_comp) == 12, f"12 ON demotes with `in` and completion figures (in={len(on_in)}, completion={len(on_comp)})")
    check(len(off_in) == 12, f"12 OFF demotes with `in` figures ({len(off_in)})")
    print(f"day31 ON  demote `in`:                          {stats(on_in)}  raw {on_in}")
    print(f"day31 ON  `from submission to completion`:      {stats(on_comp)}  raw {on_comp}")
    print(f"day31 ON  `in` minus completion:                {stats(on_delta)}  raw {[round(x, 1) for x in on_delta]}")
    print(f"day31 OFF demote `in`:                          {stats(off_in)}  raw {off_in}")
    print("day31 `D2H receipt` timing: the line carries none by format (issuer, seq, epochs, items, complete, require, checksums_sha256, retired); nothing to tabulate, nothing missing")
    return {"on_in": on_in, "on_comp": on_comp, "on_delta": on_delta, "off_in": off_in}


def main():
    args = sys.argv[1:]
    if args and args[0] == "--reread":
        reread_day31(Path(args[1]))
        return
    root = Path(args[0])
    day31 = Path(args[1]) if len(args) > 1 else root.parent / "rtx5090-day31"
    cell = root / "hashwc" / "hash-wc"
    ev = cell / "ev"
    print(f"== day-33 hash-wc cell: {cell}")
    # The collector's own lock.json carries `acquired`; the cell's ev/LOCK.json (tools/tier-lock-proof.py on the
    # inherited fd) carries the lock path, owner, device and inode. The clause: acquired on /tmp/memra-5090.lock,
    # and the proof taken inside the cell names the same lock file (same device and inode).
    coll = json.loads((cell / "lock.json").read_text()) if (cell / "lock.json").exists() else {}
    proof = json.loads((ev / "LOCK.json").read_text()) if (ev / "LOCK.json").exists() else {}
    same = proof.get("lock") == coll.get("lock") == "/tmp/memra-5090.lock" and (proof.get("device"), proof.get("inode")) == (coll.get("device"), coll.get("inode"))
    check(coll.get("acquired") is True and same, f"lock proof: collector {coll.get('lock')} acquired={coll.get('acquired')} owner={coll.get('owner')}; cell proof owner={proof.get('owner')} same file={same} (inode {proof.get('inode')})")
    cj = cell / "CELL.jsonl"
    status = [json.loads(l).get("status") for l in cj.read_text().splitlines() if l.strip()] if cj.exists() else []
    check("executed-not-qualified" in status, f"collector status {status}")
    pd = (ev / "pinned-default.log").read_text(errors="replace") if (ev / "pinned-default.log").exists() else ""
    m = re.search(r'PINNED-DEFAULT device="([^"]*)" kind=(\S+) flags=(\d+)', pd)
    check(bool(m) and m.group(2) == "write-combined", f"premise: {m.group(0) if m else 'no PINNED-DEFAULT device= line'}")
    pinned = m.group(2) if m else "unknown"
    for f in ("compute-apps.before.csv", "compute-apps.after.csv"):
        rows = [l for l in (ev / f).read_text(errors="replace").splitlines()[1:] if l.strip()] if (ev / f).exists() else ["absent"]
        check(rows == [], f"{f}: no compute app ({rows})")

    res = {}
    for label, kinds, prefix in (
        ("micro-54m", ("cached", "wc", "heap"), "HASH-MICRO rule"),
        ("twostep-54m", ("wc", "wc_copy"), "HASH-MICRO two-step rule"),
        ("micro-160m", ("cached", "wc", "heap"), "HASH-MICRO rule"),
        ("twostep-160m", ("wc", "wc_copy"), "HASH-MICRO two-step rule"),
    ):
        path = ev / f"{label}.log"
        if not path.exists():
            check(False, f"{label}.log present")
            continue
        rc = (ev / f"{label}.exit").read_text().strip() if (ev / f"{label}.exit").exists() else "?"
        rules, kv, passes, extra = parse_micro(path, kinds, prefix)
        check(rc == "0", f"{label}: exit {rc}")
        check(len(rules) == 1, f"{label}: one `{prefix}` line")
        check(all(len(v) == 10 for v in passes.values()), f"{label}: 10 passes per kind " + str({k: len(v) for k, v in passes.items()}))
        if kv:
            for k in kinds:
                check(abs(float(kv[f"{k}_ms"]) - med(passes[k])) < 0.01, f"{label}: {k} median agrees with the pass lines ({kv[f'{k}_ms']} ms)")
            check(kv.get("digest_equal") == "true", f"{label}: digests equal across kinds")
            if "wc_bit_cached" in kv:
                check(kv.get("wc_bit_cached") == "false" and kv.get("wc_bit_wc") == "true", f"{label}: driver flags: WC bit on the WC buffer only")
            else:
                check(kv.get("wc_bit_wc") == "true" and kv.get("wc_bit_copy") == "false", f"{label}: driver flags: WC bit on the WC buffer, none on the copy")
            expected_bytes = "54800000" if label.endswith("54m") else "167772160"
            check(kv.get("bytes") == expected_bytes and kv.get("n_per_order") == "5", f"{label}: bytes={kv.get('bytes')} n_per_order={kv.get('n_per_order')}")
        res[label] = (kv, passes, extra)
        print(f"{label} rule: {rules[0] if rules else 'none'}")

    text, d = regime(cell)
    print(text)
    for f in ("card.before.csv", "card.after.csv"):
        if (ev / f).exists():
            print(f"{f}: " + " | ".join((ev / f).read_text(errors="replace").splitlines()[1:2]))
    for f in ("loadavg.before.txt", "loadavg.after.txt"):
        if (ev / f).exists():
            print(f"{f}: {(ev / f).read_text().strip()}")

    d31 = reread_day31(day31)

    print("== medians (ms), N=10 pooled per kind (N=5 per order), this card, this sitting")
    print("| size | cached | wc (single pass) | heap | wc_copy (memcpy + hash) | memcpy step | hash step |")
    print("|---|---|---|---|---|---|---|")
    table = {}
    for size, m_label, t_label in (("54,800,000 B", "micro-54m", "twostep-54m"), ("160 MiB", "micro-160m", "twostep-160m")):
        mk, mp, _ = res.get(m_label, ({}, {}, {}))
        tk, tp, tx = res.get(t_label, ({}, {}, {}))
        row = {
            "cached": med(mp.get("cached", [])), "wc": med(mp.get("wc", [])), "heap": med(mp.get("heap", [])),
            "wc_single_two": med(tp.get("wc", [])), "wc_copy": med(tp.get("wc_copy", [])),
            "memcpy": med(tx.get("wc_copy", {}).get("memcpy_ms", [])), "hash": med(tx.get("wc_copy", {}).get("hash_ms", [])),
            "wc_range": mk.get("wc_range", "?"), "wc_copy_range": tk.get("wc_copy_range", "?"),
        }
        table[size] = row
        print(f"| {size} | {row['cached']:.3f} | {row['wc']:.3f} (two-step invocation's own wc arm {row['wc_single_two']:.3f}) | {row['heap']:.3f} | {row['wc_copy']:.3f} | {row['memcpy']:.3f} | {row['hash']:.3f} |")

    # The pre-registered clauses (DAY33.md section 1), applied verbatim.
    small = table.get("54,800,000 B", {})
    on_in_max = max(d31["on_in"]) if d31["on_in"] else float("nan")
    on_in_med = med(d31["on_in"])
    delta_med = med(d31["on_delta"])
    delta_max = max(d31["on_delta"]) if d31["on_delta"] else float("nan")
    wc_single_min = min(res["twostep-54m"][1]["wc"] + res["micro-54m"][1]["wc"]) if "twostep-54m" in res and "micro-54m" in res else float("nan")
    wc_copy_min = min(res["twostep-54m"][1]["wc_copy"]) if "twostep-54m" in res else float("nan")
    cached_med = small.get("cached", float("nan"))
    h1_single = wc_single_min <= on_in_max
    h1_twostep = wc_copy_min <= on_in_max
    print("== clauses")
    print(f"clause H1-single: the fastest single-pass WC hash at 54,800,000 B ({wc_single_min:.1f} ms, N=20 across both invocations) fits inside the slowest day-31 ON demote `in` ({on_in_max:.1f} ms): {h1_single}")
    print(f"clause H1-twostep: the fastest memcpy+hash of the WC buffer at 54,800,000 B ({wc_copy_min:.1f} ms, N=10) fits inside the slowest day-31 ON demote `in` ({on_in_max:.1f} ms): {h1_twostep}")
    two_cached = 2 * cached_med
    print(f"arithmetic (no clause): two cached passes at 54,800,000 B = {two_cached:.1f} ms; one = {cached_med:.1f}; day-31 ON `in` minus completion median {delta_med:.1f} (max {delta_max:.1f}); ON `in` median {on_in_med:.1f}")
    if h1_single or h1_twostep:
        h1 = "H1 stands (a WC read route fits the ON demote's wall time)"
    else:
        h1 = "H1 refuted (no WC read route fits the ON demote's wall time; H2 or H3 stands, separated by the code census, not by this cell)"
    admissible = not FAIL
    print(
        f"DAY33 HASH-WC VERDICT: 54.8MB cached {small.get('cached', float('nan')):.1f} wc {small.get('wc', float('nan')):.1f} heap {small.get('heap', float('nan')):.1f} wc_copy {small.get('wc_copy', float('nan')):.1f} (N=10 each); "
        f"160MiB cached {table.get('160 MiB', {}).get('cached', float('nan')):.1f} wc {table.get('160 MiB', {}).get('wc', float('nan')):.1f} heap {table.get('160 MiB', {}).get('heap', float('nan')):.1f} wc_copy {table.get('160 MiB', {}).get('wc_copy', float('nan')):.1f} (N=10 each); "
        f"day31 ON in median {on_in_med:.1f} max {on_in_max:.1f} (N=12) in_minus_completion median {delta_med:.1f} (N=12); "
        f"H1-single fits={h1_single} H1-twostep fits={h1_twostep} -> {h1}; pinned={pinned}; admissible={admissible}"
    )
    print(f"checks: {len(OK)} ok, {len(FAIL)} FAIL")


if __name__ == "__main__":
    main()
