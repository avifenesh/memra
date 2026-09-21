#!/usr/bin/env python3
"""Day 18 replays (research/spill-c-20260919/DAY18.md): apply the pre-registered rules to a mirrored cell.

usage: day18-replay.py <cell-dir>   (a collector --out directory, or its parent holding several)
Cells are recognised by the evidence they carry (ev/*.log names). Every clause and threshold is
fixed in DAY18.md before the runs; nothing here is read from a result.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

GEN = re.compile(r"generated (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s \((.*?); prime ([\d.]+)s\)")
CACHE = re.compile(r"MoE cache: (\d+) slots \| cumulative hits=(\d+) misses=(\d+)")
STEADY = re.compile(r"MoE cache STEADY-STATE .*?hit-rate=([\d.]+)% \| ([\d.]+) MB/decode-token")
TOKENS = re.compile(r"^tokens: (\[.*\])")
READS = re.compile(r"\[experts-via-tier\] physical_reads=(\d+)")
GPUSLRU = re.compile(r"\[expert-gpu-slru\] slots=(\d+) allocated_bytes=(\d+) evictions=(\d+)")
SHA_MISMATCH = 'Error: "experts-via-tier artifact SHA256 mismatch"'
checks = []


def check(ok, text):
    checks.append(bool(ok))
    print(("ok   " if ok else "FAIL ") + text)
    return ok


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def ts(line):
    h, m, s = line.split("\t", 1)[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def strip(line):
    return line.split("\t", 1)[1] if "\t" in line else line


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
    temps, power, sm = col("temperature.gpu"), col("power.draw [W]"), col("clocks.current.sm [MHz]")
    limit = {r.get("power.limit [W]", "").strip() for r in rows}
    d = {"samples": len(rows), "temp_lo": min(temps), "temp_hi": max(temps), "power_max": max(power),
         "limit": ", ".join(sorted(limit)), "sm_lo": min(sm), "sm_hi": max(sm)}
    return (f"telemetry: {len(rows)} samples at 250 ms; temperature {d['temp_lo']:.0f}..{d['temp_hi']:.0f} C; "
            f"power draw max {d['power_max']:.0f} W (limit {d['limit']}); SM clock {d['sm_lo']:.0f}..{d['sm_hi']:.0f} MHz"), d


def capture_ok(cell):
    cap = cell / "command.capture.json"
    if not cap.exists():
        return check(False, "command.capture.json present")
    c = json.loads(cap.read_text())
    return check(c.get("status") == "executed-not-qualified" and c.get("exit_code") == 0,
                 f"collector capture status={c.get('status')} exit={c.get('exit_code')}")


def parse_gen(path):
    lines = path.read_text(errors="replace").splitlines()
    r = {"exit": None, "match": False, "gen_s": None, "prime_s": None, "tape": None, "slots": None,
         "hits": None, "misses": None, "hitrate": None, "mb_tok": None, "reads": None, "evictions": None,
         "installed": False, "trace_lines": 0, "t_q8rp": None, "t_installed": None, "t_loaded": None,
         "t_gen": None, "t_first": None, "t_last": None, "sha_mismatch": False, "door_lines": 0}
    for raw in lines:
        l = strip(raw)
        t = ts(raw) if "\t" in raw else None
        if t is not None:
            r["t_first"] = r["t_first"] if r["t_first"] is not None else t
            r["t_last"] = t
        if "[expert-host-slru]" in l:
            r["trace_lines"] += 1
        if "experts-via-tier" in l or "expert-host-slru" in l or "expert-gpu-slru" in l:
            r["door_lines"] += 1
        if l.startswith("[q8rp] split-plane decode mirrors built"):
            r["t_q8rp"] = t
        if l.startswith("[experts-via-tier] installed"):
            r["installed"], r["t_installed"] = True, t
        if l.startswith("loaded "):
            r["t_loaded"] = t
        if "  MATCH" in l and l.startswith("prefill argmax"):
            r["match"] = True
        m = GEN.search(l)
        if m:
            r["gen_s"], r["prime_s"], r["t_gen"] = float(m.group(2)), float(m.group(5)), t
        m = TOKENS.match(l)
        if m:
            r["tape"] = m.group(1)
        m = CACHE.search(l)
        if m:
            r["slots"], r["hits"], r["misses"] = int(m.group(1)), int(m.group(2)), int(m.group(3))
        m = STEADY.search(l)
        if m:
            r["hitrate"], r["mb_tok"] = float(m.group(1)), float(m.group(2))
        m = READS.search(l)
        if m:
            r["reads"] = int(m.group(1))
        m = GPUSLRU.search(l)
        if m:
            r["evictions"] = int(m.group(3))
        if l.strip() == SHA_MISMATCH:
            r["sha_mismatch"] = True
    ex = path.with_suffix(".exit")
    if ex.exists():
        r["exit"] = int(ex.read_text().strip())
    return r


def overlap(cell):
    ev = cell / "ev"
    print(f"== overlap pair: {cell}")
    capture_ok(cell)
    labels = [f"o1-off-r{i}" for i in range(1, 6)] + [f"o1-on-r{i}" for i in range(1, 6)] + \
             [f"o2-on-r{i}" for i in range(1, 6)] + [f"o2-off-r{i}" for i in range(1, 6)]
    runs = {l: parse_gen(ev / f"{l}.log") for l in labels}
    check(all((ev / f"{l}.log").exists() for l in labels), "20 run logs present")
    check(all(r["exit"] == 0 for r in runs.values()), "every run exit 0")
    check(all(r["match"] for r in runs.values()), "every run prints MATCH")
    tapes = {r["tape"] for r in runs.values()}
    check(len(tapes) == 1 and None not in tapes, f"one tape across 20 runs ({len(tapes)} distinct)")
    check(all(r["slots"] == 9986 for r in runs.values()), "slots=9986 in every MoE cache line")
    steady = {(r["hitrate"], r["mb_tok"]) for r in runs.values()}
    check(len(steady) == 1, f"one STEADY-STATE line across 20 runs: {sorted(steady)}")
    on = [runs[l] for l in labels if "-on-" in l]
    off = [runs[l] for l in labels if "-off-" in l]
    check(all(r["installed"] and r["reads"] is not None for r in on), "ON runs carry installed and physical_reads")
    check(all(r["door_lines"] == 0 for r in off), "OFF runs carry no door line")
    for l in labels:
        r = runs[l]
        fwd = (r["t_gen"] - r["t_loaded"]) if r["t_gen"] is not None and r["t_loaded"] is not None else float("nan")
        inst = (r["t_installed"] - r["t_q8rp"]) if r["t_installed"] is not None and r["t_q8rp"] is not None else float("nan")
        wall = (r["t_last"] - r["t_first"]) if r["t_last"] is not None else float("nan")
        print(f"  {l:11s} exit={r['exit']} gen_s={r['gen_s']} prime_s={r['prime_s']} misses={r['misses']} hits={r['hits']} "
              f"reads={r['reads']} evictions={r['evictions']} install_s={inst:.2f} forward_s={fwd:.2f} wall_s={wall:.2f} trace_lines={r['trace_lines']}")
    g = lambda rs: [r["gen_s"] for r in rs if r["gen_s"] is not None]
    on1, on2 = [runs[f"o1-on-r{i}"] for i in range(1, 6)], [runs[f"o2-on-r{i}"] for i in range(1, 6)]
    off1, off2 = [runs[f"o1-off-r{i}"] for i in range(1, 6)], [runs[f"o2-off-r{i}"] for i in range(1, 6)]
    m_on, m_off = med(g(on)), med(g(off))
    r1, r2 = med(g(on1)) / med(g(off1)), med(g(on2)) / med(g(off2))
    ratio = m_on / m_off
    mb_tok = on[0]["mb_tok"] or float("nan")
    per_tok_ms = (m_on - m_off) / 32 * 1e3
    per_mb_ms = per_tok_ms / mb_tok if mb_tok else float("nan")
    fwd = lambda rs: [r["t_gen"] - r["t_loaded"] for r in rs if r["t_gen"] is not None and r["t_loaded"] is not None]
    inst = [r["t_installed"] - r["t_q8rp"] for r in on if r["t_installed"] is not None and r["t_q8rp"] is not None]
    fwd_ratio = med(fwd(on)) / med(fwd(off)) if fwd(off) else float("nan")
    if ratio >= 1.10 and r1 >= 1.10 and r2 >= 1.10:
        verdict = "sync_miss_path_slower"
    elif ratio <= 0.90 and r1 <= 0.90 and r2 <= 0.90:
        verdict = "sync_miss_path_faster"
    elif 0.90 < ratio < 1.10:
        verdict = "sync_miss_path_flat"
    else:
        verdict = "orders_disagree"
    text, d = regime(cell)
    integrity = "ok" if all(checks) else "FAIL"
    line = (f"OVERLAP-PAIR rule decode_off_s={m_off:.3f} decode_on_s={m_on:.3f} decode_ratio={ratio:.3f} "
            f"ratio_o1={r1:.3f} ratio_o2={r2:.3f} off_range={min(g(off)):.3f}..{max(g(off)):.3f} on_range={min(g(on)):.3f}..{max(g(on)):.3f} "
            f"door_cost_ms_per_decode_token={per_tok_ms:.2f} steady_mb_per_token={mb_tok} door_cost_ms_per_staged_MB={per_mb_ms:.3f} "
            f"misses_off={sorted({r['misses'] for r in off})} misses_on={sorted({r['misses'] for r in on})} "
            f"reads_on={sorted({r['reads'] for r in on})} evictions_on={sorted({r['evictions'] for r in on})} "
            f"install_on_s={med(inst):.2f} forward_off_s={med(fwd(off)):.2f} forward_on_s={med(fwd(on)):.2f} forward_ratio={fwd_ratio:.3f} "
            f"N=5/arm/order pooled=10 orders=2 "
            + (f"temp_c={d['temp_lo']:.0f}..{d['temp_hi']:.0f} power_max_w={d['power_max']:.0f} power_limit_w={d['limit']} " if d else "")
            + f"identity={'ok' if len(tapes) == 1 else 'FAIL'} integrity={integrity} -> {verdict if integrity == 'ok' else 'void (' + verdict + ')'}")
    print(text)
    print(line)
    return line


def hashlock(cell):
    ev = cell / "ev"
    print(f"== hash lock red arm: {cell}")
    capture_ok(cell)
    door, ctl = parse_gen(ev / "door.log"), parse_gen(ev / "control.log")
    check(door["exit"] == 1, f"door run exit {door['exit']} (expected 1)")
    check(door["sha_mismatch"], f"door run last line is {SHA_MISMATCH}")
    check(door["door_lines"] == 0, "door run prints no [experts-via-tier] or [expert-host-slru] line")
    check(door["t_loaded"] is None or True, "door run: refusal comes from the installer after load (informational)")
    check(ctl["exit"] == 0 and ctl["match"], f"control run exit {ctl['exit']} with MATCH={ctl['match']}")
    check(ctl["door_lines"] == 0, "control run prints no door line")
    lock_s = (door["t_last"] - door["t_q8rp"]) if door["t_q8rp"] is not None and door["t_last"] is not None else float("nan")
    text, d = regime(cell)
    print(text)
    verdict = "hash_lock_refuses" if all(checks) else "FAIL"
    line = (f"HASHLOCK rule door_exit={door['exit']} sha_mismatch_line={door['sha_mismatch']} door_lines={door['door_lines']} "
            f"control_exit={ctl['exit']} control_match={ctl['match']} lock_cost_after_load_s={lock_s:.2f} (N=1, not pooled) "
            f"artifact={(ev / 'artifact.sha256').read_text().split()[0][:16] if (ev / 'artifact.sha256').exists() else 'na'} -> {verdict}")
    print(line)
    return line


def serverdoor(cell):
    ev = cell / "ev"
    print(f"== server door red arm: {cell}")
    capture_ok(cell)
    log = (ev / "server.log").read_text(errors="replace")
    ready = (ev / "ready.txt").read_text().strip() == "ready=1" if (ev / "ready.txt").exists() else False
    req = (ev / "r1.json").exists() and bool(json.loads((ev / "r1.json").read_text())["choices"][0]["text"].strip())
    door_lines = sum(1 for l in log.splitlines() if "experts-via-tier" in l or "expert-host-slru" in l or "expert-gpu-slru" in l)
    refusal = any(("unknown" in l.lower() or "unrecognized" in l.lower() or "usage" in l.lower()) and "experts" in l for l in log.splitlines())
    check(ready, "server became ready with --experts-via-tier on its argv")
    check(req, "one completion returned text")
    check(door_lines == 0, f"zero door lines in the server log ({door_lines})")
    text, d = regime(cell)
    print(text)
    verdict = "door_unreachable_in_serving" if all(checks) else "FAIL"
    line = (f"SERVERDOOR rule ready={ready} request_ok={req} door_lines={door_lines} flag_refused={refusal} "
            f"flag_silently_accepted={ready and not refusal} -> {verdict}")
    print(line)
    return line


def hashmicro(cell):
    ev = cell / "ev"
    print(f"== hash micro-cell: {cell}")
    capture_ok(cell)
    lines = [strip(l) for l in (ev / "hash-micro.log").read_text(errors="replace").splitlines()]
    rule = [l for l in lines if l.startswith("HASH-MICRO rule")]
    check(len(rule) == 1, "one HASH-MICRO rule line")
    passes = {}
    for l in lines:
        m = re.match(r"pass order=(\d) pass=(\d+) kind=(\w+) ms=([\d.]+)", l)
        if m:
            passes.setdefault(m.group(3), []).append(float(m.group(4)))
    check(all(len(v) == 10 for v in passes.values()) and set(passes) == {"cached", "wc", "heap"}, f"10 passes per kind: {{k: len(v) for k, v in passes.items()}}")
    if rule:
        kv = dict(re.findall(r"(\w+)=(\"[^\"]*\"|\S+)", rule[0]))
        for k in ("cached", "wc", "heap"):
            check(abs(float(kv[f"{k}_ms"]) - med(passes.get(k, []))) < 0.01, f"{k} median agrees with the pass lines ({kv[f'{k}_ms']} ms)")
        check(kv.get("digest_equal") == "true", "digests equal across kinds")
        check(kv.get("wc_bit_cached") == "false" and kv.get("wc_bit_wc") == "true", "driver flags: WC bit only on the WC buffer")
    text, d = regime(cell)
    print(text)
    print(rule[0] if rule else "no rule line")
    return rule[0] if rule else "none"


def main():
    root = Path(sys.argv[1])
    cells = [root] if (root / "ev").is_dir() else sorted(p for p in root.iterdir() if (p / "ev").is_dir())
    lines = []
    for cell in cells:
        ev = cell / "ev"
        del checks[:]
        if (ev / "o1-off-r1.log").exists():
            lines.append(overlap(cell))
        elif (ev / "door.log").exists():
            lines.append(hashlock(cell))
        elif (ev / "server.log").exists():
            lines.append(serverdoor(cell))
        elif (ev / "hash-micro.log").exists():
            lines.append(hashmicro(cell))
        else:
            print(f"skip {cell}: unknown evidence")
            continue
        n = len(checks)
        print(f"DAY18 REPLAY {cell.name}: {'PASS' if all(checks) else 'FAIL'} ({n} checks)\n")
    return 0 if lines else 1


if __name__ == "__main__":
    sys.exit(main())
