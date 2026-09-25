#!/usr/bin/env python3
"""Day 40 reader for cell `attrib` (research/spill-c-20260919/DAY40.md section 3, registered before any run).

Reads the stamped run logs under <cell>/ev and prints the integrity checks and readings R1 to R6. Nothing is
estimated: every figure is a median over the runs named, from lines the binary printed. No clause or bound moves
after a run.

usage: day40-attrib.py <cell-dir> [--rig NAME]
"""
import re
import statistics
import sys
from pathlib import Path

N_TOKENS = 32
NS_KEYS = ["validate_ns", "demand_ns", "reserve_ns", "enqueue_ns", "copy_gpu_ns", "drain_ns", "sync2_ns",
           "finish_ns", "miss_total_ns", "inner_demand_ns", "trace_ns", "pread_ns", "stage_ns", "alloc_ns",
           "step_ns", "verify_ns", "publish_ns", "retire_ns", "collect_ns"]
COUNT_KEYS = ["admits", "gpu_hits", "gpu_misses", "host_hits", "host_misses", "reads", "stages", "steps",
              "verified", "copy_events", "event_errors"]
PHASES = ["gate", "generate", "warm", "window", "close"]
SERIAL = ["validate_ns", "demand_ns", "reserve_ns", "enqueue_ns", "drain_ns", "sync2_ns", "finish_ns"]


def ts_seconds(stamp):
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_run(log):
    run = {"lines": [], "stages": {}, "install": None}
    for raw in log.read_text(errors="replace").splitlines():
        stamp, _, line = raw.partition("\t")
        run["lines"].append((stamp, line))
        if "MATCH" in line and "argmax" in line:
            run["match"] = line.strip()
        if line.startswith("tokens: "):
            run["tokens"] = line.strip()
        m = re.match(r"generated (\d+) tokens in ([0-9.]+)s", line)
        if m:
            run["gen_s"] = float(m.group(2))
            run["gen_n"] = int(m.group(1))
        m = re.match(r"MoE cache: (\d+) slots", line)
        if m:
            run["slots"] = int(m.group(1))
        if line.startswith("MoE cache STEADY-STATE ("):
            run["steady"] = line.strip()
        m = re.match(r"MoE cache STEADY-STATE window: (\d+) decode steps in ([0-9.]+)s", line)
        if m:
            run["window_s"] = float(m.group(2))
            run["window_n"] = int(m.group(1))
        if line.startswith("[experts-via-tier] installed"):
            run["installed_t"] = ts_seconds(stamp)
        if line.startswith("[q8rp] split-plane decode mirrors built"):
            run["q8rp_t"] = ts_seconds(stamp)
        m = re.search(r"physical_reads=(\d+)", line)
        if m:
            run["physical_reads"] = int(m.group(1))
        m = re.match(r"\[experts-via-tier\] stages phase=(\w+) (.*)", line)
        if m:
            run["stages"][m.group(1)] = dict(
                (k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", m.group(2)))
        m = re.match(r"\[experts-via-tier\] install (.*)", line)
        if m:
            run["install"] = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", m.group(1)))
        if "[experts-via-tier]" in line or "[expert-host-slru]" in line or "[expert-gpu-slru]" in line:
            run["door_lines"] = run.get("door_lines", 0) + 1
    return run


def med(values):
    return statistics.median(values) if values else float("nan")


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r[1-5].log")):
        label = log.stem
        order, arm, _ = label.split("-")
        run = parse_run(log)
        run["exit"] = int((ev / f"{label}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        runs[label] = run
    fails = []

    def check(ok, name):
        if not ok:
            fails.append(name)

    check(len(runs) == 30, f"runs={len(runs)} (30 expected)")
    for label, r in runs.items():
        check(r["exit"] == 0, f"{label} exit={r['exit']}")
        check("match" in r and r["match"].endswith("MATCH"), f"{label} no MATCH")
        check(r.get("slots") == 9986, f"{label} slots={r.get('slots')}")
        check(r.get("gen_n") == N_TOKENS and r.get("window_n") == N_TOKENS, f"{label} token counts")
        if r["arm"] == "off":
            check(r.get("door_lines", 0) == 0, f"{label} OFF printed a door line")
        else:
            check("installed_t" in r and "physical_reads" in r, f"{label} no installed/physical_reads line")
        if r["arm"] == "ons":
            check(all(p in r["stages"] for p in ("gate", "generate", "warm", "window")), f"{label} stage lines")
            check(r["install"] is not None, f"{label} no install line")
        else:
            check(not r["stages"] and r["install"] is None, f"{label} stage or install line without the flag")
    tapes = {r.get("tokens") for r in runs.values()}
    steadies = {r.get("steady") for r in runs.values()}
    check(len(tapes) == 1, f"tapes={len(tapes)}")
    check(len(steadies) == 1, f"steady lines={len(steadies)}")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY40 ATTRIB CHECKS rig={rig} runs={len(runs)} integrity={integrity}" +
          ("" if not fails else " failed=" + "; ".join(fails)))

    def arm_values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    # R1: the door's cost on the gen-only decode and on the steady window.
    r1 = {}
    for key in ("gen_s", "window_s"):
        for order in (None, "o1", "o2"):
            on, off = med(arm_values("on", key, order)), med(arm_values("off", key, order))
            r1[(key, order)] = (on - off) * 1000 / N_TOKENS
    print("DAY40 R1 door_ms_per_token gen: pooled={:.2f} o1={:.2f} o2={:.2f} | window: pooled={:.2f} o1={:.2f} o2={:.2f}"
          " | medians gen off={:.3f} on={:.3f} ons={:.3f} window off={:.3f} on={:.3f} ons={:.3f} (N=10 per arm)".format(
              r1[("gen_s", None)], r1[("gen_s", "o1")], r1[("gen_s", "o2")],
              r1[("window_s", None)], r1[("window_s", "o1")], r1[("window_s", "o2")],
              med(arm_values("off", "gen_s")), med(arm_values("on", "gen_s")), med(arm_values("ons", "gen_s")),
              med(arm_values("off", "window_s")), med(arm_values("on", "window_s")), med(arm_values("ons", "window_s"))))
    window_door = r1[("window_s", None)]

    # R2: per stage per decode token, the steady window only, ONS runs.
    ons = [r for r in runs.values() if r["arm"] == "ons" and "window" in r["stages"] and "warm" in r["stages"]]
    per_token = {}
    for key in NS_KEYS + COUNT_KEYS:
        vals = [(r["stages"]["window"].get(key, 0) - r["stages"]["warm"].get(key, 0)) / N_TOKENS for r in ons]
        per_token[key] = med(vals)
    ms = {k: per_token[k] / 1e6 for k in NS_KEYS}
    print("DAY40 R2 per_token_ms " + " ".join(f"{k[:-3]}={ms[k]:.3f}" for k in NS_KEYS) + f" (N={len(ons)} ONS runs)")
    print("DAY40 R2 per_token_counts " + " ".join(f"{k}={per_token[k]:.1f}" for k in COUNT_KEYS))
    ranked = sorted(((ms[k], k[:-3]) for k in SERIAL), reverse=True)
    print("DAY40 R2 serial_rank " + " > ".join(f"{name}={v:.3f}" for v, name in ranked))

    # R3: coverage.
    serial = sum(ms[k] for k in SERIAL)
    parts = sum(ms[k] for k in ("demand_ns", "reserve_ns", "enqueue_ns", "drain_ns", "sync2_ns", "finish_ns"))
    print("DAY40 R3 serial_ms_per_token={:.3f} window_door_ms_per_token={:.3f} coverage={:.3f} miss_total={:.3f}"
          " miss_parts_excl_validate={:.3f} residual={:.3f}".format(
              serial, window_door, serial / window_door if window_door else float("nan"), ms["miss_total_ns"],
              parts, ms["miss_total_ns"] - parts))

    # R4: the split inside demand and finish.
    print("DAY40 R4 demand={:.3f} = inner_demand {:.3f} + trace {:.3f} + other {:.3f}; inner_demand: stage {:.3f}"
          " (alloc {:.3f}) step {:.3f} (pread {:.3f}) verify {:.3f} publish {:.3f}; finish={:.3f}: retire {:.3f}"
          " collect {:.3f}".format(
              ms["demand_ns"], ms["inner_demand_ns"], ms["trace_ns"],
              ms["demand_ns"] - ms["inner_demand_ns"] - ms["trace_ns"], ms["stage_ns"], ms["alloc_ns"],
              ms["step_ns"], ms["pread_ns"], ms["verify_ns"], ms["publish_ns"], ms["finish_ns"], ms["retire_ns"],
              ms["collect_ns"]))

    # R5: the instrument's own cost and its clause.
    instrument = (med(arm_values("ons", "window_s")) - med(arm_values("on", "window_s"))) * 1000 / N_TOKENS
    bound = min(0.05 * window_door, 2.0) if window_door == window_door else float("nan")
    verdict = "perturbed" if instrument > bound else "within_bound"
    print(f"DAY40 R5 instrument_ms_per_token={instrument:.3f} bound={bound:.3f} (min of 5% of {window_door:.3f} and 2.0)"
          f" -> {verdict}")

    # R6: install.
    inst = [r["install"] for r in ons if r["install"]]
    on_install = [r["installed_t"] - r["q8rp_t"] for r in runs.values()
                  if r["arm"] == "on" and "installed_t" in r and "q8rp_t" in r]
    print("DAY40 R6 install_s ONS sha={:.2f} catalog={:.2f} records={:.2f} setup={:.2f} | ON install_s={:.2f}"
          " (installed minus q8rp line, N={})".format(
              med([i["sha_ns"] for i in inst]) / 1e9, med([i["catalog_ns"] for i in inst]) / 1e9,
              med([i["records_ns"] for i in inst]) / 1e9, med([i["setup_ns"] for i in inst]) / 1e9,
              med(on_install), len(on_install)))
    print(f"DAY40 ATTRIB rig={rig} integrity={integrity} window_door_ms_per_token={window_door:.2f}"
          f" serial_ms_per_token={serial:.2f} instrument={verdict} top={ranked[0][1]}")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
