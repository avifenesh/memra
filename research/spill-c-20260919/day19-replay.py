#!/usr/bin/env python3
"""Day 19 replay (research/spill-c-20260919/DAY19.md): the pre-registered rule for cell `serverdoor19`.

usage: day19-replay.py <cell-dir>   (a collector --out directory holding ev/)
Every clause is fixed in DAY19.md before the run; nothing here is read from a result. Door lines are
counted by the bracketed tags (`[experts-via-tier]`, `[expert-host-slru]`, `[expert-gpu-slru]`), the
day-18 correction: the refusal line carries the bare substring `--experts-via-tier` and is not a door line.
"""
import csv
import json
import sys
from pathlib import Path

DOOR_TAGS = ("[experts-via-tier]", "[expert-host-slru]", "[expert-gpu-slru]")
FLAG_REFUSAL = '[server] FATAL: unknown argument "--experts-via-tier" refused at boot'
ENV_ON_LINE = "[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1)"
ENV_BAD_LINE = 'MEMRA_KV_HOST_CONTRACTS="on" refused: the host tier contracts door takes exactly `1` (on) or `0`'
BOOT_LINE = "[server] build:"
checks = []


def check(ok, text):
    checks.append(bool(ok))
    print(("ok   " if ok else "FAIL ") + text)
    return ok


def regime(cell):
    sampler = cell / "command.gpu.csv"
    if not sampler.exists():
        return "telemetry: command.gpu.csv absent"
    rows = list(csv.DictReader(sampler.open(errors="replace"), skipinitialspace=True))

    def col(name):
        out = []
        for r in rows:
            v = r.get(name)
            try:
                out.append(float(v.split()[0]))
            except (AttributeError, ValueError, IndexError):
                pass
        return out
    temps, power = col("temperature.gpu"), col("power.draw [W]")
    limit = ", ".join(sorted({r.get("power.limit [W]", "").strip() for r in rows}))
    return (f"telemetry: {len(rows)} samples at 250 ms; temperature {min(temps):.0f}..{max(temps):.0f} C; "
            f"power draw max {max(power):.0f} W (limit {limit})")


def arm(ev, label):
    log = (ev / f"{label}.log").read_text(errors="replace")
    lines = log.splitlines()
    rc = int((ev / f"{label}.exit").read_text().strip())
    ready = (ev / f"{label}.ready").read_text().strip() == "ready=1"
    r1 = ev / f"{label}.r1.json"
    req = r1.exists() and bool(json.loads(r1.read_text())["choices"][0]["text"].strip())
    return {"rc": rc, "ready": ready, "req": req, "lines": lines,
            "door_lines": sum(1 for l in lines if any(t in l for t in DOOR_TAGS)),
            "booted": any(BOOT_LINE in l for l in lines)}


def main(cell):
    ev = cell / "ev"
    print(f"== {cell.name}: {cell}")
    cap = cell / "command.capture.json"
    if check(cap.exists(), "command.capture.json present"):
        c = json.loads(cap.read_text())
        check(c.get("status") == "executed-not-qualified" and c.get("exit_code") == 0,
              f"collector capture status={c.get('status')} exit={c.get('exit_code')}")
    f, on, bad = arm(ev, "flag"), arm(ev, "envon"), arm(ev, "envbad")
    # flag arm: refused at boot, token named, exit 2, nothing booted, never ready, no door line.
    f_refused = any(FLAG_REFUSAL in l for l in f["lines"])
    check(f["rc"] == 2, f"flag arm exit 2 (got {f['rc']})")
    check(f_refused, "flag arm log carries the refusal naming --experts-via-tier")
    check(not f["booted"], "flag arm never printed the build identity (refused before boot)")
    check(not f["ready"], "flag arm never became ready")
    check(f["door_lines"] == 0, f"flag arm prints no bracketed door line ({f['door_lines']})")
    check(len([l for l in f["lines"] if l.strip()]) == 1, f"flag arm log is the one refusal line ({len(f['lines'])} lines)")
    # envon arm: the door as an environment variable behaves as documented: boots ON, serves.
    on_line = any(ENV_ON_LINE in l for l in on["lines"])
    check(on["ready"], "envon arm became ready with MEMRA_KV_HOST_CONTRACTS=1")
    check(on["req"], "envon arm returned one completion with text")
    check(on_line, "envon arm log carries the door ON boot line")
    check(not any("FATAL" in l for l in on["lines"]), "envon arm has no FATAL line")
    check(on["door_lines"] == 0, f"envon arm prints no MoE door line ({on['door_lines']})")
    # envbad arm: the documented strict parse refusal, before readiness.
    bad_line = any(ENV_BAD_LINE in l for l in bad["lines"])
    check(bad["rc"] != 0, f"envbad arm exit non-zero (got {bad['rc']})")
    check(bad_line, 'envbad arm log carries the MEMRA_KV_HOST_CONTRACTS="on" refusal')
    check(not bad["ready"], "envbad arm never became ready")
    print(regime(cell))
    verdict = "flag_refused; env_door_documented" if all(checks) else "FAIL"
    line = (f"SERVERDOOR19 rule flag_exit={f['rc']} flag_refused={f_refused} flag_booted={f['booted']} "
            f"flag_ready={f['ready']} flag_door_lines={f['door_lines']} envon_ready={on['ready']} "
            f"envon_request_ok={on['req']} envon_door_line={on_line} envon_exit={on['rc']} envbad_exit={bad['rc']} "
            f"envbad_refused={bad_line} envbad_ready={bad['ready']} -> {verdict}")
    print(line)
    print(f"DAY19 REPLAY {cell.name}: {'PASS' if all(checks) else 'FAIL'} ({len(checks)} checks)")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
