#!/usr/bin/env python3
"""batch-oom-recover verdict reader for the Max-Q kit (kit/PREREG.md): one VERDICT line.

usage: maxq-verdict.py --card LABEL --gate DIR [--gate DIR ...] --serve ROOT [--vmm GREEN RED]
                       [--run-dir OUT] [--expect-name NAME] [--log FILE]

--gate     a health-fault-gate receipt dir (VERDICTS.txt, j-ctrl/, j/, j-red/); the registration's j1 and j2.
--serve    day49d-run.sh's receipt root (boots/<name>.arm.txt, boots/<name>/server.log, client.jsonl).
--vmm      the j-vmm gate dirs, green then red (the Workstation receipt only; not applicable on a Max-Q).
--run-dir  the kit run's out dir: THERMAL-STOP.txt, PIN-VIOLATION.txt, STOP and card.csv are read from it.
--log      where the registered serve reader's own output goes (default: <serve>/read-serve.log).

The registered readers run unchanged: research/spill-b-20260919/day49d-read.py `serve` (and `vmm` when given).
Its per-boot lines are copied to stdout, then the gate lines, then one line:

  VERDICT batch-oom-recover card=<label> ... -> ON | OFF | NOT-READ

ON: every registered clause passed (the recovery is this card's default by the Workstation rule).
OFF: the run is valid and a registered clause read FAIL (the card keeps the chunk's error arm).
NOT-READ: a registered unit is missing, did not boot, did not reach a batched chunk of >= 2 sessions, an
arm's identity does not match its label, or the run is invalid (thermal stop, pin violation, the
server's box-wide gpu-watch latched, a card other than --expect-name).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
READER = HERE / "research/spill-b-20260919/day49d-read.py"
BOOTS = ("O1-off", "O1-on", "O2-on", "O2-off", "P1-off", "P1-on", "P2-on", "P2-off")
ON_ARMS = {"on", "on-plain"}
BOOT_LINE = re.compile(r"\[admit-oom\] batch OOM recovery: (ON|OFF)")
CRITICAL = "[gpu-watch] CRITICAL"


def text(p):
    try:
        return Path(p).read_text(errors="replace")
    except OSError:
        return ""


def recovery_state(server_log, env_text):
    """ON/OFF from the boot line (the kit's source); the Workstation tree had the env door instead."""
    m = BOOT_LINE.search(server_log)
    if m:
        return m.group(1), "boot-line"
    if not server_log:
        return None, "no-log"
    return ("ON" if "MEMRA_BATCH_OOM_RECOVER=1" in env_text else "OFF"), "env-door"


def read_gate(d):
    v = {}
    for l in text(Path(d) / "VERDICTS.txt").splitlines():
        m = re.match(r"HFG \((j|j-red)\) ", l)
        if m:
            v[m.group(1)] = l.strip()
    state, src = recovery_state(text(Path(d) / "j-ctrl" / "server.log"), text(Path(d) / "j-ctrl" / "env.txt"))
    crit = sum(text(Path(d) / u / "server.log").count(CRITICAL) for u in ("j-ctrl", "j", "j-red"))
    return v, state, src, crit


def word(line):
    return line.rsplit("-> ", 1)[-1].strip() if "-> " in line else "absent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--gate", action="append", default=[])
    ap.add_argument("--serve", required=True)
    ap.add_argument("--vmm", nargs=2, metavar=("GREEN", "RED"))
    ap.add_argument("--run-dir")
    ap.add_argument("--expect-name")
    ap.add_argument("--log")
    a = ap.parse_args()

    notread, fail, notes = [], [], []

    # Validity of the run itself (kit-only clauses; PREREG.md section 3).
    if a.run_dir:
        rd = Path(a.run_dir)
        for name in ("THERMAL-STOP.txt", "PIN-VIOLATION.txt", "STOP"):
            if (rd / name).exists():
                notread.append(f"{name}: {text(rd / name).splitlines()[0] if text(rd / name) else 'present'}")
        if a.expect_name:
            card = text(rd / "card.csv")
            if a.expect_name not in card:
                notread.append(f"card.csv does not name {a.expect_name!r}")

    # Gates (DAY49 1.8 arm j, run from the on tree).
    gate_parts = []
    j_pass = jr_pass = 0
    for d in a.gate:
        v, state, src, crit = read_gate(d)
        label = os.path.basename(os.path.normpath(d))
        jw, jrw = word(v.get("j", "")), word(v.get("j-red", ""))
        gate_parts.append(f"{label}:j={jw},j-red={jrw},recovery={state}({src})")
        for k in ("j", "j-red"):
            if k in v:
                print(f"  {label} {v[k]}")
        if crit:
            notread.append(f"{label}: {crit} gpu-watch CRITICAL line(s) (box-wide canary)")
        if state != "ON":
            notread.append(f"{label}: the gate's control boot reads recovery {state} ({src}), not the on arm")
        for k, w in (("j", jw), ("j-red", jrw)):
            if w == "PASS":
                if k == "j":
                    j_pass += 1
                else:
                    jr_pass += 1
            elif w == "FAIL" and "boot failed" not in v.get(k, ""):
                fail.append(f"{label} {k} FAIL")
            else:
                notread.append(f"{label} {k}: {w if w != 'FAIL' else 'boot failed'}")
    if len(a.gate) != 2:
        notread.append(f"{len(a.gate)} gate units given; the registration runs 2 (j1, j2)")

    # j-vmm (the Workstation receipt only).
    vmm_word = "n/a (on-demand KV planes are off on this card; MEMRA_KV_ALLOCATOR is retired)"
    if a.vmm:
        r = subprocess.run([sys.executable, str(READER), "vmm", a.card, a.vmm[0], a.vmm[1]],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout + r.stderr)
        roles = dict(re.findall(r"role=(green|red) .*-> (PASS|FAIL)$", r.stdout, re.M))
        vmm_word = f"green={roles.get('green', 'absent')},red={roles.get('red', 'absent')}"
        for role in ("green", "red"):
            if roles.get(role) == "FAIL":
                fail.append(f"j-vmm {role} FAIL")
            elif roles.get(role) != "PASS":
                notread.append(f"j-vmm {role} absent")

    # Serving shape: the registered reader, unchanged.
    serve = Path(a.serve)
    log = Path(a.log) if a.log else serve / "read-serve.log"
    r = subprocess.run([sys.executable, str(READER), "serve", a.card, str(serve)], capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr)
    sys.stdout.write(r.stdout + r.stderr)
    lines = {}
    for l in r.stdout.splitlines():
        m = re.match(r"DAY49D SERVE card=\S+ boot=(\S+) arm=(\S+) .*-> (.+)$", l)
        if m:
            lines[m.group(1)] = (m.group(2), m.group(3).strip(), l)
    on_pass = on_n = off_n = 0
    before = []
    for b in BOOTS:
        if b not in lines:
            notread.append(f"boot {b}: absent")
            continue
        arm, w, l = lines[b]
        bd = serve / "boots" / b
        slog = text(bd / "server.log")
        state, src = recovery_state(slog, text(serve / "boots" / f"{b}.arm.txt"))
        want = "ON" if arm in ON_ARMS else "OFF"
        if state != want:
            notread.append(f"boot {b}: arm {arm} reads recovery {state} ({src}), want {want}")
        if slog.count(CRITICAL):
            notread.append(f"boot {b}: gpu-watch CRITICAL (box-wide canary)")
        if w.startswith("NOT REACHED"):
            notread.append(f"boot {b}: not reached")
            continue
        m = re.search(r"rows=(\d+) status=(\{[^}]*\})", l)
        rows = int(m.group(1)) if m else 0
        if rows == 0:
            notread.append(f"boot {b}: no client rows")
            continue
        if arm in ON_ARMS:
            on_n += 1
            if w == "PASS":
                on_pass += 1
            else:
                fail.append(f"boot {b} {w}")
        else:
            off_n += 1
            before.append(f"{b}:{m.group(2).replace(' ', '')}")
    routes = dict(re.findall(r"SERVE-ROUTE card=\S+ route=(\S+) boots=\d+ reached=(\d+)", r.stdout))

    verdict = "NOT-READ" if notread else ("OFF" if fail else "ON")
    head = (f"VERDICT batch-oom-recover card={a.card} gates=[{' '.join(gate_parts)}] gate_j_pass={j_pass}/{len(a.gate)} "
            f"gate_j_red_pass={jr_pass}/{len(a.gate)} j_vmm={vmm_word} serve_on_pass={on_pass}/4 "
            f"serve_off_read={off_n}/4 before=[{' '.join(before)}] "
            f"routes_reached=plain:{routes.get('plain', 0)}/4,spec-default:{routes.get('spec-default', 0)}/4")
    why = "; ".join(notread + fail)
    print(f"{head}{' why=[' + why + ']' if why else ''} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
