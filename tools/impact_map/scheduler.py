#!/usr/bin/env python3
"""Quiet-window scheduler: run the delta set on each device when the device is free, bank the
evidence, never disturb other work.

    python3 tools/impact_map/scheduler.py plan --base origin/main --head HEAD --out Q.json
    python3 tools/impact_map/scheduler.py wait --device rtx5090 [--max-wait 6h]
    python3 tools/impact_map/scheduler.py run  --device rtx5090 --queue Q.json --out DIR
    python3 tools/impact_map/scheduler.py grant --device rtxpro6000 --gpus 1,3 --until 2026-11-03T18:40
    python3 tools/impact_map/scheduler.py request --device rtxpro6000 --queue Q.json

Windows per device class:

- rtx5090 (the owner's laptop): the window is open when `gpulock status` says free, the shared
  build lock (~/.cache/memra-rig-build.lock) is free and the 1-minute load is below
  --max-load (default: a quarter of the CPUs, so a busy desktop is never joined). The build then
  runs under that lock at nice 15 with 2 jobs, and the GPU phase under `gpulock run`, which also
  takes /tmp/memra-5090.lock.
- gb10 (the DGX Sparks): `sparklock run --unit U --wait` holds the unit; sparklock already treats
  an unleased GPU process or container as busy, so a benchmark another session runs is never
  joined.
- rtxpro6000 (a shared workstation): the cards belong to other work. A window is granted, not
  detected: `request` writes what is needed and for how long; whoever relays the
  agreement records it with `grant` (GPUs and a hard stop); `run` starts only inside a grant
  long enough for the queue, pins CUDA_VISIBLE_DEVICES to the granted cards and stops 5 minutes
  before the hard stop.
- h100, b200: best effort (owner, 2026-11-04): no window and no cell is owed.

Every wait is bounded (--max-wait); a device that never opens is reported, not retried forever.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delta  # noqa: E402

STATE = Path(os.environ.get("MEMRA_QUAL_STATE", Path.home() / ".local/state/memra-qual"))
BUILD_LOCK = Path.home() / ".cache/memra-rig-build.lock"
SAFETY = dt.timedelta(minutes=5)
POLL_S = 60


def dur(text):
    unit = text[-1]
    n = float(text[:-1])
    return dt.timedelta(seconds=n * {"s": 1, "m": 60, "h": 3600}[unit])


def sh(*cmd, check=False, **kw):
    return subprocess.run(list(cmd), capture_output=True, text=True, check=check, **kw)


def rig_window(max_load):
    """(open, why) for the local RTX 5090."""
    if shutil.which("gpulock") is None:
        return False, "gpulock missing"
    st = sh("gpulock", "status").stdout.strip()
    if "free" not in st:
        return False, f"gpu: {st}"
    load = os.getloadavg()[0]
    if load > max_load:
        return False, f"load {load:.1f} > {max_load:.1f}"
    probe = sh("flock", "-n", str(BUILD_LOCK), "true")
    if probe.returncode != 0:
        return False, "shared build lock held"
    return True, f"gpu free, load {load:.1f}, build lock free"


def spark_window(unit):
    if shutil.which("sparklock") is None:
        return False, "sparklock missing"
    st = sh("sparklock", "status", "--unit", unit).stdout.strip()
    return ("free" in st and "busy" not in st), st


def grant_path(device):
    return STATE / "grants" / f"{device}.json"


def granted_window(device, need):
    p = grant_path(device)
    if not p.exists():
        return False, "no grant recorded", None
    g = json.loads(p.read_text())
    until = dt.datetime.fromisoformat(g["until"])
    now = dt.datetime.now(until.tzinfo)
    left = until - SAFETY - now
    if left <= dt.timedelta(0):
        return False, f"grant ended at {g['until']}", None
    if left < need:
        return False, f"grant leaves {left} < needed {need}", None
    return True, f"granted GPUs {g['gpus']} until {g['until']}", g


def wait(device, *, max_wait, max_load, unit, need):
    deadline = time.monotonic() + max_wait.total_seconds()
    while True:
        if device == "rtx5090":
            ok, why = rig_window(max_load)
            grant = None
        elif device == "gb10":
            ok, why = spark_window(unit)
            grant = None
        elif device == "rtxpro6000":
            ok, why, grant = granted_window(device, need)
        else:
            return False, f"{device}: best effort; no cell is owed", None
        if ok:
            return True, why, grant
        if time.monotonic() >= deadline:
            return False, f"no window within {max_wait}: last state: {why}", None
        time.sleep(POLL_S)


def cmd_plan(a):
    rep = delta.delta(a.repo, a.base, a.head)
    pl = delta.plan(a.repo, a.base, rep)
    q = {"base": a.base, "head": sh("git", "-C", str(a.repo), "rev-parse", a.head).stdout.strip(),
         "summary": delta.summarize(rep), "runs": pl["runs"], "uncovered": pl["uncovered"], "proven": pl["proven"]}
    Path(a.out).write_text(json.dumps(q, indent=1, default=sorted) + "\n")
    for d, ms in q["runs"].items():
        print(f"{d}: {', '.join(ms)}")
    return 0


def cmd_wait(a):
    ok, why, _ = wait(a.device, max_wait=dur(a.max_wait), max_load=a.max_load, unit=a.unit, need=dur(a.need))
    print(("open: " if ok else "closed: ") + why)
    return 0 if ok else 3


def cmd_grant(a):
    p = grant_path(a.device)
    p.parent.mkdir(parents=True, exist_ok=True)
    until = dt.datetime.fromisoformat(a.until)
    if until.tzinfo is None:
        until = until.astimezone()
    p.write_text(json.dumps({"gpus": a.gpus, "until": until.isoformat(), "by": a.by}) + "\n")
    print(f"grant recorded: {a.device} GPUs {a.gpus} until {until.isoformat()}")
    return 0


def cmd_request(a):
    q = json.loads(Path(a.queue).read_text())
    runs = q["runs"].get(a.device, {})
    p = STATE / "requests" / f"{a.device}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    req = {"device": a.device, "head": q["head"], "models": runs, "estimate": a.need,
           "text": f"memra qualification needs {a.need} on one {a.device} for {q['head'][:12]}: "
                   + "; ".join(f"{m} ({', '.join(v['tests'])})" for m, v in runs.items())}
    p.write_text(json.dumps(req, indent=1) + "\n")
    print(req["text"])
    return 0


def cmd_run(a):
    q = json.loads(Path(a.queue).read_text())
    runs = q["runs"].get(a.device)
    if not runs:
        print(f"{a.device}: nothing to run")
        return 0
    ok, why, grant = wait(a.device, max_wait=dur(a.max_wait), max_load=a.max_load, unit=a.unit, need=dur(a.need))
    if not ok:
        print(f"{a.device}: {why}")
        return 3
    print(f"{a.device}: window open ({why})")
    kinds = set()
    for v in runs.values():
        kinds.update(v["tests"])
    ev_kinds = sorted({"battery" if k in ("run-gen-argmax", "run-spec", "kernel-check") else
                       "serving" if k == "serve-smoke" else "perf" if k == "perf-ab" else k for k in kinds}
                      & {"battery", "serving", "perf"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roster_ids = {l.split("\t")[1] for l in (a.repo / "tools/release-roster.tsv").read_text().splitlines()
                  if l.strip() and not l.startswith("#")}
    extra = sorted(m for m in runs if m not in roster_ids)
    if extra:
        # the device battery runs the release roster; the plan's other models are advisory runs the
        # lane owner adds to the PR body (their family's own gates), listed here so none is lost
        (out / "advisory.json").write_text(json.dumps({m: runs[m] for m in extra}, indent=1) + "\n")
        print(f"{a.device}: advisory (not in the battery roster): {', '.join(extra)}")
    cmd = [sys.executable, str(a.repo / "tools/cell_evidence.py"), "run", "--device", a.device, "--commit", q["head"],
           "--kinds", ",".join(ev_kinds), "--base", q["base"], "--out", str(out / "evidence")]
    env = dict(os.environ)
    if grant:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in grant["gpus"])
        until = dt.datetime.fromisoformat(grant["until"]) - SAFETY
        timeout = max(60, int((until - dt.datetime.now(until.tzinfo)).total_seconds()))
    else:
        timeout = int(dur(a.need).total_seconds() * 2)
    if a.device == "gb10":
        if not a.ssh_host:
            print("gb10: pass --ssh-host (the unit's ssh alias from your local config)")
            return 2
        cmd = ["sparklock", "run", "--unit", a.unit, "--ttl", a.need, "--for", "memra delta qualification",
               "--wait", "10m", "--", "ssh", a.ssh_host, " ".join(cmd)]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, env=env, timeout=timeout)
    if r.returncode != 0:
        print(f"{a.device}: evidence run failed (exit {r.returncode}); raw logs under {out}")
        return r.returncode
    return subprocess.run([sys.executable, str(a.repo / "tools/cell_evidence.py"), "bank", "--out", str(out / "evidence")]).returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "wait", "run", "grant", "request"):
        p = sub.add_parser(name)
        p.add_argument("--repo", type=Path, default=HERE.parent.parent)
        if name == "plan":
            p.add_argument("--base", default="origin/main")
            p.add_argument("--head", default="HEAD")
            p.add_argument("--out", required=True)
            continue
        p.add_argument("--device", required=True, choices=("rtx5090", "rtxpro6000", "gb10", "h100", "b200", "sm89"))
        p.add_argument("--max-wait", default="6h")
        p.add_argument("--max-load", type=float, default=(os.cpu_count() or 8) / 4)
        p.add_argument("--unit", default="both", help="sparklock unit name for gb10")
        p.add_argument("--ssh-host", help="ssh alias of the gb10 unit (local config, never tracked)")
        p.add_argument("--need", default="2h", help="window length the queue needs")
        if name in ("run", "request"):
            p.add_argument("--queue", required=True)
        if name == "run":
            p.add_argument("--out", required=True)
        if name == "grant":
            p.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",")], required=True)
            p.add_argument("--until", required=True, help="hard stop, ISO local time")
            p.add_argument("--by", default="relay", help="who granted (session name)")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "wait": cmd_wait, "run": cmd_run, "grant": cmd_grant, "request": cmd_request}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
