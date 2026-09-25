#!/usr/bin/env python3
"""M1 B1: KV ObjectStore storage cells on the proven path (M1-PREREG.md B1).

  run --bench storage-bench --root DIR --proof PRIVATE_PROOF.json --out DIR --rig pro-single
      --lock-fd N [--rounds 10] [--sizes ...] [--stub-no-lock]

Per size: one restore object written once (buffered roundtrip, fsync'd by the store); then ten
rounds, odd forward and even reversed over the three modes (buffered, uncached = O_DIRECT reads,
direct = O_DIRECT reads and writes), each round running one fresh-directory `roundtrip` visit and
one cold `restore` visit per mode. Every visit: identity triple checked, cold regime on the
object's files for restore, the 250 ms sampler on the proof's leaves, stdout and stderr straight
into files, then parsing of the saved files only (StorageSample JSON plus the stage line).
Scored: read throughput = valid_bytes / io_ns (both phases) and write throughput =
valid_bytes / (put_ns + commit_ns) (roundtrip). The registered B3 verdict rule is applied per
size and phase against buffered. Development characterization of the tier store, not spill speed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "tools/tier-battery.py").exists())


def load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load("battery", ROOT / "tools/tier-battery.py")
CACHE = load("regime", HERE / "m1-cache-regime.py")
RUNNER = load("runner", HERE / "m1-spill-runner.py")
SAMPLE = load("sampler", HERE / "m1-host-sampler.py")
SAMPLER = HERE / "m1-host-sampler.py"
SIZES = [264, 4096, 4097, 1048576, 116654080, 933232640]
MODES = ["buffered", "uncached", "direct"]
STAGE = re.compile(r"^\[storage-bench\] stages put_ns=(\d+) commit_ns=(\d+) lease_ns=(\d+) "
                   r"read_ns=(\d+) verify_ns=(\d+) total_ns=(\d+)$", re.M)


def files_under(path):
    return sorted(str(p) for p in Path(path).rglob("*") if p.is_file())


def parse(stdout_text, stderr_text):
    lines = [l for l in stdout_text.splitlines() if l.startswith("{")]
    sample = json.loads(lines[-1]) if lines else None
    m = STAGE.search(stderr_text)
    stages = dict(zip(("put_ns", "commit_ns", "lease_ns", "read_ns", "verify_ns", "total_ns"),
                      map(int, m.groups()))) if m else None
    return sample, stages


def visit(args, bench, phase, mode, size, obj, vdir, identity, leaves, top):
    vdir.mkdir(parents=True)
    rec = {"phase": phase, "mode": mode, "size": size}
    B.require(B.filesystem_identity(Path(args.root)) == identity, "identity changed before a visit")
    if phase == "restore":
        rec["regime_ok"] = CACHE.cold(files_under(obj))
    host = vdir / "host.jsonl"
    with (vdir / "sampler.err").open("xb") as serr:
        sampler = subprocess.Popen([sys.executable, str(SAMPLER), "sample", "--devices", ",".join(leaves + [top]),
                                    "--out", str(host)], stdout=subprocess.DEVNULL, stderr=serr)
    deadline = time.monotonic() + 10
    while not (host.exists() and host.read_text().count("\n") >= 2):
        B.require(sampler.poll() is None and time.monotonic() < deadline, "sampler did not start")
        time.sleep(0.02)
    with (vdir / "stdout.log").open("xb") as out, (vdir / "stderr.log").open("xb") as err:
        child = subprocess.Popen([str(bench), phase, str(obj), str(size), mode], stdout=out, stderr=err)
    _, status, usage = os.wait4(child.pid, 0)
    code = os.waitstatus_to_exitcode(status)
    time.sleep(0.3)
    sampler.send_signal(signal.SIGTERM)
    sampler.wait(timeout=10)
    sample, stages = parse((vdir / "stdout.log").read_text(errors="replace"),
                           (vdir / "stderr.log").read_text(errors="replace"))
    problems = []
    if code != 0:
        problems.append(f"exit {code}")
    if sample is None or stages is None:
        problems.append("no sample or no stage line")
    else:
        if sample["status"] != "byte-exact" or sample["fallbacks"] != 0:
            problems.append(f"status {sample['status']} fallbacks {sample['fallbacks']}")
        if sample["io_ns"] != stages["read_ns"]:
            problems.append("io_ns differs from the stage read time")
    tel = SAMPLE.validate(host, leaves + [top], 500)
    rows = [json.loads(l) for l in host.read_text().splitlines() if l.strip()]
    ticks = [r for r in rows[1:] if r.get("kind") == "tick"]
    own = {"read_bytes": usage.ru_inblock * 512, "write_bytes": usage.ru_oublock * 512}
    rec.update(exit_code=code, sample=sample, stages=stages, problems=problems, telemetry_ok=not tel,
               contamination=RUNNER.contamination(ticks, leaves, own), own_io=own,
               cpu_s=usage.ru_utime + usage.ru_stime,
               raw_sha256={n: RUNNER.sha(vdir / n) for n in ("stdout.log", "stderr.log")})
    c = rec["contamination"]
    rec["clean_timing"] = bool(rec["telemetry_ok"] and rec.get("regime_ok", True) and c is not None
                               and c["foreign_share"] <= args.contamination_limit)
    rec["scored"] = bool(rec["clean_timing"] and not problems)
    if sample and stages and stages["read_ns"]:
        rec["read_bytes_per_s"] = sample["valid_bytes"] / (stages["read_ns"] / 1e9)
    if sample and stages and phase == "roundtrip" and stages["put_ns"] + stages["commit_ns"]:
        rec["write_bytes_per_s"] = sample["valid_bytes"] / ((stages["put_ns"] + stages["commit_ns"]) / 1e9)
    (vdir / "visit.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def verdict(pairs_fwd, pairs_rev):
    ratios = pairs_fwd + pairs_rev
    if min(len(pairs_fwd), len(pairs_rev)) < 4:
        return "insufficient", None
    med = statistics.median(ratios)
    up = [sum(r > 1 for r in p) for p in (pairs_fwd, pairs_rev)]
    down = [sum(r < 1 for r in p) for p in (pairs_fwd, pairs_rev)]
    if med >= 1.05 and min(up) >= 4:
        return "winner", med
    if med <= 0.95 and min(down) >= 4:
        return "loser", med
    return "flat", med


def summarize(visits, sizes, rounds):
    out = {}
    for size in sizes:
        for phase in ("roundtrip", "restore"):
            for metric in (("read_bytes_per_s", "write_bytes_per_s") if phase == "roundtrip" else ("read_bytes_per_s",)):
                cell = {}
                base = {v["round"]: v for v in visits if v["size"] == size and v["phase"] == phase
                        and v["mode"] == "buffered" and v["scored"] and v.get(metric)}
                for mode in MODES:
                    mine = [v for v in visits if v["size"] == size and v["phase"] == phase and v["mode"] == mode]
                    vals = [v[metric] for v in mine if v["scored"] and v.get(metric)]
                    entry = {"n_scored": len(vals), "n": len(mine),
                             "median": statistics.median(vals) if vals else None,
                             "min": min(vals) if vals else None, "max": max(vals) if vals else None}
                    if mode != "buffered":
                        fwd = [v[metric] / base[v["round"]][metric] for v in mine
                               if v["scored"] and v.get(metric) and v["round"] in base and v["round"] % 2 == 1]
                        rev = [v[metric] / base[v["round"]][metric] for v in mine
                               if v["scored"] and v.get(metric) and v["round"] in base and v["round"] % 2 == 0]
                        entry["verdict_vs_buffered"], entry["median_ratio"] = verdict(fwd, rev)
                    cell[mode] = entry
                out[f"{phase}/{size}/{metric}"] = cell
    return out


def run(args):
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.stub_no_lock:
        B.require(Path(args.bench).name.startswith("m1-stub") or args.bench.endswith("debug/storage-bench"),
                  "--stub-no-lock is only for a stub or a debug build")
        lock = {"stub": True}
    else:
        proc = subprocess.run([sys.executable, str(ROOT / "tools/tier-lock-proof.py"), "--fd", str(args.lock_fd),
                               "--lock", B.LOCKS[args.rig]], pass_fds=(args.lock_fd,), capture_output=True, text=True)
        B.require(proc.returncode == 0, f"lock proof failed: {proc.stderr.strip()}")
        lock = json.loads(proc.stdout)
    proof, identity, leaves, top = RUNNER.proof_view(args.proof)
    root = Path(args.root)
    B.require(B.filesystem_identity(root) == identity, "B1 root is not on the proven filesystem")
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else SIZES
    (args.out / "identity.json").write_text(json.dumps({"bench_sha256": RUNNER.sha(args.bench), "script_sha256": RUNNER.sha(__file__),
                                                        "lock": lock, "sizes": sizes, "rounds": args.rounds,
                                                        "leaves": leaves, "top": top, "qualified": False}, indent=1) + "\n")
    visits = []
    for size in sizes:
        obj = root / f"restore-{size}"
        prep = subprocess.run([str(args.bench), "roundtrip", str(obj), str(size), "buffered"],
                              capture_output=True, text=True)
        (args.out / f"prepare-{size}.log").write_text(prep.stdout + "\n--- stderr ---\n" + prep.stderr)
        B.require(prep.returncode == 0, f"restore object for {size} failed; see prepare-{size}.log")
        for r in range(args.rounds):
            order = MODES if r % 2 == 0 else MODES[::-1]
            for mode in order:
                for phase in ("roundtrip", "restore"):
                    target = root / f"rt-{size}-r{r + 1}-{mode}" if phase == "roundtrip" else obj
                    v = visit(args, args.bench, phase, mode, size, target,
                              args.out / f"{size}/r{r + 1:02d}-{mode}-{phase}", identity, leaves, top)
                    v["round"] = r + 1
                    (args.out / f"{size}/r{r + 1:02d}-{mode}-{phase}/visit.json").write_text(json.dumps(v, indent=1) + "\n")
                    visits.append(v)
                    if phase == "roundtrip":
                        shutil.rmtree(target)
                    print(f"M1-B1 size={size} r{r + 1} {mode} {phase} scored={v['scored']} "
                          f"read={v.get('read_bytes_per_s', 0) / 1e6:.1f}MB/s problems={v['problems']}", flush=True)
        shutil.rmtree(obj)
    summary = {"cells": summarize(visits, sizes, args.rounds), "visits": len(visits),
               "failed": sum(1 for v in visits if v["problems"]), "qualified": False}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for key, cell in summary["cells"].items():
        print("M1-B1-CELL " + key + " " + " ".join(
            f"{m}={c['median'] / 1e6:.1f}MB/s(n={c['n_scored']})" + (f",{c['verdict_vs_buffered']}" if "verdict_vs_buffered" in c else "")
            for m, c in cell.items() if c["median"] is not None), flush=True)
    return 0 if not summary["failed"] else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for name in ("--bench", "--root", "--proof", "--out"):
        r.add_argument(name, required=True)
    r.add_argument("--rig", choices=["pro-single", "rtx5090"], default="pro-single")
    r.add_argument("--lock-fd", type=int)
    r.add_argument("--rounds", type=int, default=10)
    r.add_argument("--sizes")
    r.add_argument("--contamination-limit", type=float, default=0.02)
    r.add_argument("--stub-no-lock", action="store_true")
    args = ap.parse_args(argv)
    B.require(args.stub_no_lock or args.lock_fd is not None, "--lock-fd (inherited canonical lock) required")
    B.require(args.stub_no_lock or (args.rounds == 10 and not args.sizes and args.contamination_limit == 0.02),
              "the registered protocol is 10 rounds over the six sizes at a 2% co-tenancy limit")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
