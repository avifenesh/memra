#!/usr/bin/env python3
"""M1 B0 device envelope on the proven path (M1-PREREG.md B0 and B5); fio is an instrument here.

  plan                         print every fio invocation in registered order (no I/O)
  run --dir P --proof PRIVATE_PROOF.json --out DIR --rig pro-single|rtx5090 --lock-fd N
      [--fio fio] [--size 16G] [--stub-no-lock]

Order: (1) prepare one lane-owned 16 GiB file with sequential direct writes plus fsync (no holes);
(2) the descriptive grid, N=1 per cell: random reads at 4096 / 450560 / 557056 / 860160 / 1048576
bytes, depth 1 / 2 / 16, engines psync (numjobs = depth, the worker's threads), io_uring and
libaio (iodepth = depth), engine order reversed on alternate cells; (3) the scored io_uring
screen: 557056-byte reads at depth 2 and 16, psync threads vs io_uring, 5 AB plus 5 BA visits;
(4) sequential write envelope, 4 MiB, direct and buffered-plus-fsync, 30 s each, N=1. Every fio
writes its own json+ file first; parsing reads only those files. CPU per GiB comes from the fio
process's wait4 rusage (all fio threads, --thread), device bytes from the 250 ms sampler on the
proof's leaves. The screen verdict is the registered B5 rule. Never "memra spill speed".
"""
import argparse
import json
import os
from pathlib import Path
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
RUNNER = load("runner", HERE / "m1-spill-runner.py")
SAMPLER = HERE / "m1-host-sampler.py"
SIZES = [4096, 450560, 557056, 860160, 1048576]
DEPTHS = [1, 2, 16]
ENGINES = ["psync", "io_uring", "libaio"]
SCREEN_BS, SCREEN_DEPTHS, SCREEN_PAIRS = 557056, [2, 16], 5


def job(name, fio_file, rw, bs, engine, depth, runtime, extra=()):
    jobs = depth if engine == "psync" else 1
    iodepth = 1 if engine == "psync" else depth
    return {"name": name, "rw": rw, "bs": bs, "engine": engine, "depth": depth,
            "argv": ["--name=m1", f"--filename={fio_file}", "--size=SIZE", "--direct=1", f"--rw={rw}",
                     f"--bs={bs}", f"--ioengine={engine}", f"--iodepth={iodepth}", f"--numjobs={jobs}",
                     "--thread", "--group_reporting", "--time_based", f"--runtime={runtime}",
                     "--ramp_time=2", "--randrepeat=0", "--norandommap", *extra]}


def plan(fio_file):
    steps = [{"name": "prepare", "rw": "write", "bs": 4 << 20, "engine": "psync", "depth": 1,
              "argv": ["--name=m1-prep", f"--filename={fio_file}", "--size=SIZE", "--direct=1", "--rw=write",
                       f"--bs={4 << 20}", "--ioengine=psync", "--end_fsync=1"]}]
    cell = 0
    for bs in SIZES:
        for depth in DEPTHS:
            order = ENGINES if cell % 2 == 0 else ENGINES[::-1]
            for engine in order:
                steps.append(job(f"grid-{bs}-{depth}-{engine}", fio_file, "randread", bs, engine, depth, 10))
            cell += 1
    for depth in SCREEN_DEPTHS:
        for rnd in range(SCREEN_PAIRS):
            for order in ("AB", "BA"):
                pair = ["psync", "io_uring"] if order == "AB" else ["io_uring", "psync"]
                for engine in pair:
                    steps.append(job(f"screen-d{depth}-r{rnd + 1}-{order}-{engine}", fio_file, "randread",
                                     SCREEN_BS, engine, depth, 10))
    steps.append(job("write-direct", fio_file, "write", 4 << 20, "psync", 1, 30))
    buffered = job("write-buffered-fsync", fio_file, "write", 4 << 20, "psync", 1, 30, ("--end_fsync=1",))
    buffered["argv"] = [a for a in buffered["argv"] if a != "--direct=1"] + ["--direct=0"]
    steps.append(buffered)
    return steps


def parse(path):
    data = json.loads(Path(path).read_text())
    j = data["jobs"][0]
    side = j["write"] if j["write"]["io_bytes"] and not j["read"]["io_bytes"] else j["read"]
    clat = side.get("clat_ns", {}).get("percentile", {})
    return {"bw_bytes": side["bw_bytes"], "iops": side["iops"], "io_bytes": side["io_bytes"],
            "runtime_ms": side["runtime"], "clat_p50_ns": clat.get("50.000000"),
            "clat_p99_ns": clat.get("99.000000"), "usr_cpu": j.get("usr_cpu"), "sys_cpu": j.get("sys_cpu")}


def screen_verdict(rows):
    """B5: io_uring at least 1.05x psync throughput with 4 of 5 rounds agreeing in each order, or
    matched throughput (median ratio at least 0.97) at no more than 0.9x the CPU per GiB."""
    out = {}
    for depth in SCREEN_DEPTHS:
        ratios, cpu, by_order = [], [], {"AB": [], "BA": []}
        for rnd in range(1, SCREEN_PAIRS + 1):
            for order in ("AB", "BA"):
                p = rows.get(f"screen-d{depth}-r{rnd}-{order}-psync")
                u = rows.get(f"screen-d{depth}-r{rnd}-{order}-io_uring")
                if p and u:
                    ratios.append(u["bw_bytes"] / p["bw_bytes"])
                    by_order[order].append(ratios[-1])
                    cpu.append(u["cpu_s_per_gib"] / p["cpu_s_per_gib"] if p["cpu_s_per_gib"] else None)
        if len(ratios) < 2 * SCREEN_PAIRS:
            out[f"depth{depth}"] = {"verdict": "incomplete", "pairs": len(ratios)}
            continue
        med, med_cpu = statistics.median(ratios), statistics.median([c for c in cpu if c is not None])
        faster = med >= 1.05 and all(sum(r > 1 for r in by_order[o]) >= 4 for o in ("AB", "BA"))
        cheaper = 0.97 <= med and med_cpu <= 0.9
        out[f"depth{depth}"] = {"verdict": "io_uring-justified" if (faster or cheaper) else "io_uring-not-justified",
                                "median_bw_ratio": med, "median_cpu_ratio": med_cpu, "pairs": len(ratios),
                                "ratios": ratios}
    return out


def run(args):
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.stub_no_lock:
        B.require(Path(args.fio).name.startswith("m1-stub"), "--stub-no-lock is only for the stub fio")
        lock = {"stub": True}
    else:
        proc = subprocess.run([sys.executable, str(ROOT / "tools/tier-lock-proof.py"), "--fd", str(args.lock_fd),
                               "--lock", B.LOCKS[args.rig]], pass_fds=(args.lock_fd,), capture_output=True, text=True)
        B.require(proc.returncode == 0, f"lock proof failed: {proc.stderr.strip()}")
        lock = json.loads(proc.stdout)
    proof, identity, leaves, top = RUNNER.proof_view(args.proof)
    B.require(B.filesystem_identity(Path(args.dir)) == identity, "fio directory is not on the proven filesystem")
    fio_file = Path(args.dir) / "m1-fio-envelope.bin"
    B.require(not fio_file.exists(), "a previous envelope file exists; remove it first")
    version = subprocess.run([args.fio, "--version"], capture_output=True, text=True).stdout.strip()
    (args.out / "identity.json").write_text(json.dumps({"fio": version, "size": args.size, "lock": lock,
                                                        "script_sha256": RUNNER.sha(__file__), "leaves": leaves,
                                                        "qualified": False}, indent=1) + "\n")
    rows = {}
    try:
        for step in plan(fio_file):
            raw = args.out / f"{step['name']}.json"
            argv = [args.fio, *[a.replace("SIZE", args.size) for a in step["argv"]],
                    "--output-format=json+", f"--output={raw}"]
            # Sampler first, so its device window covers the whole fio run (CPU comes from wait4).
            host = args.out / f"{step['name']}.host.jsonl"
            with (args.out / f"{step['name']}.host.jsonl.err").open("xb") as serr:
                sampler = subprocess.Popen([sys.executable, str(SAMPLER), "sample", "--devices", ",".join(leaves + [top]),
                                            "--out", str(host)], stdout=subprocess.DEVNULL, stderr=serr)
            deadline = time.monotonic() + 10
            while not (host.exists() and host.read_text().count("\n") >= 2):
                B.require(sampler.poll() is None and time.monotonic() < deadline, "sampler did not start")
                time.sleep(0.02)
            with (args.out / f"{step['name']}.stderr").open("xb") as err:
                child = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=err)
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
            sampler.send_signal(signal.SIGTERM)
            sampler.wait(timeout=10)
            B.require(child.returncode == 0, f"fio failed in {step['name']}; see {step['name']}.stderr")
            row = parse(raw)
            cpu_s = usage.ru_utime + usage.ru_stime
            row.update(name=step["name"], rw=step["rw"], bs=step["bs"], engine=step["engine"], depth=step["depth"],
                       cpu_s=cpu_s, cpu_s_per_gib=cpu_s / (row["io_bytes"] / (1 << 30)) if row["io_bytes"] else None)
            rows[step["name"]] = row
            with (args.out / "rows.jsonl").open("a") as rows_out:
                rows_out.write(json.dumps(row) + "\n")
            print(f"M1-FIO {step['name']} bw={row['bw_bytes'] / 1e6:.1f}MB/s iops={row['iops']:.0f} "
                  f"cpu_s/GiB={row['cpu_s_per_gib']}", flush=True)
    finally:
        if fio_file.exists():
            fio_file.unlink()
    grid_reads = [r for n, r in rows.items() if n.startswith("grid-")]
    summary = {"steps": len(rows), "screen": screen_verdict(rows), "qualified": False,
               "sustained_read_bytes_per_s_max": max(r["bw_bytes"] for r in grid_reads) if grid_reads else None,
               "headroom_70pct_bytes_per_s": 0.7 * max(r["bw_bytes"] for r in grid_reads) if grid_reads else None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for depth, s in summary["screen"].items():
        print(f"M1-B5-SCREEN {depth}: {s['verdict']} median_bw_ratio={s.get('median_bw_ratio')} "
              f"median_cpu_ratio={s.get('median_cpu_ratio')}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    for name in ("--dir", "--proof", "--out"):
        r.add_argument(name, required=True)
    r.add_argument("--rig", choices=["pro-single", "rtx5090"], default="pro-single")
    r.add_argument("--lock-fd", type=int)
    r.add_argument("--fio", default="fio")
    r.add_argument("--size", default="16G")
    r.add_argument("--stub-no-lock", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        steps = plan(Path("/scratch/spill-f/m1-fio-envelope.bin"))
        for s in steps:
            print(" ".join(["fio", *s["argv"]]).replace("SIZE", "16G"))
        print(f"# {len(steps)} fio invocations")
        return 0
    B.require(args.stub_no_lock or args.lock_fd is not None, "--lock-fd (inherited canonical lock) required")
    B.require(args.size == "16G" or args.stub_no_lock, "the registered envelope file is 16G")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
