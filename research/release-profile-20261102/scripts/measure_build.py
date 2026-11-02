#!/usr/bin/env python3
"""One build measurement for the release-profile A/B.

usage: measure_build.py ARM KIND RUN [extra cargo args...]
  ARM  A = workspace default release profile, B = opt-level 3 + fat LTO + codegen-units 1
  KIND cold   : delete the arm's target dir, then build
       touch-engine / touch-server : touch one source file, then rebuild
       any other value : build as-is (KIND is just a label)
Runs under the shared rig build lock at nice 15 with 2 jobs and no sccache.
Writes one JSON line to results.jsonl and the raw log under logs/.
"""
import glob, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

BASE = Path.home() / ".cache/memra-relprofile"
WT = BASE / "wt"
arm, kind, run = sys.argv[1], sys.argv[2], sys.argv[3]
extra = sys.argv[4:] or ["-p", "memra-server"]
target = BASE / f"target-{arm}"
cfg = []
if arm == "B":
    cfg = ["--config", 'profile.release.lto="fat"', "--config", "profile.release.codegen-units=1",
           "--config", "profile.release.opt-level=3"]
if kind == "cold" and target.exists():
    shutil.rmtree(target)
if kind == "touch-engine":
    os.utime(WT / "crates/memra-engine/src/lib.rs")
if kind == "touch-server":
    os.utime(WT / "crates/memra-server/src/main.rs")

env = dict(os.environ, CARGO_BUILD_JOBS="2", RUSTC_WRAPPER="", CARGO_TARGET_DIR=str(target),
           MEMRA_CUDA_ARCH="120a")
logs = BASE / "logs"; logs.mkdir(exist_ok=True)
stamp = time.strftime("%Y%m%dT%H%M%S")
log = logs / f"{stamp}-{arm}-{kind}-{run}.log"
cmd = ["flock", str(Path.home() / ".cache/memra-rig-build.lock"), "nice", "-n", "15",
       "/usr/bin/time", "-v", "cargo", *(["test", "--no-run"] if kind.startswith("testbuild") else
                                          ["test"] if kind.startswith("testrun") else ["build"]),
       "--release", "--locked", *cfg, *extra]

def coretemp():
    best = 0
    for f in glob.glob("/sys/class/hwmon/hwmon*/name"):
        if open(f).read().strip() == "coretemp":
            for t in glob.glob(os.path.dirname(f) + "/temp*_input"):
                best = max(best, int(open(t).read()) // 1000)
    return best

def mem_used_mb():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":"); m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) // 1024

samples = []
stop = threading.Event()
def sampler():
    while not stop.is_set():
        rss = 0
        for p in glob.glob("/proc/[0-9]*/status"):
            try:
                txt = open(p).read()
            except OSError:
                continue
            name = txt.split("\n", 1)[0].split()[-1]
            if name in ("rustc", "nvcc", "cicc", "ptxas", "cc", "ld", "ld.lld", "cc1plus", "fatbinary"):
                for line in txt.splitlines():
                    if line.startswith("VmRSS:"):
                        rss += int(line.split()[1]); break
        samples.append((time.time(), rss // 1024, mem_used_mb(), coretemp()))
        stop.wait(0.5)

profile = open("/sys/firmware/acpi/platform_profile").read().strip()
load0 = open("/proc/loadavg").read().split()[:3]
th = threading.Thread(target=sampler, daemon=True)
with open(log, "w") as fh:
    fh.write(" ".join(cmd) + "\n"); fh.flush()
    # wall time measured inside the lock: time -v reports it; ours includes lock wait
    t0 = time.time(); th.start()
    rc = subprocess.run(cmd, cwd=WT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    t1 = time.time(); stop.set(); th.join()
text = log.read_text()
def tv(key):
    for line in text.splitlines():
        if line.strip().startswith(key):
            return line.split(":", 1)[1].strip() if key != "Elapsed" else line.rsplit(" ", 1)[-1]
    return None
el = tv("Elapsed")
parts = [float(x) for x in el.split(":")] if el else []
wall = parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else (parts[0] * 60 + parts[1] if parts else None)
binary = target / "release/memra-server"
row = {"arm": arm, "kind": kind, "run": run, "extra": extra, "rc": rc, "stamp": stamp,
       "wall_s": wall, "user_s": float(tv("User time") or 0), "sys_s": float(tv("System time") or 0),
       "max_single_rss_mb": int(tv("Maximum resident set size") or 0) // 1024,
       "peak_compiler_rss_sum_mb": max((s[1] for s in samples), default=0),
       "peak_mem_used_mb": max((s[2] for s in samples), default=0),
       "max_coretemp_c": max((s[3] for s in samples), default=0),
       "platform_profile": profile, "platform_profile_end": open("/sys/firmware/acpi/platform_profile").read().strip(),
       "loadavg_start": load0, "lock_plus_build_s": round(t1 - t0, 1),
       "server_bytes": binary.stat().st_size if binary.exists() else None, "log": str(log)}
with open(BASE / "results.jsonl", "a") as out:
    out.write(json.dumps(row) + "\n")
with open(BASE / "samples" / f"{stamp}-{arm}-{kind}-{run}.tsv" if (BASE / "samples").mkdir(exist_ok=True) is None else "", "w") as s:
    for x in samples:
        s.write("\t".join(map(str, x)) + "\n")
print(json.dumps(row))
sys.exit(rc)
