#!/usr/bin/env python3
"""Anonymous-memory peak per B3 arm on the 5090 (M1-PREREG.md section D, bounded amendment).

Run only as the collector's --execute child, inside the capped regime's 20 GiB scope:
  m1-anon-peak.py --arms-lock b3-arms.lock.json --binary run-gen --artifact FILE
                  --out DIR --lock-fd @COLLECTOR_LOCK_FD@

One cold visit per arm, the B3 arm environment verbatim. `/proc/<pid>/status` is sampled every
250 ms for VmRSS, RssAnon, RssFile, RssShmem, VmLck and VmPin; the peak of RssAnon + RssShmem is
the process memory the bounded cgroup must hold besides page cache. ru_maxrss cannot be used: it
counts the file-backed mapped pages, and run-gen maps the whole artifact (smoke: 17,802,168 kB).
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load("runner", "m1-spill-runner.py")
FIELDS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck", "VmPin")


def status(pid):
    out = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in FIELDS:
                out[key + "_kB"] = int(rest.split()[0])
    except (FileNotFoundError, ProcessLookupError):
        return None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    for name in ("--arms-lock", "--binary", "--artifact", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--lock-fd", type=int, required=True)
    p.add_argument("--visit-timeout-s", type=int, default=1800)
    a = p.parse_args()
    proc = subprocess.run([sys.executable, str(R.ROOT / "tools/tier-lock-proof.py"), "--fd", str(a.lock_fd),
                           "--lock", R.B.LOCKS["rtx5090"]], pass_fds=(a.lock_fd,), capture_output=True, text=True)
    R.B.require(proc.returncode == 0, f"lock proof failed: {proc.stderr.strip()}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lock = json.loads(Path(a.arms_lock).read_text())
    R.lock_placement["expected"] = lock.get("expected_placement")
    ident = {"tool_sha256": R.sha(__file__), "binary_sha256": R.sha(a.binary), "arms_lock_sha256": R.sha(a.arms_lock),
             "lock": json.loads(proc.stdout), "qualified": False}
    (out / "identity.json").write_text(json.dumps(ident, indent=1) + "\n")
    peaks = {}
    for arm in lock["arms"]:
        vdir = out / arm["name"]
        vdir.mkdir()
        cold = R.CACHE.cold([a.artifact])
        env = R.arm_env(lock, arm)
        with (vdir / "run.log").open("xb") as log:
            child = subprocess.Popen([a.binary, a.artifact], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=R.ROOT)
        deadline = time.monotonic() + a.visit_timeout_s
        with (vdir / "status.jsonl").open("x") as samples:
            peak = {}
            while child.poll() is None and time.monotonic() < deadline:
                s = status(child.pid)
                if s:
                    samples.write(json.dumps({"mono_ns": time.monotonic_ns(), **s}) + "\n")
                    for k, v in s.items():
                        peak[k] = max(peak.get(k, 0), v)
                    both = s.get("RssAnon_kB", 0) + s.get("RssShmem_kB", 0)
                    peak["anon_plus_shmem_kB"] = max(peak.get("anon_plus_shmem_kB", 0), both)
                time.sleep(0.25)
        if child.poll() is None:
            child.kill()
        code = child.wait()
        text = (vdir / "run.log").read_text(errors="replace")
        problems = R.correctness(arm, R.parse_log(text), None, int(lock["common_env"]["MEMRA_NGEN"]),
                                 int(lock["artifact"].get("overread_bytes_per_read", 4096)))
        peaks[arm["name"]] = {"exit_code": code, "cold_ok": cold, "peak": peak, "correctness_problems": problems}
        print(f"M1-ANON-PEAK arm={arm['name']} exit={code} cold={cold} "
              f"anon_plus_shmem_kB={peak.get('anon_plus_shmem_kB')} VmRSS_kB={peak.get('VmRSS_kB')}", flush=True)
    worst = max(v["peak"].get("anon_plus_shmem_kB", 0) for v in peaks.values())
    result = {"arms": peaks, "max_anon_plus_shmem_bytes": worst * 1024,
              "bounded_memory_max_bytes": worst * 1024 + 7_000_000_000,
              "all_exit_zero": all(v["exit_code"] == 0 for v in peaks.values()),
              "all_correct": all(not v["correctness_problems"] for v in peaks.values())}
    (out / "anon-peak.json").write_text(json.dumps(result, indent=1) + "\n")
    print("RESULT " + json.dumps({k: result[k] for k in ("max_anon_plus_shmem_bytes", "bounded_memory_max_bytes",
                                                         "all_exit_zero", "all_correct")}), flush=True)
    return 0 if result["all_exit_zero"] and result["all_correct"] else 3


if __name__ == "__main__":
    sys.exit(main())
