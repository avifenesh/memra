#!/usr/bin/env python3
"""5090 half driver (M1-PREREG.md section D): one collector cell per B3 round, idle-gated.

  rounds --regime capped|bounded --memory-max BYTES --out DIR [--rounds 1..10] [--smoke]
  rounds --regime g2 --memory-max BYTES --out DIR      (the G2 5090 half: one idle-gated cell)
  rounds --regime handoff --memory-max BYTES --out DIR --size-bytes N --host-mb M [--rounds 1-10]
      (OWED 18, section E: round k is one cycle pair, buffered,direct for odd k, reversed for even)

For each round: wait until `/tmp/memra-5090.lock` is free (non-blocking probe) and no compute
application is on the card, recording every wait with the blocking processes; then run the
round as its own collector cell inside `systemd-run --user --scope -p CPUQuota=1200% -p
MemoryMax=<BYTES> -p MemorySwapMax=0` (the rig rule; page cache is charged to that cgroup).
If the collector loses the lock race it is retried after another wait. Every wait and attempt
goes to `waits.jsonl`. The pooled verdict is computed afterwards by `m1-b3-pool.py`.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "tools/tier-battery.py").exists())
LOCK = "/tmp/memra-5090.lock"
ART = "/data/ai-ml/hf-models/qwen36-35b-a3b-mtp-gguf-5bc3e238/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
BIN = "/home/avifenesh/spill-f-5090/bin/run-gen"
PROOF = "/home/avifenesh/.local/share/memra-lane-f-private/rtx5090/m1-proof.json"
PUBLIC_PROOF = HERE / "rtx5090/proof/PROOF.json"
PROBE = "/home/avifenesh/spill-f-5090/bin/h2d-probe"
BIN18 = "/home/avifenesh/spill-f-5090/bin18"
B2_PROMPTS = "/home/avifenesh/spill-f-5090/b2-prompts.jsonl"
B2_SCRATCH = "/data/cache/spill-f-b2"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def apps():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    return [l.strip() for l in out.stdout.splitlines() if l.strip()] if out.returncode == 0 else ["nvidia-smi failed"]


def lock_free():
    with open(LOCK, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def wait_idle(log, label):
    start, blockers = time.monotonic(), []
    while True:
        a, free = apps(), lock_free()
        if free and not a:
            break
        seen = {"lock_free": free, "apps": a}
        if seen not in blockers:
            blockers.append(seen)
        time.sleep(30)
    waited = time.monotonic() - start
    log.write(json.dumps({"utc": now(), "event": "idle", "for": label, "waited_s": round(waited, 1),
                          "blockers": blockers}) + "\n")
    log.flush()
    return waited


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--regime", choices=["capped", "bounded", "g2", "handoff"], required=True)
    ap.add_argument("--memory-max", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", default="1-10")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--size-bytes", type=int)
    ap.add_argument("--host-mb", type=int)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = map(int, a.rounds.split("-")) if "-" in a.rounds else (int(a.rounds), int(a.rounds))
    with (out / "waits.jsonl").open("a") as log:
        for k in range(lo, hi + 1):
            cell = out / f"round-{k:02d}"
            for attempt in range(1, 20):
                wait_idle(log, f"round {k} attempt {attempt}")
                target = cell if attempt == 1 else out / f"round-{k:02d}-attempt{attempt}"
                argv = ["systemd-run", "--user", "--scope", "-q", "-p", "CPUQuota=1200%",
                        "-p", f"MemoryMax={a.memory_max}", "-p", "MemorySwapMax=0",
                        sys.executable, str(ROOT / "tools/tier-battery.py"), "--rig", "rtx5090", "--timeout", "10800",
                        "--external-lock", "--out", str(target), "--execute", sys.executable]
                if a.regime == "handoff":
                    order = "buffered,direct" if k % 2 else "direct,buffered"
                    argv[-4:-4] = ["--storage-root", B2_SCRATCH, "--storage-proof", str(PUBLIC_PROOF)]
                    argv += [str(HERE / "m1-handoff-driver.py"), "run", "--gate", BIN18 + "/kv-handoff-gate",
                             "--server", BIN18 + "/memra-server", "--artifact", ART, "--prompts", B2_PROMPTS,
                             "--proof", PROOF, "--scratch", B2_SCRATCH, "--out", str(target / "visits"),
                             "--size-bytes", str(a.size_bytes), "--host-mb", str(a.host_mb), "--rig", "rtx5090",
                             "--lock-fd", "@COLLECTOR_LOCK_FD@", "--io-schedule", order]
                elif a.regime == "g2":
                    argv += [str(HERE / "m1-g2-5090.py"), "--probe", PROBE, "--out", str(target / "visits"),
                             "--lock-fd", "@COLLECTOR_LOCK_FD@"]
                else:
                    argv[-4:-4] = ["--storage-root", "/data/cache", "--storage-proof", str(PUBLIC_PROOF)]
                    argv += [str(HERE / "m1-spill-runner.py"), "run",
                             "--arms-lock", str(HERE / "m1-prereg/b3-arms.lock.json"),
                             "--regime", "cold", "--binary", BIN, "--artifact", ART, "--proof", PROOF,
                             "--out", str(target / "visits"), "--rig", "rtx5090", "--lock-fd", "@COLLECTOR_LOCK_FD@",
                             "--gpu-cotenant-gate"] + (["--bound-residency-check"] if a.regime == "bounded" else [])
                    argv += ["--rounds", "1", "--smoke"] if a.smoke else ["--only-round", str(k)]
                t0 = time.monotonic()
                with (out / f"round-{k:02d}.driver-attempt{attempt}.log").open("xb") as dl:
                    rc = subprocess.run(argv, stdout=dl, stderr=subprocess.STDOUT, cwd=ROOT).returncode
                text = (out / f"round-{k:02d}.driver-attempt{attempt}.log").read_text(errors="replace")
                lost = rc != 0 and "Resource temporarily unavailable" in text and not (target / "visits").exists()
                log.write(json.dumps({"utc": now(), "event": "cell", "round": k, "attempt": attempt, "rc": rc,
                                      "seconds": round(time.monotonic() - t0, 1), "lost_lock_race": lost,
                                      "dir": target.name}) + "\n")
                log.flush()
                print(f"M1-5090 regime={a.regime} round={k} attempt={attempt} rc={rc} lost_lock_race={lost}", flush=True)
                if not lost:
                    break
                time.sleep(10)
            if a.smoke or a.regime == "g2":
                break


if __name__ == "__main__":
    main()
