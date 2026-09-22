#!/usr/bin/env python3
"""gpu-probe-recovery-gate.py: a missed nvidia-smi probe degrades; only the miss bound or a real fault latches.

The incident (memra#516; production B200 box, 2026-09-13): the `[gpu-watch]` canary hung past its
10 s deadline right after `Engine ready` while the engine captured CUDA graphs, `mark_gpu_fault`
latched, and `/health` answered 503 for the rest of the process's life although `nvidia-smi`
answered in 40 ms from a shell on the same box. 28 minutes of customer-visible outage from one
missed probe. Since #516 a steady-state hang is a MISS: the process is degraded and live, `/health`
publishes `worker.gpu_probe.{degraded, miss_streak, last_ok_age_ms, degraded_reason, latched_reason}`,
an answering probe clears the degradation, and the fatal fault latches when the streak reaches
`MEMRA_GPU_PROBE_MISSES`. Fatal Xid, ECC and row-remap findings latch on first sight, and a latched
fault is never cleared by an answer.

Fault injection: a fake `nvidia-smi` first on `PATH`. It answers the canary's own query from a
per-boot script (`ok`, `hang` = sleep past the deadline, `ecc` = uncorrected ECC 5) and passes
every other invocation to the real tool. The real `memra-server` boots the 9B on the plain route
with `MEMRA_GPU_WATCH_S=2 MEMRA_GPU_PROBE_TIMEOUT_S=2 MEMRA_GPU_PROBE_MISSES=3`; the gate polls
`/health` every 250 ms and records the sequence it saw.
Arms, each its own boot:
  A recover:  script ok, hang, hang, ok...   -> 200 with degraded=true streak 1 then 2, then
              degraded=false streak 0 with a fresh last_ok_age_ms; latched_reason null throughout.
  B latch:    script ok, hang, hang, hang, ok... -> 503 whose detail names 3 consecutive probes;
              still 503 with latched_reason after later answering probes.
  C fatal:    script ok, ecc, ok...          -> 503 naming ECC at once; later answers do not clear it.
Verdict line:
  GPU-PROBE-RECOVERY: recover=PASS|FAIL latch=PASS|FAIL fatal=PASS|FAIL -> PASS|FAIL
Exit 0 = PASS; 1 = a clause failed; 2 = REFUSED (lock, port, boot).

usage: gpu-probe-recovery-gate.py --model GGUF --bin memra-server --out NEW_DIR [--port N]
Lock: the canonical rig lock (`MEMRA_GPU_LOCK`, else `/tmp/memra-gpu.lock` or `/tmp/memra-5090.lock`)
held for the whole gate; under local-ci (`MEMRA_CI_LOCK_HELD=1`) the run's own hold is honored.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")

FAKE_SMI = r'''#!/bin/bash
# Fake nvidia-smi for tools/gpu-probe-recovery-gate.py (memra#516). Scripted answers for the
# canary's own query; every other invocation goes to the real tool.
DIR="$(cd "$(dirname "$0")" && pwd)"
case "$*" in
  *ecc.errors*|*memory.used*)
    n=$(cat "$DIR/calls" 2>/dev/null || echo 0); echo $((n+1)) > "$DIR/calls"
    step=$(sed -n "$((n+1))p" "$DIR/script"); [ -z "$step" ] && step=ok
    echo "$(date +%T) call=$((n+1)) step=$step" >> "$DIR/trace"
    case "$step" in
      hang) sleep 30; exit 0 ;;
      ecc)  echo "2026/09/22 00:00:00.000, 5, No, No"; exit 0 ;;
      *)    echo "2026/09/22 00:00:00.000, 0, No, No"; exit 0 ;;
    esac ;;
  *)
    REAL="$(cat "$DIR/real")"; [ -x "$REAL" ] && exec "$REAL" "$@"; exit 1 ;;
esac
'''


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


def take_lock():
    if os.environ.get("MEMRA_CI_LOCK_HELD") == "1":
        return None
    path = os.environ.get("MEMRA_GPU_LOCK") or next((p for p in LOCKS if os.path.exists(p)), LOCKS[1])
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        refuse(f"{path} is held by another campaign")
    return fd


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def health(port: int):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None
    except Exception:
        return None, None


class Boot:
    def __init__(self, a, name: str, script: list[str]):
        self.a, self.name, self.script = a, name, script
        self.dir = Path(a.out) / name
        self.dir.mkdir(parents=True)
        fake = self.dir / "fake-smi"
        fake.mkdir()
        real = shutil.which("nvidia-smi")
        if not real:
            refuse("no real nvidia-smi on PATH (the fake passes non-canary calls to it)")
        (fake / "real").write_text(real)
        (fake / "script").write_text("\n".join(script) + "\n")
        (fake / "calls").write_text("0")
        smi = fake / "nvidia-smi"
        smi.write_text(FAKE_SMI)
        smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
        self.fake = fake
        self.proc = None

    def boot(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        env["PATH"] = f"{self.fake}:{env.get('PATH', '')}"
        env.update(
            {
                "MEMRA_COMPAT": "openai",
                "MEMRA_MODELS": f"gate={self.a.model}",
                "MEMRA_ADDR": f"127.0.0.1:{self.a.port}",
                "MEMRA_SERVE_SPEC": "0",
                "MEMRA_GPU_WATCH_S": "2",
                "MEMRA_GPU_PROBE_TIMEOUT_S": "2",
                "MEMRA_GPU_PROBE_MISSES": "3",
            }
        )
        self.logf = open(self.dir / "server.log", "w")
        self.proc = subprocess.Popen([self.a.bin], env=env, stdout=self.logf, stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                self.stop()
                refuse(f"{self.name}: memra-server exited {code} during load; see {self.dir}/server.log")
            st, body = health(self.a.port)
            if st in (200, 503) and body and (body.get("worker") or {}).get("phase") in ("idle", "busy"):
                return
            time.sleep(0.5)
        self.stop()
        refuse(f"{self.name}: memra-server did not become ready in 600 s")

    def observe(self, seconds: float) -> list[dict]:
        seen = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            st, body = health(self.a.port)
            probe = ((body or {}).get("worker") or {}).get("gpu_probe") or {}
            seen.append(
                {
                    "t": round(time.monotonic() - t0, 2),
                    "status": st,
                    "degraded": probe.get("degraded"),
                    "miss_streak": probe.get("miss_streak"),
                    "last_ok_age_ms": probe.get("last_ok_age_ms"),
                    "latched": probe.get("latched_reason"),
                    "detail": (body or {}).get("detail"),
                }
            )
            time.sleep(0.25)
        return seen

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.logf.close()
        try:
            (self.dir / "fake-smi-trace.txt").write_text((self.fake / "trace").read_text())
        except FileNotFoundError:
            pass


def run_arm(a, name, script, seconds) -> list[dict]:
    b = Boot(a, name, script)
    b.boot()
    try:
        seen = b.observe(seconds)
    finally:
        b.stop()
    (b.dir / "observed.json").write_text(json.dumps(seen, indent=1))
    log = (b.dir / "server.log").read_text(errors="replace")
    (b.dir / "gpu-watch-lines.txt").write_text("\n".join(l for l in log.splitlines() if "[gpu-watch]" in l))
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18516)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        refuse(f"{out} exists; --out must be a NEW directory")
    out.mkdir(parents=True)
    if not port_free(a.port):
        refuse(f"port {a.port} busy")
    lock_fd = take_lock()
    verdicts = {}
    try:
        # Each hang costs interval (2 s) + deadline (2 s); an answer costs interval (2 s).
        recover = run_arm(a, "A-recover", ["ok", "hang", "hang", "ok"], 22)
        streaks = [s["miss_streak"] for s in recover]
        saw1 = any(s["status"] == 200 and s["degraded"] is True and s["miss_streak"] == 1 for s in recover)
        saw2 = any(s["status"] == 200 and s["degraded"] is True and s["miss_streak"] == 2 for s in recover)
        last = recover[-1]
        cleared = (
            last["status"] == 200
            and last["degraded"] is False
            and last["miss_streak"] == 0
            and last["last_ok_age_ms"] is not None
            and last["last_ok_age_ms"] < 6000
        )
        never503 = all(s["status"] == 200 and s["latched"] is None for s in recover)
        verdicts["recover"] = saw1 and saw2 and cleared and never503
        print(f"  A recover: streak_seen={sorted(set(x for x in streaks if x is not None))} saw1={saw1} saw2={saw2} cleared={cleared} never503={never503}", flush=True)

        latch = run_arm(a, "B-latch", ["ok", "hang", "hang", "hang", "ok", "ok", "ok"], 30)
        first503 = next((s for s in latch if s["status"] == 503), None)
        stays = latch[-1]["status"] == 503 and latch[-1]["latched"] is not None
        names = bool(first503 and first503["detail"] and "3 consecutive" in first503["detail"])
        before = all(s["status"] == 200 for s in latch[: latch.index(first503)]) if first503 else False
        verdicts["latch"] = first503 is not None and names and stays and before
        print(f"  B latch: first503_at={first503 and first503['t']} names_streak={names} stays_latched={stays} live_before={before}", flush=True)

        fatal = run_arm(a, "C-fatal", ["ok", "ecc", "ok", "ok"], 14)
        f503 = next((s for s in fatal if s["status"] == 503), None)
        ecc = bool(f503 and f503["detail"] and "ECC" in f503["detail"])
        fstays = fatal[-1]["status"] == 503 and fatal[-1]["latched"] is not None and "ECC" in fatal[-1]["latched"]
        verdicts["fatal"] = f503 is not None and ecc and fstays
        print(f"  C fatal: first503_at={f503 and f503['t']} names_ecc={ecc} stays_latched={fstays}", flush=True)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
    ok = all(verdicts.values())
    (out / "receipt.json").write_text(json.dumps({"model": a.model, "bin": a.bin, "verdicts": verdicts, "verdict": "PASS" if ok else "FAIL"}, indent=2))
    print(
        "GPU-PROBE-RECOVERY: " + " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in verdicts.items()) + f" -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
