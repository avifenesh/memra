#!/usr/bin/env python3
"""Same-pass admission gate (memra#1255): two cold DFlash requests admitted in one admission pass
must not both be charged against the same effective-free reading.

A cold DFlash session allocates nothing in `admit`: its target caches and its drafter KV plane are
built by `dspark_prime_start` in the tick that follows. On main, the gate admitted the second
request of one pass on a reading that still held the first request's whole charge, and the two
charges then spent the transient reserve between them (memra#1255: a step OOM at c = 2 on the
agentic replay, RTX 5090 Laptop, DFlash2).

The cell, on one card and one boot:

1. Learn: each of two cold prompts (A and B, about the same length, distinct text) alone on the
   idle card, to read its charge from the server's own lines (`request cost` + `draft-state
   charge`), its prompt length and its greedy text, and the boot-calibrated transient floor.
2. Keep the worker busy: one short greedy request (the keeper) decodes through the rest of the
   cell, so the pair arrives during a tick and queues for one admission pass, and `/metrics` is
   fresh (an idle worker does not refresh it).
3. Place: hold a ballast (plain driver allocations in a child process) so effective free beside
   the keeper lies between `charge_A + floor` and `charge_A + charge_B + floor` (`--place`):
   each request fits alone, the two together do not.
4. Fire A and B together (two threads), greedy.

Verdicts, read from the server log after the pair was fired:

- V1 SAME-PASS: the pair met in one admission pass: `admit` ran for both (two `[spec-k]` lines
  with the pair's prompt lengths) before the first `[prime-yield]` line, or a `[admission] settle
  wait:` line printed before it. Otherwise the run reads NOT-PLACED, not PASS.
- V2 NO-OVERLAP: the pass did not admit both: fails when `admit` ran for both before any prime
  chunk, i.e. both were charged against a reading that held room for one.
- V3 NO-OOM: no `out of memory` line, no panic or worker respawn, and the server is alive.
- V4 SERVED: both requests answered 200 with the same text as the learn pass of the same prompt
  (greedy; a wait does not change the numeric program).

Exit 0 when V1 to V4 pass, 1 when a verdict fails, 2 when the cell could not be run (refused),
3 when the pair could not be placed in one pass (NOT-PLACED).

usage: same-pass-admission-gate.py --bin BIN --model GGUF --draft DIR --out DIR [--words N]
       [--max-tokens N] [--ctx N] [--port N] [--external-lock FD | --gpu-lock PATH]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MIB = 1 << 20
MB = 1_000_000  # the server's admission lines print decimal megabytes
WORDS = (
    "amber basalt cobalt delta ember falcon garnet harbor indigo jasper kestrel lumen "
    "marble nectar orchid pewter quartz raven saffron timber umber velvet willow yonder "
    "zenith anchor beacon cinder dune ferry glacier hollow isle juniper knoll lagoon "
    "meadow north orbit prairie quarry ridge summit tundra upland valley wharf"
).split()

RE_COST = re.compile(r"\[admission\] request cost: model=\"[^\"]+\" ctx=(\d+) path=(\w+) = .* = (\d+)MB$")
RE_DRAFT = re.compile(r"\[admission\] per-session draft-state charge: model=\"[^\"]+\" \+(\d+)MB \(")
RE_FLOOR = re.compile(r"\[admit-cal\] boot calibration done: .*? transient floor (\d+)MB \(static was")
OOM_LINES = ("out of memory", "CUDA_ERROR_OUT_OF_MEMORY")
BAD_LINES = ("panicked", "[worker] FATAL", "[worker] respawn")
PRIME_LINE = "[prime-yield]"
WORKER_ADMIT_LINE = "[spec-k] "  # `admit` decided the request's K: the worker admitted it
SETTLE_LINE = "[admission] settle wait:"


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


def prose(seed: int, words: int) -> str:
    """Deterministic distinct text: no two seeds share a prefix, so no prefix entry or parked
    session can serve the other request."""
    out = [f"Record {seed}."]
    x = seed * 2654435761 % (1 << 32) or 1
    for _ in range(words):
        x = (x * 1103515245 + 12345) % (1 << 31)
        out.append(WORDS[x % len(WORDS)])
    return " ".join(out) + "\nSummarize the record above in one paragraph."


def gpu_row() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,enforced.power.limit",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    name, used, total, power = [p.strip() for p in out.split(",")]
    return {"name": name, "used_mib": int(used), "total_mib": int(total), "enforced_power_limit_w": power}


def ballast_hold() -> None:
    """Child mode: hold plain driver allocations (no pool, no stream): one more chunk per stdin
    number, the newest freed on a `-` line."""
    import ctypes

    cuda = ctypes.CDLL("libcuda.so.1")
    dev = ctypes.c_int()
    ctx = ctypes.c_void_p()
    for name, rc in (
        ("cuInit", cuda.cuInit(0)),
        ("cuDeviceGet", cuda.cuDeviceGet(ctypes.byref(dev), 0)),
        ("cuCtxCreate", cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)),
    ):
        if rc != 0:
            refuse(f"ballast: {name} failed rc={rc}")
    chunks: list[tuple[int, int]] = []
    print("BALLAST READY total=0", flush=True)
    for line in sys.stdin:
        if line.strip() == "-":
            if chunks:
                ptr, _ = chunks.pop()
                cuda.cuMemFree_v2(ctypes.c_ulonglong(ptr))
        else:
            nbytes = int(line)
            ptr = ctypes.c_ulonglong()
            rc = cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
            if rc != 0:
                print(f"BALLAST FAILED rc={rc} bytes={nbytes}", flush=True)
                continue
            chunks.append((ptr.value, nbytes))
        print(f"BALLAST HELD total={sum(n for _, n in chunks)}", flush=True)


class Ballast:
    """One process holding plain device memory."""

    def __init__(self, log: Path):
        self.log = log
        self.total = 0
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--ballast-hold"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        )
        self._expect("BALLAST READY")

    def _expect(self, prefix: str) -> str:
        line = self.proc.stdout.readline().strip()
        with self.log.open("a") as f:
            f.write(line + "\n")
        if not line.startswith(prefix):
            self.stop()
            refuse(f"ballast: wanted {prefix!r}, got {line!r}")
        return line

    def set(self, total: int) -> bool:
        while self.total > total:
            self._send("-")
        return total <= self.total or self._send(str(total - self.total))

    def _send(self, line: str) -> bool:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        answer = self.proc.stdout.readline().strip()
        with self.log.open("a") as f:
            f.write(answer + "\n")
        if answer.startswith("BALLAST FAILED"):
            return False
        if not answer.startswith("BALLAST HELD"):
            self.stop()
            refuse(f"ballast: wanted 'BALLAST HELD', got {answer!r}")
        self.total = int(answer.split("total=")[1])
        return True

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class Server:
    def __init__(self, binary: str, model: str, draft: str, port: int, ctx: int, log: Path):
        self.binary, self.model, self.draft, self.port, self.ctx, self.log = binary, model, draft, port, ctx, log
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"

    def boot(self) -> None:
        if self.probe():
            refuse(f"port {self.port} already serving; refusing to boot over it")
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_") or k == "MEMRA_GPU_LOCK"}
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "MEMRA_COMPAT": "openai",
                "MEMRA_MODELS": f"gate={self.model}",
                "MEMRA_ADDR": f"127.0.0.1:{self.port}",
                "MEMRA_CTX": str(self.ctx),
                "MEMRA_MAX_SESSIONS": "4",
                # Every request a cold prime: no prefix entry or parked session moves the
                # headroom between the learn pass and the pair or changes a request's route.
                "MEMRA_PREFIX_CACHE_MB": "0",
                "MEMRA_REUSE_POOL": "0",
                "MEMRA_TIMEOUT_MS_MAX": "240000",
                "MEMRA_DSPARK_SPEC": "1",
                "MEMRA_DSPARK_DRAFT": self.draft,
            }
        )
        with self.log.open("wb") as log:
            self.proc = subprocess.Popen([self.binary], stdout=log, stderr=subprocess.STDOUT, env=env)
        for _ in range(300):
            if self.probe():
                return
            if self.proc.poll() is not None:
                refuse(f"server died during boot (exit {self.proc.returncode}); see {self.log}")
            time.sleep(2)
        self.stop()
        refuse("server never became ready")

    def probe(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/v1/models", timeout=2):
                return True
        except Exception:
            return False

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def metrics(self) -> dict:
        with urllib.request.urlopen(f"{self.base}/metrics", timeout=10) as f:
            return json.load(f)

    def complete(self, prompt: str, max_tokens: int) -> dict:
        body = {"model": "gate", "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0, "seed": 1234, "timeout_ms": 240000}
        req = urllib.request.Request(
            f"{self.base}/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as f:
                status, payload = f.status, json.load(f)
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read().decode(errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
        usage = payload.get("usage") or {}
        return {
            "status": status,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "text": payload["choices"][0]["text"] if status == 200 else None,
            "error": None if status == 200 else payload,
        }


def effective_free(m: dict) -> int:
    return int(m["cuda_driver_free_bytes"]) + int(m["cuda_pool_cached_bytes"])


def log_lines(path: Path, start: int) -> list[str]:
    return path.read_text(errors="replace").splitlines()[start:]


def charge_of(lines: list[str]) -> int | None:
    """The first admission's charge in `lines`, in bytes: request cost plus the draft plane."""
    cost = draft = None
    for line in lines:
        if draft is None and (m := RE_DRAFT.search(line)):
            draft = int(m.group(1)) * MB
        if cost is None and (m := RE_COST.search(line)):
            if m.group(2) != "spec":
                return None
            cost = int(m.group(3)) * MB
            break
    return None if cost is None else cost + (draft or 0)


def pair_shape(lines: list[str], prompt_lens: set[int]) -> dict:
    """The pair's admissions against the first prime line after them. A pair admission is a
    `[spec-k]` line (`admit` decided its K) for one of the pair's prompt lengths; the keeper's
    short prompt never matches. A prime line is a `[prime-yield]` line, which only a prime
    chunk prints (the keeper primed before the pair arrived)."""
    def pair_admit(line: str) -> bool:
        m = re.search(r" prompt=(\d+) ", line)
        return line.startswith(WORKER_ADMIT_LINE) and m is not None and int(m.group(1)) in prompt_lens
    admitted = [i for i, l in enumerate(lines) if pair_admit(l)]
    primes = [i for i, l in enumerate(lines) if l.startswith(PRIME_LINE)]
    settles = [i for i, l in enumerate(lines) if l.startswith(SETTLE_LINE)]
    first_prime = primes[0] if primes else len(lines)
    overlap = len(admitted) >= 2 and admitted[1] < first_prime
    settled = bool(settles) and settles[0] < first_prime
    return {
        "pair_admit_lines": admitted[:6],
        "first_prime_line": first_prime if primes else None,
        "settle_lines": settles[:6],
        "placed": overlap or settled,
        "both_admitted_before_prime": overlap,
        "settle_waits": len(settles),
        "oom_lines": sum(1 for l in lines if any(p in l for p in OOM_LINES)),
        "bad_lines": sum(1 for l in lines if any(p in l for p in BAD_LINES)),
    }


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["--ballast-hold"]:
        ballast_hold()
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--words", type=int, default=3600, help="prose words per prompt (about 1.3 tokens each)")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--port", type=int, default=18176)
    ap.add_argument("--keeper-tokens", type=int, default=1200,
                    help="the keeper's max_tokens: it must outlast the placement and the pair")
    ap.add_argument("--place", type=float, default=0.5,
                    help="where in [charge_A + floor, charge_A + charge_B + floor) to place free")
    ap.add_argument("--external-lock", type=int, help="an already-held GPU lock on this fd")
    ap.add_argument("--gpu-lock", default="/tmp/memra-5090.lock")
    args = ap.parse_args(argv)
    if args.external_lock is None:
        lock = open(args.gpu_lock, "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            refuse(f"{args.gpu_lock} is held")
    args.out.mkdir(parents=True, exist_ok=True)
    serve_log = args.out / "serve.log"
    summary: dict = {"gpu": gpu_row(), "bin": args.bin, "words": args.words, "max_tokens": args.max_tokens}
    prompts = {"A": prose(11, args.words), "B": prose(23, args.words)}
    server = Server(args.bin, args.model, args.draft, args.port, args.ctx, serve_log)
    ballast = None
    rc = 0
    try:
        server.boot()
        boot_text = serve_log.read_text(errors="replace")
        m = RE_FLOOR.search(boot_text)
        if m is None:
            refuse("no boot calibration line (transient floor)")
        floor = int(m.group(1)) * MB
        # 1. Learn each request's charge and greedy text alone on the idle card.
        learn = {}
        for name, prompt in prompts.items():
            start = len(serve_log.read_text(errors="replace").splitlines())
            row = server.complete(prompt, args.max_tokens)
            time.sleep(1.0)
            charge = charge_of(log_lines(serve_log, start))
            if row["status"] != 200 or charge is None:
                refuse(f"learn {name}: status {row['status']}, charge {charge} (not a spec charge?)")
            learn[name] = {"charge": charge, **row}
        summary["floor"] = floor
        summary["learn"] = {k: {kk: v[kk] for kk in ("charge", "status", "prompt_tokens", "completion_tokens")}
                            for k, v in learn.items()}
        # 2. A keeper: one short greedy request decoding through the cell, so the worker is in a
        # tick when the pair arrives (both queue for one admission pass) and /metrics is fresh
        # (an idle worker does not refresh it).
        keeper: dict = {}
        keeper_thread = threading.Thread(
            target=lambda: keeper.update(server.complete(prose(5, 120), args.keeper_tokens)))
        keeper_thread.start()
        time.sleep(4.0)
        busy_free = effective_free(server.metrics())
        lo = learn["A"]["charge"] + floor
        hi = learn["A"]["charge"] + learn["B"]["charge"] + floor
        target = int(lo + (hi - lo) * args.place)
        summary.update({"busy_free": busy_free, "one_fits_at": lo, "two_fit_at": hi, "target_free": target})
        if busy_free < target:
            refuse(f"effective free {busy_free / MB:.0f}MB beside the keeper is under the target "
                   f"{target / MB:.0f}MB")
        ballast = Ballast(args.out / "ballast.log")
        if not ballast.set(busy_free - target):
            refuse(f"the driver could not give a {(busy_free - target) / MB:.0f}MB ballast")
        time.sleep(1.0)
        summary["ballast"] = ballast.total
        summary["free_before_pair"] = effective_free(server.metrics())
        print(f"placed: beside the keeper {busy_free / MB:.0f}MB, ballast {ballast.total / MB:.0f}MB, "
              f"after {summary['free_before_pair'] / MB:.0f}MB; one fits at {lo / MB:.0f}MB, two at "
              f"{hi / MB:.0f}MB, target {target / MB:.0f}MB", flush=True)
        if not keeper_thread.is_alive():
            refuse("the keeper finished before the pair was placed; raise --keeper-tokens")
        # 3. Fire the pair together.
        start = len(serve_log.read_text(errors="replace").splitlines())
        results: dict = {}
        barrier = threading.Barrier(2)

        def fire(name: str) -> None:
            barrier.wait()
            results[name] = server.complete(prompts[name], args.max_tokens)

        threads = [threading.Thread(target=fire, args=(n,)) for n in prompts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        keeper_thread.join()
        time.sleep(1.0)
        alive = server.proc is not None and server.proc.poll() is None
        shape = pair_shape(log_lines(serve_log, start),
                           {learn[n]["prompt_tokens"] for n in prompts})
        summary["pair"] = {n: {k: r[k] for k in ("status", "prompt_tokens", "completion_tokens", "error")}
                           for n, r in results.items()}
        summary["shape"] = shape
        summary["alive"] = alive
        summary["keeper"] = {k: keeper.get(k) for k in ("status", "completion_tokens")}
        same_text = all(results[n]["text"] == learn[n]["text"] for n in prompts)
        verdicts = {
            "V1 SAME-PASS": shape["placed"],
            "V2 NO-OVERLAP": not shape["both_admitted_before_prime"],
            "V3 NO-OOM": shape["oom_lines"] == 0 and shape["bad_lines"] == 0 and alive,
            "V4 SERVED": all(r["status"] == 200 for r in results.values()) and same_text,
        }
        summary["same_text"] = same_text
        summary["verdicts"] = verdicts
        for k, v in verdicts.items():
            print(f"{k}: {'PASS' if v else 'FAIL'}", flush=True)
        print(f"settle waits {shape['settle_waits']}, oom lines {shape['oom_lines']}", flush=True)
        if not verdicts["V1 SAME-PASS"]:
            print("NOT-PLACED: the pair did not meet in one admission pass", flush=True)
            rc = 3
        elif not all(verdicts.values()):
            rc = 1
    finally:
        if ballast is not None:
            ballast.stop()
        server.stop()
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
