#!/usr/bin/env python3
"""drafter-admission-gate.py: a DFlash request is never admitted with its draft plane uncharged.

The defect (memra#1173): when a spec request's full charge (session + draft state + reserve)
does not fit, the VRAM gate reads headroom again without the draft-state term and admits the
request when that fits. On the MTP route the term is a draft-graph capture, which the
pre-capture gate refuses later. On the DFlash route the term is the drafter KV plane, which
every drafter session allocates when it is created: the request was admitted with a plane
nothing had charged (`[admission] eager-arm admit: ... draft-state NMB uncharged`). For a
drafter with no sliding layer the plane is `(cap + block) x 40,960 B`: 1,229 MB at a 30k cap.
The fix refuses such a request its drafter (`[admission] drafter refused: ... arm A`) and serves
it plain. The receipt names the arm and prints the three numbers it decided on:

  idle-fits        idle card, drained: without <= available < with
  idle-short       idle card, drained: available < without (the plain charge is smaller still:
                   a DFlash walker is charged the whole prompt's stack, a plain prime one take)
  busy-fits        a peer active: without <= available < with
  busy-unfundable  a peer active: available < without, and the full charge cannot fit even
                   after the peer retires

Serving shape, one card, the real `memra-server` with a DFlash drafter attached
(`MEMRA_DSPARK_SPEC=1`, `MEMRA_DSPARK_DRAFT`, `MEMRA_SPEC_GATE_LOW=2` so a request beside one
peer still speculates), prefix cache and whole-session pools off so every request is a cold
prime. Three boots:

  reference boot   `MEMRA_SERVE_SPEC=0`: the request greedy and sampled (seed 1234), the plain
                   reference texts.
  measured boot    the request greedy, then sampled, on the idle card: full admissions, the
                   drafter reference texts, cost C and draft state D from the server's lines.
                   Then a BALLAST process (plain cuMemAlloc chunks through libcuda, resized
                   between requests) sets effective free (driver free + pool cached) for each
                   step, aimed from the last receipt's own numbers (from C + D + reserve until
                   one exists) and corrected from each step's outcome:
                     idle window   the middle of [without, with), greedy then sampled
                     idle band     --band-mb under `without`
                     busy window   the same middle while a peer request decodes
                     busy band     --busy-band-mb under `without` while a peer decodes (under
                                   the peer's own charge the request would wait for it)
                   A step whose ballast the driver cannot give is skipped and named.
  replay boot      `MEMRA_STEP_OOM_FAULT=1`: the request's first step reports a synthetic CUDA
                   OOM, the session parks and replays. The gate prices a replay from its K,
                   which is 0 for every DFlash session.
No admission door is touched.

Verdicts (every number from the server's own lines, /metrics and nvidia-smi):
  V1 window_reached    two idle steps landed in the window (an `eager-arm admit` line or an
                       `idle-fits` receipt). Otherwise REFUSED: the cell did not test the arm.
  V2 plane_charged     zero `[admission] eager-arm admit:` lines.
  V3 served_plain      every step with a refusal receipt answered 200 with tokens and no
                       `usage.spec`, printed `[dspark] declined: drafter refused at admission`,
                       and moved /metrics `served_plain` and `admission_drafter_refusals` by one.
  V4 plane_not_held    every idle refused step's peak pool use stayed at least D/2 under a full
                       admission's: the plane was not allocated.
  V5 no_oom            no `out of memory` line outside the replay boot's synthetic one, no 5xx,
                       the server alive with /health 200, no `panicked`, `[worker] FATAL` or
                       `[worker] respawn` line.
  V6 same_text         an idle refused greedy step's text equals the drafter reference and the
                       plain reference; an idle refused sampled step's text equals the plain
                       reference; a step that speculated wrote its drafter reference. (Busy
                       steps decode beside a peer and are not compared.)
  V7 every_arm         each of the four arms was taken at least once, and each receipt's
                       numbers agree with its arm.
  V8 replay_charged    the replay boot's request answered 200, its fault fired and it parked,
                       and it did not speculate on a plain charge: `usage.spec` only if its last
                       `request cost` line reads `path=spec`.
On the unfixed tree V2, V3, V4, V7 and V8 fail (the idle band answers HTTP 400).

Without --draft the boot serves the MTP route, where the second arm stays: M2 every window
step printed `eager-arm admit` and the pre-capture gate's `capture skipped pre-attempt`
refusal (the capture state the request was not charged for was not allocated), M3 it answered
200 with tokens, plus V1, V5 and V6. Peak pool use is printed as information only there: a
capture's state is smaller than the difference between the eager and the captured draft
programs' transients. No band, busy or replay step runs.

`summary.json` carries per step: effective free before the request, the ballast held, the
minimum effective free and driver free seen during the request (the headroom the admission
left), peak pool use, status, tokens, whether the request speculated, and the parsed receipt.

Exit 0 = every verdict held; 1 = a verdict failed; 2 = REFUSED (precondition: lock, port,
a ballast the driver cannot hold, the window not reached).

usage: drafter-admission-gate.py [--external-lock FD] --model GGUF [--draft DIR] --bin memra-server \\
           --out NEW_DIR [--port N] [--ctx N] [--prompt-words N] [--max-tokens N] \\
           [--max-steps N] [--band-mb N] [--busy-band-mb N]
Lock: the canonical rig lock only (`/tmp/memra-gpu.lock` or `/tmp/memra-5090.lock`), held for
the whole cell; `--external-lock FD` inherits the caller's FD (`gpulock run` passes FD 9) and
is verified with tools/tier-lock-proof.py before anything binds a port or boots a server.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")
MIB = 1 << 20
MB = 1_000_000  # the server's admission lines print decimal megabytes
# SPEC_SHRINK_RESERVE (worker.rs): the transient floor a spec-capable model reserves until the
# boot calibration measures its own. Used only to place the first ballast step.
STATIC_RESERVE = 1536 * MIB
WORDS = (
    "amber basalt cobalt delta ember falcon garnet harbor indigo jasper kestrel lumen "
    "marble nectar orchid pewter quartz raven saffron timber umber velvet willow yonder "
    "zenith anchor beacon cinder dune ferry glacier hollow isle juniper knoll lagoon "
    "meadow north orbit prairie quarry ridge summit tundra upland valley wharf"
).split()

RE_COST = re.compile(r"\[admission\] request cost: model=\"[^\"]+\" ctx=(\d+) path=(\w+) = .* = (\d+)MB$")
RE_DRAFT = re.compile(r"\[admission\] per-session draft-state charge: model=\"[^\"]+\" \+(\d+)MB \(")
RE_FLOOR = re.compile(r"\[admit-cal\] boot calibration done: .*? transient floor (\d+)MB \(static was")
RE_RECEIPT = re.compile(
    r"\[admission\] drafter refused: .* draft plane (\d+)MB does not fit \(available (\d+)MB, "
    r"required (\d+)MB with it, (\d+)MB without(?:; arm ([a-z-]+))?\)"
)
EAGER = "[admission] eager-arm admit:"
REFUSED = "[admission] drafter refused:"
DECLINED = "[dspark] declined: drafter refused at admission"
CAPTURE_SKIPPED = "capture skipped pre-attempt"
FAULT_FIRED = "MEMRA_STEP_OOM_FAULT fired"
SYNTHETIC_OOM = "synthetic"
BAD_LINES = ("out of memory", "CUDA_ERROR_OUT_OF_MEMORY", "panicked", "[worker] FATAL", "[worker] respawn")
ARMS = ("idle-fits", "idle-short", "busy-fits", "busy-unfundable")


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


def prose(seed: int, words: int) -> str:
    rng = random.Random(seed)
    out = [f"Inventory ledger {seed}."]
    n = 0
    item = 1
    while n < words:
        k = rng.randint(5, 9)
        out.append(f"Item {item}: " + " ".join(rng.choice(WORDS) for _ in range(k)) + ".")
        n += k + 2
        item += 1
    out.append("Summary of the ledger in one sentence:")
    return "\n".join(out)


def gpu_row() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,enforced.power.limit",
         "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    name, used, total, power = [p.strip() for p in out.split(",")]
    return {"name": name, "used_mib": int(used), "total_mib": int(total), "enforced_power_limit_w": power}


def ballast_hold() -> None:
    """Child mode: hold plain driver allocations (no pool, no stream): one more chunk per
    stdin number, the newest freed on a `-` line."""
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
        if line.strip() == "-":  # free the newest chunk
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
    """One process holding plain device memory, grown between requests."""

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

    def set(self, total: int) -> bool:
        """Hold `total` bytes: grow by one chunk, or free the newest chunks and regrow. False
        when the driver cannot give the growth: the ballast takes driver-free memory only, and
        what the server's pool holds cached is not the ballast's to take."""
        while self.total > total:
            self._send("-")
        return total <= self.total or self._send(str(total - self.total))

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class Server:
    def __init__(self, binary: str, model: str, draft: str | None, port: int, ctx: int, log: Path,
                 spec: bool = True, extra_env: dict | None = None):
        self.binary, self.model, self.draft, self.port, self.ctx, self.log = binary, model, draft, port, ctx, log
        self.spec = spec
        self.extra_env = extra_env or {}
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"

    def boot(self) -> None:
        if self.probe():
            refuse(f"port {self.port} already serving; refusing to boot over it")
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "MEMRA_COMPAT": "openai",
                "MEMRA_MODELS": f"gate={self.model}",
                "MEMRA_ADDR": f"127.0.0.1:{self.port}",
                "MEMRA_CTX": str(self.ctx),
                "MEMRA_MAX_SESSIONS": "4",
                # Every request a cold prime: no prefix entry or parked session moves the
                # headroom between steps or changes a request's route.
                "MEMRA_PREFIX_CACHE_MB": "0",
                "MEMRA_REUSE_POOL": "0",
                "MEMRA_TIMEOUT_MS_MAX": "240000",
            }
        )
        if self.draft is not None:
            env.update({"MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": self.draft})
        if not self.spec:
            env["MEMRA_SERVE_SPEC"] = "0"  # the plain reference boot
        env.update(self.extra_env)
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

    def health(self) -> int:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=10) as f:
                return f.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return 0

    def complete(self, prompt: str, max_tokens: int, sampled: bool = False) -> dict:
        body = {"model": "gate", "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0.7 if sampled else 0, "seed": 1234, "timeout_ms": 240000}
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
            "speculated": usage.get("spec") is not None,
            "sampled": sampled,
            "text": payload["choices"][0]["text"] if status == 200 else None,
            "error": None if status == 200 else payload,
        }


def effective_free(m: dict) -> int:
    return int(m["cuda_driver_free_bytes"]) + int(m["cuda_pool_cached_bytes"])


class Sampler(threading.Thread):
    """/metrics every 50 ms while one request runs: the headroom the admission left."""

    def __init__(self, server: Server):
        super().__init__(daemon=True)
        self.server = server
        self.halt = threading.Event()
        self.min_effective = self.min_driver = None
        self.peak_pool_used = 0
        self.samples = 0

    def run(self) -> None:
        while not self.halt.is_set():
            try:
                m = self.server.metrics()
            except Exception:
                self.halt.wait(0.05)
                continue
            eff, drv = effective_free(m), int(m["cuda_driver_free_bytes"])
            self.min_effective = eff if self.min_effective is None else min(self.min_effective, eff)
            self.min_driver = drv if self.min_driver is None else min(self.min_driver, drv)
            self.peak_pool_used = max(self.peak_pool_used, int(m["cuda_pool_used_bytes"]))
            self.samples += 1
            self.halt.wait(0.05)


class Peer(threading.Thread):
    """One streamed generation held open: the busy card. Short enough that its own admission
    charge (what its retirement could give back) stays under the busy band's shortfall, long
    enough to outlast the measured request's pricing by seconds."""

    def __init__(self, server: Server, max_tokens: int = 1000):
        super().__init__(daemon=True)
        self.server, self.max_tokens = server, max_tokens
        self.started, self.halt = threading.Event(), threading.Event()
        self.tokens = 0
        self.error = None

    def run(self) -> None:
        body = {"model": "gate", "prompt": "Write a long, detailed story about a lighthouse keeper.\n",
                "max_tokens": self.max_tokens, "temperature": 0, "stream": True, "timeout_ms": 240000}
        req = urllib.request.Request(
            f"{self.server.base}/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as f:
                for raw in f:
                    if raw.startswith(b"data:") and b"[DONE]" not in raw:
                        self.tokens += 1
                        self.started.set()
                    if self.halt.is_set():
                        break  # closing the stream aborts the session
        except Exception as exc:  # recorded, never swallowed
            self.error = repr(exc)
        self.started.set()


def parse_receipt(lines: list[str]) -> dict | None:
    for l in lines:
        m = RE_RECEIPT.search(l)
        if m:
            return {"plane_mb": int(m.group(1)), "available_mb": int(m.group(2)), "with_mb": int(m.group(3)),
                    "without_mb": int(m.group(4)), "arm": m.group(5)}
    return None


def receipt_consistent(r: dict) -> bool:
    """The receipt's numbers agree with the arm it names."""
    if r["arm"] in ("idle-fits", "busy-fits"):
        return r["without_mb"] <= r["available_mb"] < r["with_mb"]
    if r["arm"] in ("idle-short", "busy-unfundable"):
        return r["available_mb"] < r["without_mb"]
    return False


def run_step(server: Server, step, kind: str, words: int, max_tokens: int, ballast_total: int,
             sampled: bool = False, before_free: int | None = None) -> dict:
    """One request. `before_free` is effective free as the caller set it (an idle worker does
    not refresh /metrics, so the reading before a request is the caller's arithmetic)."""
    before = server.metrics()
    offset = server.log.stat().st_size
    sampler = Sampler(server)
    sampler.start()
    # One prompt for every step: the caches are off, and V6 compares texts across steps.
    resp = server.complete(prose(7000, words), max_tokens, sampled)
    sampler.halt.set()
    sampler.join(timeout=5)
    threading.Event().wait(0.5)
    after = server.metrics()
    with server.log.open("rb") as f:
        f.seek(offset)
        lines = f.read().decode(errors="replace").splitlines()
    costs = [(m.group(2), int(m.group(3))) for l in lines if (m := RE_COST.search(l))]
    return {
        "step": step,
        "kind": kind,
        "ballast_bytes": ballast_total,
        "effective_free_before": effective_free(before) if before_free is None else before_free,
        "pool_used_before": int(before["cuda_pool_used_bytes"]),
        "min_effective_free": sampler.min_effective,
        "min_driver_free": sampler.min_driver,
        "peak_pool_used": sampler.peak_pool_used,
        "samples": sampler.samples,
        "effective_free_after": effective_free(after),
        "served_plain_delta": int(after.get("served_plain", 0)) - int(before.get("served_plain", 0)),
        "served_dspark_delta": int(after.get("served_dspark", 0)) - int(before.get("served_dspark", 0)),
        "drafter_refusals_delta": (
            int(after["admission_drafter_refusals"]) - int(before["admission_drafter_refusals"])
            if "admission_drafter_refusals" in after else None
        ),
        "eager_lines": [l for l in lines if EAGER in l],
        "refused_lines": [l for l in lines if REFUSED in l],
        "receipt": parse_receipt(lines),
        "declined_lines": [l for l in lines if DECLINED in l],
        "capture_skipped_lines": [l for l in lines if CAPTURE_SKIPPED in l],
        "fault_lines": [l for l in lines if FAULT_FIRED in l],
        "cost_lines": costs,
        "spec_cost_mb": next((c for p, c in reversed(costs) if p == "spec"), None),
        "last_cost_path": costs[-1][0] if costs else None,
        "draft_mb": next((int(m.group(1)) for l in lines if (m := RE_DRAFT.search(l))), None),
        **resp,
    }


def in_window(row: dict) -> bool:
    """The step fit only without its draft state: main's eager line, or a fits receipt."""
    r = row.get("receipt")
    return bool(row["eager_lines"]) or bool(r and r["arm"] in ("idle-fits", "busy-fits", None))


def verdicts(rows: list[dict], draft_bytes: int, log_text: str, alive: bool, health: int, refs: dict,
             full: dict, dflash: bool = True, replay: dict | None = None, replay_log: str = "") -> dict:
    """The verdicts over the recorded steps: pure, so the gate's teeth are testable
    (tools/test_drafter_admission_gate.py).

    `rows` are the ballast steps (kind: idle-window, idle-band, busy-window, busy-band); `full`
    is a full speculating admission on the idle card whose /metrics reading before it is live
    (the boot's second request: the worker publishes none before its first); `refs` holds the
    four reference texts (`spec_greedy`, `spec_sampled`, `plain_greedy`, `plain_sampled`);
    `replay` is the replay boot's request (DFlash only)."""
    window = [r for r in rows if r["kind"] == "idle-window" and in_window(r)]
    refused = [r for r in rows if r["refused_lines"]]
    idle_refused = [r for r in refused if r["kind"].startswith("idle")]
    v = {}
    v["V1 window_reached"] = (len(window) >= 2, f"{len(window)} idle window steps of {len(rows)} steps (want 2)")
    eager = log_text.count(EAGER)
    full_peak = full["peak_pool_used"] - full["pool_used_before"]

    def peak(r):
        return r["peak_pool_used"] - r["pool_used_before"]

    if dflash:
        v["V2 plane_charged"] = (eager == 0, f"{eager} eager-arm admit lines on a DFlash boot")
        plain = [
            r["status"] == 200 and (r["completion_tokens"] or 0) > 0 and not r["speculated"]
            and len(r["declined_lines"]) == 1 and r["served_plain_delta"] == 1
            and r["drafter_refusals_delta"] == 1
            for r in refused
        ]
        v["V3 served_plain"] = (
            bool(plain) and all(plain),
            "; ".join(
                f"step {r['step']}: status {r['status']} tokens {r['completion_tokens']} speculated "
                f"{r['speculated']} declined {len(r['declined_lines'])} served_plain +{r['served_plain_delta']} "
                f"refusals +{r['drafter_refusals_delta']}"
                for r in refused
            ) or "no step was refused its drafter",
        )
        held = [peak(r) <= full_peak - draft_bytes // 2 for r in idle_refused]
        v["V4 plane_not_held"] = (
            bool(held) and all(held),
            f"full admission peak pool use {full_peak / MB:.0f}MB, draft plane {draft_bytes / MB:.0f}MB; idle "
            f"refused steps " + (", ".join(f"{peak(r) / MB:.0f}MB" for r in idle_refused) or "none"),
        )
    else:
        refusals = [bool(r["eager_lines"]) and bool(r["capture_skipped_lines"]) for r in window]
        v["M2 capture_refused"] = (
            bool(refusals) and all(refusals),
            "; ".join(
                f"step {r['step']}: eager-arm lines {len(r['eager_lines'])}, capture refusals "
                f"{len(r['capture_skipped_lines'])}"
                for r in window
            ),
        )
        served = [r["status"] == 200 and (r["completion_tokens"] or 0) > 0 for r in window]
        v["M3 served"] = (
            bool(served) and all(served),
            "; ".join(f"step {r['step']}: status {r['status']} tokens {r['completion_tokens']} speculated "
                      f"{r['speculated']}" for r in window),
        )
        # Not a verdict on this route: a capture's state is small beside the difference between
        # the eager and the captured draft programs' own transients (39 MB against 157 MB on
        # Qwen3.8-27B), so peak pool use cannot show its absence. M2's refusal line does.
        v["INFO peak_pool_use"] = (
            True,
            f"full admission peak pool use {full_peak / MB:.0f}MB, draft state {draft_bytes / MB:.0f}MB; window "
            f"steps " + ", ".join(f"{peak(r) / MB:.0f}MB" for r in window),
        )
    bad = {b: log_text.count(b) for b in BAD_LINES if b in log_text}
    fivexx = [r["status"] for r in rows if r["status"] >= 500]
    v["V5 no_oom"] = (
        not bad and not fivexx and alive and health == 200,
        f"bad lines {bad}, 5xx {fivexx}, alive {alive}, /health {health}",
    )
    # Idle steps that took an arm (window steps and idle refusals), each once.
    compared = window + [r for r in idle_refused if r not in window]
    same = []
    for r in compared:
        if r["speculated"]:
            same.append(r["text"] == (refs["spec_sampled"] if r["sampled"] else refs["spec_greedy"]))
        elif r["sampled"]:
            same.append(r["text"] is not None and r["text"] == refs["plain_sampled"])
        else:
            same.append(r["text"] is not None and r["text"] == refs["spec_greedy"] == refs["plain_greedy"])
    v["V6 same_text"] = (
        bool(same) and all(same),
        "; ".join(
            f"step {r['step']} {'sampled' if r['sampled'] else 'greedy'} speculated {r['speculated']}: "
            f"{'same' if ok else 'DIFFERENT'}"
            for r, ok in zip(compared, same)
        )
        + f"; greedy references agree {refs['spec_greedy'] == refs['plain_greedy']}; sampled references agree "
        f"{refs['spec_sampled'] == refs['plain_sampled']}",
    )
    if dflash:
        with_receipt = [r for r in refused if r["receipt"]]
        seen = {a: sum(1 for r in with_receipt if r["receipt"]["arm"] == a) for a in ARMS}
        consistent = [receipt_consistent(r["receipt"]) for r in with_receipt]
        kinds_ok = all(
            (r["receipt"]["arm"] or "").startswith("idle" if r["kind"].startswith("idle") else "busy")
            for r in with_receipt
        )
        v["V7 every_arm"] = (
            all(n >= 1 for n in seen.values()) and bool(consistent) and all(consistent) and kinds_ok,
            f"receipts per arm {seen}; "
            + "; ".join(
                f"step {r['step']} {r['kind']}: {r['receipt']['arm']} available {r['receipt']['available_mb']}MB "
                f"with {r['receipt']['with_mb']}MB without {r['receipt']['without_mb']}MB "
                f"{'ok' if receipt_consistent(r['receipt']) else 'INCONSISTENT'}"
                for r in with_receipt
            )
            + "".join(f"; step {r['step']} {r['kind']}: status {r['status']}, no receipt"
                      for r in rows if not r["refused_lines"] and r["kind"] != "idle-window"),
        )
        if replay is not None:
            synthetic_only = all(SYNTHETIC_OOM in l for l in replay_log.splitlines()
                                 if "out of memory" in l or "CUDA_ERROR_OUT_OF_MEMORY" in l)
            parked = bool(replay["fault_lines"]) and len(replay["cost_lines"]) >= 2
            charged = (not replay["speculated"]) or replay["last_cost_path"] == "spec"
            v["V8 replay_charged"] = (
                replay["status"] == 200 and parked and charged and synthetic_only,
                f"status {replay['status']}, fault fired {len(replay['fault_lines'])}, cost lines "
                f"{replay['cost_lines']}, speculated {replay['speculated']}, only the synthetic OOM {synthetic_only}",
            )
    return v


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["--ballast-hold"]:
        ballast_hold()
        return
    lock_fd = None
    lock_owner = "internal-canonical"
    if argv[:1] == ["--external-lock"]:
        if len(argv) < 2 or not argv[1].isdigit():
            refuse("inherited lock FD required after --external-lock")
        lock_fd, lock_owner = int(argv[1]), "collector"
        argv = argv[2:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=18173)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--prompt-words", type=int, default=13500)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--band-mb", type=int, default=150)
    ap.add_argument("--busy-band-mb", type=int, default=400)
    ap.add_argument("--gpu-lock", default=os.environ.get("MEMRA_GPU_LOCK", LOCKS[1]))
    args = ap.parse_args(argv)
    if lock_fd is None:
        if args.gpu_lock not in LOCKS:
            refuse(f"non-canonical GPU lock {args.gpu_lock}")
        fh = open(args.gpu_lock, "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            refuse("canonical GPU lock busy")
        lock_fd = fh.fileno()
    proof = subprocess.run(
        [sys.executable, str(HERE / "tier-lock-proof.py"), "--fd", str(lock_fd), "--lock", args.gpu_lock,
         "--owner", lock_owner],
        capture_output=True, text=True, pass_fds=(lock_fd,),
    )
    if proof.returncode != 0:
        refuse(f"lock proof failed: {proof.stdout.strip()} {proof.stderr.strip()}")
    for path, what in ((args.model, "model"), (args.bin, "server binary")):
        if not Path(path).is_file():
            refuse(f"{what} missing: {path}")
    if args.draft is not None and not Path(args.draft).is_dir():
        refuse(f"drafter export missing: {args.draft}")
    if args.out.exists():
        refuse(f"--out must be new: {args.out}")
    args.out.mkdir(parents=True)

    dflash = args.draft is not None
    gpu = gpu_row()
    words, ntok = args.prompt_words, args.max_tokens
    steps_file = args.out / "steps.jsonl"
    pause = threading.Event().wait

    def record(row: dict) -> dict:
        with steps_file.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    refs: dict = {}
    # The plain references: the same two requests on a boot that never speculates.
    ref_server = Server(args.bin, args.model, args.draft, args.port, args.ctx, args.out / "server-plain-ref.log",
                        spec=False)
    try:
        ref_server.boot()
        for sampled in (False, True):
            r = ref_server.complete(prose(7000, words), ntok, sampled)
            if r["status"] != 200 or r["speculated"]:
                refuse(f"plain reference request failed or speculated: {json.dumps(r)[:400]}")
            refs["plain_sampled" if sampled else "plain_greedy"] = r["text"]
    finally:
        ref_server.stop()
    pause(3.0)

    # A request beside one peer still speculates (the busy steps need the draft-state term).
    extra = {"MEMRA_SPEC_GATE_LOW": "2"} if dflash else {}
    server = Server(args.bin, args.model, args.draft, args.port, args.ctx, args.out / "server.log", extra_env=extra)
    ballast = None
    rows: list[dict] = []
    alive, health = False, 0
    try:
        server.boot()
        first = record(run_step(server, 0, "full", words, ntok, 0))
        if first["status"] != 200 or not first["speculated"] or first["eager_lines"] or first["refused_lines"]:
            refuse(f"step 0 must be a full speculating admission on the idle card: {json.dumps(first)[:600]}")
        if first["spec_cost_mb"] is None or not first["draft_mb"]:
            refuse("step 0 printed no request-cost or draft-state line")
        refs["spec_greedy"] = first["text"]
        full = record(run_step(server, 0, "full", words, ntok, 0, sampled=True))
        if full["status"] != 200 or not full["speculated"] or full["eager_lines"] or full["refused_lines"]:
            refuse(f"the sampled reference must be a full speculating admission: {json.dumps(full)[:600]}")
        refs["spec_sampled"] = full["text"]
        # The second request's cost line, when it printed one, carries the fixed residual the
        # first request taught the cost model; the first's does not.
        draft = first["draft_mb"] * MB
        floor = RE_FLOOR.search(server.log.read_text(errors="replace"))
        reserve = int(floor.group(1)) * MIB if floor else STATIC_RESERVE  # that line prints MiB
        est = {"with": (full["spec_cost_mb"] or first["spec_cost_mb"]) * MB + draft + reserve}
        est["without"] = est["with"] - draft
        first_estimate = dict(est)
        # Effective free with no ballast, as the last request left it. An idle worker does not
        # refresh /metrics, so what the ballast process takes is accounted here, not read back:
        # its CUDA context first (a few hundred MB, read from nvidia-smi), then its chunks.
        used_before = gpu_row()["used_mib"]
        ballast = Ballast(args.out / "ballast.log")
        pause(1.0)
        context_bytes = max(0, gpu_row()["used_mib"] - used_before) * MIB
        base = {"free": effective_free(server.metrics()) - context_bytes}  # idle, no ballast chunks

        def learn(row: dict) -> None:
            """Move the estimate to what the step showed: the receipt's own numbers, else the
            newest spec charge; and re-base idle free on the reading the request left."""
            if row["receipt"]:
                est["with"], est["without"] = row["receipt"]["with_mb"] * MB, row["receipt"]["without_mb"] * MB
            elif row["spec_cost_mb"] is not None:
                est["with"] = row["spec_cost_mb"] * MB + draft + reserve
                est["without"] = est["with"] - draft
            base["free"] = row["effective_free_after"] + row["ballast_bytes"]

        skipped: list[str] = []

        def skip(step: int, kind: str, target: int) -> None:
            skipped.append(f"step {step} {kind}: not placed, the driver could not give the ballast for "
                           f"{target / MB:.0f}MB effective free")
            print(skipped[-1], flush=True)

        def idle_step(step: int, kind: str, target: int, sampled: bool = False) -> dict | None:
            if not ballast.set(max(0, base["free"] - target)):
                skip(step, kind, target)
                return None
            pause(1.0)
            row = run_step(server, step, kind, words, ntok, ballast.total, sampled,
                           before_free=base["free"] - ballast.total)
            row["target_effective_free"] = target
            record(row)
            rows.append(row)
            print(
                f"step {step} {kind}: free {row['effective_free_before'] / MB:.0f}MB (ballast "
                f"{ballast.total / MB:.0f}MB) status {row['status']} {'sampled' if sampled else 'greedy'} "
                f"speculated {row['speculated']} receipt {row['receipt'] and row['receipt']['arm']} "
                f"eager {len(row['eager_lines'])}",
                flush=True,
            )
            learn(row)
            return row

        def idle_again() -> None:
            """The card idle again after a peer: no active session, then a settled reading."""
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and int(server.metrics().get("active_sessions", 0)) != 0:
                pause(0.5)
            pause(1.0)

        def busy_step(step: int, kind: str, offset_from_without: int) -> dict | None:
            peer = Peer(server)
            peer.start()
            if not peer.started.wait(90) or peer.error or not peer.is_alive():
                print(f"step {step} {kind}: the peer did not start ({peer.error})", flush=True)
                peer.halt.set()
                return None
            pause(1.0)
            live = effective_free(server.metrics())  # the peer keeps /metrics fresh
            target = est["without"] + offset_from_without
            if not ballast.set(max(0, ballast.total + live - target)):
                skip(step, kind, target)
                peer.halt.set()
                peer.join(timeout=60)
                idle_again()
                return None
            pause(1.0)
            before_free = effective_free(server.metrics())
            row = run_step(server, step, kind, words, ntok, ballast.total, before_free=before_free)
            row["target_effective_free"] = target
            row["peer_tokens_at_end"] = peer.tokens
            row["peer_alive_at_end"] = peer.is_alive()
            peer.halt.set()
            peer.join(timeout=60)
            idle_again()
            row["effective_free_after"] = effective_free(server.metrics())
            record(row)
            rows.append(row)
            print(
                f"step {step} {kind}: free {before_free / MB:.0f}MB (ballast {ballast.total / MB:.0f}MB) status "
                f"{row['status']} speculated {row['speculated']} receipt {row['receipt'] and row['receipt']['arm']} "
                f"eager {len(row['eager_lines'])} peer alive {row['peer_alive_at_end']}",
                flush=True,
            )
            learn(row)
            return row

        # Idle window: the middle of [without, with), greedy then sampled. A full admission
        # means the aim was high, an HTTP 400 that it was low: move a third of the plane.
        step, hits = 0, 0
        while hits < 2 and step < args.max_steps:
            step += 1
            row = idle_step(step, "idle-window", est["without"] + draft // 2, sampled=hits == 1)
            if row is None:
                break
            if in_window(row):
                hits += 1
            elif row["status"] == 200 and row["speculated"] and row["spec_cost_mb"] is None:
                est["with"] -= draft // 3
                est["without"] -= draft // 3
            elif row["status"] != 200:
                est["with"] += draft // 3
                est["without"] += draft // 3
        if dflash and hits >= 2:
            step += 1
            idle_step(step, "idle-band", est["without"] - args.band_mb * MB)
            for kind, offset in (("busy-window", draft // 2), ("busy-band", -args.busy_band_mb * MB)):
                for _ in range(2):  # one retry: the first aim reads a card the peer just changed
                    step += 1
                    row = busy_step(step, kind, offset)
                    if row is not None and (row["refused_lines"] or row["eager_lines"]):
                        break
        pause(1.0)
        alive = server.proc is not None and server.proc.poll() is None
        health = server.health() if alive else 0
    finally:
        if ballast is not None:
            ballast.stop()
        server.stop()
    log_text = server.log.read_text(errors="replace")

    replay, replay_log = None, ""
    if dflash:
        pause(3.0)
        replay_server = Server(args.bin, args.model, args.draft, args.port, args.ctx, args.out / "server-replay.log",
                               extra_env={"MEMRA_STEP_OOM_FAULT": "1"})
        try:
            replay_server.boot()
            replay = record(run_step(replay_server, "replay", "replay", words, ntok, 0))
            print(
                f"replay step: status {replay['status']} speculated {replay['speculated']} fault "
                f"{len(replay['fault_lines'])} cost lines {replay['cost_lines']}",
                flush=True,
            )
        finally:
            replay_server.stop()
        replay_log = replay_server.log.read_text(errors="replace")

    result = verdicts(rows, draft, log_text, alive, health, refs, full, dflash, replay, replay_log)
    if skipped and "V7 every_arm" in result:
        ok, detail = result["V7 every_arm"]
        result["V7 every_arm"] = (ok, detail + "; " + "; ".join(skipped))
    summary = {
        "gpu": gpu,
        "binary": args.bin,
        "model": args.model,
        "draft": args.draft,
        "route": "dflash" if dflash else "mtp",
        "ctx": args.ctx,
        "draft_plane_bytes": draft,
        "reserve_estimate_bytes": reserve,
        "first_estimate_bytes": first_estimate,
        "ballast_context_bytes": context_bytes,
        "reference_texts": refs,
        "full_admission": full,
        "steps": rows,
        "skipped_steps": skipped,
        "replay": replay,
        "verdicts": {k: {"pass": ok, "detail": detail} for k, (ok, detail) in result.items()},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with (args.out / "VERDICTS.txt").open("w") as f:
        for k, (ok, detail) in result.items():
            line = f"{k}: {detail}" if k.startswith("INFO") else f"{k}: {'PASS' if ok else 'FAIL'} ({detail})"
            print(line, flush=True)
            f.write(line + "\n")
    if not result["V1 window_reached"][0]:
        refuse("the window was not reached: fewer than two idle steps fit only without their draft state")
    sys.exit(0 if all(ok for ok, _ in result.values()) else 1)


if __name__ == "__main__":
    main()
