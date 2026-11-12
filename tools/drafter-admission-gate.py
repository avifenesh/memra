#!/usr/bin/env python3
"""drafter-admission-gate.py: a DFlash request is never admitted with its draft plane uncharged.

The defect (memra#1173): when a spec request's full charge (session + draft state + reserve)
does not fit, the VRAM gate reads headroom again without the draft-state term and admits the
request when that fits. On the MTP route the term is a draft-graph capture, which the
pre-capture gate refuses later. On the DFlash route the term is the drafter KV plane, which
every drafter session allocates when it is created: the request was admitted with a plane
nothing had charged (`[admission] eager-arm admit: ... draft-state NMB uncharged`). For a
drafter with no sliding layer the plane is `(cap + block) x 40,960 B`: 1,229 MB at a 30k cap.

Serving shape, one card, one boot of the real `memra-server` with a DFlash drafter attached
(`MEMRA_DSPARK_SPEC=1`, `MEMRA_DSPARK_DRAFT`), prefix cache and whole-session pools off so
every request is a cold prime:

  step 0   one long greedy request on the idle card: admitted in full, speculates. Its cost
           line gives C (`[admission] request cost ... = C MB`), its draft-state line gives D
           (`per-session draft-state charge ... +D MB`), and /metrics gives effective free F0
           (driver free + pool cached) and the peak pool use of a full drafter session.
  step k   a BALLAST process (one plain cuMemAlloc through libcuda, grown between requests)
           lowers effective free by about D/3 per step, starting about D/2 above the gate's
           estimated requirement C + D + reserve, and the same shape of request (a different
           text) is sent. A step lands in the WINDOW when the server prints either the unfixed
           receipt (`eager-arm admit`) or the fixed one (`drafter refused`). The gate stops
           after --window-hits such steps. No admission door is touched.

Verdicts (every number from the server's own lines, /metrics and nvidia-smi):
  V1 window_reached   at least --window-hits steps landed in the window. Otherwise REFUSED:
                      the cell did not test the arm (card too small or too large for the shape).
  V2 plane_charged    zero `[admission] eager-arm admit:` lines: no request was admitted with
                      its draft plane uncharged. RED on the unfixed tree.
  V3 served_plain     every window step answered 200 with tokens and no `usage.spec`, printed
                      `[dspark] declined: drafter refused at admission`, and /metrics
                      `served_plain` and `admission_drafter_refusals` each rose by one per step.
  V4 plane_not_held   every window step's peak pool use stayed at least D/2 under step 0's: the
                      plane was not allocated. RED on the unfixed tree (the plane is there).
  V5 no_oom           zero `out of memory` lines, zero 5xx, the server alive with /health 200,
                      no `panicked`, `[worker] FATAL` or `[worker] respawn` line.
  V6 same_text        window steps alternate a greedy and a seeded sampled (temperature 0.7)
                      request on step 0's prompt. A greedy step's text equals both references:
                      the drafter's (step 0) and the plain one (a first boot with
                      `MEMRA_SERVE_SPEC=0`). A sampled step's text equals the reference of the
                      program that served it: plain when it did not speculate, the drafter's
                      when it did. The detail line says whether the two sampled references
                      agree (they are different random programs and need not).

Without --draft the boot serves the MTP route, where the second arm stays (`--route mtp`
verdicts): M2 every window step printed `eager-arm admit` and the pre-capture gate's `capture
skipped pre-attempt` refusal, M3 it answered 200 with tokens, M4 its peak pool use stayed at
least D/2 under step 0's (the capture state the request was not charged for was not allocated),
plus V1, V5 and V6.

`summary.json` carries per step: effective free before the request, the ballast held, the
minimum effective free and driver free seen during the request (the headroom the admission
left), peak pool use, status, tokens, and whether the request speculated.

Exit 0 = every verdict held; 1 = a verdict failed; 2 = REFUSED (precondition: lock, port,
window not reached, a request refused before the window).

usage: drafter-admission-gate.py [--external-lock FD] --model GGUF [--draft DIR] --bin memra-server \\
           --out NEW_DIR [--port N] [--ctx N] [--prompt-words N] [--max-tokens N] \\
           [--window-hits N] [--max-steps N]
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
EAGER = "[admission] eager-arm admit:"
REFUSED = "[admission] drafter refused:"
DECLINED = "[dspark] declined: drafter refused at admission"
CAPTURE_SKIPPED = "capture skipped pre-attempt"
BAD_LINES = ("out of memory", "CUDA_ERROR_OUT_OF_MEMORY", "panicked", "[worker] FATAL", "[worker] respawn")


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
    """Child mode: hold plain driver allocations (no pool, no stream); one more per stdin line."""
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
    total = 0
    print("BALLAST READY total=0", flush=True)
    for line in sys.stdin:
        nbytes = int(line)
        ptr = ctypes.c_ulonglong()
        rc = cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if rc != 0:
            print(f"BALLAST FAILED rc={rc} bytes={nbytes}", flush=True)
            continue
        total += nbytes
        print(f"BALLAST HELD total={total}", flush=True)


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

    def add(self, nbytes: int) -> None:
        self.proc.stdin.write(f"{nbytes}\n")
        self.proc.stdin.flush()
        self.total = int(self._expect("BALLAST HELD").split("total=")[1])

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
                 spec: bool = True):
        self.binary, self.model, self.draft, self.port, self.ctx, self.log = binary, model, draft, port, ctx, log
        self.spec = spec
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
                time.sleep(0.05)
                continue
            eff, drv = effective_free(m), int(m["cuda_driver_free_bytes"])
            self.min_effective = eff if self.min_effective is None else min(self.min_effective, eff)
            self.min_driver = drv if self.min_driver is None else min(self.min_driver, drv)
            self.peak_pool_used = max(self.peak_pool_used, int(m["cuda_pool_used_bytes"]))
            self.samples += 1
            time.sleep(0.05)


def run_step(server: Server, step: int, words: int, max_tokens: int, ballast_total: int,
             sampled: bool = False) -> dict:
    before = server.metrics()
    offset = server.log.stat().st_size
    sampler = Sampler(server)
    sampler.start()
    # One prompt for every step: the caches are off, and V6 compares texts across steps.
    resp = server.complete(prose(7000, words), max_tokens, sampled)
    sampler.halt.set()
    sampler.join(timeout=5)
    time.sleep(0.5)
    after = server.metrics()
    with server.log.open("rb") as f:
        f.seek(offset)
        lines = f.read().decode(errors="replace").splitlines()
    return {
        "step": step,
        "ballast_bytes": ballast_total,
        "effective_free_before": effective_free(before),
        "driver_free_before": int(before["cuda_driver_free_bytes"]),
        "pool_used_before": int(before["cuda_pool_used_bytes"]),
        "min_effective_free": sampler.min_effective,
        "min_driver_free": sampler.min_driver,
        "peak_pool_used": sampler.peak_pool_used,
        "samples": sampler.samples,
        "served_plain_delta": int(after.get("served_plain", 0)) - int(before.get("served_plain", 0)),
        "served_dspark_delta": int(after.get("served_dspark", 0)) - int(before.get("served_dspark", 0)),
        "drafter_refusals_delta": (
            int(after["admission_drafter_refusals"]) - int(before["admission_drafter_refusals"])
            if "admission_drafter_refusals" in after else None
        ),
        "eager_lines": [l for l in lines if EAGER in l],
        "refused_lines": [l for l in lines if REFUSED in l],
        "declined_lines": [l for l in lines if DECLINED in l],
        "capture_skipped_lines": [l for l in lines if CAPTURE_SKIPPED in l],
        "cost_mb": next((int(m.group(3)) for l in lines if (m := RE_COST.search(l))), None),
        "cost_path": next((m.group(2) for l in lines if (m := RE_COST.search(l))), None),
        "ctx": next((int(m.group(1)) for l in lines if (m := RE_COST.search(l))), None),
        "draft_mb": next((int(m.group(1)) for l in lines if (m := RE_DRAFT.search(l))), None),
        **resp,
    }


def in_window(row: dict) -> bool:
    return bool(row["eager_lines"] or row["refused_lines"])


def verdicts(rows: list[dict], window_hits: int, draft_bytes: int, log_text: str, alive: bool, health: int,
             refs: dict, dflash: bool = True) -> dict:
    """The verdicts over the recorded steps: pure, so the gate's teeth are testable.

    `rows[0]` is the full greedy admission; `refs` holds the four reference texts
    (`spec_greedy`, `spec_sampled`, `plain_greedy`, `plain_sampled`)."""
    full, window = rows[0], [r for r in rows[1:] if in_window(r)]
    v = {}
    v["V1 window_reached"] = (
        len(window) >= window_hits,
        f"{len(window)} window steps of {len(rows) - 1} (want {window_hits})",
    )
    eager = log_text.count(EAGER)
    if dflash:
        v["V2 plane_charged"] = (eager == 0, f"{eager} eager-arm admit lines on a DFlash boot")
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
    if dflash:
        plain = [
            r["status"] == 200 and (r["completion_tokens"] or 0) > 0 and not r["speculated"]
            and len(r["declined_lines"]) == 1 and r["served_plain_delta"] == 1
            and r["drafter_refusals_delta"] == 1
            for r in window
        ]
        v["V3 served_plain"] = (
            bool(plain) and all(plain),
            "; ".join(
                f"step {r['step']}: status {r['status']} tokens {r['completion_tokens']} speculated "
                f"{r['speculated']} declined {len(r['declined_lines'])} served_plain +{r['served_plain_delta']} "
                f"refusals +{r['drafter_refusals_delta']}"
                for r in window
            ),
        )
    full_peak = full["peak_pool_used"] - full["pool_used_before"]
    held = [(r["peak_pool_used"] - r["pool_used_before"]) <= full_peak - draft_bytes // 2 for r in window]
    v["V4 plane_not_held" if dflash else "M4 capture_not_held"] = (
        bool(held) and all(held),
        f"full step peak pool use {full_peak / MB:.0f}MB, plane {draft_bytes / MB:.0f}MB; window steps "
        + ", ".join(f"{(r['peak_pool_used'] - r['pool_used_before']) / MB:.0f}MB" for r in window),
    )
    bad = {b: log_text.count(b) for b in BAD_LINES if b in log_text}
    fivexx = [r["status"] for r in rows if r["status"] >= 500]
    v["V5 no_oom"] = (
        not bad and not fivexx and alive and health == 200,
        f"bad lines {bad}, 5xx {fivexx}, alive {alive}, /health {health}",
    )
    same = []
    for r in window:
        if r["sampled"]:
            want = refs["spec_sampled"] if r["speculated"] else refs["plain_sampled"]
            same.append(r["text"] is not None and r["text"] == want)
        else:
            same.append(r["text"] is not None and r["text"] == refs["spec_greedy"] == refs["plain_greedy"])
    v["V6 same_text"] = (
        bool(same) and all(same),
        "; ".join(
            f"step {r['step']} {'sampled' if r['sampled'] else 'greedy'} speculated {r['speculated']}: "
            f"{'same' if ok else 'DIFFERENT'}"
            for r, ok in zip(window, same)
        )
        + f"; greedy references agree {refs['spec_greedy'] == refs['plain_greedy']}; sampled references agree "
        f"{refs['spec_sampled'] == refs['plain_sampled']}",
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
    ap.add_argument("--prompt-words", type=int, default=9000)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--window-hits", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=12)
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
    refs: dict = {}
    # The plain references: the same two requests on a boot that never speculates.
    ref_server = Server(args.bin, args.model, args.draft, args.port, args.ctx, args.out / "server-plain-ref.log",
                        spec=False)
    try:
        ref_server.boot()
        for sampled in (False, True):
            r = ref_server.complete(prose(7000, args.prompt_words), args.max_tokens, sampled)
            if r["status"] != 200 or r["speculated"]:
                refuse(f"plain reference request failed or speculated: {json.dumps(r)[:400]}")
            refs["plain_sampled" if sampled else "plain_greedy"] = r["text"]
    finally:
        ref_server.stop()
    time.sleep(3.0)

    server = Server(args.bin, args.model, args.draft, args.port, args.ctx, args.out / "server.log")
    ballast = None
    rows: list[dict] = []
    try:
        server.boot()
        rows.append(run_step(server, 0, args.prompt_words, args.max_tokens, 0))
        full = rows[0]
        (args.out / "steps.jsonl").write_text(json.dumps(full) + "\n")
        if full["status"] != 200 or not full["speculated"] or in_window(full):
            refuse(f"step 0 must be a full speculating admission on the idle card: {json.dumps(full)[:600]}")
        if full["cost_mb"] is None or not full["draft_mb"]:
            refuse("step 0 printed no request-cost or draft-state line")
        refs["spec_greedy"] = full["text"]
        spec_sampled = run_step(server, 0, args.prompt_words, args.max_tokens, 0, sampled=True)
        with (args.out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(spec_sampled) + "\n")
        if spec_sampled["status"] != 200 or not spec_sampled["speculated"] or in_window(spec_sampled):
            refuse(f"the sampled reference must be a full speculating admission: {json.dumps(spec_sampled)[:600]}")
        refs["spec_sampled"] = spec_sampled["text"]
        cost, draft = full["cost_mb"] * MB, full["draft_mb"] * MB
        boot_text = server.log.read_text(errors="replace")
        floor = RE_FLOOR.search(boot_text)
        reserve = int(floor.group(1)) * MIB if floor else STATIC_RESERVE  # that line prints MiB
        required = cost + draft + reserve
        target = required + draft // 2
        ballast = Ballast(args.out / "ballast.log")
        for step in range(1, args.max_steps + 1):
            # The first ballast lands about D/2 above the estimated requirement; each later
            # step takes D/3 more, whatever the last request left behind.
            free = effective_free(server.metrics())
            if free > target:
                ballast.add(free - target)
                time.sleep(1.0)
            hits = sum(in_window(r) for r in rows)
            row = run_step(server, step, args.prompt_words, args.max_tokens, ballast.total, sampled=hits % 2 == 1)
            row["target_effective_free"] = target
            rows.append(row)
            with (args.out / "steps.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"step {step}: free {row['effective_free_before'] / MB:.0f}MB (target {target / MB:.0f}MB) "
                f"status {row['status']} {'sampled' if row['sampled'] else 'greedy'} speculated "
                f"{row['speculated']} window {in_window(row)}",
                flush=True,
            )
            if sum(in_window(r) for r in rows) >= args.window_hits or row["status"] != 200:
                break
            target -= draft // 3
        time.sleep(1.0)
        alive = server.proc is not None and server.proc.poll() is None
        health = server.health() if alive else 0
    finally:
        if ballast is not None:
            ballast.stop()
        server.stop()

    log_text = server.log.read_text(errors="replace")
    result = verdicts(rows, args.window_hits, draft, log_text, alive, health, refs, dflash)
    summary = {
        "gpu": gpu,
        "binary": args.bin,
        "model": args.model,
        "draft": args.draft,
        "route": "dflash" if dflash else "mtp",
        "reference_texts": refs,
        "ctx": args.ctx,
        "request_cost_bytes": cost,
        "draft_plane_bytes": draft,
        "reserve_estimate_bytes": reserve,
        "required_estimate_bytes": required,
        "steps": rows,
        "verdicts": {k: {"pass": ok, "detail": detail} for k, (ok, detail) in result.items()},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with (args.out / "VERDICTS.txt").open("w") as f:
        for k, (ok, detail) in result.items():
            line = f"{k}: {'PASS' if ok else 'FAIL'} ({detail})"
            print(line, flush=True)
            f.write(line + "\n")
    if not result["V1 window_reached"][0]:
        refuse("the window was not reached: no step fit only without its draft state")
    sys.exit(0 if all(ok for ok, _ in result.values()) else 1)


if __name__ == "__main__":
    main()
