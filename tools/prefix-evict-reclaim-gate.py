#!/usr/bin/env python3
"""prefix-evict-reclaim-gate.py: an evicted prefix entry must credit admission and the driver.

The defect (memra#346, #445, #523 item 4): the admission path evicts unleased prefix-cache
entries when a request does not fit (`[admit-oom] reclaim-on-defer`), then re-reads headroom
in the SAME tick. The entries' planes are `CudaSlice`s whose drop is a stream-ordered
`cuMemFreeAsync` into the caching pool: the pool reports them used until the stream is fenced,
and even then they stay mapped in the pool where `cuMemGetInfo` cannot see them. So a 43.5 GB
eviction printed `effective free 69204MB -> 69204MB`, and on a busy box every later prefill
sat deferred behind an entry that was already gone.

Serving shape, one card, one boot of the real `memra-server` per arm:

  calibration boot (no reserve override): P1 (long, text A) seeds entry E1; P0 (short prompt,
    long greedy generation) is the busy peer; P2 (long, text B) runs beside it. Records the
    admission cost of P2 (`[admission] request cost ... = C MB`), effective free at idle (F0 =
    driver free + pool cached, from /metrics), E1 (`prefix_cache_bytes`), and the three greedy
    texts.
  measured boot: MEMRA_ADMIT_RESERVE_MB = (F0 - C2 - margin) / MiB, the documented
    teeth/diagnostics door, so that P2's admission requires F0 - margin: with E1 resident beside
    the busy peer the box is short, and after a TRUE credit it fits. F0 is effective free after
    P1 plus E1 (the /metrics publish trails the first retire, so the idle scrape reads 0), and
    the margin is clamped into the window the calibration's own readings allow. Same P1, P0, P2. P2 arrives while
    P0 is still decoding, so the idle-box drain arm (`admission-drain`) cannot mask the tick.

Assertions (every number in bytes from the server's own lines and /metrics):
  V1 reclaim_credit:    the `reclaim-on-defer ... effective free A -> B` line credits
                        B - A >= E1 - slack in the tick that evicted E1.
  V2 same_tick_admit:   P2 is admitted in that tick: no `[admit-oom] VRAM defer` line after the
                        reclaim line, no `reject averted`, HTTP 200.
  V3 driver_free_moved: the `[admit-oom] reclaim settle` line shows driver free rising by
                        >= E1 - one 2 MiB granule and `trim_released_bytes` >= the same; a
                        `pool_retained_bytes` shortfall is quoted verbatim and fails.
  V4 identity:          sha256 over (P1, P0, P2) texts is identical between the calibration boot
                        and the measured boot (pressure changes admission, never tokens). The
                        runner compares the same digest across binaries (base vs fix).

Exit 0 = every assertion held; 1 = a verdict failed (red on `main` today); 2 = REFUSED
(precondition not met: lock, port, busy-peer window, prompt too long, card too small).

usage: prefix-evict-reclaim-gate.py [--external-lock FD] --model GGUF --bin memra-server \
           --out NEW_DIR [--port N] [--prompt-words N] [--busy-tokens N] [--margin-mib N]
Lock: the canonical rig lock only (`/tmp/memra-gpu.lock` or `/tmp/memra-5090.lock`), held
for the whole cell; `--external-lock FD` inherits the collector's FD (lead ruling 5) and is
verified with tools/tier-lock-proof.py before anything binds a port or boots a server.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
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
GRANULE = 2 * MIB  # the driver's VMM allocation granularity on every card measured so far
WORDS = (
    "amber basalt cobalt delta ember falcon garnet harbor indigo jasper kestrel lumen "
    "marble nectar orchid pewter quartz raven saffron timber umber velvet willow yonder "
    "zenith anchor beacon cinder dune ferry glacier hollow isle juniper knoll lagoon "
    "meadow north orbit prairie quarry ridge summit tundra upland valley wharf"
).split()

RE_INSERT = re.compile(r"\[prefix-cache\] insert probation \((\w+)\): (\d+) tokens, ([\d.]+)MB")
RE_COST = re.compile(
    r"\[admission\] request cost: model=\"[^\"]+\" ctx=(\d+) path=(\w+) = .* = (\d+)MB$"
)
RE_RECLAIM = re.compile(
    r"\[admit-oom\] reclaim-on-defer: evicted (\d+) prefix entries \+ (\d+) plain \+ (\d+) spec "
    r"\+ (\d+) dspark parked sessions \(global LRU\); effective free (\d+)MB -> (\d+)MB"
)
RE_SETTLE = re.compile(
    r"\[admit-oom\] reclaim settle \(([^)]+)\): dev(\d+) evicted_prefix_bytes=(\d+) "
    r"pool_cached_gain_bytes=(\d+) trim_released_bytes=(\d+) driver_free_bytes (\d+) -> (\d+) "
    r"pool_reserved_bytes (\d+) -> (\d+) pool_used_bytes (\d+) -> (\d+)(.*)$"
)
RE_DEFER = re.compile(r"\[admit-oom\] VRAM defer: ")
RE_AVERTED = re.compile(r"\[admission\] reject averted: ")
RE_DRAIN = re.compile(r"\[device-trim\] reason=admission-drain (\{.*\})$")
RE_CAPACITY = re.compile(r"\[admit-oom\] capacity reject: ")


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prose(seed: int, words: int, header: str) -> str:
    rng = random.Random(seed)
    out = [header]
    n = 0
    item = 1
    while n < words:
        k = rng.randint(5, 9)
        out.append(f"Item {item}: " + " ".join(rng.choice(WORDS) for _ in range(k)) + ".")
        n += k + 2
        item += 1
    return "\n".join(out)


class Server:
    def __init__(self, binary: str, model: str, port: int, log: Path, extra_env: dict):
        self.binary, self.model, self.port, self.log = binary, model, port, log
        self.extra_env = extra_env
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
                "MEMRA_CTX": "65536",
                "MEMRA_MAX_SESSIONS": "4",
                "MEMRA_PREFIX_CACHE_MB": "6144",
                "MEMRA_SERVE_SPEC": "0",
                "MEMRA_TIMEOUT_MS_MAX": "240000",
            }
        )
        env.update(self.extra_env)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("wb") as log:
            # Same process group as the gate: the collector's timeout reaps everything.
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

    def chat(self, text: str, max_tokens: int) -> dict:
        body = {
            "model": "gate",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "timeout_ms": 240000,
        }
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=300) as f:
                status, payload = f.status, json.load(f)
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read().decode(errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
        elapsed = time.monotonic() - t0
        # Identity covers the WHOLE message (content and any reasoning channel: a short
        # greedy budget lands in the thinking channel and leaves `content` empty).
        text_out = ""
        finish = None
        if status == 200:
            choice = payload["choices"][0]
            text_out = json.dumps(choice.get("message"), sort_keys=True)
            finish = choice.get("finish_reason")
        usage = payload.get("usage") or {}
        return {
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "sent_at": t0,
            "done_at": t0 + elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish_reason": finish,
            "message_sha256": sha(text_out),
            "text": text_out,
            "error": None if status == 200 else payload,
        }


def effective_free(m: dict) -> int:
    return int(m["cuda_driver_free_bytes"]) + int(m["cuda_pool_cached_bytes"])


def mem_row(m: dict) -> dict:
    return {
        k: int(m[k])
        for k in (
            "cuda_driver_free_bytes",
            "cuda_pool_reserved_bytes",
            "cuda_pool_used_bytes",
            "cuda_pool_cached_bytes",
            "prefix_cache_bytes",
            "prefix_cache_entries",
            "active_sessions",
            "queued_requests",
        )
    }


def run_boot(
    name: str, out: Path, args, texts: dict, extra_env: dict, log_hook=None
) -> dict:
    d = out / name
    d.mkdir(parents=True)
    srv = Server(args.bin, args.model, args.port, d / "server.log", extra_env)
    srv.boot()
    rec: dict = {"name": name, "env": extra_env, "metrics": {}, "requests": {}}
    try:
        time.sleep(1.0)
        rec["metrics"]["idle"] = mem_row(srv.metrics())
        p1 = srv.chat(texts["A"], 8)
        rec["requests"]["p1"] = p1
        if p1["status"] != 200:
            refuse(f"{name}: P1 was not served (HTTP {p1['status']}): {p1['error']}")
        for _ in range(20):  # the /metrics publish trails the retire by up to 32 ticks
            time.sleep(0.25)
            m = srv.metrics()
            if int(m["prefix_cache_bytes"]) > 0 and int(m["active_sessions"]) == 0:
                break
        rec["metrics"]["after_p1"] = mem_row(m)
        if int(m["prefix_cache_bytes"]) == 0:
            refuse(f"{name}: P1 published no prefix entry (prefix_cache_bytes=0)")

        p0_box: dict = {}

        def busy() -> None:
            p0_box.update(srv.chat(texts["C"], args.busy_tokens))

        t = threading.Thread(target=busy, daemon=True)
        t.start()
        # Wait until the busy peer is ADMITTED (active_sessions >= 1); the publish lags, so
        # poll briefly and fall back to a fixed lead time.
        p2_sent = None
        for _ in range(16):
            time.sleep(0.25)
            try:
                if int(srv.metrics().get("active_sessions", 0)) >= 1:
                    break
            except Exception:
                pass
        rec["metrics"]["before_p2"] = mem_row(srv.metrics())
        p2 = srv.chat(texts["B"], 8)
        p2_sent = p2["sent_at"]
        t.join(timeout=600)
        rec["requests"]["p0"] = p0_box
        rec["requests"]["p2"] = p2
        time.sleep(1.0)
        rec["metrics"]["final"] = mem_row(srv.metrics())
        rec["busy_peer_overlap_s"] = round(p0_box.get("done_at", 0) - p2_sent, 3) if p0_box else None
    finally:
        srv.stop()
    rec["identity_sha256"] = sha(
        "\n".join(rec["requests"][k].get("text", "") for k in ("p1", "p0", "p2"))
    )
    for k in ("p1", "p0", "p2"):
        rec["requests"][k].pop("text", None)
        rec["requests"][k].pop("sent_at", None)
        rec["requests"][k].pop("done_at", None)
    (d / "cell.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def parse_log(path: Path) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    ev: dict = {
        "inserts": [],
        "costs": [],
        "reclaims": [],
        "settles": [],
        "defers_after_reclaim": 0,
        "defers_total": 0,
        "averted": 0,
        "drains": [],
        "capacity_rejects": 0,
    }
    seen_reclaim = False
    for ln in lines:
        m = RE_INSERT.search(ln)
        if m:
            ev["inserts"].append({"why": m.group(1), "tokens": int(m.group(2)), "mb": float(m.group(3))})
        m = RE_COST.search(ln)
        if m:
            ev["costs"].append({"ctx": int(m.group(1)), "path": m.group(2), "mb": int(m.group(3))})
        m = RE_RECLAIM.search(ln)
        if m:
            seen_reclaim = True
            ev["reclaims"].append(
                {
                    "line": ln.strip(),
                    "prefix": int(m.group(1)),
                    "free_before_mb": int(m.group(5)),
                    "free_after_mb": int(m.group(6)),
                }
            )
        m = RE_SETTLE.search(ln)
        if m:
            ev["settles"].append(
                {
                    "line": ln.strip(),
                    "why": m.group(1),
                    "device": int(m.group(2)),
                    "evicted_prefix_bytes": int(m.group(3)),
                    "pool_cached_gain_bytes": int(m.group(4)),
                    "trim_released_bytes": int(m.group(5)),
                    "driver_free_before": int(m.group(6)),
                    "driver_free_after": int(m.group(7)),
                    "pool_reserved_before": int(m.group(8)),
                    "pool_reserved_after": int(m.group(9)),
                    "pool_used_before": int(m.group(10)),
                    "pool_used_after": int(m.group(11)),
                    "tail": m.group(12).strip(),
                }
            )
        if RE_DEFER.search(ln):
            ev["defers_total"] += 1
            if seen_reclaim:
                ev["defers_after_reclaim"] += 1
        if RE_AVERTED.search(ln):
            ev["averted"] += 1
        m = RE_DRAIN.search(ln)
        if m:
            try:
                ev["drains"].append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                ev["drains"].append({"raw": m.group(1)})
        if RE_CAPACITY.search(ln):
            ev["capacity_rejects"] += 1
    return ev


def main() -> None:
    argv = sys.argv[1:]
    lock_fd = None
    lock_owner = "internal-canonical"
    if argv[:1] == ["--external-lock"]:
        if len(argv) < 2 or not argv[1].isdigit():
            refuse("inherited lock FD required after --external-lock")
        lock_fd, lock_owner = int(argv[1]), "collector"
        argv = argv[2:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=18113)
    ap.add_argument("--prompt-words", type=int, default=22000, help="words per long prompt (about 1.4 tokens each)")
    ap.add_argument("--busy-tokens", type=int, default=400, help="greedy generation length of the busy peer")
    ap.add_argument("--margin-mib", type=int, default=1024, help="preferred admission slack under F0; clamped into the window the calibration allows")
    ap.add_argument("--gpu-lock", default=os.environ.get("MEMRA_GPU_LOCK", "/tmp/memra-gpu.lock"))
    args = ap.parse_args(argv)

    if args.gpu_lock not in LOCKS:
        refuse("noncanonical GPU lock")
    if lock_owner == "internal-canonical":
        fh = open(args.gpu_lock, "a")  # noqa: SIM115 (held for the process lifetime)
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            refuse("canonical GPU lock busy")
        lock_fd = fh.fileno()
    proof = subprocess.run(
        [sys.executable, str(HERE / "tier-lock-proof.py"), "--fd", str(lock_fd), "--lock", args.gpu_lock, "--owner", lock_owner],
        capture_output=True, text=True, pass_fds=(lock_fd,),
    )
    if proof.returncode != 0:
        refuse(f"lock proof failed: {proof.stdout.strip()} {proof.stderr.strip()}")
    if not Path(args.model).is_file():
        refuse(f"model missing: {args.model}")
    if not os.access(args.bin, os.X_OK):
        refuse(f"server binary missing or not executable: {args.bin}")
    if args.out.exists():
        refuse(f"--out must be a new directory: {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "LOCK.json").write_text(proof.stdout)
    rig = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,power.limit,power.max_limit,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True,
    ).stdout.strip()
    binsha = hashlib.sha256(Path(args.bin).read_bytes()).hexdigest()
    (args.out / "rig.json").write_text(json.dumps({"nvidia_smi": rig, "binary_sha256": binsha, "binary": args.bin, "model": args.model}, indent=2) + "\n")

    texts = {
        "A": prose(1, args.prompt_words, "Document A. Read every item; answer with one word."),
        "B": prose(2, args.prompt_words, "Document B. Read every item; answer with one word."),
        "C": "Write the numbers from one to one thousand in English words, one per line, no commentary.",
    }
    (args.out / "prompts.json").write_text(json.dumps({k: sha(v) for k, v in texts.items()}, indent=2) + "\n")

    # ---- calibration boot -------------------------------------------------------------
    cal = run_boot("calibration", args.out, args, texts, {})
    cal_ev = parse_log(args.out / "calibration" / "server.log")
    p2_tokens = cal["requests"]["p2"]["prompt_tokens"]
    p1_tokens = cal["requests"]["p1"]["prompt_tokens"]
    if cal["requests"]["p2"]["status"] != 200 or cal["requests"]["p0"].get("status") != 200:
        refuse("calibration boot did not serve P0/P2 (no pressure applied yet): "
               f"p0={cal['requests']['p0'].get('status')} p2={cal['requests']['p2']['status']}")
    if p1_tokens + 8 + 64 >= 65536 or p2_tokens + 8 + 64 >= 65536:
        refuse(f"prompt too long for MEMRA_CTX=65536: p1={p1_tokens} p2={p2_tokens}")
    if cal_ev["reclaims"]:
        refuse("calibration boot already reclaimed (card too small for two entries at this size)")
    # The admission cap is prompt + max_tokens + a small boundary slack; pick the cost line
    # whose ctx sits nearest above P2's prompt (P1's line is 66 tokens further away).
    def nearest_cost(tokens: int):
        cands = [c for c in cal_ev["costs"] if tokens <= c["ctx"] < tokens + 1024]
        return min(cands, key=lambda c: c["ctx"] - tokens) if cands else None

    cost_p2 = nearest_cost(p2_tokens)
    if cost_p2 is None:
        refuse("no `[admission] request cost` line for the long prompt in the calibration log")
    # /metrics publishes only after the first retire, so `idle` reads 0: F0 is the effective
    # free WITH the first request's persistent workspaces warm and E1 added back, which is
    # exactly the baseline P2's admission sees in the measured boot.
    f1c = effective_free(cal["metrics"]["after_p1"])
    f2c = effective_free(cal["metrics"]["before_p2"])
    e1 = cal["metrics"]["after_p1"]["prefix_cache_bytes"]
    e0 = cal["metrics"]["before_p2"]["prefix_cache_bytes"] - e1  # the busy peer's own seed
    f0 = f1c + e1
    cost_p2_bytes = cost_p2["mb"] * 1_000_000
    # Margin window from the calibration's own readings: the reclaim must FIRE with E1
    # resident beside the busy peer (margin < F1c - F2c + E1) and P2 must FIT after a true
    # credit of E1 + E0 (margin > F1c - F2c - E0). 128 MiB slack on each side.
    peer_footprint = f1c - f2c
    margin_lo = peer_footprint - e0 + 128 * MIB
    margin_hi = peer_footprint + e1 - 128 * MIB
    if margin_lo >= margin_hi:
        refuse(f"no admissible margin: busy-peer footprint {peer_footprint} B, E1 {e1} B, E0 {e0} B")
    margin = min(max(args.margin_mib * MIB, margin_lo), margin_hi)
    reserve_mib = (f0 - cost_p2_bytes - margin) // MIB
    if reserve_mib <= 0:
        refuse(f"card too small for the design: F0={f0} cost={cost_p2_bytes} margin={margin}")
    calib = {
        "f0_effective_free_bytes": f0,
        "f1_after_p1_effective_free_bytes": f1c,
        "f2_before_p2_effective_free_bytes": f2c,
        "busy_peer_footprint_bytes": peer_footprint,
        "e0_busy_peer_entry_bytes": e0,
        "margin_bytes": margin,
        "margin_window_bytes": [margin_lo, margin_hi],
        "e1_prefix_bytes": e1,
        "p1_prompt_tokens": p1_tokens,
        "p2_prompt_tokens": p2_tokens,
        "cost_p2_mb": cost_p2["mb"],
        "reserve_mib": reserve_mib,
        "required_p2_bytes": cost_p2_bytes + reserve_mib * MIB,
        "inserts": cal_ev["inserts"],
        "costs": cal_ev["costs"],
    }
    (args.out / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n")

    # ---- measured boot ----------------------------------------------------------------
    meas = run_boot("measured", args.out, args, texts, {"MEMRA_ADMIT_RESERVE_MB": str(reserve_mib)})
    ev = parse_log(args.out / "measured" / "server.log")
    e1m = meas["metrics"]["after_p1"]["prefix_cache_bytes"]
    if meas["requests"]["p0"].get("status") != 200:
        refuse(f"busy peer P0 failed in the measured boot: {meas['requests']['p0']}")
    overlap = meas["busy_peer_overlap_s"]
    if overlap is None or overlap < 1.0:
        refuse(f"busy-box condition not met: P0 finished {overlap}s after P2 was sent; raise --busy-tokens")
    if not ev["reclaims"]:
        refuse("the measured boot never reached reclaim-on-defer; the pressure arithmetic did not bite "
               f"(F0={f0} E1={e1m} reserve={reserve_mib}MiB); see measured/server.log")

    slack = 8 * MIB  # two granules plus the two 1e6-rounded MB figures in the line
    r = ev["reclaims"][0]
    credit = (r["free_after_mb"] - r["free_before_mb"]) * 1_000_000
    v1 = r["prefix"] >= 1 and credit >= e1m - slack
    p2 = meas["requests"]["p2"]
    v2 = p2["status"] == 200 and ev["defers_after_reclaim"] == 0 and ev["averted"] == 0
    settle = next((s for s in ev["settles"] if s["why"] == "reclaim-on-defer"), None)
    if settle is None:
        v3 = False
        driver_delta = None
        released = None
    else:
        driver_delta = settle["driver_free_after"] - settle["driver_free_before"]
        released = settle["trim_released_bytes"]
        v3 = (
            driver_delta >= e1m - GRANULE
            and released >= e1m - GRANULE
            and "pool_retained_bytes" not in settle["tail"]
        )
    v4 = cal["identity_sha256"] == meas["identity_sha256"]

    if p2["status"] != 200:
        p2_verdict = f"http-{p2['status']}"
    elif ev["averted"]:
        p2_verdict = f"reject-averted-after-{ev['defers_after_reclaim']}-defer-ticks"
    elif ev["defers_after_reclaim"]:
        p2_verdict = f"deferred-{ev['defers_after_reclaim']}-ticks"
    else:
        p2_verdict = "admit-same-tick"
    ok = v1 and v2 and v3 and v4
    verdict = (
        f"PREFIX-EVICT-RECLAIM: entry_bytes={e1m} reclaim_credit_bytes={credit} "
        f"driver_free_delta_bytes={driver_delta if driver_delta is not None else 'none'} "
        f"trim_released_bytes={released if released is not None else 'none'} "
        f"p2={p2_verdict} busy_overlap_s={overlap} identity={meas['identity_sha256'][:16]} "
        f"V1={'ok' if v1 else 'FAIL'} V2={'ok' if v2 else 'FAIL'} V3={'ok' if v3 else 'FAIL'} "
        f"V4={'ok' if v4 else 'FAIL'} -> {'PASS' if ok else 'FAIL'}"
    )
    summary = {
        "verdict": verdict,
        "pass": ok,
        "binary_sha256": binsha,
        "rig": rig,
        "calibration": calib,
        "measured": {
            "e1_prefix_bytes": e1m,
            "reclaim_line": r["line"],
            "reclaim_credit_bytes": credit,
            "settle": settle,
            "defers_total": ev["defers_total"],
            "defers_after_reclaim": ev["defers_after_reclaim"],
            "reject_averted": ev["averted"],
            "admission_drain_trims": ev["drains"],
            "capacity_rejects": ev["capacity_rejects"],
            "p2": {k: v for k, v in p2.items() if k != "error"} | {"error": p2["error"]},
            "busy_peer_overlap_s": overlap,
            "metrics": meas["metrics"],
        },
        "identity": {"calibration": cal["identity_sha256"], "measured": meas["identity_sha256"], "equal": v4},
        "assertions": {"V1_reclaim_credit": v1, "V2_same_tick_admit": v2, "V3_driver_free_moved": v3, "V4_identity": v4},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "VERDICT.txt").write_text(verdict + "\n")
    print("reclaim line:", r["line"])
    if settle:
        print("settle line: ", settle["line"])
    for dr in ev["drains"]:
        print("admission-drain trim:", json.dumps(dr))
    print(verdict, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
