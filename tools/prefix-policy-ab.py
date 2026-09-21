#!/usr/bin/env python3
"""prefix-policy-ab.py: interleaved A/B of the prefix-cache eviction policy on the incident's shape.

memra#523 item 2: "Re-decide the default policy on the incident's shape (one agent loop at 105k to
168k growing about 300 tokens/turn beside promoted 30k conversations), interleaved A/B both orders
N>=5, and make the winner the naked default per door hygiene." The two arms are
`MEMRA_PREFIX_CACHE_POLICY=slru` (byte-segmented LRU with the newest-turn-fits rule, #523 item 1)
and `MEMRA_PREFIX_CACHE_POLICY=lru` (plain global LRU). The policy moves bytes' RESIDENCY, never
bytes' VALUES: every completion digest must be identical across arms, runs and the cache-off boot,
and a mismatch is a FAIL of the whole cell, not a data point.

Serving shape, one card, the real `memra-server`, plain path (`MEMRA_SERVE_SPEC=0`), an explicit
prefix budget, greedy `prompt_ids` requests, one server boot per run (the policy is a process-wide
read). The replay, scaled to the budget from the incident's ratios (cohort about 74 % of the
budget inside the 80 % protected share, loop entries 45 % to 50 %, cohort + loop entry > budget):

  1. cohort: T tenants (`cache_salt=cohort-<t>`), each a conversation of its own length, each
     sent twice; the second send is a whole-entry hit whose lease promotes the entry, so the
     cohort ends PROTECTED under slru (asserted on the second send: cached == prompt).
  2. loop: the agent tenant (`cache_salt=grow`) replays K turns, turn k+1 = turn k's ids plus
     `--grow-tokens` new ids, from `--start-tokens`, the largest prompt for which two
     consecutive turns still fit the budget together (the shape gate below).
  3. returns: after every `--return-every` loop turns one cohort tenant continues its conversation
     (its prompt plus `--grow-tokens` new ids), round robin; after the loop every cohort tenant
     continues once more. The returns are where scan resistance shows: a policy that keeps a
     returning tenant's entry serves the return from cache.

Schedule: the collector's pair vocabulary, `AB-0, BA-0, AB-1, BA-1, ...` (`paired_orders`), so
every pair's two runs are adjacent (A = slru, B = lru), N pairs per order, one lock hold, one
thermal window, one binary, the same ids and seeds in every run. A cache-off boot
(`MEMRA_PREFIX_CACHE_MB=0`) replays the same plan first and gives every request its cold digest.

Per request: `usage.prompt_tokens` and `usage.prompt_tokens_details.cached_tokens` from the
response, computed = prompt - cached (the billable cost), the server's `[prefix-cache]` window
(insert / hit / evict segment / demote / typed refusals), the `[spec-k]` admission receipt, the
`[ttft]` receipt (`MEMRA_TTFT_TRACE=1`, `first_decode_ms`), client E2E, settled `/metrics`, and
sha256 of the completion text.

Verdict rule (pre-registered in research/spill-b-20260919/DAY15.md before any run):
  primary    total computed tokens over the whole replay, lower is better;
  secondary  cohort tenants' cached_tokens on their returns (scan resistance), stated always;
  a policy WINS only if it is better on the primary at EVERY pair in BOTH orders; otherwise
  INCONCLUSIVE (a tie at any pair is not "better"). Digest identity is a precondition: any
  request whose digest differs across runs or from the cold boot fails the cell (exit 1).

Exit 0 = every digest identical and a verdict line was produced (WINNER or INCONCLUSIVE);
1 = a digest differed; 2 = REFUSED (lock, port, shape, boot policy, an unserved request).
Fewer than 5 pairs per order is a SMOKE cell: it runs and records but prints no verdict.

usage: prefix-policy-ab.py [--external-lock FD] --model GGUF --bin memra-server --out NEW_DIR \
           [--port N] [--budget-mib 2048] [--cohort-tokens 7800,8000,8200,8400] [--turns 12] \
           [--start-tokens 27300] [--grow-tokens 300] [--return-every 3] [--pairs 5] \
           [--ctx 32768] [--max-tokens 8]
Lock: the canonical rig lock only (`/tmp/memra-gpu.lock` or `/tmp/memra-5090.lock`), held for
the whole cell; `--external-lock FD` inherits the collector's FD and is verified with
tools/tier-lock-proof.py before anything binds a port or boots a server.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")
MIB = 1 << 20
ARMS = {"A": "slru", "B": "lru"}
POLICY_BOOT_WORD = {"slru": "byte-SLRU", "lru": "plain-LRU"}

RE_ON = re.compile(r"\[prefix-cache\] on: budget \d+MB \((\d+) B,.*policy ([^,]+),")
RE_INSERT = re.compile(r"\[prefix-cache\] insert (?:probation )?\(([^)]+)\): (\d+) tokens, ([\d.]+)MB")
RE_HIT = re.compile(r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens from cache")
RE_EVICT = re.compile(
    r"\[prefix-cache\] evict \((?:snapshot preflight(?:, )?)?(Probation|Protected)? ?(?:LRU)?\): (\d+) tokens, ([\d.]+)MB"
)
RE_REFUSED = re.compile(r"\[prefix-cache\] insert refused: (.*)$")
RE_SKIPPED = re.compile(r"\[prefix-cache\] (?:snapshot skipped|skip pinned): (.*)$")
RE_DEMOTE = re.compile(r"\[prefix-cache\] demote \(protected bytes\): ([\d.]+)MB")
RE_SPECK = re.compile(r"\[spec-k\] model=.* prompt=(\d+) cached=(\d+) lcp=(\d+)")
RE_TTFT = re.compile(r"\[ttft\] id=(\S+) .*prompt_tokens=(\d+) outcome=(\S+) .*prime_ms=(\S+) .*first_decode_ms=(\S+) .*total_ms=(\S+)")


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = max(0, min(len(v) - 1, round(p / 100 * (len(v) - 1))))
    return v[k]


def gpu_sample() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
    ).stdout.strip()
    keys = ("temperature_c", "power_w", "power_limit_w", "clocks_sm_mhz", "clocks_mem_mhz", "memory_used_mib")
    parts = [p.strip() for p in out.split(",")]
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **{k: num(p) if i < len(parts) else None for i, (k, p) in enumerate(zip(keys, parts + [""] * 6))}}


class Server:
    def __init__(self, binary: str, model: str, port: int, log: Path, budget_mib: int, ctx: int, policy: str | None):
        self.binary, self.model, self.port, self.log = binary, model, port, log
        self.budget_mib, self.ctx, self.policy = budget_mib, ctx, policy
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"
        self.offset = 0

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
                "MEMRA_PREFIX_CACHE_MB": str(self.budget_mib),
                "MEMRA_SERVE_SPEC": "0",
                "MEMRA_TIMEOUT_MS_MAX": "240000",
                "MEMRA_TTFT_TRACE": "1",
            }
        )
        # Both arms are EXPLICIT: never inherit a launcher's policy or share.
        env.pop("MEMRA_PREFIX_CACHE_POLICY", None)
        env.pop("MEMRA_PREFIX_CACHE_PROTECTED_PCT", None)
        if self.policy is not None:
            env["MEMRA_PREFIX_CACHE_POLICY"] = self.policy
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("wb") as log:
            # Same process group as the harness: the collector's timeout reaps everything.
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
        # The port must be free before the next boot probes it.
        for _ in range(50):
            if not self.probe():
                return
            time.sleep(0.2)

    def metrics(self) -> dict:
        with urllib.request.urlopen(f"{self.base}/metrics", timeout=10) as f:
            return json.load(f)

    def settled_metrics(self) -> dict | None:
        """/metrics after the last retire has published and two reads agree (bounded wait)."""
        last = None
        for _ in range(40):
            try:
                m = self.metrics()
            except Exception:
                time.sleep(0.25)
                continue
            row = mem_row(m)
            if row["active_sessions"] == 0 and row["queued_requests"] == 0 and row == last:
                return row
            last = row
            time.sleep(0.25)
        return last

    def new_log_lines(self) -> list[str]:
        data = self.log.read_bytes()
        chunk = data[self.offset :]
        self.offset = len(data)
        return chunk.decode(errors="replace").splitlines()

    def complete(self, ids: list[int], salt: str, max_tokens: int) -> dict:
        body = {
            "model": "gate",
            "prompt_ids": ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "cache_salt": salt,
            "timeout_ms": 240000,
        }
        req = urllib.request.Request(
            f"{self.base}/v1/completions",
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
        text_out, finish = "", None
        if status == 200:
            choice = payload["choices"][0]
            text_out = choice.get("text") if "text" in choice else json.dumps(choice.get("message"), sort_keys=True)
            finish = choice.get("finish_reason")
        usage = payload.get("usage")
        if usage is None and status == 200:
            usage = {
                "prompt_tokens": payload.get("prompt_tokens"),
                "completion_tokens": payload.get("completion_tokens"),
                "prompt_tokens_details": {"cached_tokens": payload.get("cached_tokens")},
            }
        usage = usage or {}
        return {
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish_reason": finish,
            "text_sha256": sha(text_out or ""),
            "error": None if status == 200 else payload,
        }


def mem_row(m: dict) -> dict:
    return {
        k: int(m.get(k, 0))
        for k in (
            "cuda_driver_free_bytes",
            "cuda_pool_cached_bytes",
            "prefix_cache_bytes",
            "prefix_cache_entries",
            "prefix_cache_evictions",
            "prefix_cache_hits",
            "prefix_cache_misses",
            "prefix_cache_inserts",
            "prefix_cache_hit_tokens",
            "prefix_cache_skips_budget",
            "prefix_cache_skips_pinned",
            "active_sessions",
            "queued_requests",
        )
    }


def parse_window(lines: list[str]) -> dict:
    ev: dict = {"inserts": [], "hits": [], "evicts": [], "refused": [], "skipped": [], "demotes": 0, "lines": [], "route": [], "ttft": None}
    for ln in lines:
        if "[prefix-cache]" in ln or "[spec-k] model=" in ln:
            ev["lines"].append(ln.strip())
        m = RE_TTFT.search(ln)
        if m:
            ev["ttft"] = {
                "id": m.group(1), "prompt_tokens": int(m.group(2)), "outcome": m.group(3),
                "prime_ms": num(m.group(4)), "first_decode_ms": num(m.group(5)), "total_ms": num(m.group(6)),
            }
            continue
        m = RE_SPECK.search(ln)
        if m:
            ev["route"].append({"prompt": int(m.group(1)), "cached": int(m.group(2)), "lcp": int(m.group(3)), "line": ln.strip()})
            continue
        m = RE_INSERT.search(ln)
        if m:
            ev["inserts"].append({"why": m.group(1), "tokens": int(m.group(2)), "mb": float(m.group(3))})
            continue
        m = RE_HIT.search(ln)
        if m:
            ev["hits"].append({"hit": int(m.group(1)), "prompt": int(m.group(2))})
            continue
        m = RE_EVICT.search(ln)
        if m:
            ev["evicts"].append({"segment": m.group(1) or "unstated", "tokens": int(m.group(2)), "mb": float(m.group(3))})
            continue
        m = RE_REFUSED.search(ln)
        if m:
            ev["refused"].append(ln.strip())
            continue
        m = RE_SKIPPED.search(ln)
        if m:
            ev["skipped"].append(ln.strip())
            continue
        if RE_DEMOTE.search(ln):
            ev["demotes"] += 1
    return ev


def fit_bytes_per_token(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least squares bytes = fixed + per_token * tokens over (tokens, bytes) points."""
    n = len(points)
    sx = sum(t for t, _ in points)
    sy = sum(b for _, b in points)
    sxx = sum(t * t for t, _ in points)
    sxy = sum(t * b for t, b in points)
    den = n * sxx - sx * sx
    if den == 0:
        return 0.0, sy / n
    per_token = (n * sxy - sx * sy) / den
    fixed = (sy - per_token * sx) / n
    return per_token, fixed


def loop_ids(tokens: int) -> list[int]:
    # Stable, well inside the vocab, non-repeating over 7000 ids; the same generator as the
    # day-14 twin so the loop's turn-1 prompt is comparable across the two gates.
    return [3000 + (i % 7000) for i in range(tokens)]


def cohort_ids(tenant: int, tokens: int) -> list[int]:
    # Each cohort tenant has its own id range: no tenant shares a prefix with another or with the loop.
    return [70000 + tenant * 10000 + (i % 7000) for i in range(tokens)]


def growth_ids(seq: int, tokens: int) -> list[int]:
    # The appended answer plus the next message, distinct per growth step.
    return [20000 + ((seq * 1000 + j) % 9000) for j in range(tokens)]


def build_plan(cohort_tokens: list[int], start: int, grow: int, turns: int, return_every: int) -> list[dict]:
    """The request plan every boot replays: role, tenant, salt, ids (deterministic from the args)."""
    plan: list[dict] = []
    conv = {t: cohort_ids(t, n) for t, n in enumerate(cohort_tokens, start=1)}
    for t in conv:
        for role in ("seed1", "seed2"):
            plan.append({"role": role, "tenant": t, "salt": f"cohort-{t}", "ids": conv[t], "turn": None})
    ids = loop_ids(start)
    seq = 0
    tenants = list(conv)
    for k in range(1, turns + 1):
        if k > 1:
            ids = ids + growth_ids(k, grow)
        plan.append({"role": "loop", "tenant": 0, "salt": "grow", "ids": ids, "turn": k})
        if return_every and k % return_every == 0:
            t = tenants[seq % len(tenants)]
            seq += 1
            conv[t] = conv[t] + growth_ids(100 + seq, grow)
            plan.append({"role": "return", "tenant": t, "salt": f"cohort-{t}", "ids": conv[t], "turn": k})
    for t in tenants:
        seq += 1
        conv[t] = conv[t] + growth_ids(100 + seq, grow)
        plan.append({"role": "final", "tenant": t, "salt": f"cohort-{t}", "ids": conv[t], "turn": turns})
    for i, p in enumerate(plan):
        p["idx"] = i
    return plan


def replay(srv: Server, plan: list[dict], max_tokens: int, label: str) -> list[dict]:
    rows = []
    before = srv.settled_metrics()
    for p in plan:
        r = srv.complete(p["ids"], p["salt"], max_tokens)
        time.sleep(0.5)
        window = parse_window(srv.new_log_lines())
        after = srv.settled_metrics()
        if before is None or after is None:
            refuse(f"{label} request {p['idx']} ({p['role']} {p['salt']}): /metrics never settled")
        if r["status"] != 200:
            refuse(f"{label} request {p['idx']} ({p['role']} {p['salt']}, {len(p['ids'])} ids) was not served (HTTP {r['status']}): {r['error']}")
        if r["prompt_tokens"] != len(p["ids"]):
            refuse(f"{label} request {p['idx']}: prompt_tokens {r['prompt_tokens']} != {len(p['ids'])} ids")
        cached = r["cached_tokens"] or 0
        rows.append({
            "idx": p["idx"], "role": p["role"], "tenant": p["tenant"], "salt": p["salt"], "turn": p["turn"],
            "prompt_ids": len(p["ids"]),
            "prompt_tokens": r["prompt_tokens"], "cached_tokens": cached, "computed_tokens": r["prompt_tokens"] - cached,
            "completion_tokens": r["completion_tokens"], "finish_reason": r["finish_reason"],
            "elapsed_s": r["elapsed_s"], "text_sha256": r["text_sha256"],
            "ttft_first_decode_ms": (window["ttft"] or {}).get("first_decode_ms"),
            "prime_ms": (window["ttft"] or {}).get("prime_ms"),
            "window": window, "metrics_before": before, "metrics_after": after,
        })
        before = after
    return rows


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
    ap.add_argument("--port", type=int, default=18115)
    ap.add_argument("--budget-mib", type=int, default=2048)
    ap.add_argument("--cohort-tokens", default="7800,8000,8200,8400")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--start-tokens", type=int, default=27300)
    ap.add_argument("--grow-tokens", type=int, default=300)
    ap.add_argument("--return-every", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=5, help="pairs per order (AB and BA each); < 5 is a smoke cell without a verdict")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=8)
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
    cohort_tokens = [int(t) for t in args.cohort_tokens.split(",") if t]
    if len(cohort_tokens) < 2 or len(set(cohort_tokens)) != len(cohort_tokens):
        refuse("--cohort-tokens needs at least two distinct lengths (the bytes(tokens) fit)")
    if args.turns < 2 or args.pairs < 1 or args.return_every < 0:
        refuse("--turns >= 2, --pairs >= 1, --return-every >= 0")
    last_tokens = args.start_tokens + (args.turns - 1) * args.grow_tokens
    longest = max(last_tokens, max(cohort_tokens) + args.grow_tokens * (2 + args.turns // max(1, args.return_every or args.turns)))
    if longest + args.max_tokens + 64 >= args.ctx:
        refuse(f"the longest prompt ({longest} tokens) does not fit MEMRA_CTX={args.ctx}")
    smoke = args.pairs < 5
    args.out.mkdir(parents=True)
    (args.out / "LOCK.json").write_text(proof.stdout)
    rig = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,power.limit,power.max_limit,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True,
    ).stdout.strip()
    binsha = hashlib.sha256(Path(args.bin).read_bytes()).hexdigest()
    model_sha_file = Path(args.model + ".sha256")
    (args.out / "rig.json").write_text(json.dumps({
        "nvidia_smi": rig, "binary_sha256": binsha, "binary": args.bin, "model": args.model,
        "model_sha256_sidecar": model_sha_file.read_text().split()[0] if model_sha_file.is_file() else None,
    }, indent=2) + "\n")

    plan = build_plan(cohort_tokens, args.start_tokens, args.grow_tokens, args.turns, args.return_every)
    (args.out / "plan.json").write_text(json.dumps(
        {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "requests": [{k: v for k, v in p.items() if k != "ids"} | {"prompt_ids": len(p["ids"]), "ids_sha256": sha(",".join(map(str, p["ids"])))} for p in plan]},
        indent=2) + "\n")

    # ---- cold reference: the cache OFF, the same plan, every request's cold digest -----------
    srv = Server(args.bin, args.model, args.port, args.out / "cold" / "server.log", 0, args.ctx, None)
    srv.boot()
    try:
        boot_lines = srv.new_log_lines()
        if any(RE_ON.search(ln) for ln in boot_lines):
            refuse("the cold boot armed the prefix cache; MEMRA_PREFIX_CACHE_MB=0 must disable it")
        cold_rows = replay(srv, plan, args.max_tokens, "cold")
    finally:
        srv.stop()
    if any(r["cached_tokens"] or r["window"]["inserts"] or r["window"]["hits"] for r in cold_rows):
        refuse("the prefix cache took part in the cache-off boot")
    cold_digest = {r["idx"]: r["text_sha256"] for r in cold_rows}
    (args.out / "cold" / "rows.json").write_text(json.dumps(cold_rows, indent=2) + "\n")

    # ---- the interleaved runs ------------------------------------------------------------------
    schedule = [(f"{order}-{i}", order, pos, ARMS[letter]) for i in range(args.pairs) for order in ("AB", "BA") for pos, letter in enumerate(order)]
    runs: list[dict] = []
    shape: dict | None = None
    budget_bytes = args.budget_mib * MIB
    protected_share = budget_bytes * 80 // 100
    for run_no, (pair_id, order, pos, arm) in enumerate(schedule, start=1):
        run_dir = args.out / "runs" / f"{run_no:02d}-{pair_id}-{arm}"
        srv = Server(args.bin, args.model, args.port, run_dir / "server.log", args.budget_mib, args.ctx, arm)
        gpu_start = gpu_sample()
        t_start = time.monotonic()
        srv.boot()
        try:
            boot_lines = srv.new_log_lines()
            on = next((RE_ON.search(ln) for ln in boot_lines if RE_ON.search(ln)), None)
            if on is None:
                refuse(f"run {run_no} ({arm}): no `[prefix-cache] on:` boot line; the cache did not arm")
            budget = int(on.group(1))
            policy = on.group(2).strip()
            if POLICY_BOOT_WORD[arm] not in policy:
                refuse(f"run {run_no}: the boot reported policy {policy!r}, expected {POLICY_BOOT_WORD[arm]!r} for arm {arm}")
            if budget != budget_bytes:
                refuse(f"run {run_no}: budget {budget} B differs from the requested {args.budget_mib} MiB")
            rows = replay(srv, plan, args.max_tokens, f"run {run_no} ({arm})")
            final = srv.settled_metrics()
        finally:
            srv.stop()
        gpu_end = gpu_sample()
        # Cohort promotion: every second seed send is a whole-entry hit (both arms).
        for r in rows:
            if r["role"] == "seed2" and (r["cached_tokens"] != r["prompt_tokens"] or not r["window"]["hits"]):
                refuse(f"run {run_no} ({arm}): cohort tenant {r['tenant']} did not promote: second send cached={r['cached_tokens']} of {r['prompt_tokens']}")
        if shape is None:
            # The pressure arithmetic, read from this binary's own insert lines on the first run.
            points = []
            for r in rows:
                if r["role"] == "seed1":
                    ins = [i for i in r["window"]["inserts"] if i["tokens"] == r["prompt_tokens"]]
                    if not ins:
                        refuse(f"run {run_no}: cohort tenant {r['tenant']} published no entry ({r['window']['lines']})")
                    points.append((ins[0]["tokens"], ins[0]["mb"] * 1e6))
            cohort_row = next(r for r in rows if r["role"] == "seed2" and r["tenant"] == len(cohort_tokens))["metrics_after"]
            per_token, fixed = fit_bytes_per_token(points)
            est = lambda tokens: fixed + per_token * tokens  # noqa: E731
            shape = {
                "budget_bytes": budget_bytes,
                "protected_share_bytes": protected_share,
                "cohort_bytes_after_seed": cohort_row["prefix_cache_bytes"],
                "cohort_entries_after_seed": cohort_row["prefix_cache_entries"],
                "cohort_evictions_after_seed": cohort_row["prefix_cache_evictions"],
                "cohort_points": points,
                "fit_bytes_per_token": per_token,
                "fit_fixed_bytes": fixed,
                "turn1_entry_estimate_bytes": est(args.start_tokens),
                "last_turn_entry_estimate_bytes": est(last_tokens),
                "cohort_within_protected_share": cohort_row["prefix_cache_bytes"] <= protected_share,
                "turn1_exceeds_free_share": cohort_row["prefix_cache_bytes"] + est(args.start_tokens) > budget_bytes,
                "every_turn_fits_budget": est(last_tokens) <= budget_bytes,
                "two_last_turns_fit_together": est(last_tokens - args.grow_tokens) + est(last_tokens) <= budget_bytes,
                "turn1_share_of_budget": est(args.start_tokens) / budget_bytes,
                "last_turn_share_of_budget": est(last_tokens) / budget_bytes,
                "cohort_share_of_budget": cohort_row["prefix_cache_bytes"] / budget_bytes,
            }
            (args.out / "shape.json").write_text(json.dumps(shape, indent=2) + "\n")
            if cohort_row["prefix_cache_entries"] != len(cohort_tokens) or cohort_row["prefix_cache_evictions"] != 0:
                refuse(f"the cohort is not intact after seeding: {cohort_row}")
            if not shape["cohort_within_protected_share"]:
                refuse(f"cohort {cohort_row['prefix_cache_bytes']} B exceeds the protected share {protected_share} B")
            if not shape["turn1_exceeds_free_share"]:
                refuse(f"no pressure at turn 1: cohort + turn-1 entry {shape['turn1_entry_estimate_bytes']:.0f} <= budget")
            if not shape["every_turn_fits_budget"]:
                refuse(f"the last turn's entry {shape['last_turn_entry_estimate_bytes']:.0f} B exceeds the budget")
            if not shape["two_last_turns_fit_together"]:
                refuse("two consecutive loop entries do not fit the budget together: that is the leased-refusal shape, not the incident's")
        loop = [r for r in rows if r["role"] == "loop"]
        returns = [r for r in rows if r["role"] in ("return", "final")]
        ttft_loop = [r["ttft_first_decode_ms"] for r in loop if r["ttft_first_decode_ms"] is not None]
        ttft_all = [r["ttft_first_decode_ms"] for r in rows if r["ttft_first_decode_ms"] is not None]
        run = {
            "run": run_no, "pair_id": pair_id, "order": order, "position": pos, "arm": arm,
            "policy_line": next(ln.strip() for ln in boot_lines if RE_ON.search(ln)),
            "wall_s": round(time.monotonic() - t_start, 1),
            "gpu_start": gpu_start, "gpu_end": gpu_end,
            "computed_tokens": sum(r["computed_tokens"] for r in rows),
            "computed_tokens_loop": sum(r["computed_tokens"] for r in loop),
            "computed_tokens_returns": sum(r["computed_tokens"] for r in returns),
            "loop_cached": [r["cached_tokens"] for r in loop],
            "loop_cold_after_1": sum(1 for r in loop[1:] if r["cached_tokens"] == 0),
            "return_cached": [{"role": r["role"], "tenant": r["tenant"], "after_turn": r["turn"], "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"]} for r in returns],
            "return_cached_sum": sum(r["cached_tokens"] for r in returns),
            "evictions": sum(len(r["window"]["evicts"]) for r in rows),
            "protected_evictions": sum(1 for r in rows for e in r["window"]["evicts"] if e["segment"] == "Protected"),
            "demotes": sum(r["window"]["demotes"] for r in rows),
            "refused_or_skipped": [ln for r in rows for ln in r["window"]["refused"] + r["window"]["skipped"]],
            "ttft_loop_ms": {"n": len(ttft_loop), "p50": pct(ttft_loop, 50), "p95": pct(ttft_loop, 95)},
            "ttft_all_ms": {"n": len(ttft_all), "p50": pct(ttft_all, 50), "p95": pct(ttft_all, 95)},
            "e2e_all_s": {"n": len(rows), "p50": pct([r["elapsed_s"] for r in rows], 50), "p95": pct([r["elapsed_s"] for r in rows], 95)},
            "digests_identical_to_cold": sum(1 for r in rows if r["text_sha256"] == cold_digest[r["idx"]]),
            "final_metrics": final,
            "rows": rows,
        }
        runs.append(run)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        print(f"run {run_no:02d} {pair_id} {arm}: computed={run['computed_tokens']} loop_cold_after_1={run['loop_cold_after_1']} "
              f"return_cached={run['return_cached_sum']} evictions={run['evictions']} digests==cold {run['digests_identical_to_cold']}/{len(rows)} "
              f"ttft_loop p50={run['ttft_loop_ms']['p50']} p95={run['ttft_loop_ms']['p95']} ms wall={run['wall_s']}s "
              f"temp {gpu_start['temperature_c']}->{gpu_end['temperature_c']} C", flush=True)

    # ---- digest identity: precondition -------------------------------------------------------
    digest_ok = []
    mismatches = []
    for p in plan:
        seen = {cold_digest[p["idx"]]} | {run["rows"][p["idx"]]["text_sha256"] for run in runs}
        digest_ok.append(len(seen) == 1)
        if len(seen) != 1:
            mismatches.append({"idx": p["idx"], "role": p["role"], "salt": p["salt"], "digests": sorted(seen)})
    identical = all(digest_ok)

    # ---- the pairs -----------------------------------------------------------------------------
    pairs = []
    for i in range(args.pairs):
        for order in ("AB", "BA"):
            pid = f"{order}-{i}"
            a = next(r for r in runs if r["pair_id"] == pid and r["arm"] == "slru")
            b = next(r for r in runs if r["pair_id"] == pid and r["arm"] == "lru")
            pairs.append({
                "pair_id": pid, "order": order,
                "slru_computed": a["computed_tokens"], "lru_computed": b["computed_tokens"],
                "diff_slru_minus_lru": a["computed_tokens"] - b["computed_tokens"],
                "slru_better": a["computed_tokens"] < b["computed_tokens"],
                "lru_better": b["computed_tokens"] < a["computed_tokens"],
                "slru_return_cached": a["return_cached_sum"], "lru_return_cached": b["return_cached_sum"],
                "slru_loop_cold_after_1": a["loop_cold_after_1"], "lru_loop_cold_after_1": b["loop_cold_after_1"],
                "slru_ttft_loop_p50_ms": a["ttft_loop_ms"]["p50"], "lru_ttft_loop_p50_ms": b["ttft_loop_ms"]["p50"],
                "slru_ttft_loop_p95_ms": a["ttft_loop_ms"]["p95"], "lru_ttft_loop_p95_ms": b["ttft_loop_ms"]["p95"],
                "slru_temp_c": [a["gpu_start"]["temperature_c"], a["gpu_end"]["temperature_c"]],
                "lru_temp_c": [b["gpu_start"]["temperature_c"], b["gpu_end"]["temperature_c"]],
            })
    n_pairs = len(pairs)
    slru_all = all(p["slru_better"] for p in pairs)
    lru_all = all(p["lru_better"] for p in pairs)
    if smoke:
        winner = "SMOKE"
    elif slru_all:
        winner = "WINNER=slru"
    elif lru_all:
        winner = "WINNER=lru"
    else:
        winner = "INCONCLUSIVE"

    def arm_stats(arm: str) -> dict:
        rs = [r for r in runs if r["arm"] == arm]
        comp = [r["computed_tokens"] for r in rs]
        ret = [r["return_cached_sum"] for r in rs]
        cold = [r["loop_cold_after_1"] for r in rs]
        tt = [x for r in rs for rr in r["rows"] if rr["role"] == "loop" and (x := rr["ttft_first_decode_ms"]) is not None]
        tt_all = [x for r in rs for rr in r["rows"] if (x := rr["ttft_first_decode_ms"]) is not None]
        e2e = [rr["elapsed_s"] for r in rs for rr in r["rows"]]
        temps = [t for r in rs for t in (r["gpu_start"]["temperature_c"], r["gpu_end"]["temperature_c"]) if t is not None]
        return {
            "N_runs": len(rs),
            "computed_tokens": {"median": statistics.median(comp), "min": min(comp), "max": max(comp), "values": comp},
            "return_cached_sum": {"median": statistics.median(ret), "min": min(ret), "max": max(ret), "values": ret},
            "loop_cold_after_1": {"median": statistics.median(cold), "max": max(cold), "values": cold},
            "ttft_loop_first_decode_ms": {"n": len(tt), "p50": pct(tt, 50), "p95": pct(tt, 95)},
            "ttft_all_first_decode_ms": {"n": len(tt_all), "p50": pct(tt_all, 50), "p95": pct(tt_all, 95)},
            "e2e_all_s": {"n": len(e2e), "p50": pct(e2e, 50), "p95": pct(e2e, 95)},
            "temperature_c": {"min": min(temps) if temps else None, "max": max(temps) if temps else None},
            "evictions": [r["evictions"] for r in rs],
            "protected_evictions": [r["protected_evictions"] for r in rs],
            "refused_or_skipped_lines": sum(len(r["refused_or_skipped"]) for r in rs),
        }

    stats = {arm: arm_stats(arm) for arm in ("slru", "lru")}
    all_temps = [t for r in runs for t in (r["gpu_start"]["temperature_c"], r["gpu_end"]["temperature_c"]) if t is not None]
    thermal = {
        "power_limit_w": runs[0]["gpu_start"]["power_limit_w"] if runs else None,
        "temperature_c_min": min(all_temps) if all_temps else None,
        "temperature_c_max": max(all_temps) if all_temps else None,
        "samples": "run boundaries here; the collector's command.gpu.csv is the 250 ms record",
    }
    verdict = (
        f"PREFIX-POLICY-AB: budget_bytes={budget_bytes} cohort_tenants={len(cohort_tokens)} "
        f"cohort_bytes={shape['cohort_bytes_after_seed'] if shape else 0} turns={args.turns} start_tokens={args.start_tokens} grow={args.grow_tokens} "
        f"return_every={args.return_every} pairs_per_order={args.pairs} runs={len(runs)} requests_per_run={len(plan)} "
        f"digests_identical={sum(digest_ok)}/{len(plan)} "
        f"computed_tokens slru_median={stats['slru']['computed_tokens']['median']:.0f} lru_median={stats['lru']['computed_tokens']['median']:.0f} (N={stats['slru']['N_runs']} each) "
        f"pairs_slru_better={sum(p['slru_better'] for p in pairs)}/{n_pairs} pairs_lru_better={sum(p['lru_better'] for p in pairs)}/{n_pairs} "
        f"ties={sum(1 for p in pairs if not p['slru_better'] and not p['lru_better'])}/{n_pairs} "
        f"return_cached slru_median={stats['slru']['return_cached_sum']['median']:.0f} lru_median={stats['lru']['return_cached_sum']['median']:.0f} "
        f"loop_cold_after_1 slru_max={stats['slru']['loop_cold_after_1']['max']} lru_max={stats['lru']['loop_cold_after_1']['max']} "
        f"refusals slru={stats['slru']['refused_or_skipped_lines']} lru={stats['lru']['refused_or_skipped_lines']} "
        f"temp_c={thermal['temperature_c_min']}..{thermal['temperature_c_max']} power_limit_w={thermal['power_limit_w']} "
        f"-> {'DIGEST-FAIL' if not identical else winner}"
    )

    runs_table = ["| run | pair | order | arm | computed tokens | loop computed | returns computed | loop cold after 1 | return cached sum | evictions (protected) | demotes | refusals | ttft loop p50/p95 ms | e2e p50/p95 s | digests == cold | temp C start->end | wall s |",
                  "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | ---: |"]
    for r in runs:
        runs_table.append(
            f"| {r['run']} | {r['pair_id']} | {r['order']} | {r['arm']} | {r['computed_tokens']} | {r['computed_tokens_loop']} | {r['computed_tokens_returns']} | "
            f"{r['loop_cold_after_1']} | {r['return_cached_sum']} | {r['evictions']} ({r['protected_evictions']}) | {r['demotes']} | {len(r['refused_or_skipped'])} | "
            f"{r['ttft_loop_ms']['p50']}/{r['ttft_loop_ms']['p95']} | {r['e2e_all_s']['p50']}/{r['e2e_all_s']['p95']} | {r['digests_identical_to_cold']}/{len(r['rows'])} | "
            f"{r['gpu_start']['temperature_c']}->{r['gpu_end']['temperature_c']} | {r['wall_s']} |"
        )
    pairs_table = ["| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |",
                   "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for p in pairs:
        better = "slru" if p["slru_better"] else "lru" if p["lru_better"] else "tie"
        pairs_table.append(
            f"| {p['pair_id']} | {p['order']} | {p['slru_computed']} | {p['lru_computed']} | {p['diff_slru_minus_lru']} | {better} | "
            f"{p['slru_return_cached']} | {p['lru_return_cached']} | {p['slru_loop_cold_after_1']} | {p['lru_loop_cold_after_1']} | "
            f"{p['slru_ttft_loop_p50_ms']}/{p['slru_ttft_loop_p95_ms']} | {p['lru_ttft_loop_p50_ms']}/{p['lru_ttft_loop_p95_ms']} | "
            f"{p['slru_temp_c'][0]}->{p['slru_temp_c'][1]} | {p['lru_temp_c'][0]}->{p['lru_temp_c'][1]} |"
        )
    # Per-request table for one run of each arm (the first pair), the residency story in tokens.
    req_table = ["| idx | role | tenant | turn | prompt | cold digest[:16] | slru cached | slru computed | slru events | lru cached | lru computed | lru events |",
                 "| ---: | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |"]
    first_a = next((r for r in runs if r["arm"] == "slru"), None)
    first_b = next((r for r in runs if r["arm"] == "lru"), None)

    def events(row: dict) -> str:
        w = row["window"]
        parts = [f"hit {h['hit']}" for h in w["hits"]]
        parts += [f"insert {i['tokens']}" for i in w["inserts"]]
        parts += [f"evict {e['tokens']} ({e['segment']})" for e in w["evicts"]]
        parts += [f"demote x{w['demotes']}"] if w["demotes"] else []
        parts += [f"REFUSED x{len(w['refused']) + len(w['skipped'])}"] if (w["refused"] or w["skipped"]) else []
        return ", ".join(parts) or "none"

    if first_a and first_b:
        for p in plan:
            ra, rb = first_a["rows"][p["idx"]], first_b["rows"][p["idx"]]
            req_table.append(
                f"| {p['idx']} | {p['role']} | {p['salt']} | {p['turn'] if p['turn'] is not None else '-'} | {len(p['ids'])} | {cold_digest[p['idx']][:16]} | "
                f"{ra['cached_tokens']} | {ra['computed_tokens']} | {events(ra)} | {rb['cached_tokens']} | {rb['computed_tokens']} | {events(rb)} |"
            )

    summary = {
        "verdict": verdict,
        "winner": None if smoke or not identical else winner,
        "smoke": smoke,
        "digests_identical": identical,
        "digest_mismatches": mismatches,
        "rule": {
            "primary": "total computed tokens (prompt_tokens - cached_tokens summed over every request of the replay), lower is better",
            "secondary": "cohort tenants' cached_tokens on their returns (return and final requests), stated always",
            "win": "a policy wins only if it is better on the primary at every pair in both orders; a tie at any pair is not better; otherwise inconclusive",
            "precondition": "every completion digest identical across arms, runs and the cache-off boot",
        },
        "binary_sha256": binsha,
        "rig": rig,
        "thermal": thermal,
        "shape": shape,
        "plan_requests": len(plan),
        "schedule": [{"run": i + 1, "pair_id": s[0], "order": s[1], "position": s[2], "arm": s[3]} for i, s in enumerate(schedule)],
        "arms": stats,
        "pairs": pairs,
        "runs": [{k: v for k, v in r.items() if k != "rows"} for r in runs],
        "cold": {"rows": [{k: v for k, v in r.items() if k not in ("window", "metrics_before", "metrics_after")} for r in cold_rows]},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "RUNS.md").write_text("\n".join(runs_table) + "\n")
    (args.out / "PAIRS.md").write_text("\n".join(pairs_table) + "\n")
    (args.out / "REQUESTS.md").write_text("\n".join(req_table) + "\n")
    (args.out / "VERDICT.txt").write_text(verdict + "\n")
    print("\n".join(pairs_table))
    for m in mismatches:
        print("digest mismatch:", json.dumps(m))
    print(verdict, flush=True)
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
