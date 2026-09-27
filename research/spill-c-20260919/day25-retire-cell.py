#!/usr/bin/env python3
"""Day 25 retire-seam settle cell (spill-c DAY25.md pre-registration, fixed before the run).

The day-16 stall shape (lane A's `stall_cell.py`: a streaming TENANT decodes one token per tick on a
`MEMRA_SERVE_SPEC=0` boot; the INTRUDER fires at the tenant's 24th token; order 1 = (idle, arm) x N,
order 2 = (arm, idle) x N) with a CAPTURE-SEEDING intruder that retires at once: a warm hit on a
5120-token entry E0, deepened by exactly 64 fresh on-grid tokens (5184 = 162 x 32), `max_tokens=1`.
Its prefill-done seed publishes a fresh 5184-token entry (`prefix_seed_deepens`: 64 >= the 64-token
floor); with the door ON that seed is a `capture submitted off the tick` on the copy stream; with the
door OFF the same seed is the tick program's synchronous `insert (seed)`.

Two arms, one receipt each:

- `plain`: the shape as briefed. The worker's iteration is tick-top poll, admission, abort sweep,
  prime wave (the seed at prefill-done), decode, retire; the intruder's first token (its
  `max_tokens=1` finish) is sampled one iteration after its prefill-done, so its capture is settled by
  the NEXT iteration's tick-top poll before any retire: the retire-seam settle is not on this path
  (the local 9B smoke read `settled_by_poll` for every capture). This arm bounds the door's whole
  cost at the shape (submission, poll, publish) and records where each capture settled.
- `coincide`: the seam exercised. A VICTIM stream (a fresh 40-token prompt under the 64-token seed
  floor, so it never seeds, `max_tokens=300`) is started at the tenant's 12th token; at the 24th
  token the intruder is posted and the victim's socket is closed right after. The disconnect abort
  sweep retires the victim in the iteration that admits and primes the intruder, so that iteration's
  retire block finds `finished` non-empty and a `Capturing` entry, and the lead's settle
  (`host_capture_settle_pending(.., ContractWait::Block, "a session retire")`, revuto on #634) runs
  BLOCKING: the publish line reads `settled synchronously by a session retire`. A run whose two
  arrivals straddle an iteration boundary reads `tick-top poll` instead and is counted as a miss.
  Both door arms carry the victim's abort and cache drop; only the settle differs.

Prompts are calibrated through `/v1/tokenize` (no seed happens at calibration): the base to exactly
`--base-tokens` tokens, each run's extension to exactly `--base-tokens + --deepen`, fresh words per
run and per arm, so every arm run publishes a fresh entry (the wrapper's prefix budget holds them
all; no eviction demote joins either arm). One untimed setup post of the base publishes E0.

Per run the harness records the tenant's ITLs (p50, max, stall = max - p50, and the TWO-TICK stall
= ITL[max] + ITL[max + 1] - 2 p50), the intruder's wall and `usage`, the victim's tokens and abort
line, and the server log's capture lines: `capture submitted off the tick (seed): .. (X MB)`,
`capture published off the tick (seed): .. complete after P poll(s), C ms from submission to
completion, .. (SETTLED_BY)`, `insert (seed): N tokens` (printed by both the tick program's seed and
the capture's publish), `[prefix-host] demote:` and any refusal/latch/disabled line. The rule line is
fixed here and replayed from receipt.json by `--replay`; the thresholds live in
`day25-retire-reading.py`. Client-side only: stdlib, no engine binary, no GPU access of its own.

    day25-retire-cell.py --port P --server-log LOG --out DIR --tag TAG --arm plain|coincide [--n 5]
                         [--base-tokens 5120] [--deepen 64]
    day25-retire-cell.py --replay DIR/receipt.json
"""
import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import re
import socket
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "stall_cell", os.path.join(HERE, "..", "spill-a-20260919", "stall_cell.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

VICTIM_AT = 12        # the victim stream starts when the tenant's 12th token has arrived
VICTIM_WORDS = 36     # about 40 tokens with its header: under the 64-token seed floor
VICTIM_MAX_TOKENS = 300
SETTLED_RETIRE = "settled synchronously by a session retire"
SETTLED_POLL = "tick-top poll"
BAD = ("capture refused", "capture dropped", "TIER DISABLED", "latched", "demote failed", "promote failed")
RX_PUB = re.compile(r"capture published off the tick \((\w[\w-]*)\): (\d+) tokens complete after (\d+) "
                    r"poll\(s\), ([0-9.]+)ms from submission to completion, ([0-9.]+)ms to publication \(([^)]*)\)")
RX_SUB = re.compile(r"capture submitted off the tick \((\w[\w-]*)\): (\d+) tokens, (\d+) planes \(([0-9.]+)MB\)")
RX_INS = re.compile(r"\[prefix-cache\] insert \((\w[\w-]*)\): (\d+) tokens, ([0-9.]+)MB")


def tokenize(port, prompt):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("POST", "/v1/tokenize", body=json.dumps({"model": "gate", "prompt": prompt}),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    if r.status != 200:
        raise RuntimeError(f"tokenize HTTP {r.status}: {data[:200]!r}")
    return int(json.loads(data)["count"])


FILLERS = "a an of to in on it is at by or as be we he up so no do if my".split()


def calibrate(port, make, target, guess, count=None):
    """`make(n)` builds a prompt from n words (more words, more tokens). Bracket then bisect to the
    largest n whose count is at or under `target`, then append single-token fillers one at a time
    until the count is exactly `target`, skipping a filler the tokenizer makes two tokens of (the
    27B's tokenizer makes about a quarter of the words two tokens, so a fixed step of one word can
    straddle the target: day 25's first box attempt oscillated 5183/5185). `count` replaces
    `/v1/tokenize` in the unit check."""
    count = count or (lambda text: tokenize(port, text))
    tried = []

    def cnt(n):
        c = count(make(n))
        tried.append((n, c))
        return c

    lo = max(1, guess)
    c = cnt(lo)
    guard = 0
    while c > target and guard < 64:
        guard += 1
        if lo == 1:
            raise RuntimeError(f"calibration underflow: {tried}")
        lo = max(1, lo - max(1, c - target))
        c = cnt(lo)
    if c == target:
        return make(lo), tried
    hi = lo + max(1, target - c)
    ch = cnt(hi)
    while ch <= target and guard < 64:
        guard += 1
        lo, c = hi, ch
        if c == target:
            return make(lo), tried
        hi = hi + max(1, target - c)
        ch = cnt(hi)
    while hi - lo > 1 and guard < 64:
        guard += 1
        mid = (lo + hi) // 2
        cm = cnt(mid)
        if cm == target:
            return make(mid), tried
        if cm < target:
            lo, c = mid, cm
        else:
            hi = mid
    text = make(lo)
    for w in FILLERS * 3:
        cand = f"{text} {w}"
        cc = count(cand)
        tried.append((f"+{w}", cc))
        if cc == target:
            return cand, tried
        if cc < target:
            text, c = cand, cc
    raise RuntimeError(f"calibration did not converge: {tried}")


def capture_lines(tail):
    pub, sub, ins, bad = [], [], [], []
    aborts = 0
    for ln in tail.splitlines():
        m = RX_PUB.search(ln)
        if m:
            pub.append({"why": m.group(1), "tokens": int(m.group(2)), "polls": int(m.group(3)),
                        "copy_ms": float(m.group(4)), "publish_ms": float(m.group(5)), "settled_by": m.group(6)})
        m = RX_SUB.search(ln)
        if m:
            sub.append({"why": m.group(1), "tokens": int(m.group(2)), "planes": int(m.group(3)), "mb": float(m.group(4))})
        m = RX_INS.search(ln)
        if m:
            ins.append({"why": m.group(1), "tokens": int(m.group(2)), "mb": float(m.group(3))})
        if "[abort] client disconnected" in ln:
            aborts += 1
        if any(b in ln for b in BAD):
            bad.append(ln[:200])
    return pub, sub, ins, bad, aborts


class Victim:
    """A streaming session that exists to be aborted: opened in a thread, closed from the caller."""

    def __init__(self, port, prompt):
        self.port, self.prompt = port, prompt
        self.conn = None
        self.tokens = 0
        self.started = threading.Event()
        self.first_token = threading.Event()
        self.error = None
        self.closed_at = None
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        try:
            self.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
            body = {"model": "gate", "prompt": self.prompt, "max_tokens": VICTIM_MAX_TOKENS,
                    "temperature": 0, "stream": True}
            self.conn.request("POST", "/v1/completions", body=json.dumps(body),
                              headers={"Content-Type": "application/json"})
            r = self.conn.getresponse()
            self.started.set()
            if r.status != 200:
                self.error = f"victim HTTP {r.status}"
                return
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                    continue
                ev = json.loads(line[5:].strip())
                ch = ev.get("choices") or []
                if ch and ch[0].get("text"):
                    self.tokens += 1
                    self.first_token.set()
        except Exception as e:  # the close we issue lands here; recorded
            self.error = repr(e)
        finally:
            self.started.set()
            self.first_token.set()

    def close(self):
        self.closed_at = time.monotonic()
        try:
            if self.conn is not None and self.conn.sock is not None:
                self.conn.sock.shutdown(socket.SHUT_RDWR)
                self.conn.sock.close()
        except Exception as e:
            self.error = (self.error or "") + f" close:{e!r}"

    def record(self):
        self.th.join(timeout=30)
        return {"tokens_before_close": self.tokens, "error": self.error}


def stream_tenant_with_marks(port, arrivals, marks, error):
    """A's tenant stream, with an event per marked token count (12 and 24) instead of one."""
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        body = {"model": "gate", "prompt": A.TENANT_PROMPT, "max_tokens": A.TENANT_MAX_TOKENS,
                "temperature": 0, "stream": True}
        c.request("POST", "/v1/completions", body=json.dumps(body),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            error.append(f"tenant HTTP {r.status}: {r.read()[:200]!r}")
            return
        text = []
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            now = time.monotonic()
            ev = json.loads(payload)
            ch = ev.get("choices") or []
            if ch and ch[0].get("text"):
                arrivals.append(now)
                text.append(ch[0]["text"])
                if len(arrivals) in marks:
                    marks[len(arrivals)].set()
        c.close()
        arrivals.append(("text", "".join(text)))
    except Exception as e:  # recorded, never inferred
        error.append(f"tenant: {e!r}")
    finally:
        for ev in marks.values():
            ev.set()


def one_run(port, arm, mode, run_id, prompt, log_path, log_off):
    arrivals, error = [], []
    marks = {VICTIM_AT: threading.Event(), A.FIRE_AT: threading.Event()}
    th = threading.Thread(target=stream_tenant_with_marks, args=(port, arrivals, marks, error), daemon=True)
    t_start = time.monotonic()
    th.start()
    intruder, victim = None, None
    if arm != "idle":
        vic = None
        if mode == "coincide":
            marks[VICTIM_AT].wait(timeout=600)
            vic = Victim(port, f"victim {run_id}: {A.words(VICTIM_WORDS, 7000 + run_id)}")
            vic.first_token.wait(timeout=120)
        marks[A.FIRE_AT].wait(timeout=600)
        t_i = time.monotonic()
        try:
            # Post first, close second: the admission sweep precedes the abort sweep in the worker's
            # iteration, so this order makes the two arrivals land in one iteration.
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
            c.request("POST", "/v1/completions",
                      body=json.dumps({"model": "gate", "prompt": prompt, "max_tokens": 1, "temperature": 0}),
                      headers={"Content-Type": "application/json"})
            if vic is not None:
                vic.close()
            r = c.getresponse()
            data = r.read()
            wall = (time.monotonic() - t_i) * 1e3
            c.close()
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {data[:200]!r}")
            usage = json.loads(data).get("usage", {})
            intruder = {"fired_at_tenant_token": A.FIRE_AT, "fired_at_ms": (t_i - t_start) * 1e3, "wall_ms": wall,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens",
                                                                                         usage.get("cached_tokens"))}
            if vic is not None:
                intruder["victim_close_after_post_ms"] = (vic.closed_at - t_i) * 1e3
        except Exception as e:  # recorded, never inferred
            intruder = {"error": repr(e), "fired_at_ms": (t_i - t_start) * 1e3}
        if vic is not None:
            victim = vic.record()
    th.join(timeout=900)
    text, times = "", []
    for a in arrivals:
        if isinstance(a, tuple):
            text = a[1]
        else:
            times.append(a)
    itl = [(times[i + 1] - times[i]) * 1e3 for i in range(len(times) - 1)]
    tail, new_off = A.log_tail(log_path, log_off)
    pub, sub, ins, bad, aborts = capture_lines(tail)
    run = {
        "arm": arm, "run_id": run_id, "tenant_tokens": len(times),
        "ttft_ms": (times[0] - t_start) * 1e3 if times else None,
        "tenant_wall_ms": (times[-1] - t_start) * 1e3 if times else None,
        "itl_ms": [round(x, 3) for x in itl],
        "tenant_text_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
        "intruder": intruder, "victim": victim, "aborts": aborts, "errors": error + bad,
        "captures_published": pub, "captures_submitted": sub, "inserts": ins,
        "server_demote_ms": A.server_ms(tail, "demote"),
        "server_log_lines": [ln[:240] for ln in tail.splitlines()
                             if "[prefix-host]" in ln or "[prefix-cache]" in ln or "[abort]" in ln][:40],
    }
    if itl:
        p50 = statistics.median(itl)
        i_max = max(range(len(itl)), key=lambda i: itl[i])
        run["p50"] = p50
        run["max"] = itl[i_max]
        run["max_index"] = i_max
        run["stall_ms"] = itl[i_max] - p50
        nxt = itl[i_max + 1] if i_max + 1 < len(itl) else None
        run["next_itl_ms"] = nxt
        run["two_tick_stall_ms"] = (itl[i_max] + nxt - 2 * p50) if nxt is not None else None
    return run, new_off


def summarize(runs):
    idle = [r for r in runs if r["arm"] == "idle"]
    arm = [r for r in runs if r["arm"] != "idle"]
    pool_idle = [x for r in idle for x in r["itl_ms"]]
    pool_arm = [x for r in arm for x in r["itl_ms"]]
    stalls = [r["stall_ms"] for r in arm if "stall_ms" in r]
    two = [r["two_tick_stall_ms"] for r in arm if r.get("two_tick_stall_ms") is not None]
    walls = [r["intruder"]["wall_ms"] for r in arm if r.get("intruder") and "wall_ms" in r["intruder"]]
    pubs = [p for r in arm for p in r["captures_published"]]
    med = lambda xs: statistics.median(xs) if xs else None
    return {
        "idle": {"runs": len(idle), "p50": A.pct(pool_idle, .5), "p95": A.pct(pool_idle, .95),
                 "p99": A.pct(pool_idle, .99), "max": max(pool_idle) if pool_idle else None, "n": len(pool_idle)},
        "arm": {"runs": len(arm), "p50": A.pct(pool_arm, .5), "p95": A.pct(pool_arm, .95),
                "p99": A.pct(pool_arm, .99), "max": max(pool_arm) if pool_arm else None, "n": len(pool_arm)},
        "stall_ms": {"per_run": stalls, "median": med(stalls), "min": min(stalls) if stalls else None,
                     "max": max(stalls) if stalls else None},
        "two_tick_stall_ms": {"per_run": two, "median": med(two)},
        "intruder_wall_ms": {"per_run": walls, "median": med(walls)},
        "intruder_prompt_tokens": [r["intruder"].get("prompt_tokens") for r in arm if r.get("intruder")],
        "intruder_cached_tokens": [r["intruder"].get("cached_tokens") for r in arm if r.get("intruder")],
        "captures_published": len(pubs),
        "settled_by_retire": sum(1 for p in pubs if p["settled_by"] == SETTLED_RETIRE),
        "settled_by_poll": sum(1 for p in pubs if p["settled_by"] == SETTLED_POLL),
        "settled_by_other": sorted({p["settled_by"] for p in pubs if p["settled_by"] not in (SETTLED_RETIRE, SETTLED_POLL)}),
        "copy_ms": {"per_capture": [p["copy_ms"] for p in pubs], "median": med([p["copy_ms"] for p in pubs])},
        "copy_ms_retire_settled": [p["copy_ms"] for p in pubs if p["settled_by"] == SETTLED_RETIRE],
        "stall_ms_retire_settled": [r["stall_ms"] for r in arm if "stall_ms" in r
                                    and any(p["settled_by"] == SETTLED_RETIRE for p in r["captures_published"])],
        "victim_close_after_post_ms": [round(r["intruder"]["victim_close_after_post_ms"], 2) for r in arm
                                       if r.get("intruder") and "victim_close_after_post_ms" in r["intruder"]],
        "capture_polls": [p["polls"] for p in pubs],
        "captures_submitted_mb": [s["mb"] for r in arm for s in r["captures_submitted"]],
        "seed_inserts": sum(1 for r in arm for i in r["inserts"] if i["why"] == "seed"),
        "seed_tokens": sorted({i["tokens"] for r in arm for i in r["inserts"] if i["why"] == "seed"}
                              | {s["tokens"] for r in arm for s in r["captures_submitted"]}),
        "victim_aborts": sum(r["aborts"] for r in arm),
        "victim_tokens_before_close": [r["victim"]["tokens_before_close"] for r in arm if r.get("victim")],
        "server_demote_ms": [x for r in arm for x in r["server_demote_ms"]],
        "tenant_text_shas": sorted({r["tenant_text_sha"] for r in runs}),
        "errors": [e for r in runs for e in r["errors"]]
        + [r["intruder"]["error"] for r in arm if r.get("intruder") and "error" in r["intruder"]],
    }


def rule_line(tag, mode, n, s):
    f = lambda v: "na" if v is None or v != v else f"{v:.1f}"
    r1 = lambda xs: [round(x, 1) for x in xs]
    return (f"RETIRE rule cell={tag} arm={mode} n_per_order={n} pooled={n * 2} "
            f"idle_runs={s['idle']['runs']} idle_p50={f(s['idle']['p50'])} idle_p95={f(s['idle']['p95'])} "
            f"idle_p99={f(s['idle']['p99'])} idle_max={f(s['idle']['max'])} "
            f"arm_runs={s['arm']['runs']} arm_p50={f(s['arm']['p50'])} arm_p95={f(s['arm']['p95'])} "
            f"arm_p99={f(s['arm']['p99'])} arm_max={f(s['arm']['max'])} "
            f"stall_median={f(s['stall_ms']['median'])} stall_min={f(s['stall_ms']['min'])} stall_max={f(s['stall_ms']['max'])} "
            f"two_tick_stall_median={f(s['two_tick_stall_ms']['median'])} "
            f"intruder_wall_median={f(s['intruder_wall_ms']['median'])} intruder_wall_ms={r1(s['intruder_wall_ms']['per_run'])} "
            f"intruder_prompt_tokens={s['intruder_prompt_tokens']} intruder_cached_tokens={s['intruder_cached_tokens']} "
            f"captures_published={s['captures_published']} settled_by_retire={s['settled_by_retire']} "
            f"settled_by_poll={s['settled_by_poll']} settled_by_other={s['settled_by_other']} "
            f"copy_ms={r1(s['copy_ms']['per_capture'])} copy_ms_median={f(s['copy_ms']['median'])} "
            f"copy_ms_retire_settled={r1(s['copy_ms_retire_settled'])} stall_ms_retire_settled={r1(s['stall_ms_retire_settled'])} "
            f"victim_close_after_post_ms={s['victim_close_after_post_ms']} "
            f"capture_polls={s['capture_polls']} captures_submitted_mb={r1(s['captures_submitted_mb'])} "
            f"seed_inserts={s['seed_inserts']} seed_tokens={s['seed_tokens']} "
            f"victim_aborts={s['victim_aborts']} victim_tokens_before_close={s['victim_tokens_before_close']} "
            f"server_demote_ms={r1(s['server_demote_ms'])} "
            f"tenant_text_identical={len(s['tenant_text_shas']) == 1} errors={len(s['errors'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int)
    ap.add_argument("--server-log")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tag", default="retire")
    ap.add_argument("--arm", choices=["plain", "coincide"], default="plain")
    ap.add_argument("--base-tokens", type=int, default=5120)
    ap.add_argument("--deepen", type=int, default=64)
    ap.add_argument("--replay")
    a = ap.parse_args()
    if a.replay:
        rec = json.load(open(a.replay))
        line = rule_line(rec["tag"], rec["mode"], rec["n_per_order"], summarize(rec["runs"]))
        print(line)
        ok = line == rec["rule_line"]
        print("RETIRE REPLAY:", "PASS (replay agrees with the harness's rule line)" if ok else "FAIL (rule line differs)")
        return 0 if ok else 1
    if not (a.port and a.server_log and a.out):
        ap.error("--port --server-log --out are required")
    os.makedirs(a.out, exist_ok=True)
    log_off = os.path.getsize(a.server_log)
    setup = {}
    salt = 3000 if a.arm == "plain" else 4000
    # Calibration through /v1/tokenize (no seed happens here): the base to exactly base_tokens.
    base, tried = calibrate(a.port, lambda n: f"survey 5000: {A.words(n, 5000)}", a.base_tokens, a.base_tokens - 4)
    setup["base_calibration"] = tried
    # One extension per arm run (2 x n), each to exactly base_tokens + deepen; fresh words per run and per arm.
    exts = []
    for k in range(2 * a.n):
        ext, tried_k = calibrate(a.port, lambda m, k=k: f"{base} {A.words(m, salt + k)}", a.base_tokens + a.deepen, a.deepen)
        exts.append(ext)
        setup.setdefault("ext_calibration", []).append(tried_k)
    # Untimed setup: the base publishes E0 (or hits it if the other arm's invocation published it already).
    resp, wall = A.post(a.port, {"model": "gate", "prompt": base, "max_tokens": 1, "temperature": 0})
    tail, log_off = A.log_tail(a.server_log, log_off)
    pub, sub, ins, bad, _ = capture_lines(tail)
    setup["e0"] = {"wall_ms": wall, "usage": resp.get("usage"), "captures_published": pub,
                   "captures_submitted": sub, "inserts": ins, "bad": bad}
    # The ON arm's E0 seed may itself be a capture pending at the tick top; give it a tick to publish
    # before the timed window (untimed, recorded), so the first timed run starts with no capture in flight.
    time.sleep(1.0)
    tail, log_off = A.log_tail(a.server_log, log_off)
    pub, sub, ins, bad, _ = capture_lines(tail)
    setup["after_e0"] = {"captures_published": pub, "inserts": ins, "bad": bad}
    print(f"  setup ({a.arm}): base usage={setup['e0']['usage']} seed_lines={len(ins) + len(setup['e0']['inserts'])} "
          f"captures={len(pub) + len(setup['e0']['captures_published'])}", flush=True)
    runs = []
    run_id = 0
    arm_ix = 0
    seq = [("idle", a.arm)] * a.n + [(a.arm, "idle")] * a.n
    for order, pair in enumerate(seq):
        for arm in pair:
            run_id += 1
            prompt = None
            if arm != "idle":
                prompt = exts[arm_ix]
                arm_ix += 1
            r, log_off = one_run(a.port, arm, a.arm, run_id, prompt, a.server_log, log_off)
            r["order"] = 1 if order < a.n else 2
            runs.append(r)
            cap = r["captures_published"]
            print(f"  run {run_id:2d} order {r['order']} {arm:8s} tokens={r['tenant_tokens']} "
                  f"p50={r.get('p50', float('nan')):.1f} max={r.get('max', float('nan')):.1f} "
                  f"stall={r.get('stall_ms', float('nan')):.1f} two_tick={r.get('two_tick_stall_ms') or float('nan'):.1f} "
                  f"intruder={r['intruder'] and (round(r['intruder'].get('wall_ms', float('nan')), 1), r['intruder'].get('prompt_tokens'), r['intruder'].get('cached_tokens'))} "
                  f"captures={[(p['copy_ms'], p['settled_by']) for p in cap]} seeds={[i['tokens'] for i in r['inserts']]} "
                  f"victim={r['victim']} aborts={r['aborts']} demote={r['server_demote_ms']} errors={len(r['errors'])}", flush=True)
    s = summarize(runs)
    line = rule_line(a.tag, a.arm, a.n, s)
    rec = {"tag": a.tag, "mode": a.arm, "n_per_order": a.n, "fire_at": A.FIRE_AT, "victim_at": VICTIM_AT,
           "tenant_max_tokens": A.TENANT_MAX_TOKENS, "tenant_prompt": A.TENANT_PROMPT,
           "base_tokens": a.base_tokens, "deepen": a.deepen,
           "harness_a_sha256": hashlib.sha256(open(_spec.origin, "rb").read()).hexdigest(),
           "setup": setup, "runs": runs, "summary": s, "rule_line": line}
    json.dump(rec, open(os.path.join(a.out, "receipt.json"), "w"), indent=1)
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
