#!/usr/bin/env python3
"""prefix-newest-turn-fits-gate.py: a growing conversation's newest turn fits the prefix cache.

The defect (memra#523 items 1 and 3): under the segmented policy that was the default until
2026-09-21 a newly published entry could be its own eviction victim, or the snapshot preflight
could refuse it outright, whenever the free share beside a PROMOTED cohort was smaller than the
entry. A growing long-context conversation then ran cold on every turn (`cached_tokens=0`, a full
re-prefill, the incident's 90-130 s ticks and 408s) while other tenants' entries sat protected,
and the only trace was one once-announced `snapshot skipped` line. `cached_tokens` decides the
customer's bill and whether the cache engaged at all, so the gate reads it from the response.
Since memra#523 item 2 (`docs/decisions/PREFIX-CACHE-POLICY.md`) the only policy is plain LRU:
the gate's subject is the DEFAULT policy, whatever it is, and the boot line must report it.

Serving shape, one card, two boots of the real `memra-server` per cell, plain path
(`MEMRA_SERVE_SPEC=0`), a SMALL explicit prefix budget, default policy. The CALIBRATION boot
runs with the prefix cache OFF (`MEMRA_PREFIX_CACHE_MB=0`) and replays the identical request
sequence, so each twin turn's own retained footprint (the bytes a request of that length
keeps after it retires, with no cache activity at all) is measured on the same binary and the
same card; the MEASURED boot then arms the cache with the small budget:

  1. cohort: a second tenant (`cache_salt=cohort`) sends three prompts of different lengths, each
     twice; the second send is a whole-entry hit (asserted: `cached_tokens == prompt_tokens` on
     the second send), so the cohort is a reused, recently touched set when the growth starts.
  2. twin: the growing tenant (`cache_salt=grow`) replays an 8-turn conversation, turn k+1 =
     turn k's prompt ids plus a fixed number of new ids (a real conversation appends the answer
     and the next message; the cache mechanics are the same, and exact ids make
     `cached_tokens` a closed-form expectation).

  The pressure arithmetic is checked from the server's own lines, not assumed: the cohort's
  `insert` lines give a bytes(tokens) fit for this artifact, and the gate REFUSES (exit 2)
  unless cohort <= 80 % of the budget, cohort + turn-1 entry > budget, and every turn's entry
  <= budget. That is exactly the incident's shape scaled to a small budget.

THE CAPTURE LAW (memra#602, research/spill-b-20260919/DAY17.md, fixed 2026-09-21): every entry the
prefix cache publishes lands on the GDN prime grid. The cold prime's `prime_cache` calls all start
on the 32-token WY-chunk grid; a restore at an OFF-grid entry end followed by one off-grid suffix
prime call is a second numeric program (`align_prime_ranges_to_gdn`, hybrid_forward.rs), and on
this gate's day-16 shape it flipped a greedy near-tie at turn 10 (restored `"_\t\t\"\t\t\"\t"`,
cold `"_\n"`, generated token 2) while the mechanics passed. The prompt-end seed now publishes at
the largest grid-aligned length not exceeding the prompt end that leaves at least PRIME_MIN_T (16)
prompt tokens behind it (`seed_capture_boundary`, worker.rs; the LCP and message-boundary captures
already did), so the published entry of a prompt of P tokens has `capture_len(P)` tokens, a hit
restores exactly that many, and `cached_tokens` reports the restored (aligned) length, never P.
`--grid` names the grid (the engine's `gdn_chunk_size()` default, 32); a server on another grid
fails V6 loudly instead of passing by accident.

Assertions (bytes from the server's `[prefix-cache]` lines and `/metrics`; every clause is a
verdict, none is relaxed for a binary):
  V1 cached:      every turn k >= 2 reports `usage.prompt_tokens_details.cached_tokens` EQUAL to
                  the entry turn k-1 published (its `insert (seed): N tokens` line): the previous
                  turn's entry was restored, whole, and billed as restored. (Until day 18 this read
                  `>= turn k-1's prompt_tokens`, which only an off-grid prompt-end entry satisfies.)
  V2 lines:       turn 1 publishes exactly one `insert (seed)` (no `insert refused`, no
                  `snapshot skipped`, no `seed REFUSED`); every turn k >= 2 has exactly one
                  `[prefix-cache] hit:` line whose restored length is turn k-1's published length
                  AND publishes exactly one entry of its own; no turn of the growing tenant is
                  refused or skipped.
  V3 effective:   after EVERY turn, the calibration boot's effective free
                  (`cuda_driver_free_bytes + cuda_pool_cached_bytes`) equals the measured boot's
                  effective free plus the cache's resident bytes (`prefix_cache_bytes`), within a
                  slack: the cache's only cost to effective free is what it holds, so every
                  evicted byte came back. An eviction that did not return its bytes would leave
                  the measured boot short by the evicted entry. This is #523 item 4's honesty (the
                  day-13 gate's V3) under the capacity-eviction shape, stated on states, not on
                  per-turn deltas: round 1 of day 14 asserted `consumed == grew` and failed on the
                  fix by exactly the base's cache-free consumption (1,111,666,004 B on turn 1);
                  round 2 subtracted the calibration turn's consumed bytes and failed only on turn
                  1 by 149,094,400 B, the amount by which the cache-off cohort phase (six cold
                  sends) had already grown the retained footprint that the cache-on cohort (three
                  cold, three restored) had not. The per-turn deltas stay in the table as report.
  V4 cohort:      at least one eviction of a cohort entry (`ns "cohort"`) during the growing
                  tenant's turns, and no turn evicts the entry it just published: the room came
                  from the reused cohort, never from the entry itself.
  V5 identity:    EVERY turn's completion text (sha256) equals the calibration boot's cold
                  completion of the same prompt: the restored render is the cold render. Until
                  day 18 this was a recorded column, not a verdict; on the day-16 lru shape
                  (`--cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000
                  --grow-tokens 150`) it read 11/12 on `main` (turn 10) while V1..V4 passed.
  V6 grid:        every `insert (seed)` of the measured boot (cohort and turns) has exactly
                  `capture_len(prompt_tokens)` tokens, and every hit's restored length is a
                  multiple of `--grid`: the capture law above, checked on the server's own
                  lines. Red on an off-grid prompt-end seed even when V5 happens to hold (a
                  near-tie that did not flip on one card is not identity).
  Recorded, not judged: the per-turn `[primeseg] call start=.. grid_off=..` receipts
  (MEMRA_DEBUG_PRIMESEG=1, an existing diagnostic) and the count of off-grid prime-call starts.
  V3 premise:     V3 compares the two boots' states, so it presumes they retain the SAME
                  non-prefix-cache device state after every send. The admission's
                  `[admit-oom] reclaim-on-defer` releases parked plain/spec/dspark sessions when
                  a request's cost plus the reserve floor exceeds effective free; when it fires
                  in one boot's window and not in the other's (the cache-on boot carries the
                  cache's resident bytes, so on a card close to the floor it crosses first), the
                  premise does not hold and V3 cannot be decided. The gate then REFUSES (exit 2),
                  typed `REFUSED: V3 premise: ...`, naming the windows, the released sessions
                  per boot and the card at each boot (driver free, compute-apps); the verdict
                  line V1..V6 would have printed is kept in summary.json under
                  `verdict_under_broken_premise`, never printed as the verdict. Two boots that
                  release identically keep V3 evaluated exactly as above; V3's clause, form and
                  slack are unchanged (spill-b day 29: the local RTX 5090 read `V3=FAIL` in both
                  door arms with a constant `-410352980` B error, one cohort-shaped parked plain
                  session released by the cache-on boot's turn-2 reclaim beside a 1.4 GB
                  co-tenant, while the 96 GB card read `-> PASS`).

Exit 0 = every assertion held; 1 = a verdict failed (red on `main` before day 14: `cached_tokens=0`
on every turn after the once-announced `snapshot skipped` line; red on `main` before day 18 on
V5 and V6: the off-grid prompt-end seed); 2 = REFUSED (lock, port, shape).

usage: prefix-newest-turn-fits-gate.py [--external-lock FD] --model GGUF --bin memra-server \
           --out NEW_DIR [--port N] [--budget-mib 1024] [--cohort-tokens 2800,3000,3200] \
           [--turns 8] [--start-tokens 9200] [--grow-tokens 300] [--grid 32]
Lock: the canonical rig lock only (`/tmp/memra-gpu.lock` or `/tmp/memra-5090.lock`), held for
the whole cell; `--external-lock FD` inherits the collector's FD (lead ruling 5) and is verified
with tools/tier-lock-proof.py before anything binds a port or boots a server.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")
MIB = 1 << 20
V3_SLACK = 64 * MIB
# The engine's constants the capture law is stated in (worker.rs / hybrid_forward.rs): a capture
# boundary leaves at least PRIME_MIN_T prompt tokens behind it (the W1 floor: a shorter remainder
# rides tokenwise decode_step, another program) and an entry is never shorter than
# PREFIX_CACHE_MIN_TOKENS. The grid itself is `--grid` (Engine::gdn_chunk_size(), 32 shipped).
PRIME_MIN_T = 16
PREFIX_CACHE_MIN_TOKENS = 64

RE_ON = re.compile(r"\[prefix-cache\] on: budget \d+MB \((\d+) B,.*policy ([^,]+),")
# `insert (seed)` since memra#523 item 2; `insert probation (seed)` on the segmented binaries.
RE_INSERT = re.compile(r"\[prefix-cache\] insert (?:probation )?\(([^)]+)\): (\d+) tokens, ([\d.]+)MB")
RE_HIT = re.compile(r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens from cache")
# `evict (LRU)`, `evict (snapshot preflight, LRU)`; segmented binaries: `evict (Protected LRU)`,
# `evict (snapshot preflight, Probation LRU)`; the pre-fix main: `evict (snapshot preflight)`.
RE_EVICT = re.compile(
    r"\[prefix-cache\] evict \((?:snapshot preflight(?:, )?)?(Probation|Protected)? ?(?:LRU)?\): (\d+) tokens, ([\d.]+)MB"
)
RE_NS = re.compile(r'ns "([^"]+)"')
RE_REFUSED = re.compile(r"\[prefix-cache\] insert refused: (.*)$")
RE_SKIPPED = re.compile(r"\[prefix-cache\] snapshot skipped: (.*)$")
# The grid law's own typed refusal (memra#602): counted with the refusals above.
RE_SEED_REFUSED = re.compile(r"\[prefix-cache\] seed REFUSED \(grid\): (.*)$")
# Prime-call receipts (MEMRA_DEBUG_PRIMESEG=1): the call start's offset from the grid.
RE_PRIMESEG = re.compile(r"\[primeseg\] call start=(\d+) take=(\d+) grid_off=(\d+)")
RE_DEMOTE = re.compile(r"\[prefix-cache\] demote \(protected bytes\): ([\d.]+)MB")
RE_RECLAIM = re.compile(r"\[admit-oom\] reclaim-on-defer: ")
# The same line, with its counts: parked sessions released per pool and the effective free it moved.
RE_RECLAIM_PARKED = re.compile(
    r"\[admit-oom\] reclaim-on-defer: evicted (\d+) prefix entries \+ (\d+) plain \+ (\d+) spec \+ (\d+) dspark "
    r"parked sessions \(global LRU\); effective free (\d+)MB -> (\d+)MB"
)
PARKED_POOLS = ("plain", "spec", "dspark")
# The per-request route receipt (`[glm5-spec] route=plain ... cold=1 restored=0 reason=...`) is
# printed by draft-capable models on every request; it is recorded per turn so a cold turn can
# be quoted from the receipt, never a verdict input (V1 reads `cached_tokens` from the response).
RE_ROUTE = re.compile(r"\[glm5-spec\] route=\S+ .*cold=(\d) restored=(\d)")
# Models without that route line print the spec admission receipt per request instead
# (`[spec-k] model=... prompt=N cached=M lcp=L ...`); recorded for the same reason.
RE_SPECK = re.compile(r"\[spec-k\] model=.* prompt=(\d+) cached=(\d+) lcp=(\d+)")


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def capture_len(prompt_tokens: int, grid: int) -> int | None:
    """The entry length the prompt-end seed publishes for a prompt of `prompt_tokens` (memra#602).

    The largest multiple of `grid` not exceeding the prompt end whose remainder is 0 or at least
    PRIME_MIN_T (a remainder of 1..PRIME_MIN_T-1 steps down one grid unit, the W1 floor the
    engine's `grid_align_boundary_within` applies to every capture); `None` when that length is
    under PREFIX_CACHE_MIN_TOKENS (the server refuses, typed: `seed REFUSED (grid)`).
    """
    if prompt_tokens % grid == 0:
        return prompt_tokens
    b = prompt_tokens // grid * grid
    while b >= grid and prompt_tokens - b < PRIME_MIN_T:
        b -= grid
    return b if b >= PREFIX_CACHE_MIN_TOKENS else None


class Server:
    def __init__(self, binary: str, model: str, port: int, log: Path, budget_mib: int):
        self.binary, self.model, self.port, self.log = binary, model, port, log
        self.budget_mib = budget_mib
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"
        self.offset = 0
        self.card_at_boot: dict | None = None

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
                "MEMRA_CTX": "16384",
                "MEMRA_MAX_SESSIONS": "4",
                "MEMRA_PREFIX_CACHE_MB": str(self.budget_mib),
                "MEMRA_SERVE_SPEC": "0",
                "MEMRA_TIMEOUT_MS_MAX": "240000",
                # Prime-call receipts for the record (an existing documented diagnostic): the
                # grid law is about call STARTS, and V6 is only interpretable next to them.
                "MEMRA_DEBUG_PRIMESEG": "1",
            }
        )
        # The subject is the DEFAULT policy: never inherit a launcher's value for the doors that
        # used to select one (removed 2026-09-21; a binary that still reads them must not see them).
        env.pop("MEMRA_PREFIX_CACHE_POLICY", None)
        env.pop("MEMRA_PREFIX_CACHE_PROTECTED_PCT", None)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.card_at_boot = card_listing()
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

    def complete(self, ids: list[int], salt: str, max_tokens: int = 8) -> dict:
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
        # compat mode carries the OpenAI usage object; the native shape carries the same
        # worker truth as flat prompt_tokens/cached_tokens fields.
        usage = payload.get("usage")
        if usage is None and status == 200:
            usage = {
                "prompt_tokens": payload.get("prompt_tokens"),
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
            "cuda_pool_reserved_bytes",
            "cuda_pool_used_bytes",
            "cuda_pool_cached_bytes",
            "prefix_cache_bytes",
            "prefix_cache_entries",
            "prefix_cache_evictions",
            "prefix_cache_skips_budget",
            "prefix_cache_skips_pinned",
            "active_sessions",
            "queued_requests",
        )
    }


def effective_free(row: dict) -> int:
    return row["cuda_driver_free_bytes"] + row["cuda_pool_cached_bytes"]


def parse_window(lines: list[str]) -> dict:
    ev: dict = {"inserts": [], "hits": [], "evicts": [], "refused": [], "skipped": [], "demotes": 0, "reclaims": 0, "lines": [], "route": [], "primeseg": [], "parked_releases": []}
    for ln in lines:
        if "[prefix-cache]" in ln or "[admit-oom]" in ln or "[glm5-spec] route=" in ln or "[spec-k] model=" in ln or "[primeseg] call" in ln:
            ev["lines"].append(ln.strip())
        m = RE_PRIMESEG.search(ln)
        if m:
            ev["primeseg"].append({"start": int(m.group(1)), "take": int(m.group(2)), "grid_off": int(m.group(3))})
            continue
        m = RE_ROUTE.search(ln)
        if m:
            ev["route"].append({"kind": "route", "cold": int(m.group(1)), "restored": int(m.group(2)), "line": ln.strip()})
            continue
        m = RE_SPECK.search(ln)
        if m:
            ev["route"].append({"kind": "spec-k", "prompt": int(m.group(1)), "cached": int(m.group(2)), "lcp": int(m.group(3)), "line": ln.strip()})
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
            ns = RE_NS.search(ln)
            ev["evicts"].append({"segment": m.group(1) or "unstated", "tokens": int(m.group(2)), "mb": float(m.group(3)), "ns": ns.group(1) if ns else ""})
            continue
        m = RE_REFUSED.search(ln)
        if m:
            ev["refused"].append(ln.strip())
            continue
        m = RE_SKIPPED.search(ln)
        if m:
            ev["skipped"].append(ln.strip())
            continue
        m = RE_SEED_REFUSED.search(ln)
        if m:
            ev["refused"].append(ln.strip())
            continue
        if RE_DEMOTE.search(ln):
            ev["demotes"] += 1
        if RE_RECLAIM.search(ln):
            ev["reclaims"] += 1
            m = RE_RECLAIM_PARKED.search(ln)
            if m:
                ev["parked_releases"].append({
                    "prefix": int(m.group(1)), "plain": int(m.group(2)), "spec": int(m.group(3)), "dspark": int(m.group(4)),
                    "effective_free_before_mb": int(m.group(5)), "effective_free_after_mb": int(m.group(6)), "line": ln.strip(),
                })
            else:
                # A reclaim line the detailed shape did not parse (engine wording drift): the
                # premise is then UNREADABLE for this window, never "released nothing" (revuto on
                # #633). `premise_readable()` turns it into a typed refusal.
                ev.setdefault("reclaim_unparsed", []).append(ln.strip())
    return ev


def parked_released(window: dict) -> tuple[int, int, int]:
    """Parked sessions a window's reclaim-on-defer lines released, per pool (plain, spec, dspark)."""
    return tuple(sum(r[pool] for r in window.get("parked_releases", [])) for pool in PARKED_POOLS)


def window_reclaims_unparsed(window: dict) -> list[str]:
    """Reclaim lines the detailed shape did not parse: every reclaim must account for its releases."""
    unparsed = list(window.get("reclaim_unparsed", []))
    if window.get("reclaims", 0) != len(window.get("parked_releases", [])) + len(unparsed):
        unparsed.append(f"<{window.get('reclaims', 0)} reclaim line(s), {len(window.get('parked_releases', []))} parsed>")
    return unparsed


def premise_unreadable_windows(cal: dict, rec: dict) -> list[dict]:
    """Windows in either boot whose reclaim lines the detailed shape did not parse."""
    rows = []
    for c, m in zip(cal["cohort"], rec["cohort"]):
        for j, (cw, mw) in enumerate(zip(c.get("windows", []), (m["window_first"], m["window_second"])), start=1):
            for boot, w in (("calibration", cw), ("measured", mw)):
                bad = window_reclaims_unparsed(w)
                if bad:
                    rows.append({"who": f"cohort {c['tokens']} send {j}", "boot": boot, "lines": bad})
    for ct, mt in zip(cal["turns"], rec["turns"]):
        for boot, w in (("calibration", ct["window"]), ("measured", mt["window"])):
            bad = window_reclaims_unparsed(w)
            if bad:
                rows.append({"who": f"turn {mt['turn']}", "boot": boot, "lines": bad})
    return rows


def gate_outcome(others_ok: bool, v3: bool, premise_ok: bool, premise_readable: bool) -> tuple[str, int]:
    """The exit rule (revuto on #633): a failed V1/V2/V4/V5/V6 is a verdict FAIL (exit 1) whatever
    the premise says, because asymmetric parked-session releases affect V3 only; a broken or
    unreadable premise with every other clause holding is a typed refusal (exit 2), V3 undecided;
    otherwise the verdict stands on V3 (exit 0 or 1)."""
    if not others_ok:
        return ("verdict", 1)
    if not (premise_ok and premise_readable):
        return ("refuse", 2)
    return ("verdict", 0 if v3 else 1)


def v3_premise_rows(cal: dict, rec: dict) -> list[dict]:
    """Window by window (each cohort send, then each turn), what each boot's reclaim released.

    V3's state equation presumes the two boots retain the same parked sessions after every send;
    a window whose releases differ between the boots breaks that premise for every later state.
    """
    rows = []
    for c, m in zip(cal["cohort"], rec["cohort"]):
        for j, (cw, mw) in enumerate(zip(c.get("windows", []), (m["window_first"], m["window_second"])), start=1):
            rows.append({"who": f"cohort {c['tokens']} send {j}", "calibration": parked_released(cw), "measured": parked_released(mw)})
    for ct, mt in zip(cal["turns"], rec["turns"]):
        rows.append({"who": f"turn {mt['turn']}", "calibration": parked_released(ct["window"]), "measured": parked_released(mt["window"])})
    for r in rows:
        r["equal"] = tuple(r["calibration"]) == tuple(r["measured"])
    return rows


def card_listing() -> dict:
    """The card right now: driver free of total (MiB) and the compute-apps listing (read only)."""
    def q(*args: str) -> str:
        return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    apps = [ln.strip() for ln in q("--query-compute-apps=pid,process_name,used_memory").splitlines() if ln.strip()]
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "memory_free_total_mib": q("--query-gpu=memory.free,memory.total"), "compute_apps": apps}


def card_brief(card: dict | None) -> str:
    if not card:
        return "card not sampled"
    apps = []
    for a in card["compute_apps"]:
        parts = [p.strip() for p in a.split(",")]
        if len(parts) >= 3:
            apps.append(f"{Path(parts[1]).name} {parts[2]} MiB")
    return f"free/total MiB {card['memory_free_total_mib']}, compute-apps {len(apps)} [{'; '.join(apps)}]"


def v3_premise_refusal(rows: list[dict], cal_card: dict | None, meas_card: dict | None, budget: int) -> str:
    broken = [r for r in rows if not r["equal"]]
    detail = "; ".join(
        f"{r['who']}: calibration released plain/spec/dspark {'/'.join(map(str, r['calibration']))}, measured {'/'.join(map(str, r['measured']))}"
        for r in broken
    )
    return (
        f"V3 premise: `[admit-oom] reclaim-on-defer` released parked sessions in one boot without the counterpart "
        f"in the other on {len(broken)} window(s) ({detail}); the state equation compares boots whose retained "
        f"parked-session sets differ, so V3 is not decidable on this card shape (budget {budget} B, MEMRA_CTX 16384, "
        f"MEMRA_MAX_SESSIONS 4); card at the calibration boot: {card_brief(cal_card)}; at the measured boot: "
        f"{card_brief(meas_card)}; the clause verdicts are kept in summary.json (verdict_under_broken_premise), not printed as a verdict"
    )


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


def ids_for(tokens: int) -> list[int]:
    # Stable, well inside the vocab, non-repeating over the longest prompt used here.
    return [3000 + (i % 7000) for i in range(tokens)]


def cohort_ids(tokens: int) -> list[int]:
    # A distinct id range: the cohort shares no prefix with the twin.
    return [i + 60000 for i in ids_for(tokens)]


def twin_ids(start_tokens: int, grow_tokens: int, turns: int) -> list[list[int]]:
    """Turn k+1 is turn k's ids plus the appended ids (the previous answer plus the next message)."""
    out = []
    ids = ids_for(start_tokens)
    for k in range(1, turns + 1):
        if k > 1:
            ids = ids + [20000 + ((k * 1000 + j) % 9000) for j in range(grow_tokens)]
        out.append(ids)
    return out


def calibrate_footprint(args, cohort_tokens: list[int], prompts: list[list[int]]) -> dict:
    """The cache-off boot: the identical request sequence, no prefix cache, per-turn consumed bytes."""
    srv = Server(args.bin, args.model, args.port, args.out / "calibration" / "server.log", 0)
    srv.boot()
    cal: dict = {"boot_lines": [], "cohort": [], "turns": [], "card_at_boot": srv.card_at_boot}
    try:
        boot_lines = srv.new_log_lines()
        cal["boot_lines"] = [ln.strip() for ln in boot_lines if "[prefix-cache]" in ln]
        if any(RE_ON.search(ln) for ln in boot_lines):
            refuse("the calibration boot armed the prefix cache; MEMRA_PREFIX_CACHE_MB=0 must disable it")
        for n in cohort_tokens:
            ids = cohort_ids(n)
            sends = []
            windows = []
            for _ in range(2):
                r = srv.complete(ids, "cohort")
                time.sleep(0.5)
                windows.append(parse_window(srv.new_log_lines()))
                if r["status"] != 200:
                    refuse(f"calibration: cohort prompt of {n} tokens was not served: {r['error']}")
                sends.append(r)
            cal["cohort"].append({"tokens": n, "first": sends[0], "second": sends[1], "windows": windows, "metrics_after": srv.settled_metrics()})
        for k, ids in enumerate(prompts, start=1):
            before = srv.settled_metrics()
            r = srv.complete(ids, "grow")
            time.sleep(0.5)
            window = parse_window(srv.new_log_lines())
            after = srv.settled_metrics()
            if before is None or after is None:
                refuse(f"calibration turn {k}: /metrics never settled")
            if r["status"] != 200:
                refuse(f"calibration turn {k} was not served (HTTP {r['status']}): {r['error']}")
            if before["prefix_cache_bytes"] or after["prefix_cache_bytes"] or window["inserts"] or window["hits"]:
                refuse(f"calibration turn {k}: the prefix cache took part in a cache-off boot")
            cal["turns"].append({
                "turn": k,
                "prompt_ids": len(ids),
                "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"],
                "elapsed_s": r["elapsed_s"],
                "finish_reason": r["finish_reason"],
                "text_sha256": r["text_sha256"],
                "window": window,
                "metrics_before": before,
                "metrics_after": after,
                "effective_free_consumed": effective_free(before) - effective_free(after),
            })
    finally:
        srv.stop()
    return cal


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
    ap.add_argument("--port", type=int, default=18114)
    ap.add_argument("--budget-mib", type=int, default=1024)
    ap.add_argument("--cohort-tokens", default="2800,3000,3200")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--start-tokens", type=int, default=9200)
    ap.add_argument("--grow-tokens", type=int, default=300)
    ap.add_argument("--grid", type=int, default=32, help="the GDN prime grid (Engine::gdn_chunk_size(), 32 shipped)")
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
    if args.turns < 2:
        refuse("--turns must be at least 2")
    if args.grid < PRIME_MIN_T or args.grid % PRIME_MIN_T:
        refuse(f"--grid {args.grid} is not a multiple of PRIME_MIN_T={PRIME_MIN_T}")
    if any(capture_len(n, args.grid) is None for n in cohort_tokens) or any(capture_len(len(p), args.grid) is None for p in twin_ids(args.start_tokens, args.grow_tokens, args.turns)):
        refuse("a cohort or twin prompt is too short for a grid-aligned entry (capture_len is None); the shape needs longer prompts")
    last_tokens = args.start_tokens + (args.turns - 1) * args.grow_tokens
    if last_tokens + 8 + 64 >= 16384:
        refuse(f"the last turn ({last_tokens} tokens) does not fit MEMRA_CTX=16384")
    args.out.mkdir(parents=True)
    (args.out / "LOCK.json").write_text(proof.stdout)
    rig = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,power.limit,power.max_limit,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True,
    ).stdout.strip()
    binsha = hashlib.sha256(Path(args.bin).read_bytes()).hexdigest()
    (args.out / "rig.json").write_text(json.dumps({"nvidia_smi": rig, "binary_sha256": binsha, "binary": args.bin, "model": args.model}, indent=2) + "\n")

    prompts = twin_ids(args.start_tokens, args.grow_tokens, args.turns)
    cal = calibrate_footprint(args, cohort_tokens, prompts)
    (args.out / "calibration.json").write_text(json.dumps(cal, indent=2) + "\n")
    footprint = {t["turn"]: t["effective_free_consumed"] for t in cal["turns"]}
    cal_eff_after = {t["turn"]: effective_free(t["metrics_after"]) for t in cal["turns"]}
    cold_sha = {t["turn"]: t["text_sha256"] for t in cal["turns"]}

    srv = Server(args.bin, args.model, args.port, args.out / "measured" / "server.log", args.budget_mib)
    srv.boot()
    rec: dict = {"boot": {}, "cohort": [], "turns": []}
    try:
        boot_lines = srv.new_log_lines()
        on = next((RE_ON.search(ln) for ln in boot_lines if RE_ON.search(ln)), None)
        if on is None:
            refuse("no `[prefix-cache] on:` boot line; the cache did not arm")
        budget = int(on.group(1))
        policy = on.group(2).strip()
        rec["boot"] = {"budget_bytes": budget, "policy": policy, "line": next(ln.strip() for ln in boot_lines if RE_ON.search(ln)), "card_at_boot": srv.card_at_boot}
        if "PLAIN-LRU" not in policy.upper():
            refuse(f"the boot did not report the plain-LRU default (policy {policy!r}); the gate's subject is the default policy")
        if budget != args.budget_mib * MIB:
            refuse(f"budget {budget} B differs from the requested {args.budget_mib} MiB")
        cohort_cap = budget * 80 // 100  # the incident's cohort share of the budget

        # ---- 1. the protected cohort -----------------------------------------------------
        points: list[tuple[int, float]] = []
        grid_rows: list[dict] = []  # V6: every published length and every restored length
        for n in cohort_tokens:
            ids = cohort_ids(n)
            first = srv.complete(ids, "cohort")
            time.sleep(0.5)
            w1 = parse_window(srv.new_log_lines())
            second = srv.complete(ids, "cohort")
            time.sleep(0.5)
            w2 = parse_window(srv.new_log_lines())
            row = srv.settled_metrics()
            rec["cohort"].append({"tokens": n, "first": first, "second": second, "window_first": w1, "window_second": w2, "metrics_after": row})
            if first["status"] != 200 or second["status"] != 200:
                refuse(f"cohort prompt of {n} tokens was not served: {first['error'] or second['error']}")
            ins = [i for i in w1["inserts"] if i["why"] == "seed"]
            if len(ins) != 1:
                refuse(f"cohort prompt of {n} tokens published {len(ins)} seed entries, expected one (window: {w1['lines']})")
            points.append((ins[0]["tokens"], ins[0]["mb"] * 1e6))
            # The second send must restore the entry the first one published, whole (the
            # published length, which the grid law makes `capture_len(n)`: judged by V6 below).
            if second["cached_tokens"] != ins[0]["tokens"] or not w2["hits"] or w2["hits"][0]["hit"] != ins[0]["tokens"]:
                refuse(f"cohort promotion did not happen for {n} tokens: second send cached={second['cached_tokens']} of {second['prompt_tokens']}, published {ins[0]['tokens']}")
            grid_rows.append({"who": f"cohort {n}", "kind": "insert", "tokens": ins[0]["tokens"], "expected": capture_len(first["prompt_tokens"], args.grid), "prompt_tokens": first["prompt_tokens"]})
            grid_rows.append({"who": f"cohort {n}", "kind": "hit", "tokens": second["cached_tokens"], "expected": ins[0]["tokens"], "prompt_tokens": second["prompt_tokens"]})
        cohort_row = rec["cohort"][-1]["metrics_after"]
        if cohort_row is None:
            refuse("/metrics never settled after the cohort")
        cohort_bytes = cohort_row["prefix_cache_bytes"]
        if cohort_row["prefix_cache_entries"] != len(cohort_tokens) or cohort_row["prefix_cache_evictions"] != 0:
            refuse(f"the cohort is not intact: {cohort_row}")
        if any(w["demotes"] for c in rec["cohort"] for w in (c["window_first"], c["window_second"])):
            refuse("a cohort entry was demoted during seeding; the cohort exceeds the protected share")
        per_token, fixed = fit_bytes_per_token(points)
        est = lambda tokens: fixed + per_token * tokens  # noqa: E731
        e1 = est(args.start_tokens)
        e_last = est(last_tokens)
        shape = {
            "budget_bytes": budget,
            "cohort_cap_bytes": cohort_cap,
            "cohort_bytes": cohort_bytes,
            "cohort_points": points,
            "fit_bytes_per_token": per_token,
            "fit_fixed_bytes": fixed,
            "turn1_entry_estimate_bytes": e1,
            "last_turn_entry_estimate_bytes": e_last,
            "cohort_within_80pct": cohort_bytes <= cohort_cap,
            "turn1_exceeds_free_share": cohort_bytes + e1 > budget,
            "every_turn_fits_budget": e_last <= budget,
            "two_turns_fit_budget": est(last_tokens - args.grow_tokens) + e_last <= budget,
        }
        (args.out / "shape.json").write_text(json.dumps(shape, indent=2) + "\n")
        if not shape["cohort_within_80pct"]:
            refuse(f"cohort {cohort_bytes} B exceeds 80 % of the budget ({cohort_cap} B): not the incident's shape")
        if not shape["turn1_exceeds_free_share"]:
            refuse(f"no pressure at turn 1: cohort {cohort_bytes} + turn-1 entry {e1:.0f} <= budget {budget}; raise --start-tokens or lower --budget-mib")
        if not shape["every_turn_fits_budget"]:
            refuse(f"the last turn's entry {e_last:.0f} B exceeds the budget {budget} B; every turn must be insertable")

        # ---- 2. the 8-turn twin ------------------------------------------------------------
        prev_prompt_tokens = None
        prev_published = None
        for k, ids in enumerate(prompts, start=1):
            before = srv.settled_metrics()
            r = srv.complete(ids, "grow")
            time.sleep(0.5)
            window = parse_window(srv.new_log_lines())
            after = srv.settled_metrics()
            if before is None or after is None:
                refuse(f"turn {k}: /metrics never settled")
            eff_before, eff_after = effective_free(before), effective_free(after)
            grew = after["prefix_cache_bytes"] - before["prefix_cache_bytes"]
            consumed = eff_before - eff_after
            seed_inserts = [i for i in window["inserts"] if i["why"] == "seed"]
            published = seed_inserts[0]["tokens"] if len(seed_inserts) == 1 else None
            expected_capture = capture_len(r["prompt_tokens"], args.grid) if r["prompt_tokens"] else None
            turn = {
                "turn": k,
                "prompt_ids": len(ids),
                "status": r["status"],
                "prompt_tokens": r["prompt_tokens"],
                "cached_tokens": r["cached_tokens"],
                "prev_prompt_tokens": prev_prompt_tokens,
                "published_tokens": published,
                "seed_inserts": len(seed_inserts),
                "expected_capture_tokens": expected_capture,
                "prev_published_tokens": prev_published,
                "primeseg": window["primeseg"],
                "off_grid_calls": sum(1 for c in window["primeseg"] if c["grid_off"] != 0),
                "elapsed_s": r["elapsed_s"],
                "finish_reason": r["finish_reason"],
                "text_sha256": r["text_sha256"],
                "error": r["error"],
                "window": window,
                "metrics_before": before,
                "metrics_after": after,
                "effective_free_before": eff_before,
                "effective_free_after": eff_after,
                "effective_free_consumed": consumed,
                "footprint_calibration_bytes": footprint[k],
                "prefix_bytes_grew": grew,
                "inserted_bytes_from_line": int(sum(i["mb"] for i in window["inserts"]) * 1e6),
                "evicted_bytes_from_lines": int(sum(e["mb"] for e in window["evicts"]) * 1e6),
                "delta_error_bytes": (consumed - footprint[k] - grew) if window["evicts"] else None,
                "calibration_effective_free_after": cal_eff_after[k],
                "v3_state_error_bytes": cal_eff_after[k] - (eff_after + after["prefix_cache_bytes"]),
                "cold_text_sha256": cold_sha[k],
                "text_identical_to_cold": r["text_sha256"] == cold_sha[k],
            }
            rec["turns"].append(turn)
            if r["status"] != 200:
                refuse(f"turn {k} was not served (HTTP {r['status']}): {r['error']}")
            if published is not None:
                grid_rows.append({"who": f"turn {k}", "kind": "insert", "tokens": published, "expected": expected_capture, "prompt_tokens": r["prompt_tokens"]})
            if k > 1:
                grid_rows.append({"who": f"turn {k}", "kind": "hit", "tokens": r["cached_tokens"], "expected": prev_published, "prompt_tokens": r["prompt_tokens"]})
            prev_prompt_tokens = r["prompt_tokens"]
            prev_published = published
        rec["final_metrics"] = srv.settled_metrics()
    finally:
        srv.stop()

    # ---- verdicts -------------------------------------------------------------------------
    turns = rec["turns"]
    # V1: the previous turn's PUBLISHED entry was restored, whole (equality, not >=: see the header).
    v1_rows = [t["cached_tokens"] is not None and t["prev_published_tokens"] is not None and t["cached_tokens"] == t["prev_published_tokens"] for t in turns[1:]]
    v1 = all(v1_rows)
    cold_after_1 = sum(1 for t in turns[1:] if (t["cached_tokens"] or 0) == 0)

    def publishes(t: dict) -> bool:
        return t["seed_inserts"] == 1  # the length is V6's clause

    v2_rows = []
    for t in turns:
        w = t["window"]
        clean = not w["refused"] and not w["skipped"]
        if t["turn"] == 1:
            v2_rows.append(clean and publishes(t))
        else:
            hit_ok = len(w["hits"]) == 1 and t["prev_published_tokens"] is not None and w["hits"][0]["hit"] == t["prev_published_tokens"]
            v2_rows.append(clean and hit_ok and publishes(t))
    v2 = all(v2_rows)
    v3_rows = [abs(t["v3_state_error_bytes"]) <= V3_SLACK for t in turns]
    v3 = bool(v3_rows) and all(v3_rows)
    cohort_evictions = sum(1 for t in turns for e in t["window"]["evicts"] if e["ns"] == "cohort")
    self_evictions = sum(1 for t in turns for e in t["window"]["evicts"] if e["ns"] == "grow" and t["published_tokens"] is not None and e["tokens"] == t["published_tokens"])
    total_evictions = sum(len(t["window"]["evicts"]) for t in turns)
    refused_lines = [ln for t in turns for ln in t["window"]["refused"] + t["window"]["skipped"]]
    v4 = cohort_evictions >= 1 and self_evictions == 0
    # V5: the restored render IS the cold render, on every turn (turn 1 is cold in both boots and
    # pins that the cache-on cold prime, now stopped on its seed boundary, is the cache-off one).
    v5_rows = [t["text_identical_to_cold"] for t in turns]
    v5 = bool(v5_rows) and all(v5_rows)
    # V6: the capture law on the server's own lines (cohort and turns): every published entry is
    # capture_len(prompt), every restored length is the published one and a grid multiple.
    v6_rows = []
    for row in grid_rows:
        on_law = row["tokens"] is not None and row["expected"] is not None and row["tokens"] == row["expected"]
        on_grid = row["tokens"] is not None and row["tokens"] % args.grid == 0
        row["ok"] = on_law and on_grid
        v6_rows.append(row["ok"])
    v6 = bool(v6_rows) and all(v6_rows)
    off_grid_calls = sum(t["off_grid_calls"] for t in turns)
    ok = v1 and v2 and v3 and v4 and v5 and v6
    verdict = (
        f"PREFIX-NEWEST-TURN-FITS: budget_bytes={rec['boot']['budget_bytes']} cohort_bytes={shape['cohort_bytes']} "
        f"turns={len(turns)} cold_turns_after_1={cold_after_1} cached_ok={sum(v1_rows)}/{len(v1_rows)} "
        f"lines_ok={sum(v2_rows)}/{len(v2_rows)} evictions={total_evictions} cohort_evictions={cohort_evictions} self_evictions={self_evictions} "
        f"refused_or_skipped={len(refused_lines)} effective_free_ok={sum(v3_rows)}/{len(v3_rows)} "
        f"identity_ok={sum(v5_rows)}/{len(v5_rows)} grid_ok={sum(v6_rows)}/{len(v6_rows)} grid={args.grid} off_grid_calls={off_grid_calls} "
        f"V1={'ok' if v1 else 'FAIL'} V2={'ok' if v2 else 'FAIL'} V3={'ok' if v3 else 'FAIL'} V4={'ok' if v4 else 'FAIL'} "
        f"V5={'ok' if v5 else 'FAIL'} V6={'ok' if v6 else 'FAIL'} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    table = ["| turn | prompt_tokens | cached_tokens | prev published | published | expected capture | off-grid calls | server receipt | hit line | insert | evict (segment) | refused/skipped | effective free after | resident prefix bytes | cache-off effective free after | V3 error (state) | consumed | footprint (cache-off) | grew | delta error | elapsed s | text sha256[:16] | == cold |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for t in turns:
        w = t["window"]
        hits = ", ".join("{hit} of {prompt}".format(**h) for h in w["hits"]) or "none"
        inserts = ", ".join("{tokens} tok {mb}MB ({why})".format(**i) for i in w["inserts"]) or "none"
        evicts = ", ".join("{tokens} tok {mb}MB ({segment}, ns {ns})".format(**e) for e in w["evicts"]) or "none"
        prev = t["prev_published_tokens"] if t["prev_published_tokens"] is not None else "-"
        derr = t["delta_error_bytes"] if t["delta_error_bytes"] is not None else "-"
        route = ", ".join(
            ("cold={cold} restored={restored}" if r["kind"] == "route" else "cached={cached} lcp={lcp}").format(**r) for r in w["route"]
        ) or "none"
        table.append(
            f"| {t['turn']} | {t['prompt_tokens']} | {t['cached_tokens']} | {prev} | {t['published_tokens']} | {t['expected_capture_tokens']} | {t['off_grid_calls']} | {route} | {hits} | {inserts} | {evicts} | "
            f"{len(w['refused']) + len(w['skipped'])} | {t['effective_free_after']} | {t['metrics_after']['prefix_cache_bytes']} | "
            f"{t['calibration_effective_free_after']} | {t['v3_state_error_bytes']} | {t['effective_free_consumed']} | "
            f"{t['footprint_calibration_bytes']} | {t['prefix_bytes_grew']} | {derr} | {t['elapsed_s']} | {t['text_sha256'][:16]} | "
            f"{'yes' if t['text_identical_to_cold'] else 'NO'} |"
        )
    premise_rows = v3_premise_rows(cal, rec)
    premise_ok = all(r["equal"] for r in premise_rows)
    unreadable = premise_unreadable_windows(cal, rec)
    premise_readable = not unreadable
    others_ok = v1 and v2 and v4 and v5 and v6
    outcome, exit_code = gate_outcome(others_ok, v3, premise_ok, premise_readable)
    if not premise_readable:
        refusal = ("V3 premise unreadable: " + "; ".join(f"{r['who']} ({r['boot']}): {' | '.join(r['lines'])}" for r in unreadable)
                   + " (a reclaim-on-defer line the gate's detailed shape did not parse; engine wording drift?)")
    elif not premise_ok:
        refusal = v3_premise_refusal(premise_rows, cal.get("card_at_boot"), rec["boot"].get("card_at_boot"), rec["boot"]["budget_bytes"])
    else:
        refusal = None
    if outcome == "verdict" and refusal is not None:
        # a non-V3 clause failed: the verdict prints (exit 1) and the premise note rides beside it
        verdict += " (V3 premise: " + ("unreadable" if not premise_readable else "broken") + ", V3 undecided)"
    summary = {
        "verdict": verdict if outcome == "verdict" else f"REFUSED: {refusal}",
        "verdict_under_broken_premise": None if outcome == "verdict" else verdict,
        "pass": outcome == "verdict" and exit_code == 0,
        "exit_code": exit_code,
        "v3_premise": {"holds": premise_ok, "readable": premise_readable, "rows": premise_rows, "unreadable": unreadable},
        "binary_sha256": binsha,
        "rig": rig,
        "boot": rec["boot"],
        "shape": shape,
        "cohort": rec["cohort"],
        "turns": turns,
        "calibration": {"cohort": cal["cohort"], "turns": cal["turns"], "boot_lines": cal["boot_lines"], "card_at_boot": cal.get("card_at_boot")},
        "turns_identical_to_cold": sum(1 for t in turns if t["text_identical_to_cold"]),
        "final_metrics": rec.get("final_metrics"),
        "refused_or_skipped_lines": refused_lines,
        "assertions": {"V1_cached": v1, "V2_lines": v2, "V3_effective_free": v3, "V4_cohort_evicted_never_self": v4, "V5_identity": v5, "V6_grid": v6},
        "capture_law": {"grid": args.grid, "prime_min_t": PRIME_MIN_T, "prefix_cache_min_tokens": PREFIX_CACHE_MIN_TOKENS, "rows": grid_rows},
        "off_grid_calls": off_grid_calls,
        "v3_slack_bytes": V3_SLACK,
        "v3_form": "state: calibration effective free after turn k == measured effective free after turn k + resident prefix bytes",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "TURNS.md").write_text("\n".join(table) + "\n")
    (args.out / "VERDICT.txt").write_text(summary["verdict"] + "\n")
    print(rec["boot"]["line"])
    print(f"calibration (cache off): footprint per turn {[footprint[t['turn']] for t in turns]} B; "
          f"measured completions identical to cold on {summary['turns_identical_to_cold']}/{len(turns)} turns")
    for ln in refused_lines:
        print("refusal:", ln)
    for t in turns:
        for ln in t["window"]["lines"]:
            print(f"turn {t['turn']}:", ln)
    print("\n".join(table))
    for r in premise_rows:
        if not r["equal"]:
            print(f"premise: {r['who']}: calibration released plain/spec/dspark {r['calibration']}, measured {r['measured']}")
    for r in unreadable:
        print(f"premise unreadable: {r['who']} ({r['boot']}): {' | '.join(r['lines'])}")
    if outcome == "refuse":
        refuse(refusal)
    print(verdict, flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
