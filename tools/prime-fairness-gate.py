#!/usr/bin/env python3
"""prime-fairness-gate.py: one long cold prime must not hold the worker tick for its whole prompt.

The incident (memra#521; darklanes ops 2026-09-05): one tenant's 135k-token cold prefill ran in
ONE worker tick (`tick_max_ms 92542`), nothing else on the box was served for 90-130 s, and every
queued request from three other tenants hit the queue deadline for 11.5 h while every monitor
stayed green. The mechanism that ends it on the walker routes (GDN MTP prime, DFlash, GLM plain and
spec) is the owned `PrimeWalker` behind `MEMRA_PRIME_YIELD`: the prime advances ONE frozen chunk per
tick, the worker returns to drain arrivals and give each admitted peer one bounded quantum, then
resumes. Both arms execute the same frozen range tape, so the bytes are the same; only the
interleaving changes.

Serving shape, one card, one boot per arm of the real `memra-server` on its spec route with the
concurrency demotion pinned off (`MEMRA_SPEC_GATE_LOW=64 HIGH=65`, so both arms route every request
the same way and the byte comparison measures the yield, not the route), greedy natural-text
`prompt` requests on `/v1/completions` (the tokenizer is calibrated per boot through
`usage.prompt_tokens`, so the token targets are met within 10%), streaming so first-token time is
what a client sees. Each prompt opens with a case number and a plant word of its own and closes
by asking for them back, so the greedy continuation is a function of the primed prompt; a request
that generates fewer than `--min-tokens` (16) tokens, or a cell whose long prompt and two cold peers
do not produce three distinct outputs, REFUSES the run because the byte clause would be vacuous.
  seed: a `--seed-tokens` (4,096) prompt completes cold, so its prefix entry exists (the cache hit).
  cell: at t=0 a `--long` token (default 131,072) cold prompt starts; at +2 s and every +3 s the
        peers start: two cold short prompts (`--peer` tokens) and the seeded prompt again (a cache
        hit). `MEMRA_MAX_SESSIONS=4` admits exactly the long prompt and its three peers.
Clauses, every one a verdict, on the `--yield-values` arms (default `0,1`; the gate is the same
whatever the binary's default is, because the value is set explicitly):
  V1 bytes:      every request's greedy completion text is identical across the arms (the one
                 numerical program law: a yield is not a program change).
  V2 peers:      on the yielding arm every peer's first token arrives within `--peer-ttft-bar`
                 seconds (default 8, twice the 9B's measured 4096-token chunk wall plus its own
                 work) and the peers' p95 first-token time is at most half the non-yielding arm's.
  V3 tick:       `/health` `worker.tick_max_ms` on the yielding arm is at most `--tick-bar-ms`
                 (default 6000) after the cell; the non-yielding arm's is reported beside it.
  V4 completion: every request finishes (`finish_reason` present) on both arms; the long prime's
                 first-token penalty on the yielding arm is reported, never asserted.
  V5 engaged:    the yielding boot's log carries the `[prime-walk] ... supported=true yield_door=true`
                 line and at least one `[prime-yield]` line; a quiet arm is not a passing arm
                 (darklanes#630 compared OFF against OFF for a boot).
Verdict line:
  PRIME-FAIRNESS: long=L peers=N bytes=yes|NO peer_p95=OFF/ON s peer_max=OFF/ON s tick_max=OFF/ON ms
  long_ttft=OFF/ON s yields=Y -> PASS|FAIL
Exit 0 = PASS; 1 = a clause failed; 2 = REFUSED (lock, port, boot, an unserved seed).

Service shape (`--shape service`, the saved-prime service interval of memra#521): the A/B of two
binaries (or two settings) on a decode-heavy cell. `--arm LABEL=BIN[:YIELD]` twice replaces `--bin`
and `--yield-values`; the first arm is the base, the second the candidate, YIELD `unset` leaves
`MEMRA_PRIME_YIELD` out of the environment (the shipped default). The cell: `--decoders` (2) cold
`--peer`-token prompts with `--decoder-tokens` (4096) outputs start at t=0; at +`--lead` (3) s the
`--long` cold prime starts; at +`--second-at` (10) s a second `--second-long` (32768) cold prime; at
+2 s more one cold `--peer`-token peer with `--max-tokens` outputs. `MEMRA_MAX_SESSIONS` admits all
of them. Every streamed event is timestamped. Reported per arm, p50/p95/p99 over all boots: the
decoders' event gaps (ITL) and token rate inside each prime's window (prime start to its first
event, cut at the decoder's own end), the decoders' TPOT, every request's TTFT and E2E. Clauses,
declared before the cell runs:
  S1 bytes:      every request's greedy text identical across every boot of both arms.
  S2 completion: every request finishes on every boot.
  S3 engaged:    every boot of both arms logs `[prime-walk] supported=true yield_door=true` and
                 at least one `[prime-yield]` (skipped for an arm run with YIELD `0`).
  S4 service:    the candidate's median decoder token rate inside the long prime's window is at
                 least `--rate-bar` (2.0) times the base's.
  S5 cost:       each prime's median TTFT on the candidate is at most `--cost-bar` (2.0) times the
                 base's plus `--cost-slack` (5) s (the structural bound: C-S plus one tick per chunk).
  S6 window:     every decoder is still decoding when the long prime starts on every boot, or the
                 service clause would measure nothing (REFUSED otherwise).
Raw per boot: cell JSON with every request's event times, server log, `nvidia-smi` 250 ms telemetry
CSV. The receipt carries the sha256 of every binary and of the model file, and both boot orders.
Verdict line:
  PRIME-SERVICE: base=LABEL cand=LABEL reps=R bytes=yes|NO rate=B/C tok/s long_ttft=B/C s
  second_ttft=B/C s itl_p99=B/C ms -> PASS|FAIL

usage: prime-fairness-gate.py --model GGUF --bin memra-server --out NEW_DIR [--port N]
           [--long 131072] [--peer 2048] [--seed-tokens 4096] [--max-tokens 32] [--min-tokens 16]
           [--yield-values 0,1] [--reps 1] [--peer-ttft-bar 8] [--tick-bar-ms 6000]
       prime-fairness-gate.py --shape service --model GGUF --arm base=BIN_A --arm cand=BIN_B
           --out NEW_DIR [--reps 6] [--decoders 2] [--decoder-tokens 4096] [--second-long 32768]
           [--rate-bar 2.0] [--cost-bar 2.0] [--cost-slack 5] [--telemetry-gpu UUID]
Lock: the canonical rig lock (`MEMRA_GPU_LOCK`, else `/tmp/memra-gpu.lock` or `/tmp/memra-5090.lock`)
held for the whole gate; under local-ci (`MEMRA_CI_LOCK_HELD=1`) the run's own hold is honored.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")


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


SENTENCE = (
    "The community garden has morning sunlight, two raised beds, a nearby water tap, compost, "
    "labels, gloves, and volunteers who can help each morning. "
)


def text_for(target_tokens: int, tokens_per_sentence: float, seed: int) -> str:
    """Natural repeated text of about `target_tokens` tokens (calibrated per boot). The lead names
    a case number and a plant word that appear nowhere else, and the tail asks for exactly those
    two facts back, so the greedy continuation is a function of the primed prompt (the state this
    door reorders) and differs between requests; a shared task line at the end made every request
    echo the same 32 tokens (review round 2), which compared bytes without comparing the prime."""
    words = ["garden", "orchard", "nursery", "greenhouse", "allotment", "vineyard"]
    word = words[seed % len(words)]
    lead = f"Case {seed}: reference notes for planning a community {word}. "
    body = SENTENCE.replace("garden", word)
    n = max(1, int(target_tokens / tokens_per_sentence))
    tail = (
        f"\n\nQuestion for case {seed}: state the case number and the plant word from the first "
        f"line of these notes, then give one sentence of advice for the {word}.\nAnswer: Case"
    )
    return lead + body * n + tail


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[max(0, int(round(q * len(xs) + 0.5)) - 1)] if xs else float("nan")


class Server:
    def __init__(self, binary: str, model: str, port: int, log: Path, yield_value: str, sessions: int = 4):
        self.binary, self.model, self.port, self.log = binary, model, port, log
        self.yield_value, self.sessions = yield_value, sessions
        self.proc: subprocess.Popen | None = None

    def boot(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        env.update(
            {
                "MEMRA_COMPAT": "openai",
                "MEMRA_MODELS": f"gate={self.model}",
                "MEMRA_ADDR": f"127.0.0.1:{self.port}",
                "MEMRA_MAX_SESSIONS": str(self.sessions),
                "MEMRA_TIMEOUT_MS_MAX": "600000",
                "MEMRA_TICK_TRACE": "1",
                # Pin the peers' route. The spec gate demotes to plain decode by concurrency
                # (LOW=2 HIGH=4 by default), and the two arms admit the peers at different
                # active counts, so without the pin a peer takes K=3 on one arm and K=0 on the
                # other and the byte comparison measures the route, not the yield. Both arms run
                # every request on the spec route with these bounds.
                "MEMRA_SPEC_GATE_LOW": "64",
                "MEMRA_SPEC_GATE_HIGH": "65",
            }
        )
        if self.yield_value != "unset":
            env["MEMRA_PRIME_YIELD"] = self.yield_value
        self.logf = open(self.log, "w")
        self.proc = subprocess.Popen([self.binary], env=env, stdout=self.logf, stderr=subprocess.STDOUT)
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                self.stop()
                refuse(f"memra-server exited {code} during load; see {self.log}")
            if self.health() is not None:
                return
            time.sleep(0.5)
        # A refusal must not leave a loaded server on the card (and on the port) for the next
        # stage of the battery: stop the child first, then refuse.
        self.stop()
        refuse("memra-server did not become healthy in 900 s")

    def get(self, path: str):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read())
            except Exception:
                return e.code, None
        except Exception:
            return None, None

    def health(self):
        st, body = self.get("/health")
        return body if st == 200 else None

    def count_tokens(self, text: str) -> int:
        """`usage.prompt_tokens` of a one-token completion: the server's own tokenizer count."""
        body = {"model": "gate", "prompt": text, "max_tokens": 1, "temperature": 0, "stream": False}
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return int(json.load(r)["usage"]["prompt_tokens"])

    def complete(self, prompt: str, salt: str, max_tokens: int) -> dict:
        body = {
            "model": "gate",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache_salt": salt,
            "timeout_ms": 600000,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        # `ttft_s` is the first choice event (a token or an immediate finish): what a client sees
        # as the response starting. `first_token_s` is the first non-empty piece, `None` when the
        # model's first token is EOS (synthetic id prompts do that).
        out = {
            "status": None,
            "text": "",
            "finish": None,
            "error": None,
            "ttft_s": None,
            "first_token_s": None,
            "total_s": None,
            "usage": None,
            "start": None,
            "events": [],
        }
        t0 = time.monotonic()
        out["start"] = t0
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                out["status"] = r.status
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        out["error"] = obj["error"]
                        continue
                    if obj.get("usage"):
                        out["usage"] = obj["usage"]
                    for ch in obj.get("choices", []):
                        if out["ttft_s"] is None:
                            out["ttft_s"] = time.monotonic() - t0
                        piece = ch.get("text")
                        if piece is None:
                            piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            out["events"].append(time.monotonic() - t0)
                            if out["first_token_s"] is None:
                                out["first_token_s"] = time.monotonic() - t0
                            out["text"] += piece
                        if ch.get("finish_reason"):
                            out["finish"] = ch["finish_reason"]
        except urllib.error.HTTPError as e:
            out["status"] = e.code
            try:
                out["error"] = json.loads(e.read()).get("error")
            except Exception:
                pass
        except Exception as e:
            out["error"] = {"transport": repr(e)}
        out["total_s"] = time.monotonic() - t0
        out["sha"] = hashlib.sha256(out["text"].encode()).hexdigest()[:16]
        return out

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.logf.close()


def prompts_for(srv: Server, a) -> dict:
    """Calibrate the tokenizer once per boot, then build the four prompts to their token targets."""
    probe = SENTENCE * 64
    per_sentence = (srv.count_tokens(probe) - 1) / 64.0
    if per_sentence <= 0:
        refuse("tokenizer calibration returned no tokens")
    prompts = {
        "seed": text_for(a.seed_tokens, per_sentence, 5210),
        "long": text_for(a.long, per_sentence, 521),
        "peer-a": text_for(a.peer, per_sentence, 5211),
        "peer-b": text_for(a.peer, per_sentence, 5212),
    }
    counts = {k: srv.count_tokens(v) for k, v in prompts.items()}
    if counts["long"] < a.long * 0.9:
        refuse(f"long prompt calibrated to {counts['long']} tokens, below 90% of {a.long}")
    return {"prompts": prompts, "tokens": counts, "tokens_per_sentence": per_sentence}


def run_cell(srv: Server, a, prompts: dict) -> dict:
    """Seed, then the long prime with its staggered peers; every request's timing and bytes."""
    seed = srv.complete(prompts["seed"], "seed", a.max_tokens)
    if seed["finish"] is None or seed["error"]:
        refuse(f"seed request did not finish: {json.dumps(seed)[:300]}")
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def go(name, prompt, salt):
        r = srv.complete(prompt, salt, a.max_tokens)
        with lock:
            results[name] = r

    t_cell = time.monotonic()
    threads = [threading.Thread(target=go, args=("long", prompts["long"], "long"))]
    threads[0].start()
    peers = [
        ("peer-cold-a", prompts["peer-a"], "peer-a"),
        ("peer-cold-b", prompts["peer-b"], "peer-b"),
        ("peer-hit", prompts["seed"], "seed"),
    ]
    for k, (name, prompt, salt) in enumerate(peers):
        time.sleep(2.0 if k == 0 else 3.0)
        t = threading.Thread(target=go, args=(name, prompt, salt))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    time.sleep(1.0)
    health = srv.health() or {}
    return {
        "seed": seed,
        "cell_s": time.monotonic() - t_cell,
        "requests": results,
        "tick_max_ms": (health.get("worker") or {}).get("tick_max_ms"),
        "generation": (health.get("worker") or {}).get("generation"),
    }


def run_service_cell(srv: Server, a, prompts: dict) -> dict:
    """Decoders first, then the long prime, a second prime and a cold peer; every event timed."""
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def go(name, prompt, salt, max_tokens):
        r = srv.complete(prompt, salt, max_tokens)
        with lock:
            results[name] = r

    plan = [(f"decoder-{k}", prompts[f"decoder-{k}"], f"decoder-{k}", a.decoder_tokens, 0.0) for k in range(a.decoders)]
    plan += [
        ("long", prompts["long"], "long", a.max_tokens, a.lead),
        ("second", prompts["second"], "second", a.max_tokens, a.lead + a.second_at),
        ("peer", prompts["peer"], "peer", a.max_tokens, a.lead + a.second_at + 2.0),
    ]
    t_cell = time.monotonic()
    threads = []
    for name, prompt, salt, max_tokens, at in plan:
        delay = t_cell + at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        t = threading.Thread(target=go, args=(name, prompt, salt, max_tokens))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    time.sleep(1.0)
    health = srv.health() or {}
    for r in results.values():
        r["start"] -= t_cell
    return {
        "cell_s": time.monotonic() - t_cell,
        "requests": results,
        "tick_max_ms": (health.get("worker") or {}).get("tick_max_ms"),
        "generation": (health.get("worker") or {}).get("generation"),
    }


def service_prompts(srv: Server, a) -> dict:
    probe = SENTENCE * 64
    per_sentence = (srv.count_tokens(probe) - 1) / 64.0
    if per_sentence <= 0:
        refuse("tokenizer calibration returned no tokens")
    prompts = {f"decoder-{k}": text_for(a.peer, per_sentence, 5230 + k) for k in range(a.decoders)}
    prompts.update(
        long=text_for(a.long, per_sentence, 521),
        second=text_for(a.second_long, per_sentence, 5220),
        peer=text_for(a.peer, per_sentence, 5221),
    )
    counts = {k: srv.count_tokens(v) for k, v in prompts.items()}
    if counts["long"] < a.long * 0.9 or counts["second"] < a.second_long * 0.9:
        refuse(f"prime prompts calibrated to {counts['long']}/{counts['second']} tokens, below 90% of target")
    return {"prompts": prompts, "tokens": counts, "tokens_per_sentence": per_sentence}


def decoder_window(dec: dict, prime: dict) -> dict:
    """The decoder's service inside one prime's window: from the prime's start to its first event,
    cut at the decoder's own last event. Times are cell-relative."""
    lo = prime["start"]
    hi = prime["start"] + (prime["ttft_s"] if prime["ttft_s"] is not None else prime["total_s"])
    ev = [dec["start"] + t for t in dec["events"]]
    hi = min(hi, ev[-1]) if ev else lo
    inside = [t for t in ev if lo <= t <= hi]
    gaps = [(b - a) * 1000.0 for a, b in zip(inside, inside[1:])]
    span = hi - lo
    return {"span_s": span, "events": len(inside), "rate": len(inside) / span if span > 0 else 0.0, "gaps_ms": gaps}


class Telemetry:
    """`nvidia-smi` at 250 ms for one boot, raw CSV; absent tool or GPU is recorded, never fatal."""

    def __init__(self, path: Path, gpu: str | None):
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks_event_reasons.active",
            "--format=csv",
            "-lms",
            "250",
            "-f",
            str(path),
        ]
        if gpu:
            cmd[1:1] = ["-i", gpu]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            path.write_text(f"telemetry unavailable: {e!r}\n")
            self.proc = None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def service_main(a, out: Path, arms: list[tuple[str, str, str]]) -> int:
    names = [f"decoder-{k}" for k in range(a.decoders)] + ["long", "second", "peer"]
    hashes = {"model": sha256_file(a.model), **{f"bin:{label}": sha256_file(b) for label, b, _ in arms}}
    (out / "hashes.json").write_text(json.dumps(hashes, indent=2))
    lock_fd = take_lock()
    runs: list[dict] = []
    try:
        for rep in range(a.reps):
            order = arms if rep % 2 == 0 else list(reversed(arms))
            for label, binary, yv in order:
                tag = f"rep{rep}-{label}"
                srv = Server(binary, a.model, a.port, out / f"{tag}-server.log", yv, sessions=len(names))
                tel = Telemetry(out / f"{tag}-telemetry.csv", a.telemetry_gpu)
                try:
                    srv.boot()
                    prompts = service_prompts(srv, a)
                    cell = run_service_cell(srv, a, prompts["prompts"])
                    cell["prompt_tokens"] = prompts["tokens"]
                finally:
                    srv.stop()
                    tel.stop()
                cell.update(arm=label, bin=binary, yield_value=yv, rep=rep, order=[x[0] for x in order])
                cell["log"] = log_facts(out / f"{tag}-server.log")
                (out / f"{tag}-cell.json").write_text(json.dumps(cell, indent=2))
                reqs = cell["requests"]
                short = [n for n in names if (reqs[n].get("usage") or {}).get("completion_tokens", 0) < a.min_tokens]
                if short:
                    refuse(f"{tag}: {short} generated fewer than {a.min_tokens} tokens; the byte clause would be vacuous")
                if len({reqs[n]["sha"] for n in names}) != len(names):
                    refuse(f"{tag}: the {len(names)} prompts did not produce distinct outputs")
                late = [
                    n for n in names[: a.decoders]
                    if not reqs[n]["events"] or reqs[n]["start"] + reqs[n]["events"][-1] <= reqs["long"]["start"]
                ]
                if late:
                    refuse(f"{tag}: {late} stopped decoding before the long prime started (S6)")
                win = decoder_window(reqs[names[0]], reqs["long"])
                print(
                    f"  {tag}: long ttft={reqs['long']['ttft_s']:.2f}s second ttft={reqs['second']['ttft_s']:.2f}s "
                    f"peer ttft={reqs['peer']['ttft_s']:.2f}s decoder rate in long window={win['rate']:.2f} ev/s "
                    f"tick_max_ms={cell['tick_max_ms']} yields={cell['log']['yield_lines']}",
                    flush=True,
                )
                runs.append(cell)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)

    (base, _, base_y), (cand, _, cand_y) = arms
    by = {label: [c for c in runs if c["arm"] == label] for label, _, _ in arms}
    shas = {n: sorted({c["requests"][n]["sha"] for c in runs}) for n in names}
    bytes_ok = all(len(v) == 1 for v in shas.values())
    complete_ok = all(c["requests"][n]["finish"] is not None and not c["requests"][n]["error"] for c in runs for n in names)
    engaged_ok = all(
        c["log"]["supported"] and c["log"]["yield_door_on"] and c["log"]["yield_lines"] > 0
        for c in runs if c["yield_value"] != "0"
    ) and not any(c["log"]["worker_panic"] for c in runs)

    def stats(xs: list[float]) -> dict:
        return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "p99": pct(xs, 0.99)}

    summary: dict[str, dict] = {}
    for label in by:
        cells = by[label]
        arm: dict[str, dict] = {}
        for prime in ("long", "second"):
            wins = [decoder_window(c["requests"][d], c["requests"][prime]) for c in cells for d in names[: a.decoders]]
            arm[f"decoder_rate_in_{prime}"] = stats([w["rate"] for w in wins])
            arm[f"decoder_itl_ms_in_{prime}"] = stats([g for w in wins for g in w["gaps_ms"]])
        itl = []
        tpot = []
        for c in cells:
            for d in names[: a.decoders]:
                r = c["requests"][d]
                itl += [(b - x) * 1000.0 for x, b in zip(r["events"], r["events"][1:])]
                toks = (r.get("usage") or {}).get("completion_tokens", 0)
                if toks > 1 and r["ttft_s"] is not None:
                    tpot.append((r["total_s"] - r["ttft_s"]) * 1000.0 / (toks - 1))
        arm["decoder_itl_ms"] = stats(itl)
        arm["decoder_tpot_ms"] = stats(tpot)
        for n in names:
            arm[f"{n}_ttft_s"] = stats([c["requests"][n]["ttft_s"] or float("inf") for c in cells])
            arm[f"{n}_e2e_s"] = stats([c["requests"][n]["total_s"] for c in cells])
        arm["tick_max_ms"] = stats([float(c["tick_max_ms"] or 0) for c in cells])
        arm["yield_lines"] = sum(c["log"]["yield_lines"] for c in cells)
        summary[label] = arm

    rate_b = summary[base]["decoder_rate_in_long"]["p50"]
    rate_c = summary[cand]["decoder_rate_in_long"]["p50"]
    service_ok = rate_c >= a.rate_bar * rate_b
    cost = {}
    for prime in ("long", "second"):
        tb, tc = summary[base][f"{prime}_ttft_s"]["p50"], summary[cand][f"{prime}_ttft_s"]["p50"]
        cost[prime] = {"base": tb, "cand": tc, "bar": a.cost_bar * tb + a.cost_slack, "ok": tc <= a.cost_bar * tb + a.cost_slack}
    cost_ok = all(v["ok"] for v in cost.values())
    ok = bytes_ok and complete_ok and engaged_ok and service_ok and cost_ok
    receipt = {
        "shape": "service",
        "model": a.model,
        "arms": [{"label": l, "bin": b, "yield": y} for l, b, y in arms],
        "hashes": hashes,
        "reps": a.reps,
        "orders": [c["order"] for c in runs[:: len(arms)]],
        "cell": {
            "long": a.long, "second_long": a.second_long, "peer": a.peer, "decoders": a.decoders,
            "decoder_tokens": a.decoder_tokens, "max_tokens": a.max_tokens, "lead_s": a.lead,
            "second_at_s": a.second_at, "sessions": len(names),
        },
        "declared": {"rate_bar": a.rate_bar, "cost_bar": a.cost_bar, "cost_slack_s": a.cost_slack, "min_tokens": a.min_tokens},
        "prompt_tokens": runs[0].get("prompt_tokens"),
        "shas": shas,
        "summary": summary,
        "cost": cost,
        "verdicts": {"bytes": bytes_ok, "completion": complete_ok, "engaged": engaged_ok, "service": service_ok, "cost": cost_ok},
        "verdict": "PASS" if ok else "FAIL",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(
        f"PRIME-SERVICE: base={base} cand={cand} reps={a.reps} bytes={'yes' if bytes_ok else 'NO'} "
        f"rate={rate_b:.2f}/{rate_c:.2f} ev/s long_ttft={cost['long']['base']:.2f}/{cost['long']['cand']:.2f}s "
        f"second_ttft={cost['second']['base']:.2f}/{cost['second']['cand']:.2f}s "
        f"itl_p99={summary[base]['decoder_itl_ms']['p99']:.0f}/{summary[cand]['decoder_itl_ms']['p99']:.0f}ms "
        f"engaged={'yes' if engaged_ok else 'NO'} -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


def log_facts(path: Path) -> dict:
    text = path.read_text(errors="replace")
    walk = [l for l in text.splitlines() if "[prime-walk]" in l]
    yields = [l for l in text.splitlines() if "[prime-yield]" in l]
    chunks = [l for l in text.splitlines() if "[prime-chunk]" in l]
    return {
        "prime_walk": walk[:2],
        "supported": any("supported=true" in l for l in walk),
        "yield_door_on": any("yield_door=true" in l for l in walk),
        "yield_lines": len(yields),
        "chunk_lines": len(chunks),
        "worker_panic": any("[worker] PANIC" in l for l in text.splitlines()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18521)
    ap.add_argument("--long", type=int, default=131072)
    ap.add_argument("--peer", type=int, default=2048)
    ap.add_argument("--seed-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--min-tokens", type=int, default=16, help="REFUSE when a request generates fewer (the byte clause would be vacuous)")
    ap.add_argument("--yield-values", default="0,1")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--peer-ttft-bar", type=float, default=8.0)
    ap.add_argument("--tick-bar-ms", type=int, default=6000)
    ap.add_argument("--shape", choices=("incident", "service"), default="incident")
    ap.add_argument("--arm", action="append", default=[], help="LABEL=BIN[:YIELD], twice; YIELD unset omits MEMRA_PRIME_YIELD")
    ap.add_argument("--decoders", type=int, default=2)
    ap.add_argument("--decoder-tokens", type=int, default=4096)
    ap.add_argument("--second-long", type=int, default=32768)
    ap.add_argument("--lead", type=float, default=3.0)
    ap.add_argument("--second-at", type=float, default=10.0)
    ap.add_argument("--rate-bar", type=float, default=2.0)
    ap.add_argument("--cost-bar", type=float, default=2.0)
    ap.add_argument("--cost-slack", type=float, default=5.0)
    ap.add_argument("--telemetry-gpu", help="nvidia-smi -i value for the 250 ms telemetry (default: every GPU)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        refuse(f"{out} exists; --out must be a NEW directory")
    arms_spec: list[tuple[str, str, str]] = []
    for spec in a.arm:
        label, sep, rest = spec.partition("=")
        binary, _, yv = rest.partition(":")
        if not sep or not label or not binary or not os.path.isfile(binary):
            refuse(f"--arm {spec!r}: expected LABEL=BIN[:YIELD] with an existing BIN")
        arms_spec.append((label, binary, yv or "unset"))
    if a.arm and (len(arms_spec) != 2 or arms_spec[0][0] == arms_spec[1][0]):
        refuse("--arm needs exactly two arms with distinct labels")
    if a.shape == "service" and not a.arm:
        refuse("--shape service compares two arms; pass --arm twice")
    if a.shape == "incident" and a.arm:
        refuse("the incident shape compares --yield-values on one --bin")
    out.mkdir(parents=True)
    if not port_free(a.port):
        refuse(f"port {a.port} busy")
    if a.shape == "service":
        return service_main(a, out, arms_spec)
    if not a.bin:
        refuse("--bin is required for the incident shape")
    lock_fd = take_lock()
    arms = a.yield_values.split(",")
    if len(arms) != 2:
        refuse("--yield-values needs exactly two arms, e.g. 0,1")
    runs: list[dict] = []
    calibration = None
    try:
        for rep in range(a.reps):
            order = arms if rep % 2 == 0 else list(reversed(arms))
            for arm in order:
                tag = f"rep{rep}-yield{arm}"
                srv = Server(a.bin, a.model, a.port, out / f"{tag}-server.log", arm)
                try:
                    srv.boot()
                    prompts = prompts_for(srv, a)
                    if calibration is None:
                        calibration = {k: v for k, v in prompts.items() if k != "prompts"}
                        (out / "calibration.json").write_text(json.dumps(calibration, indent=2))
                    cell = run_cell(srv, a, prompts["prompts"])
                    cell["prompt_tokens"] = prompts["tokens"]
                finally:
                    srv.stop()
                short = [
                    (n, r) for n, r in cell["requests"].items()
                    if (r.get("usage") or {}).get("completion_tokens", len(r["text"]) // 4) < a.min_tokens
                ]
                if short:
                    (out / f"{tag}-cell.json").write_text(json.dumps(cell, indent=2))
                    refuse(
                        f"{tag}: {[n for n, _ in short]} generated fewer than {a.min_tokens} tokens; "
                        "the byte clause would be vacuous"
                    )
                # The four prompts differ (case number, plant word, length), so their greedy
                # continuations must differ; identical outputs mean the model collapsed onto a
                # prompt-independent attractor and the byte clause would not see the prime.
                distinct = {cell["requests"][n]["sha"] for n in ("long", "peer-cold-a", "peer-cold-b")}
                if len(distinct) != 3:
                    (out / f"{tag}-cell.json").write_text(json.dumps(cell, indent=2))
                    refuse(
                        f"{tag}: the long prompt and the two cold peers produced only {len(distinct)} "
                        "distinct output(s); the byte clause would not depend on the primed prompt"
                    )
                cell.update(arm=arm, rep=rep, log=log_facts(out / f"{tag}-server.log"))
                (out / f"{tag}-cell.json").write_text(json.dumps(cell, indent=2))
                runs.append(cell)
                peers = [cell["requests"][n] for n in ("peer-cold-a", "peer-cold-b", "peer-hit")]
                print(
                    f"  {tag}: long ttft={cell['requests']['long']['ttft_s']:.2f}s total={cell['requests']['long']['total_s']:.2f}s "
                    f"peers ttft={[round(p['ttft_s'] or -1, 2) for p in peers]} tick_max_ms={cell['tick_max_ms']} "
                    f"yields={cell['log']['yield_lines']} chunks={cell['log']['chunk_lines']}",
                    flush=True,
                )
    finally:
        if lock_fd is not None:
            os.close(lock_fd)

    off, on = arms[0], arms[1]
    by_arm = {arm: [c for c in runs if c["arm"] == arm] for arm in arms}
    names = ["long", "peer-cold-a", "peer-cold-b", "peer-hit"]
    # V1 bytes: every request identical across every run
    shas = {n: {c["requests"][n]["sha"] for c in runs} for n in names}
    bytes_ok = all(len(s) == 1 for s in shas.values())
    # V4 completion
    complete_ok = all(c["requests"][n]["finish"] is not None and not c["requests"][n]["error"] for c in runs for n in names)

    def peer_ttfts(arm):
        return [c["requests"][n]["ttft_s"] or float("inf") for c in by_arm[arm] for n in names[1:]]

    p95_off, p95_on = pct(peer_ttfts(off), 0.95), pct(peer_ttfts(on), 0.95)
    max_off, max_on = max(peer_ttfts(off)), max(peer_ttfts(on))
    peers_ok = max_on <= a.peer_ttft_bar and p95_on <= p95_off / 2.0
    tick_off = max(c["tick_max_ms"] or 0 for c in by_arm[off])
    tick_on = max(c["tick_max_ms"] or 0 for c in by_arm[on])
    tick_ok = tick_on <= a.tick_bar_ms
    long_off = pct([c["requests"]["long"]["ttft_s"] or float("inf") for c in by_arm[off]], 0.5)
    long_on = pct([c["requests"]["long"]["ttft_s"] or float("inf") for c in by_arm[on]], 0.5)
    yields = sum(c["log"]["yield_lines"] for c in by_arm[on])
    engaged_ok = all(c["log"]["supported"] and c["log"]["yield_door_on"] and c["log"]["yield_lines"] > 0 for c in by_arm[on]) and not any(
        c["log"]["worker_panic"] for c in runs
    )
    ok = bytes_ok and peers_ok and tick_ok and complete_ok and engaged_ok
    receipt = {
        "model": a.model,
        "bin": a.bin,
        "long": a.long,
        "peer": a.peer,
        "seed_tokens": a.seed_tokens,
        "prompt_tokens": runs[0].get("prompt_tokens"),
        "min_tokens": a.min_tokens,
        "max_tokens": a.max_tokens,
        "arms": arms,
        "reps": a.reps,
        "shas": {n: sorted(s) for n, s in shas.items()},
        "peer_p95_s": {off: p95_off, on: p95_on},
        "peer_max_s": {off: max_off, on: max_on},
        "tick_max_ms": {off: tick_off, on: tick_on},
        "long_ttft_s": {off: long_off, on: long_on},
        "yields_on": yields,
        "verdicts": {"bytes": bytes_ok, "peers": peers_ok, "tick": tick_ok, "completion": complete_ok, "engaged": engaged_ok},
        "verdict": "PASS" if ok else "FAIL",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(
        f"PRIME-FAIRNESS: long={a.long} peers=3x{a.reps} bytes={'yes' if bytes_ok else 'NO'} "
        f"peer_p95={p95_off:.2f}/{p95_on:.2f}s peer_max={max_off:.2f}/{max_on:.2f}s "
        f"tick_max={tick_off}/{tick_on}ms long_ttft={long_off:.2f}/{long_on:.2f}s yields={yields} "
        f"engaged={'yes' if engaged_ok else 'NO'} -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
