#!/usr/bin/env python3
"""request-fault-gate.py: a per-request panic retires that request typed and leaves its peers whole.

The defect (memra#525): the GPU worker runs every session on one thread inside one `catch_unwind`,
so a Rust panic in ONE request's step (a bad index, an unwrap on a request-shaped edge) unwound the
whole scheduler loop: every in-flight session was dropped, every peer stream truncated after a 200,
health flipped dead, and the supervisor respawned the worker (or exited 70) for a fault that was
never the card's. Since memra#525 each per-session step, spec step, prefill call and batched decode
runs under `request_fault_guard` (worker.rs): a panic whose payload quotes no driver error and after
which the CUDA context still synchronizes, allocates and reads back is a REQUEST fault (one `[fault]`
line, `request_faults_total` += 1, the request ends with `code: worker_fault`); a driver-looking
payload or a failed probe stays a WORKER fault and takes the old ladder.

Serving shape, one card, one boot of the real `memra-server` with the fault-injection door
`MEMRA_FAULT_INJECT_CACHE_SALT=<salt>` (FLAGS.md): a request whose `cache_salt` equals the salt
panics inside its own guarded `fault-inject` step at the top of the tick. Two rounds against the same server, greedy, plain path
(`MEMRA_SERVE_SPEC=0`), every request a `stream: true` chat completion:
  control: `--peers` concurrent streams (prompts P1..Pn), texts kept.
  fault:   the same `--peers` streams plus one more carrying the salt, all started together.
Clauses, every one a verdict:
  V1 typed:     the salted stream ends with an error object whose `code` is `worker_fault` (and no
                `finish_reason` chunk), i.e. the client learns the request failed instead of reading a
                truncated 200 as a complete answer.
  V2 peers:     every peer in the fault round finishes with a `finish_reason` and its text equals the
                control round's text for the same prompt, byte for byte.
  V3 counters:  `/metrics` `request_faults_total` == 1 and `worker_respawns_total` == 0 after the
                fault round; `/health` is 200 with `worker.generation` == 0 (no respawn happened).
  V4 log:       the server log has exactly one `[fault] request=` line and it names the salted
                request's site and panic text; no `[worker] PANIC` line.
Verdict line:
  REQUEST-FAULT: peers=N typed=yes|NO identical=K/N faults=F respawns=R generation=G -> PASS|FAIL
Exit 0 = PASS; 1 = a clause failed; 2 = REFUSED (lock, port, an unserved control request).

usage: request-fault-gate.py --model GGUF --bin memra-server --out NEW_DIR [--port N] [--peers 3]
           [--max-tokens 48]
Lock: the canonical rig lock (`/tmp/memra-gpu.lock` or `/tmp/memra-5090.lock`, or `MEMRA_GPU_LOCK`)
held for the whole cell; under local-ci (`MEMRA_CI_LOCK_HELD=1`) the run's own hold is honored.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

LOCKS = ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock")
SALT = "fault-inject-525"
PROMPTS = [
    "List three facts about the Moon in one short paragraph.",
    "Explain what a hash table is to a beginner in a few sentences.",
    "Write four lines about autumn rain, plain prose, no title.",
    "Describe how a bicycle gear works in two sentences.",
    "Name two uses of copper and why it suits each.",
    "Summarize the rules of tic-tac-toe in three sentences.",
]


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", flush=True)
    sys.exit(2)


def take_lock():
    """Hold the rig lock for the whole cell. Inside local-ci the run already holds the canonical
    lock (MEMRA_CI_LOCK_HELD=1) and points inner gates at MEMRA_GPU_LOCK, its non-contending inner
    file; standalone runs take the canonical lock (both names exist: pick the one present)."""
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


class Server:
    def __init__(self, binary: str, model: str, port: int, log: Path, salt: str | None):
        self.binary, self.model, self.port, self.log, self.salt = binary, model, port, log, salt
        self.proc: subprocess.Popen | None = None

    def boot(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        env.update(
            {
                "MEMRA_COMPAT": "openai",
                "MEMRA_MODELS": f"gate={self.model}",
                "MEMRA_ADDR": f"127.0.0.1:{self.port}",
                "MEMRA_CTX": "8192",
                "MEMRA_MAX_SESSIONS": "8",
                "MEMRA_SERVE_SPEC": "0",
                "MEMRA_TIMEOUT_MS_MAX": "240000",
            }
        )
        if self.salt:
            env["MEMRA_FAULT_INJECT_CACHE_SALT"] = self.salt
        self.logf = open(self.log, "w")
        self.proc = subprocess.Popen([self.binary], env=env, stdout=self.logf, stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                refuse(f"memra-server exited {self.proc.returncode} during load; see {self.log}")
            if self.health() is not None:
                return
            time.sleep(0.5)
        refuse("memra-server did not become healthy in 600 s")

    def get(self, path: str):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
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

    def stream_chat(self, prompt: str, salt: str | None, max_tokens: int) -> dict:
        body = {
            "model": "gate",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if salt:
            body["cache_salt"] = salt
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        out = {"status": None, "text": "", "finish": None, "error": None, "chunks": 0, "done": False}
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                out["status"] = r.status
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        out["done"] = True
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        out["error"] = obj["error"]
                        continue
                    out["chunks"] += 1
                    for ch in obj.get("choices", []):
                        delta = ch.get("delta", {})
                        out["text"] += (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                        if ch.get("finish_reason"):
                            out["finish"] = ch["finish_reason"]
        except urllib.error.HTTPError as e:
            out["status"] = e.code
            try:
                out["error"] = json.loads(e.read()).get("error")
            except Exception:
                pass
        except Exception as e:  # connection dropped mid-stream: the truncated class
            out["error"] = {"transport": repr(e)}
        return out

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(45)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.logf.close()


def run_round(srv: Server, jobs: list[tuple[str, str | None]], max_tokens: int) -> list[dict]:
    results: list[dict | None] = [None] * len(jobs)

    def go(k: int) -> None:
        results[k] = srv.stream_chat(jobs[k][0], jobs[k][1], max_tokens)

    threads = [threading.Thread(target=go, args=(k,)) for k in range(len(jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18525)
    ap.add_argument("--peers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()
    # Reserve one additional prompt for the salted request below.
    if a.peers < 1 or a.peers >= len(PROMPTS):
        refuse(f"--peers must be 1..{len(PROMPTS) - 1}")
    out = Path(a.out)
    if out.exists():
        refuse(f"{out} exists; --out must be a NEW directory")
    out.mkdir(parents=True)
    if not port_free(a.port):
        refuse(f"port {a.port} busy")
    lock_fd = take_lock()

    srv = Server(a.bin, a.model, a.port, out / "server.log", SALT)
    srv.boot()
    try:
        peers = [(PROMPTS[k], None) for k in range(a.peers)]
        control = run_round(srv, peers, a.max_tokens)
        for k, r in enumerate(control):
            if r["finish"] is None or r["error"]:
                refuse(f"control peer {k} did not finish: {json.dumps(r)[:300]}")
        m0 = srv.get("/metrics")[1] or {}
        faults0 = int(m0.get("request_faults_total", -1))
        fault_round = run_round(srv, peers + [(PROMPTS[a.peers], SALT)], a.max_tokens)
        time.sleep(1.0)
        health = srv.health()
        m1 = srv.get("/metrics")[1] or {}
    finally:
        srv.stop()
        if lock_fd is not None:
            os.close(lock_fd)

    salted = fault_round[-1]
    typed = (
        salted["finish"] is None
        and salted["error"] is not None
        and salted["error"].get("code") == "worker_fault"
    )
    identical = 0
    per_peer = []
    for k in range(a.peers):
        c, f = control[k], fault_round[k]
        same = f["finish"] is not None and not f["error"] and f["text"] == c["text"]
        identical += int(same)
        per_peer.append(
            {
                "prompt": PROMPTS[k],
                "control_sha": hashlib.sha256(c["text"].encode()).hexdigest()[:16],
                "fault_sha": hashlib.sha256(f["text"].encode()).hexdigest()[:16],
                "fault_finish": f["finish"],
                "fault_error": f["error"],
                "identical": same,
            }
        )
    faults = int(m1.get("request_faults_total", -1))
    respawns = int(m1.get("worker_respawns_total", -1))
    generation = (health or {}).get("worker", {}).get("generation", -1) if health else -1
    log = (out / "server.log").read_text(errors="replace")
    fault_lines = [l for l in log.splitlines() if "[fault] request=" in l]
    worker_panic = [l for l in log.splitlines() if "[worker] PANIC" in l]
    v4 = (
        len(fault_lines) == 1
        and "site=fault-inject" in fault_lines[0]
        and "MEMRA_FAULT_INJECT_CACHE_SALT matched" in fault_lines[0]
        and not worker_panic
    )
    ok = (
        typed
        and identical == a.peers
        and faults0 == 0
        and faults == 1
        and respawns == 0
        and health is not None
        and generation == 0
        and v4
    )
    receipt = {
        "model": a.model,
        "bin": a.bin,
        "peers": a.peers,
        "max_tokens": a.max_tokens,
        "salted": salted,
        "per_peer": per_peer,
        "request_faults_total_before": faults0,
        "request_faults_total": faults,
        "worker_respawns_total": respawns,
        "health": health,
        "fault_lines": fault_lines,
        "worker_panic_lines": worker_panic,
        "verdict": "PASS" if ok else "FAIL",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    for k, p in enumerate(per_peer):
        print(f"  peer{k}: control={p['control_sha']} fault={p['fault_sha']} finish={p['fault_finish']} identical={'yes' if p['identical'] else 'NO'}")
    print(f"  salted: finish={salted['finish']} error={json.dumps(salted['error'])[:200]}")
    for l in fault_lines:
        print(f"  {l[:220]}")
    print(
        f"REQUEST-FAULT: peers={a.peers} typed={'yes' if typed else 'NO'} identical={identical}/{a.peers} "
        f"faults={faults} respawns={respawns} generation={generation} log_ok={'yes' if v4 else 'NO'} "
        f"-> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
