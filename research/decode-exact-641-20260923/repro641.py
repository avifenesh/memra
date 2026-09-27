#!/usr/bin/env python3
"""repro641.py: one boot of the memra#641 serving shape, or the peer-b solo control.

The cell is the prime-fairness gate as it stood at 5a9fd0414 (the version whose run1 non-yielding
arm produced the divergent peer-cold-b bytes): no spec-gate pin, so the placement default LOW=2
HIGH=4 applies and the peers demote to plain batched decode when four sessions are active.

  cell:  seed (4,096 ids, seed 5210, salt "seed") completes; then at t=0 the long prompt (131,072
         ids, seed 521, salt "long"); at +2 s peer-cold-a (2,048 ids, seed 5211, salt "peer-a");
         at +3 s more peer-cold-b (2,048 ids, seed 5212, salt "peer-b"); at +3 s more peer-hit (the
         seed ids, salt "seed"). Every request: max_tokens 32, temperature 0, stream, timeout
         600,000 ms, /v1/completions with prompt_ids.
  solo:  peer-cold-b alone on a fresh server (the control).

The caller holds /tmp/memra-5090.lock; this script takes no lock. One boot per invocation, so the
caller orders the arms. Writes <out>/<tag>-server.log and <out>/<tag>-cell.json and prints one line.
"""
import argparse
import hashlib
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# The reference bytes R: peer-cold-b solo, on the yielding arm, and in every probe of
# research/prime-fairness-default-20260922/raw/.
R_SHA16 = "06bfb5126effdd4c"


def ids_for(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 100_000) for _ in range(n)]


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


class Server:
    def __init__(self, binary, model, port, log: Path, yield_value):
        self.binary, self.model, self.port, self.log, self.yield_value = binary, model, port, log, yield_value
        self.proc = None

    def boot(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        env.update(
            {
                "MEMRA_COMPAT": "openai",
                "MEMRA_MODELS": f"gate={self.model}",
                "MEMRA_ADDR": f"127.0.0.1:{self.port}",
                "MEMRA_MAX_SESSIONS": "4",
                "MEMRA_TIMEOUT_MS_MAX": "600000",
                "MEMRA_TICK_TRACE": "1",
                "MEMRA_PRIME_YIELD": self.yield_value,
            }
        )
        # Diagnostic arms only (never part of the pre-registered cell): pass-through doors.
        for k in os.environ.get("REPRO641_EXTRA_ENV", "").split():
            name, _, val = k.partition("=")
            env[name] = val
        self.logf = open(self.log, "w")
        self.proc = subprocess.Popen([self.binary], env=env, stdout=self.logf, stderr=subprocess.STDOUT)
        print(f"server pid {self.proc.pid}", flush=True)
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.proc.poll() is not None:
                print(f"REFUSED: memra-server exited {self.proc.returncode} during load; see {self.log}")
                sys.exit(2)
            if self.health() is not None:
                return
            time.sleep(0.5)
        print("REFUSED: memra-server did not become healthy in 900 s")
        self.stop()
        sys.exit(2)

    def health(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=10) as r:
                return json.loads(r.read()) if r.status == 200 else None
        except Exception:
            return None

    def complete(self, ids, salt, max_tokens):
        body = {
            "model": "gate",
            "prompt_ids": ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "cache_salt": salt,
            "timeout_ms": 600000,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        out = {"status": None, "text": "", "finish": None, "error": None, "ttft_s": None, "total_s": None}
        t0 = time.monotonic()
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
                    for ch in obj.get("choices", []):
                        piece = ch.get("text")
                        if piece is None:
                            piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            if out["ttft_s"] is None:
                                out["ttft_s"] = time.monotonic() - t0
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

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(30)
        print(f"server pid {self.proc.pid} exit {self.proc.returncode}", flush=True)
        self.logf.close()


def run_cell(srv, max_tokens):
    seed_ids = ids_for(4096, 5210)
    seed = srv.complete(seed_ids, "seed", max_tokens)
    results, lock = {}, threading.Lock()

    def go(name, ids, salt):
        r = srv.complete(ids, salt, max_tokens)
        with lock:
            results[name] = r

    threads = [threading.Thread(target=go, args=("long", ids_for(131072, 521), "long"))]
    threads[0].start()
    peers = [("peer-cold-a", ids_for(2048, 5211), "peer-a"), ("peer-cold-b", ids_for(2048, 5212), "peer-b"), ("peer-hit", seed_ids, "seed")]
    for k, (name, ids, salt) in enumerate(peers):
        time.sleep(2.0 if k == 0 else 3.0)
        t = threading.Thread(target=go, args=(name, ids, salt))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return {"seed": seed, "requests": results}


def shape(log: Path):
    """The #641 prime shape: peer-b demoted to plain (K=0) and primed in a fresh B=3 concat batch
    of 1,024-row chunks, then a carried B=3 batch. Returns (on_shape, facts)."""
    text = log.read_text(errors="replace").splitlines()
    pb = [l for l in text if "[prime-batch]" in l]
    k_b = [l for l in text if '[spec-k]' in l and 'tenant="peer-b"' in l]
    fresh = [l for l in pb if re.search(r"B=3 tokens=3072 carried=0 partial=3\b", l)]
    carried = [l for l in pb if re.search(r"B=3 tokens=3072 carried=3 partial=1\b", l)]
    on = bool(k_b) and " K=0 " in k_b[0] and len(fresh) == 1 and len(carried) == 1
    return on, {"spec_k_peer_b": k_b[:1], "prime_batch": pb[:6], "panic": any("PANIC" in l for l in text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["cell", "solo"])
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--yield-value", default="0")
    ap.add_argument("--port", type=int, default=18641)
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{a.tag}-server.log"
    if log.exists():
        print(f"REFUSED: {log} exists")
        return 2
    if not port_free(a.port):
        print(f"REFUSED: port {a.port} busy")
        return 2
    srv = Server(a.bin, a.model, a.port, log, a.yield_value)
    srv.boot()
    try:
        if a.kind == "cell":
            cell = run_cell(srv, a.max_tokens)
        else:
            cell = {"requests": {"peer-cold-b": srv.complete(ids_for(2048, 5212), "peer-b", a.max_tokens)}}
    finally:
        srv.stop()
    on, facts = shape(log) if a.kind == "cell" else (None, {})
    b = cell["requests"]["peer-cold-b"]
    cell.update(kind=a.kind, tag=a.tag, yield_value=a.yield_value, on_shape=on, shape=facts, peer_b_matches_R=b["sha"] == R_SHA16,
                extra_env=os.environ.get("REPRO641_EXTRA_ENV", ""))
    (out / f"{a.tag}-cell.json").write_text(json.dumps(cell, indent=2))
    shas = {n: r["sha"] for n, r in cell["requests"].items()}
    print(f"REPRO641 {a.tag}: kind={a.kind} yield={a.yield_value} on_shape={on} peer-b sha={b['sha']} "
          f"{'== R' if b['sha'] == R_SHA16 else '!= R'} finish={b['finish']} text={b['text']!r} shas={shas}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
