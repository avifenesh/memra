#!/usr/bin/env python3
"""Owned-server GLM5 prime gates. Run on the assigned tune host under the GPU lock.

The recipe is a JSON map of environment values and a models.toml file. Credentials,
addresses and machine identity are supplied at execution, never built into this lane.
Each boot uses a random loopback port and its own ephemeral key and process group.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import time
import urllib.request
import uuid


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def gpu_empty():
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"
    ], text=True).strip()
    if apps:
        raise RuntimeError("compute applications present before launch/after cleanup")


class Boot:
    def __init__(self, args, arm, number):
        self.args = args
        self.arm = arm
        self.root = args.out / f"{number:02d}-arm{arm}"
        self.root.mkdir(parents=True, exist_ok=False)
        gpu_empty()
        self.key = secrets.token_hex(24)
        keys = self.root / "keys.toml"
        keys.write_text('[[keys]]\ntenant="glm5-prime-gate"\nsha256="' +
                        hashlib.sha256(self.key.encode()).hexdigest() + '"\n')
        keys.chmod(0o600)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = self.root / "server.log"
        env = json.loads(args.profile.read_text())
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError("profile must be a string environment map")
        env.update(MEMRA_ADDR=f"127.0.0.1:{port}", MEMRA_API_KEYS=str(keys),
                   MEMRA_MODEL_METADATA=str(args.metadata), MEMRA_PRIME_YIELD=str(arm),
                   MEMRA_PRIME_CHUNK=str(args.chunk), MEMRA_TICK_TRACE="1",
                   MEMRA_TTFT_TRACE="1")
        env.pop("MEMRA_REQUEST_LEDGER", None)
        if args.cell == "exactness":
            env.update(MEMRA_SPEC_GATE_LOW="64", MEMRA_SPEC_GATE_HIGH="128", MEMRA_SPEC_K="3")
        if env.get("MEMRA_SERVE_SPEC") != "1" or env.get("MEMRA_GLM5_SPEC") != "1":
            raise ValueError("gate recipe must select GLM5 speculation")
        if int(env.get("MEMRA_MAX_SESSIONS", "1")) < 2:
            raise ValueError("gate recipe must admit at least two sessions")
        self.nonce = uuid.uuid4().hex
        write(self.root / "identity.json", {
            "boot_nonce": self.nonce, "arm": arm, "binary_sha256": args.binary_sha256,
            "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
            "cell": args.cell, "chunk": args.chunk,
            "guard": [env.get("MEMRA_SPEC_GATE_LOW"), env.get("MEMRA_SPEC_GATE_HIGH")],
        })
        write(self.root / "profile.json", env)
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen([str(args.binary)], cwd=self.root,
            env={"PATH": os.environ["PATH"], **env}, stdout=self.log,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + args.boot_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"server exited {self.process.returncode}")
                try:
                    with self.request("/readyz", timeout=3) as response:
                        if response.status == 200:
                            print("READY", self.root, flush=True)
                            return
                except OSError:
                    pass
                time.sleep(1)
            raise TimeoutError("boot deadline")
        except BaseException:
            self.close()
            raise

    def request(self, route, body=None, timeout=None):
        return urllib.request.urlopen(urllib.request.Request(self.url + route,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}),
            timeout=timeout or self.args.request_timeout)

    def one(self, label, body):
        body = dict(body, model=self.args.model, stream=True,
                    stream_options={"include_usage": True})
        write(self.root / f"{label}-request.json", body)
        start = time.monotonic()
        text, reasoning, events = [], [], []
        usage, finish, first, done = {}, None, None, False
        try:
            with self.request("/v1/chat/completions", body) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    events.append(event)
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content") or ""
                        reason = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if content or reason:
                            first = first if first is not None else time.monotonic() - start
                        text.append(content)
                        reasoning.append(reason)
                        finish = choice.get("finish_reason") or finish
            if not done or finish not in ("length", "stop") or first is None:
                raise RuntimeError(f"incomplete SSE stream: done={done}, finish={finish}")
            spec = usage.get("spec", {})
            if not spec or not any(spec.get(k, 0) > 0 for k in ("drafted", "draft_tokens", "drafted_tokens")):
                raise RuntimeError(f"missing per-request spec engagement: {spec}")
            row = {"text": "".join(text), "reasoning": "".join(reasoning), "usage": usage,
                   "ttft_s": first, "total_s": time.monotonic() - start, "finish": finish,
                   "boot_nonce": self.nonce, "label": label}
            write(self.root / f"{label}-result.json", row)
            return row
        finally:
            write(self.root / f"{label}-events.json", events)

    def exactness(self):
        result = {}
        for restored in (False, True):
            messages = [{"role": "user", "content": "Explain the next step briefly. " +
                         "A garden has water, sunlight and two raised beds. " * 40}]
            chain = []
            for turn in range(4):
                label = f"{'restored' if restored else 'cold'}-{turn}"
                salt = "restored-chain" if restored else label
                row = self.one(label, {"messages": messages, "temperature": 0,
                                      "max_tokens": 32, "cache_salt": salt})
                cached = row["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                if restored and turn > 0 and not cached:
                    raise RuntimeError(f"restored turn {turn} did not hit prefix cache")
                if not restored and cached:
                    raise RuntimeError("cold cell unexpectedly reused a prefix")
                chain.append([row["text"], row["reasoning"]])
                messages = messages + [{"role": "assistant", "content": row["text"],
                    "reasoning_content": row["reasoning"]}, {"role": "user",
                    "content": f"Continue with another concrete example, turn {turn}."}]
            result["restored" if restored else "cold"] = chain
        long = {"messages": [{"role": "user", "content":
            "Summarize these independent garden notes briefly.\n" +
            "Water the first bed in the morning and check the second bed in the evening.\n" * 160}],
            "temperature": 0, "max_tokens": 32}
        small = {"messages": [{"role": "user", "content": "Name two colors."}],
                 "temperature": 0, "max_tokens": 32}
        c1 = [self.one("long-c1", dict(long, cache_salt="long-c1")),
              self.one("small-c1", dict(small, cache_salt="small-c1"))]
        offset = self.log_path.stat().st_size
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            long_job = pool.submit(self.one, "long-c2", dict(long, cache_salt="long-c2"))
            deadline = time.monotonic() + self.args.request_timeout
            while time.monotonic() < deadline:
                with self.log_path.open() as log:
                    log.seek(offset)
                    delta = log.read()
                if "[prime-chunk] phase=glm5-trunk" in delta:
                    break
                if long_job.done():
                    long_job.result()
                    raise RuntimeError("long request completed without a prime chunk trigger")
                time.sleep(0.01)
            else:
                raise TimeoutError("prime chunk trigger missing")
            small_job = pool.submit(self.one, "small-c2", dict(small, cache_salt="small-c2"))
            c2 = [long_job.result(), small_job.result()]
        for solo, paired in zip(c1, c2):
            if [solo["text"], solo["reasoning"]] != [paired["text"], paired["reasoning"]]:
                raise RuntimeError("paired request differs from its own c1 greedy oracle")
        if self.arm and not re.search(r"\[prime-yield\] count=[1-9]", delta + self.log_path.read_text()[offset:]):
            raise RuntimeError("ON c2 did not yield")
        result["pair"] = [[r["text"], r["reasoning"]] for r in c2]
        return result

    def latency(self):
        if self.args.long_request is None:
            raise ValueError("latency requires a pinned --long-request JSON")
        body = json.loads(self.args.long_request.read_text())
        forbidden = {"temperature", "top_p", "top_k", "min_p", "seed"}
        if forbidden.intersection(body):
            raise ValueError("latency request overrides vendor sampling")
        body["cache_salt"] = self.nonce + "-long"
        small = {"messages": [{"role": "user", "content": "Give two practical gardening tips."}],
                 "max_tokens": 128}
        jobs = []
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(24) as pool:
            long_job = pool.submit(self.one, "long", body)
            for i in range(1, 21):
                time.sleep(max(0, start + i * 5 - time.monotonic()))
                jobs.append(pool.submit(self.one, f"small-{i}", dict(small, cache_salt=f"{self.nonce}-{i}")))
            rows = [j.result() for j in jobs]
            long_row = long_job.result()  # drain beyond the 100-second arrival window
        ttfts = sorted(r["ttft_s"] for r in rows)
        twin = []
        messages = body["messages"]
        for turn in range(8):
            row = self.one(f"twin-{turn}", {"messages": messages, "max_tokens": 128,
                                          "cache_salt": self.nonce + "-twin"})
            cached = row["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if turn and not cached:
                raise RuntimeError("cache-on twin lost reuse")
            twin.append(row)
            messages = messages + [{"role": "assistant", "content": row["text"],
                "reasoning_content": row["reasoning"]}, {"role": "user", "content": "Continue briefly."}]
        return {"small_p95_ttft_s": ttfts[math.ceil(.95*len(ttfts))-1],
                "long_ttft_s": long_row["ttft_s"], "long_total_s": long_row["total_s"],
                "small_count": len(rows), "cache_twin": twin}

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=60)
        self.log.close()
        # Context unload can outlast process termination; keep the lock until empty.
        deadline = time.monotonic() + 120
        while True:
            try:
                gpu_empty()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1)
        (self.root / "keys.toml").unlink(missing_ok=True)
        write(self.root / "cleanup.json", {"pid": self.process.pid,
              "returncode": self.process.returncode, "compute_apps": [], "boot_nonce": self.nonce})


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "profile", "metadata", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--model", default="zai/glm-5.3-flash")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--cell", choices=("exactness", "latency"), required=True)
    parser.add_argument("--boot-order", default="0,1")
    parser.add_argument("--long-request", type=Path)
    parser.add_argument("--boot-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=1800)
    args = parser.parse_args()
    for key in ("binary", "profile", "metadata", "out"):
        setattr(args, key, getattr(args, key).resolve())
    with args.binary.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != args.binary_sha256:
            raise ValueError("binary hash mismatch")
    import fcntl
    with open("/tmp/memra-gpu.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = []
        for number, arm in enumerate(map(int, args.boot_order.split(","))):
            if arm not in (0, 1):
                raise ValueError("unknown arm")
            boot = Boot(args, arm, number)
            try:
                result = boot.exactness() if args.cell == "exactness" else boot.latency()
                log = boot.log_path.read_text()
                if "[glm5-acc]" not in log:
                    raise RuntimeError("missing server spec engagement")
                if re.search(r"CUDA_ERROR_OUT_OF_MEMORY|out of memory|step-OOM|panicked", log, re.I):
                    raise RuntimeError("OOM/panic in scored boot")
                write(boot.root / "summary.json", result)
                results.append((arm, result))
                print("PASS", args.cell, arm, boot.root, flush=True)
            finally:
                boot.close()
        if args.cell == "exactness" and any(r != results[0][1] for _, r in results):
            raise RuntimeError("OFF/ON greedy chains differ")
        write(args.out / "summary.json", {"cell": args.cell, "boots": results})


if __name__ == "__main__":
    main()
