#!/usr/bin/env python3
"""Collect five bounded real-endpoint serving cells, without promoting support.

Uses the existing cache grid contract. Bounded raw completions are token tapes,
not answer-quality claims. Startup and retirement wait on the owned process's
log events; the caller must supply the canonical GPU lease on an inherited FD.
"""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
import traceback

from cache_qualification import capture_len


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completion(document, maximum):
    require(isinstance(document, dict) and not document.get("error"), "error or non-object response")
    require(bool(document.get("id")) and document.get("model") == "gate", "missing id or wrong model")
    choices = document.get("choices", [])
    require(len(choices) == 1, "expected one choice")
    choice = choices[0]
    require(isinstance(choice.get("text"), str) and bool(choice["text"]), "empty token tape")
    require(choice.get("finish_reason") in ("stop", "length"), "missing finish reason")
    usage = document.get("usage", {})
    p, n, total = (usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    require(all(type(x) is int for x in (p, n, total)), "missing token usage")
    require(p > 0 and 0 < n <= maximum and total == p + n, "invalid token accounting")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    require(type(cached) is int and 0 <= cached <= p, "invalid cache accounting")
    return {"text": choice["text"], "text_sha256": hashlib.sha256(choice["text"].encode()).hexdigest(),
            "finish": choice["finish_reason"], "prompt_tokens": p, "completion_tokens": n,
            "cached_tokens": cached, "spec": usage.get("spec")}


def stream_completion(raw, maximum):
    frames = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
    require(frames and frames[-1] == "[DONE]" and frames.count("[DONE]") == 1, "missing terminal DONE")
    documents = [json.loads(frame) for frame in frames[:-1]]
    require(len(documents) > 1, "missing incremental frames")
    require(all(isinstance(d, dict) and not d.get("error") for d in documents), "stream error")
    ids = {d.get("id") for d in documents}
    require(len(ids) == 1 and None not in ids and "" not in ids, "inconsistent stream id")
    require(all(d.get("model") == "gate" for d in documents), "wrong stream model")
    choices = [c for d in documents for c in d.get("choices", [])]
    terminals = [c for c in choices if c.get("finish_reason") is not None]
    require(len(terminals) == 1 and choices[-1] is terminals[0], "invalid finish ordering")
    usages = [d["usage"] for d in documents if isinstance(d.get("usage"), dict)]
    require(usages and all(u == usages[0] for u in usages), "missing or inconsistent stream usage")
    merged = {"id": next(iter(ids)), "model": "gate", "usage": usages[0], "choices": [
        {"text": "".join(c.get("text", "") for c in choices), "finish_reason": terminals[0]["finish_reason"]}]}
    result = completion(merged, maximum)
    result["frames"] = len(documents)
    return result


class Server:
    def __init__(self, args, out, cache_mb):
        self.args, self.out = args, out
        self.key = secrets.token_hex(24)
        self.lines, self.condition, self.eof = [], threading.Condition(), False
        with socket.socket() as guard:
            guard.bind(("127.0.0.1", args.port))
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        settings = {"MEMRA_ADDR": f"127.0.0.1:{args.port}", "MEMRA_MODELS": f"gate={args.model}",
                    "MEMRA_COMPAT": "openai", "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "4",
                    "MEMRA_PREFIX_CACHE_MB": str(cache_mb), "MEMRA_KV_HOST_MB": "0"}
        env.update(settings)
        env["MEMRA_API_KEY"] = self.key
        save(out / "environment.json", settings)
        self.process = subprocess.Popen([args.server], env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        pass_fds=(args.external_lock,))
        self.log = out / "server.log"
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        with self.log.open("w") as output:
            for line in self.process.stdout:
                output.write(line)
                output.flush()
                with self.condition:
                    self.lines.append(line)
                    self.condition.notify_all()
        with self.condition:
            self.eof = True
            self.condition.notify_all()

    def event(self, pattern, start=0, timeout=180):
        with self.condition:
            found = lambda: any(pattern in line for line in self.lines[start:])
            ready = self.condition.wait_for(lambda: found() or self.eof, timeout)
            require(ready and found(), f"no server event {pattern!r} before timeout/exit")
            return [line.strip() for line in self.lines[start:] if pattern in line]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=5)
        save(self.out / "exit.json", {"pid": self.process.pid, "returncode": self.process.returncode})

    def request(self, name, body=None, route="/v1/completions", abort_frames=None, barrier=None):
        if barrier:
            barrier.wait(timeout=20)
        start = time.monotonic()
        connection = http.client.HTTPConnection("127.0.0.1", self.args.port, timeout=180)
        suffix = ".sse" if body and body.get("stream") else ".json"
        raw_path = self.out / (name + suffix)
        receipt_path = self.out / (name + ".request.json")
        save(receipt_path, {"request": body, "route": route, "started_monotonic": start,
                            "status": "request_started"})
        try:
            headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.key}
            connection.request("POST" if body is not None else "GET", route,
                               json.dumps(body) if body is not None else None, headers)
            response = connection.getresponse()
            frame_times = []
            if body and body.get("stream"):
                pieces, content_frames = [], 0
                with raw_path.open("wb") as capture:
                    while line := response.readline():
                        capture.write(line)
                        capture.flush()
                        pieces.append(line)
                        if line.startswith(b"data: {"):
                            frame = json.loads(line[6:])
                            if any(c.get("text") for c in frame.get("choices", [])):
                                content_frames += 1
                                frame_times.append(time.monotonic() - start)
                                if abort_frames and content_frames == abort_frames:
                                    break
                if abort_frames:
                    require(content_frames == abort_frames, "cancel fixture ended before disconnect")
                # Close the response as well as HTTPConnection: the response owns the socket.
                raw = b"".join(pieces)
                response.close()
            else:
                raw = response.read()
            elapsed = time.monotonic() - start
            raw_path.write_bytes(raw)
            receipt = {"request": body, "route": route, "status": response.status,
                       "started_monotonic": start, "ended_monotonic": start + elapsed,
                       "wall_seconds": elapsed, "response_file": raw_path.name,
                       "response_sha256": digest(raw_path), "content_frame_seconds": frame_times}
            save(receipt_path, receipt)
            if abort_frames:
                return receipt
            require(response.status == 200, f"{name}: HTTP {response.status}: {raw[:200]!r}")
            parsed = (stream_completion(raw.decode(), body["max_tokens"]) if suffix == ".sse"
                      else json.loads(raw))
            return parsed, receipt
        except Exception as error:
            receipt = json.loads(receipt_path.read_text())
            receipt["error"] = str(error)
            receipt["ended_monotonic"] = time.monotonic()
            save(receipt_path, receipt)
            raise
        finally:
            connection.close()


PROMPT = ("During a code review, Alice reads the patch, Bob checks the tests, and Carol verifies "
          "the documentation. They record each result before the release. " * 8
          + "List the three people and their tasks:\n")


def body(prompt=PROMPT, salt="qualification", maximum=64, **kwargs):
    return {"model": "gate", "prompt": prompt, "max_tokens": maximum,
            "temperature": 0, "seed": 7, "cache_salt": salt, **kwargs}


def cache_check(cold, warm):
    require(cold["cached_tokens"] == 0, "cold request restored cache")
    require(warm["cached_tokens"] == capture_len(cold["prompt_tokens"]), "cache grid accounting differs")
    require(warm["cached_tokens"] > 0, "warm request did not restore cache")
    same_completion(cold, warm, "cold/warm")


def same_completion(a, b, label):
    for field in ("text_sha256", "completion_tokens", "prompt_tokens", "finish"):
        require(a[field] == b[field], f"{label} {field} differs")


def rejected(call):
    try:
        call()
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return str(error)
    raise ValueError("red arm incorrectly accepted")


def collect(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    proof = subprocess.check_output(["python3", "tools/tier-lock-proof.py", "--fd", str(args.external_lock),
                                     "--lock", "/tmp/memra-5090.lock", "--owner", "collector"],
                                    pass_fds=(args.external_lock,), text=True)
    (out / "LOCK.json").write_text(proof)
    inspection = Path(args.inspection)
    artifact_lock = (inspection / "artifact.lock").read_text()
    fields = dict(line.split("=", 1) for line in artifact_lock.splitlines() if "=" in line)
    require(fields.get("source") == args.model and fields.get("family") == "qwen35",
            "inspection must identify this Qwen3.5 artifact")
    require(fields.get("binding") == fields.get("tokenizer") == "passed",
            "inspection tensor/tokenizer prerequisites failed")
    metadata = {"source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "server_sha256": digest(args.server), "cli_sha256": digest(args.cli),
                "model_sha256": digest(args.model), "model_bytes": Path(args.model).stat().st_size,
                "model_filename": Path(args.model).name, "collector_sha256": digest(__file__),
                "inspection_artifact_lock_sha256": digest(inspection / "artifact.lock"),
                "context": 8192, "max_offered_concurrency": 4, "gpu": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv"], text=True),
                "scope": "bounded serving mechanics; raw greedy tapes; no support promotion",
                "remaining_prerequisites": ["checkpoint parity", "strict runtime and rewrite receipts"],
                "inspection_prerequisites": {"tensor_binding": fields["binding"], "tokenizer": fields["tokenizer"]}}
    save(out / "manifest.json", metadata)
    results, red = {}, {}
    on = out / "cache-on"
    on.mkdir()
    server = Server(args, on, 2048)
    def cell(name, fn):
        try:
            results[name] = {"status": "passed", "detail": fn()}
        except Exception as error:
            results[name] = {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        save(on / (name + "-result.json"), results[name])
        print(name + ": " + results[name]["status"], flush=True)
    def req(name, request):
        document, receipt = server.request(name, request)
        return completion(document, request["max_tokens"]), receipt
    try:
        server.event("[server] listening on")
        models, _ = server.request("models", route="/v1/models")
        require([m["id"] for m in models["data"]] == ["gate"], "wrong endpoint identity")
        server.request("readyz", route="/readyz")

        def streaming():
            plain, _ = req("stream-control", body(salt="stream-control"))
            streamed, _ = server.request("stream", body(salt="stream", stream=True,
                                                          stream_options={"include_usage": True}))
            same_completion(streamed, plain, "stream/blocking")
            raw = (on / "stream.sse").read_text()
            red["stream_missing_done"] = rejected(lambda: stream_completion(raw.replace("data: [DONE]", ""), 64))
            red["stub_http"] = rejected(lambda: completion({"choices": [{"text": "ok"}]}, 64))
            red["bad_usage"] = rejected(lambda: completion({"id": "test", "model": "gate", "choices": [
                {"text": "x", "finish_reason": "length"}], "usage": {"prompt_tokens": 2,
                "completion_tokens": 1, "total_tokens": 99}}, 64))
            return {"blocking": plain, "stream": streamed}
        cell("streaming", streaming)

        def cache():
            cold, _ = req("cache-cold", body(salt="cache"))
            warm, _ = req("cache-warm", body(salt="cache"))
            cache_check(cold, warm)
            return {"cold": cold, "warm": warm, "expected_cached": capture_len(cold["prompt_tokens"])}
        cell("cache", cache)

        def concurrency():
            serial = [req(f"serial-{i}", body(salt=f"serial-{i}"))[0] for i in range(4)]
            barrier = threading.Barrier(4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                pending = [pool.submit(server.request, f"concurrent-{i}", body(salt=f"concurrent-{i}"),
                                       barrier=barrier) for i in range(4)]
                responses = [future.result(timeout=240) for future in pending]
            concurrent_rows = [completion(d, 64) for d, _ in responses]
            receipts = [r for _, r in responses]
            require(max(r["started_monotonic"] for r in receipts) < min(r["ended_monotonic"] for r in receipts),
                    "offered requests did not overlap")
            for c, s in zip(concurrent_rows, serial):
                same_completion(c, s, "serial/concurrent")
            return {"serial": serial, "concurrent": concurrent_rows, "all_four_offered_overlap": True}
        cell("concurrency", concurrency)

        def cancellation():
            control, _ = req("cancel-control", body(salt="cancel-control"))
            start = len(server.lines)
            cancelled = server.request("cancel-disconnect", body(salt="cancel-fault", maximum=1024,
                                                                   stream=True), abort_frames=8)
            aborted = server.event("[abort] client disconnected:", start, timeout=30)
            require(not b"data: [DONE]" in (on / "cancel-disconnect.sse").read_bytes(), "cancel stream already complete")
            recovery, _ = req("cancel-recovery", body(salt="cancel-control"))
            cache_check(control, recovery)
            server.request("cancel-readyz", route="/readyz")
            return {"control": control, "disconnected": cancelled, "abort_log": aborted, "recovery": recovery}
        cell("cancellation", cancellation)

        def long_context():
            prompt = ("Read the numbered archive. Preserve the order of all entries.\n"
                      + "The archive contains red blue green gold silver white black.\n" * 520
                      + "List the colors from the archive:\n")
            tokens, _ = server.request("long-tokenize", {"model": "gate", "prompt": prompt}, "/v1/tokenize")
            require(6000 <= tokens["count"] <= 8128, f"long fixture has {tokens['count']} tokens outside declared band")
            cold, _ = req("long-cold", body(prompt, "long"))
            warm, _ = req("long-warm", body(prompt, "long"))
            cache_check(cold, warm)
            require(cold["prompt_tokens"] == tokens["count"], "tokenize/usage mismatch")
            return {"cold": cold, "warm": warm, "context_envelope": 8192,
                    "tested_prompt_tokens": tokens["count"], "generation_budget": 64}
        cell("long_context", long_context)
    finally:
        server.close()
    off = out / "cache-off-red"
    off.mkdir()
    server = Server(args, off, 0)
    try:
        server.event("[server] listening on")
        cold, _ = req("cache-cold", body(salt="cache-off"))
        warm, _ = req("cache-warm", body(salt="cache-off"))
        require(cold["cached_tokens"] == warm["cached_tokens"] == 0, "cache-off arm unexpectedly hit")
        red["real_cache_disabled"] = rejected(lambda: cache_check(cold, warm))
    finally:
        server.close()
    save(out / "red-arms.json", red)
    raw_hashes = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}
    record = ["format\tmemra-qualification-record-v1", "family\tqwen35", "promote_to\tNativeReference"]
    for name, result in results.items():
        result["raw_files_sha256"] = raw_hashes
        path = out / (name + ".json")
        save(path, result)
        record += [f"cell.{name}\t{result['status']}", f"cell.{name}.evidence\t{path.name}",
                   f"cell.{name}.evidence_sha256\t{digest(path)}"]
    path = out / "qualification.tsv"
    path.write_text("\n".join(record) + "\n")
    checked = subprocess.run([args.cli, "model", "qualify", str(path)], capture_output=True, text=True)
    save(out / "record-validation.json", {"returncode": checked.returncode, "stdout": checked.stdout, "stderr": checked.stderr})
    require(checked.returncode == 0, "qualification record inconsistent")
    require(len(results) == 5 and all(v["status"] == "passed" for v in results.values()), "one or more serving cells failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--external-lock", type=int, required=True)
    collect(parser.parse_args())
