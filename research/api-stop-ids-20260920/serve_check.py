#!/usr/bin/env python3
"""Raw-id stop serving gate. Run under flock /tmp/memra-5090.lock.

Uses the existing native completions token tape, not a debug build or re-tokenization.
Boots the same binary again in OpenAI mode and compares exact greedy prefixes.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = "http://127.0.0.1:18091"
    prompt = "<bos><start_of_turn>user\nCount from one to ten, separated by commas.<end_of_turn>\n<start_of_turn>model\n"
    records = []

    def request(name, body, stream=False):
        data = json.dumps(body).encode()
        req = urllib.request.Request(base + "/v1/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read().decode()
        (out / (name + (".sse" if stream else ".json"))).write_text(raw)
        records.append({"name": name, "request": body,
                        "response_sha256": hashlib.sha256(raw.encode()).hexdigest()})
        return raw if stream else json.loads(raw)

    @contextlib.contextmanager
    def server(mode):
        # Refuse a foreign responder before launch, and check our child at readiness.
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 18091))
        env = dict(os.environ, MEMRA_COMPAT=mode, MEMRA_ADDR="127.0.0.1:18091",
                   MEMRA_CTX="4096", MEMRA_SERVE_SPEC="0", MEMRA_MODELS="g=" + args.model)
        with (out / (mode + "-server.log")).open("w") as log:
            proc = subprocess.Popen([args.binary], env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 240
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError("server exited before readiness: " + str(proc.returncode))
                    try:
                        with urllib.request.urlopen(base + "/readyz", timeout=2) as response:
                            if response.status == 200:
                                owner = subprocess.run(
                                    ["ss", "-ltnp", "sport = :18091"],
                                    capture_output=True, text=True, check=True)
                                assert f"pid={proc.pid}," in owner.stdout, "foreign readiness responder"
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("readiness timeout")
                    time.sleep(1)
                yield
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    body = {"model": "g", "prompt": prompt, "temperature": 0, "seed": 42, "max_tokens": 64}
    with server("native"):
        native = request("native-baseline", body)
        ids = native["tokens"]
        assert len(ids) > 3, native
        # Choose a first-occurrence interior id so expected stopping position is unambiguous.
        k = next(i for i in range(2, len(ids) - 1) if ids[i] not in ids[:i])
        prefix = request("native-prefix", dict(body, max_tokens=k))
        assert prefix["tokens"] == ids[:k], (prefix, ids, k)
    with server("openai"):
        baseline = request("openai-baseline", body)
        assert baseline["choices"][0]["text"] == native["text"]
        stopped = request("openai-stop", dict(body, stop_token_ids=[ids[k]]))
        assert stopped["choices"][0]["text"] == prefix["text"], (stopped, prefix)
        assert baseline["choices"][0]["text"].startswith(prefix["text"])
        assert stopped["choices"][0]["finish_reason"] == "stop", stopped
        assert stopped["usage"]["completion_tokens"] == k, stopped
        first = request("openai-first-stop", dict(body, stop_token_ids=[ids[0]]))
        assert first["choices"][0]["text"] == "", first
        assert first["choices"][0]["finish_reason"] == "stop", first
        assert first["usage"]["completion_tokens"] == 0, first
        raw = request("openai-stream-stop", dict(body, stop_token_ids=[ids[k]], stream=True), True)
        frames = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        assert "data: [DONE]" in raw
        text = "".join(f["choices"][0].get("text", "") for f in frames if f.get("choices"))
        assert text == prefix["text"], frames
        assert any(f.get("choices") and f["choices"][0].get("finish_reason") == "stop" for f in frames)
        usage = next(f["usage"] for f in reversed(frames) if f.get("usage"))
        assert usage["completion_tokens"] == k, usage
        if native["stop_reason"] == "Eos":
            eos = request("openai-explicit-eos", dict(body, stop_token_ids=[ids[-1]]))
            assert eos["choices"][0]["text"] == baseline["choices"][0]["text"]
            assert eos["choices"][0]["finish_reason"] == "stop"
            assert eos["usage"]["completion_tokens"] == len(ids) - 1
    result = {"result": "PASS", "prefix_tokens": k, "stop_id": ids[k],
              "baseline_tokens": len(ids), "baseline_stop_reason": native["stop_reason"],
              "binary_sha256": hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
              "requests": records}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
