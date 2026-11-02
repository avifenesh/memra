#!/usr/bin/env python3
"""Host-side serving bench for one release-profile arm on the local RTX 5090.

usage: host_bench.py ARM ROUND
Boots target-ARM/release/memra-server on the Qwen3.5-9B NVFP4 MTP GGUF and measures:
  boot (spawn -> /health), /v1/tokenize on the frozen ~131k-token chat (template + BPE +
  JSON), streamed TTFT on a short cached prompt, and greedy streams with the server
  process CPU time per generated token. Appends one JSON line to host_results.jsonl.
Run it under gpulock.
"""
import hashlib, http.client, json, os, signal, statistics, subprocess, sys, time
from pathlib import Path

BASE = Path.home() / ".cache/memra-relprofile"
arm, rnd = sys.argv[1], sys.argv[2]
TOK_ONLY = len(sys.argv) > 3 and sys.argv[3] == "tok"
MODEL = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
PORT = 8291
binary = BASE / f"target-{arm}/release/memra-server"
log = BASE / "logs" / f"{time.strftime('%Y%m%dT%H%M%S')}-serve-{arm}-{rnd}.log"
class Done(Exception):
    pass
CLK = os.sysconf("SC_CLK_TCK")
profile = open("/sys/firmware/acpi/platform_profile").read().strip()

def req(method, path, body=None, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, out

def stream_chat(body):
    """Returns (ttft_s, e2e_s, content, frames)."""
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
    data = json.dumps(dict(body, stream=True)).encode()
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body=data, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200, (r.status, r.read()[:500])
    ttft, parts, frames, usage = None, [], 0, None
    buf = b""
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        obj = json.loads(payload)
        frames += 1
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            d = ch.get("delta", {})
            piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(piece)
    e2e = time.perf_counter() - t0
    c.close()
    return ttft, e2e, "".join(parts), frames, usage

def proc_cpu(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / CLK

def thread_cpu(pid):
    out = {}
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            s = open(f"/proc/{pid}/task/{t}/stat").read()
        except OSError:
            continue
        comm = s[s.index("(") + 1:s.rindex(")")]
        f = s.rsplit(")", 1)[1].split()
        out[(t, comm)] = (int(f[11]) + int(f[12])) / CLK
    return out

env = dict(os.environ, MEMRA_COMPAT="openai", MEMRA_MODELS=f"smoke={MODEL}", MEMRA_ADDR=f"127.0.0.1:{PORT}")
lf = open(log, "w")
t_spawn = time.perf_counter()
srv = subprocess.Popen([str(binary)], env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
row = {"arm": arm, "round": rnd, "platform_profile": profile, "log": str(log),
       "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
try:
    while True:
        assert srv.poll() is None, "server died during boot; see log"
        try:
            if req("GET", "/health", timeout=2)[0] == 200:
                break
        except OSError:
            pass
        time.sleep(0.05)
        assert time.perf_counter() - t_spawn < 600, "boot timeout"
    row["boot_s"] = round(time.perf_counter() - t_spawn, 3)

    # idle CPU baseline
    c0 = proc_cpu(srv.pid); time.sleep(5); row["idle_cpu_pct"] = round((proc_cpu(srv.pid) - c0) / 5 * 100, 2)

    # tokenize the frozen long chat
    msgs = json.loads((BASE / "longchat.json").read_text())
    body = {"model": "smoke", "messages": msgs}
    times, count, tok_hash = [], None, None
    cpu_per = []
    for i in range(65 if TOK_ONLY else 23):
        c0 = proc_cpu(srv.pid)
        t0 = time.perf_counter(); st, out = req("POST", "/v1/tokenize", body); dt = time.perf_counter() - t0
        time.sleep(0.02)
        if i >= (5 if TOK_ONLY else 3):
            cpu_per.append(proc_cpu(srv.pid) - c0)
        assert st == 200, out[:500]
        o = json.loads(out)
        h = hashlib.sha256(json.dumps(o["tokens"]).encode()).hexdigest()
        assert tok_hash in (None, h)
        count, tok_hash = o["count"], h
        if i >= (5 if TOK_ONLY else 3):
            times.append(dt)
    row["tokenize_tokens"] = count
    row["tokenize_ids_sha256"] = tok_hash
    row["tokenize_ms"] = {"median": round(statistics.median(times) * 1e3, 2), "min": round(min(times) * 1e3, 2),
                          "p90": round(sorted(times)[int(0.9 * len(times)) - 1] * 1e3, 2), "n": len(times)}

    row["tokenize_server_cpu_ms"] = {"median": round(statistics.median(cpu_per) * 1e3, 1), "mean": round(statistics.mean(cpu_per) * 1e3, 1), "n": len(cpu_per)}
    if TOK_ONLY:
        row["mode"] = "tok"
        raise Done
    # short cached prompt TTFT
    short = {"model": "smoke", "messages": [{"role": "user", "content": "Reply with one word: hello."}],
             "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    for _ in range(3):
        stream_chat(short)
    tt, ee = [], []
    for _ in range(30):
        t, e, _, _, _ = stream_chat(short); tt.append(t); ee.append(e)
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    row["ttft_ms"] = {"p50": round(statistics.median(tt) * 1e3, 2), "p95": round(q(tt, 0.95) * 1e3, 2), "n": len(tt)}
    row["short_e2e_ms"] = {"p50": round(statistics.median(ee) * 1e3, 2), "n": len(ee)}

    # greedy streams: byte identity + server CPU per generated token
    greedy = {"model": "smoke", "messages": [{"role": "user", "content":
              "Write a detailed technical explanation of how a paged KV cache works in an LLM inference server."}],
              "max_tokens": 512, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
              "stream_options": {"include_usage": True}}
    reps = []
    for _ in range(3):
        c0 = proc_cpu(srv.pid); th0 = thread_cpu(srv.pid)
        t, e, content, frames, usage = stream_chat(greedy)
        c1 = proc_cpu(srv.pid); th1 = thread_cpu(srv.pid)
        (BASE / "streams").mkdir(exist_ok=True)
        (BASE / "streams" / f"{arm}-{rnd}-{len(reps)}.txt").write_text(content)
        ntok = (usage or {}).get("completion_tokens") or frames
        top = sorted(((th1[k] - th0.get(k, 0), k[1]) for k in th1), reverse=True)[:3]
        reps.append({"content_sha256": hashlib.sha256(content.encode()).hexdigest(), "tokens": ntok,
                     "wall_s": round(e, 3), "cpu_s": round(c1 - c0, 3),
                     "cpu_ms_per_token": round((c1 - c0) * 1e3 / ntok, 3),
                     "top_threads": [(n, round(v, 2)) for v, n in top]})
    row["greedy"] = reps
    row["greedy_sha256"] = reps[0]["content_sha256"]
    row["greedy_identical_within_arm"] = len({r["content_sha256"] for r in reps}) == 1
    # seeded sampled stream: host-side sampling numerics (softmax/top-p/RNG) byte identity
    sampled = dict(greedy, temperature=0.8, top_p=0.95, seed=1234, max_tokens=256)
    shas = []
    for j in range(2):
        _, _, content, _, _ = stream_chat(sampled)
        (BASE / "streams" / f"{arm}-{rnd}-sampled-{j}.txt").write_text(content)
        shas.append(hashlib.sha256(content.encode()).hexdigest())
    row["sampled_sha256"] = shas[0]
    row["sampled_identical_within_arm"] = len(set(shas)) == 1
except Done:
    pass
finally:
    os.killpg(srv.pid, signal.SIGTERM)
    try:
        srv.wait(30)
    except subprocess.TimeoutExpired:
        os.killpg(srv.pid, signal.SIGKILL); srv.wait()
    lf.close()
with open(BASE / ("host_tok_results.jsonl" if TOK_ONLY else "host_results.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row))
