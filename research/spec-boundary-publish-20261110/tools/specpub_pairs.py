#!/usr/bin/env python3
"""Warm == cold pair cells for the spec boundary publication (GB10 decode round 6, lever 5): the
pair cells of research/spec-pool-grid-20261109/spec_pool_pair_cells.py with the deep chat shape of
the split rung lane's cells, bounded turns, and the warm path read from the prefix cache's own
lines.

One warm boot serves a conversation in order (each turn exactly extends the previous prompt and
answer). Turn 1 is cold: its speculative session publishes the prompt's KV at the first burst
boundary (`[prefix-cache] insert (spec-boundary)`). Every later warm turn must hit an entry that
publisher wrote and restore a speculative session from it (`[prefix-cache] hit`, `spec restore`).
One cold boot per later turn primes that turn's prompt from scratch, and every warm answer must be
byte-equal to its cold twin.

  deepN   chat, thinking off: turn 1 is N synthetic records and a question, the later turns short
          follow-ups. N sets the depth (27 tokens a record: deep38 is about 1k tokens, deep4840
          about 131k).

Every turn is bounded (max_tokens), so no session parks at its end and the only warm path is the
prefix cache. With _TRACE=1 every boot sets MEMRA_PREFIX_SPLIT_TRACE=1: the server hashes each
published entry's trunk KV, recurrent state and boundary logits
(`[prefix-cache-split-state] role=spec-snapshot`), so the same case on two binaries shows whether
they published the same bytes. The trace synchronizes the stream at every capture, so a case
without it is the served shape and a case with it is the digest reading.

usage: specpub_pairs.py OUT CASE...   CASE = name=BINARY|deepN|T1,T2[,T3]|ENV
  T = max_tokens per turn; two values run two turns.
  ENV: space-separated VAR=VALUE for every boot, plus request keys _TEMP, _SEED, _TOPP and
  _TRACE=1 (above).
Writes OUT/<name>/ (per-turn JSON, server logs) and OUT/verdicts.jsonl; exit 1 if any case's bytes
differ or a warm turn did not restore from a spec-published entry.
PAIR_MODEL names the GGUF, CELL_PORT the port."""
import hashlib
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
# A timeout's SIGTERM unwinds through the finally blocks, so no server outlives the cell.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.path.expanduser(os.environ.get(
    "PAIR_MODEL", "~/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"))
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2", "MEMRA_TICK_TRACE": "1"}
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
LINES = {
    "insert": r"\[prefix-cache\] insert \(([a-z0-9-]+)\): (\d+) tokens, ([\d.]+)MB",
    "hit": r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens",
    "spec_restore": r"spec restore: (\d+) of (\d+) prompt tokens",
    "draft_restore": r"spec restore: (\d+) of (\d+) prompt tokens \+ draft plane from cache",
    "state": r"\[prefix-cache-split-state\] role=(\S+) why=(\S+) split=(\d+) entry_pos=\d+ "
             r"state_sha256=([0-9a-f]+) boundary_logits_sha256=([0-9a-f]+)",
    "state_error": r"\[prefix-cache-split-state\] ERROR (.*)",
    "refused": r"\[prefix-cache\] (?:spec publish|insert refused|REFUSED|skip)[^\n]*",
    # a resume that does not read the prefix cache: none may appear on a bounded turn
    "spec_parked_end": r"spec-reuse: (\d+) committed tokens resumed",
    "spec_affinity": r"spec-affinity: rewound to (\d+) of (\d+)",
    "spec_grid_rewind": r"grid-rewind: spec extension of (\d+) committed rows rewound to (\d+)",
}
BURST = re.compile(r"\[tick-spec\].* generated=(\d+)->(\d+) rounds=(\d+) drafted=(\d+) accepted=(\d+) k=(\d+)")


def deep_users(n):
    doc = "\n".join(f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}."
                    for i in range(n))
    return [doc + "\n\nName the first three records whose line ends in 7.", "Now name three more.",
            "Which of all six has the largest record number?"]


def boot(log, env):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(env)
    p = subprocess.Popen([env["_BIN"]], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise SystemExit(f"server exited {p.returncode}; see {log}")
        try:
            urllib.request.urlopen(URL + "/v1/models", timeout=3).read()
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise SystemExit("server not ready in 900 s")


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(120)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def ask(users, req, n, answers, max_tokens, out):
    body = dict(req, max_tokens=max_tokens)
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": users[i]})
        if i < n - 1:
            msgs.append({"role": "assistant", "content": answers[i]})
    body["messages"] = msgs
    body["chat_template_kwargs"] = {"enable_thinking": False}
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=3600) as resp:
        obj = json.loads(resp.read().decode())
    obj["_client_elapsed_s"] = round(time.perf_counter() - t0, 4)
    obj["_request"] = {k: v for k, v in body.items() if k != "messages"}
    obj["_prompt_sha256"] = hashlib.sha256(json.dumps(msgs).encode()).hexdigest()
    out.write_text(json.dumps(obj))
    m = obj["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""), m.get("content") or ""


def scan(log):
    text = log.read_text(errors="replace")
    out = {k: [list(m.groups()) if m.groups() else m.group(0) for m in re.finditer(rx, text)]
           for k, rx in LINES.items()}
    for k in ("hit", "spec_restore", "draft_restore", "spec_parked_end", "spec_affinity", "spec_grid_rewind"):
        out[k] = [[int(x) for x in g] for g in out[k]]
    out["insert"] = [[why, int(n), float(mb)] for why, n, mb in out["insert"]]
    out["bursts"] = [[int(x) for x in m.groups()] for m in BURST.finditer(text) if m.group(1) != m.group(2)]
    return out


def usage(path):
    u = json.loads(path.read_text()).get("usage") or {}
    return {"prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "spec": u.get("spec")}


def case(name, binary, shape, maxt, extra):
    ev = OUT / name
    ev.mkdir(parents=True, exist_ok=True)
    env = dict(COMMON, _BIN=binary)
    req = {"model": "q", "temperature": 0}
    for kv in extra.split():
        k, v = kv.split("=", 1)
        if k == "_TEMP":
            req["temperature"] = float(v)
        elif k == "_SEED":
            req["seed"] = int(v)
        elif k == "_TOPP":
            req["top_p"] = float(v)
        elif k == "_TRACE":
            if v == "1":
                env["MEMRA_PREFIX_SPLIT_TRACE"] = "1"
        else:
            env[k] = v
    assert shape.startswith("deep"), shape
    users = deep_users(int(shape[4:]))
    turns_n = len(maxt)
    later = list(range(2, turns_n + 1))
    (ev / "binary.sha").write_text(hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest() + "\n")
    p = boot(ev / "warm-server.log", env)
    warm, answers = [], []
    try:
        for t in range(1, turns_n + 1):
            full, content = ask(users, req, t, answers, maxt[t - 1], ev / f"warm-t{t}.json")
            warm.append(full)
            answers.append(content)
    finally:
        stop(p)
    cold = {}
    for t in later:
        p = boot(ev / f"cold-t{t}-server.log", env)
        try:
            cold[t], _ = ask(users, req, t, answers, maxt[t - 1], ev / f"cold-t{t}.json")
        finally:
            stop(p)
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]  # noqa: E731
    turns = {1: {"warm_sha": sha(warm[0]), "warm_len": len(warm[0]) - 1}}
    for t in later:
        turns[t] = {"equal": warm[t - 1] == cold[t], "warm_sha": sha(warm[t - 1]), "cold_sha": sha(cold[t]),
                    "warm_len": len(warm[t - 1]) - 1, "cold_len": len(cold[t]) - 1}
    w = scan(ev / "warm-server.log")
    # Every later warm turn restored a speculative session, trunk and draft plane, from an entry
    # the spec boundary publisher wrote: the restore's length is the token count of an
    # `insert (spec-boundary)` line and of no other insert, the restore line names the draft
    # plane, and nothing resumed outside the prefix cache.
    spec_published = {n for why, n, _ in w["insert"] if why == "spec-boundary"}
    other_published = {n for why, n, _ in w["insert"] if why != "spec-boundary"}
    restored = [n for n, _ in w["spec_restore"]]
    path_ok = (len(restored) == len(later) and [n for n, _ in w["hit"]] == restored
               and w["draft_restore"] == w["spec_restore"]
               and all(n in spec_published and n not in other_published for n in restored)
               and not (w["spec_parked_end"] or w["spec_affinity"] or w["spec_grid_rewind"])
               and not w["state_error"])
    cold_inserts = {t: scan(ev / f"cold-t{t}-server.log") for t in later}
    cold_clean = all(not c["hit"] and not c["spec_restore"] for c in cold_inserts.values())
    use = {f"warm-t{t}": usage(ev / f"warm-t{t}.json") for t in range(1, turns_n + 1)}
    use.update({f"cold-t{t}": usage(ev / f"cold-t{t}.json") for t in later})
    return {"case": name, "binary": binary, "shape": shape, "max_tokens": maxt, "env": extra,
            "request": req, "turns": turns, "bytes": all(turns[t]["equal"] for t in later),
            "restored_from_spec_entry": path_ok, "cold_is_cold": cold_clean,
            "warm": {k: w[k] for k in ("insert", "hit", "spec_restore", "draft_restore", "state", "state_error", "refused",
                                       "spec_parked_end", "spec_affinity", "spec_grid_rewind", "bursts")},
            "cold_state": {t: c["state"] for t, c in cold_inserts.items()},
            "cold_insert": {t: c["insert"] for t, c in cold_inserts.items()}, "usage": use}


failed = False
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    binary, shape, maxt, extra = rest.split("|", 3)
    v = case(name, os.path.expanduser(binary), shape, [int(x) for x in maxt.split(",")], extra)
    failed |= not (v["bytes"] and v["restored_from_spec_entry"] and v["cold_is_cold"])
    print(json.dumps(v), flush=True)
    with open(OUT / "verdicts.jsonl", "a") as f:
        f.write(json.dumps(v) + "\n")
sys.exit(1 if failed else 0)
