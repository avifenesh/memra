#!/usr/bin/env python3
"""The two rollback seams that keep the sampled tail pass (memra#1186): rows to compare
between two binaries.

`MEMRA_SPEC_SAMPLED_BOUNDARY=0` puts an argmax at every burst boundary and
`MEMRA_SPEC_PEN_SESSION=0` restarts the penalty window at every burst. Both define their text
by the burst boundary, so neither can be compared with plain decode: what must hold is that a
binary with the sampled pending carry writes, under each seam, the text the binary before it
wrote, with the same rounds and acceptance where the tail pass is kept.

One boot per seam setting (boundary, pen, both), the device's own K, MEMRA_SPEC_BOUNDARY_TRACE=1.
Five lone requests per boot, 240 tokens: temperature 1 on a 1.2k-token and a 34-token prompt,
temperature 0.7 with top_p 0.95, a frequency penalty of 0.4, and temperature 0.7 with top_p
0.8, top_k 20 and a presence penalty of 1.5. Each row also counts the sampled bursts that
entered on a carried token: none where the seam keeps the tail (every sampled request under
`boundary` and `both`, the two penalised ones under `pen`).

usage: seam_rows.py SERVER_BINARY MODEL.gguf OUT [PORT]
"""
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

BIN, MODEL, OUT = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 18199
URL = f"http://127.0.0.1:{PORT}"
OUT.mkdir(parents=True, exist_ok=True)


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic):
    return f"{topic.capitalize()} field notes. {notes(topic, 45)}\n\nWrite about 300 words on what these notes say about {topic}."


def short_prompt(topic):
    return f"{topic.capitalize()} in brief: write about 300 words on how a {topic} survey is planned and run."


REQS = [("a", long_prompt("volcanoes"), {"temperature": 1.0, "seed": 701}, False),
        ("b", short_prompt("meadow"), {"temperature": 1.0, "seed": 702}, False),
        ("f", long_prompt("harbors"), {"temperature": 0.7, "top_p": 0.95, "seed": 707}, False),
        ("g", short_prompt("orchard"), {"temperature": 1.0, "frequency_penalty": 0.4, "seed": 708}, True),
        ("h", long_prompt("lakes"), {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": 709}, True)]
SEAMS = {"boundary": {"MEMRA_SPEC_SAMPLED_BOUNDARY": "0"}, "pen": {"MEMRA_SPEC_PEN_SESSION": "0"},
         "both": {"MEMRA_SPEC_SAMPLED_BOUNDARY": "0", "MEMRA_SPEC_PEN_SESSION": "0"}}
bad = 0
for case, env in SEAMS.items():
    log = OUT / f"server-{case}.log"
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update({"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
              "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2", "MEMRA_SPEC_BOUNDARY_TRACE": "1", **env})
    p = subprocess.Popen([BIN], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            if p.poll() is not None:
                raise SystemExit(f"server exited {p.returncode}; see {log}")
            if time.time() - t0 > 900:
                raise SystemExit("server not ready in 900 s")
            try:
                urllib.request.urlopen(URL + "/readyz", timeout=3).read()
                break
            except Exception:
                time.sleep(1)
        for pid, content, samp, penalised in REQS:
            mark = log.stat().st_size
            body = {"model": "q", "messages": [{"role": "user", "content": content}], "max_tokens": 240, "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False}, **samp}
            req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    obj = json.loads(r.read())
                text, usage, err = obj["choices"][0]["message"]["content"] or "", obj.get("usage") or {}, None
            except Exception as exc:  # recorded and counted, never swallowed
                text, usage, err = "", {}, repr(exc)
            time.sleep(0.3)
            with open(log, errors="replace") as f:
                f.seek(mark)
                chunk = f.read()
            spec = usage.get("spec") or {}
            tail_kept = case != "pen" or penalised
            row = {"case": case, "prompt": pid, "penalised": penalised, "tail_kept": tail_kept,
                   "completion_tokens": usage.get("completion_tokens"), "acceptance": spec.get("acceptance_rate"),
                   "rounds": spec.get("rounds"), "drafted": spec.get("drafted"), "accepted": spec.get("accepted"),
                   "carries": chunk.count("[spec-carry] site=burst-entry"),
                   "boundary_draws": chunk.count("[spec-boundary] site=burst-tail-commit"),
                   "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "error": err}
            if not err and (not text or row["acceptance"] is None):
                row["error"] = "empty answer or no speculation"
            bad += bool(row["error"])
            with open(OUT / "runs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(case, pid, row["completion_tokens"], row["rounds"], row["acceptance"], "carries", row["carries"],
                  "tail draws", row["boundary_draws"], row["sha"], row["error"] or "", flush=True)
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
print("SEAM ROWS:", "errors" if bad else "written", flush=True)
sys.exit(1 if bad else 0)
