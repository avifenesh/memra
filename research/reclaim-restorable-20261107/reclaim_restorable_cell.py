#!/usr/bin/env python3
"""Served cell for the admission reclaim's restorable spare (RTX 5090, Qwen3.8-27B + DFlash2).

The sequence is the one the 2026-11-07 re-validation logged (`restorable` in its serve cells):
  r1  P+X, 1 token, greedy with presence_penalty 1.5: served plain, publishes a plain entry
  r2  P, 96 tokens, greedy: cold-primes DFlash, the boundary capture publishes a tail entry
  r3  P+X+question, 96 tokens, greedy or T=1 seed 1234, streamed (TTFT and total measured)
r3 runs under device-memory pressure, so admission's reclaim-on-defer fires. Main spared only
the longest match (the plain entry) and evicted the tail. The fix spares the tail the
restorable-hit preference takes.

Every arm is a fresh boot with the DFlash restore armed (it defaults off on the 5090), prefix
cache 4000 MB, context 8192, one session, no host tier. A binary with the three restore doors
(main before memra#1083) gets all three forced on; a binary with the one switch gets
MEMRA_DSPARK_PREFIX_RESTORE=1 alone (it refuses the retired door names at boot). The reference
boot has the prefix cache off, so r3 cold-primes DFlash.

usage: reclaim_restorable_cell.py OUT BIN_BASE BIN_FIX [ORDER]
  ORDER: comma list of arms per mode, default base,fix,fix,base,base,fix
  CELL_DOORS=base:three,fix:one (the default) says which binary takes which form; the
  served-ed28/ run predates the switch and ran both binaries with three.
Writes OUT/server-<mode>-<run>-<arm>.log, OUT/runs.jsonl, OUT/cells.txt; exit 1 on a failed check.
"""
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

OUT, BIN_BASE, BIN_FIX = sys.argv[1], sys.argv[2], sys.argv[3]
ORDER = (sys.argv[4] if len(sys.argv) > 4 else "base,fix,fix,base,base,fix").split(",")
os.makedirs(OUT, exist_ok=True)
PORT = int(os.environ.get("CELL_PORT", "18317"))
BASE_URL = f"http://127.0.0.1:{PORT}"
Q38 = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
DF2 = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-DFlash2-memra"
TRIM = "/data/ai-ml/models/q38-gguf/q38-ranks-sxc32768.gguf.txt"
FAILS = []

ENV = {"MEMRA_MODELS": f"q38={Q38}", "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "1", "MEMRA_KV_HOST_MB": "0",
       "MEMRA_MTP_SKIP": "1", "MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": DF2, "MEMRA_FRSPEC_TRIM": TRIM,
       "MEMRA_COMPAT": "openai", "MEMRA_ADDR": f"127.0.0.1:{PORT}"}
ONE_SWITCH = dict(ENV, MEMRA_PREFIX_CACHE_MB="4000", MEMRA_PREFIX_STABLE_BOUNDARY="1",
                  MEMRA_DSPARK_PREFIX_RESTORE="1")
THREE_DOORS = dict(ONE_SWITCH, MEMRA_DSPARK_PARTIAL_RESTORE="1", MEMRA_DSPARK_BOUNDARY_CAPTURE="1")
DOORS = dict(kv.split(":") for kv in os.environ.get("CELL_DOORS", "base:three,fix:one").split(","))
ARM = {arm: {"one": ONE_SWITCH, "three": THREE_DOORS}[DOORS[arm]] for arm in ("base", "fix")}
REF = dict(ENV, MEMRA_PREFIX_CACHE_MB="0")


def para(i, n):
    # tools/dspark-cold-drafter-gate.py's paragraph text (about 175 tokens per 60-word paragraph)
    words = ("memory bandwidth decode kernel scheduler prefix cache draft tail verify accept "
             "token session window boundary restore prime chunk tensor stream").split()
    return " ".join(f"{words[(i * 7 + k * 3) % len(words)]}{(i * 31 + k) % 97}" for k in range(n))


P = "Background notes.\n" + "\n".join(para(i, 60) for i in range(15)) + "\n"
X = "Session log.\n" + "\n".join(para(1000 + i, 60) for i in range(3)) + "\n"
Q = "\nQuestion: summarize the notes above in three sentences.\nAnswer:"


def cell(name, ok, why=""):
    line = f"CELL {name} {'PASS' if ok else 'FAIL'} {why}".rstrip()
    print(line, flush=True)
    with open(os.path.join(OUT, "cells.txt"), "a") as f:
        f.write(line + "\n")
    if not ok:
        FAILS.append(name)


def port_free():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


class Server:
    def __init__(self, tag, env, binary):
        if not port_free():
            raise SystemExit(f"port {PORT} already has a listener; refusing to boot {tag}")
        self.log = os.path.join(OUT, f"server-{tag}.log")
        e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        e.update(env)
        self.p = subprocess.Popen([binary], env=e, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"server {tag} exited {self.p.returncode}; see {self.log}")
            if "[server] listening" in self.text():
                try:
                    urllib.request.urlopen(BASE_URL + "/v1/models", timeout=3).read()
                    return
                except Exception:
                    pass
            time.sleep(2)
        self.stop()
        raise SystemExit(f"server {tag} not ready in 900 s")

    def text(self):
        return open(self.log, errors="replace").read()

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()


def completion(body):
    req = urllib.request.Request(BASE_URL + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def streamed(body):
    """-> (text, ttft_s, total_s, usage). TTFT = first chunk carrying text."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE_URL + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    ttft, parts, usage = None, [], None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                if ch.get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(ch["text"])
    return "".join(parts), ttft, time.perf_counter() - t0, usage


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def r3_segment(log):
    # r3's lines: everything after the third admission.
    idx = [m.start() for m in re.finditer(r"\[meter\] admit id=", log)]
    return log[idx[2]:] if len(idx) >= 3 else ""


def run(mode, temp, tag, env, binary):
    s = Server(tag, env, binary)
    try:
        r1 = completion({"model": "q38", "prompt": P + X, "max_tokens": 1, "temperature": 0, "presence_penalty": 1.5})
        r2 = completion({"model": "q38", "prompt": P, "max_tokens": 96, "temperature": 0})
        text, ttft, total, usage = streamed({"model": "q38", "prompt": P + X + Q, "max_tokens": 96,
                                             "temperature": temp, "seed": 1234})
        log = s.text()
    finally:
        s.stop()
    seg = r3_segment(log)
    flags = {
        "tail_insert": "insert (dspark-boundary)" in log,
        "spared_restorable": "reclaim spared the prompt's restorable entry" in seg,
        "spared_plain_only": "reclaim spared the prompt's prefix entry" in seg,
        "dropped_longer": "reclaim dropped the prompt's longer non-restorable entry" in seg,
        "kept_both": "reclaim kept the prompt's restorable and longer entries" in seg,
        "dropped_tail": "reclaim dropped the prompt's restorable entry" in seg,
        "climb_lines": [l for l in seg.splitlines() if "reclaim dropped the prompt" in l or "reclaim kept the prompt" in l],
        "reclaim_evicted": [l for l in seg.splitlines() if "reclaim-on-defer: evicted" in l],
        "restorable_preferred": "restorable-preferred:" in seg,
        "cache_preferred": "cache-preferred:" in seg,
        "dspark_restore": "[prefix-cache] DSPARK restore:" in seg,
        "prefix_hit": [l for l in seg.splitlines() if "[prefix-cache] hit:" in l],
    }
    rec = {"mode": mode, "tag": tag, "binary": binary, "r3_text_sha": sha(text), "r3_len": len(text),
           "r3_ttft_s": ttft, "r3_total_s": total, "r3_usage": usage,
           "prompt_tokens": [(r1.get("usage") or {}).get("prompt_tokens"),
                             (r2.get("usage") or {}).get("prompt_tokens"),
                             (usage or {}).get("prompt_tokens")],
           "flags": flags, "r3_text": text}
    with open(os.path.join(OUT, "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{tag}: ttft {ttft:.3f}s total {total:.3f}s sha {sha(text)} "
          f"{ {k: v for k, v in flags.items() if v and not isinstance(v, list)} }", flush=True)
    return rec


def main():
    summary = {}
    for mode, temp in (("greedy", 0.0), ("sampled", 1.0)):
        ref = run(mode, temp, f"{mode}-ref", REF, BIN_FIX)
        recs = {"base": [], "fix": []}
        for i, arm in enumerate(ORDER):
            recs[arm].append(run(mode, temp, f"{mode}-{i + 1}-{arm}", ARM[arm], BIN_BASE if arm == "base" else BIN_FIX))
        for arm, rs in recs.items():
            f0 = rs[0]["flags"]
            if arm == "base":
                cell(f"{mode}-base-evicts-tail",
                     all(r["flags"]["spared_plain_only"] and r["flags"]["reclaim_evicted"] and r["flags"]["cache_preferred"]
                         and not r["flags"]["dspark_restore"] for r in rs),
                     "; ".join(f0["reclaim_evicted"])[-160:])
            else:
                cell(f"{mode}-fix-spares-tail",
                     all(r["flags"]["spared_restorable"] for r in rs), "")
                cell(f"{mode}-fix-dflash-restore",
                     all(r["flags"]["dspark_restore"] and not r["flags"]["cache_preferred"] for r in rs),
                     "restorable-preferred " + str([r["flags"]["restorable_preferred"] for r in rs])
                     + " dropped-longer " + str([r["flags"]["dropped_longer"] for r in rs])
                     + " kept-both " + str([r["flags"]["kept_both"] for r in rs])
                     + " dropped-tail " + str([r["flags"]["dropped_tail"] for r in rs]))
            same = {r["r3_text_sha"] for r in rs}
            cell(f"{mode}-{arm}-stable", len(same) == 1, f"r3 sha {sorted(same)}")
            if mode == "greedy" or arm == "fix":
                cell(f"{mode}-{arm}-bytes-vs-ref", all(r["r3_text"] == ref["r3_text"] for r in rs),
                     f"ref {ref['r3_text_sha']} arm {sorted(same)}")
            summary[f"{mode}/{arm}"] = {
                "n": len(rs),
                "ttft_s": [round(r["r3_ttft_s"], 4) for r in rs],
                "total_s": [round(r["r3_total_s"], 4) for r in rs],
                "ttft_median_s": round(statistics.median(r["r3_ttft_s"] for r in rs), 4),
                "total_median_s": round(statistics.median(r["r3_total_s"] for r in rs), 4),
                "completion_tokens": [(r["r3_usage"] or {}).get("completion_tokens") for r in rs],
            }
        summary[f"{mode}/ref"] = {"ttft_s": round(ref["r3_ttft_s"], 4), "total_s": round(ref["r3_total_s"], 4),
                                  "sha": ref["r3_text_sha"]}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    sys.exit(1 if FAILS else 0)


main()
