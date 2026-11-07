#!/usr/bin/env python3
"""Serving cells for the 2026-11-06 path-coverage re-validation (RTX 5090). Each scenario boots
memra-server, drives the requests the changed code needs, checks bytes, and prints one
"CELL <name> PASS|FAIL <why>" line per check. Exit 1 when any check fails.

usage: serve_cells.py SCENARIO BIN OUTDIR [BIN_OLD]
scenarios:
  mtp-pool    9B + owntrim draft, default env: spec-cost calibration and wave-probe boot lines,
              solo vs 8-concurrent greedy identity (pooled MTP cohort, riders, cost verdicts at
              c=8), the max_tokens=1 boundary publication. Writes solo texts for plain-ref.
  plain-ref   9B with MEMRA_SERVE_SPEC=0: the same solo prompts; every greedy text must equal the
              spec boot's (the serial burst refactor and the 1-token prime cut against plain).
  wave32      9B, MEMRA_SERVE_SPEC=0, no session cap: 24 distinct greedy prompts solo, then all 24
              concurrent (17..32-row exact waves); again with MEMRA_DECODE_BATCH_CAP=16.
  dflash-demote  Q38 + DFlash2, MEMRA_DSPARK_SPEC=1, MEMRA_SPEC_DEMOTE_AT=40: one seeded sampled
              request twice per boot, two boots; all four texts equal and the demote line names
              the philox event.
  vision-ab   Q38 + MEMRA_VISION_DIR (outside.safetensors): two image requests, greedy, on BIN_OLD
              (a1e9 era) and BIN; texts must be equal.
"""
import base64
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

SCEN, BIN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BIN_OLD = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(OUT, exist_ok=True)
H = "<hf-models>"
Q9 = f"{H}/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
Q9D = f"{H}/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"
Q38 = "<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
DF2 = "<models>/q38-gguf/Qwen3.8-27B-DFlash2-memra"
VIS = os.environ.get("VIS_DIR", "<models>/q38-vision")
ORN = f"{H}/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf"
REPO = "<work>/engine-dsv4-glm"
PORT = 18291
BASE = f"http://127.0.0.1:{PORT}"
FAILS = []
TOPICS = ["why the sky is blue", "how a hash map handles collisions", "the causes of the French Revolution",
          "how vaccines train the immune system", "what a compiler's register allocator does", "how tides work",
          "the difference between TCP and UDP", "how photosynthesis stores energy", "why prime numbers matter in cryptography",
          "how a bicycle stays upright", "what causes inflation", "how GPS finds a position", "how bread rises",
          "the life cycle of a star", "how a transistor switches", "why ice floats", "how noise-cancelling headphones work",
          "what a database index is", "how bees communicate", "how a refrigerator moves heat", "why leaves change color",
          "how an electric motor turns", "what entropy means", "how a search engine ranks pages"]


def prompt(i, words=""):
    return f"Explain {TOPICS[i % len(TOPICS)]} in a few short paragraphs.{words}"


def cell(name, ok, why=""):
    line = f"CELL {name} {'PASS' if ok else 'FAIL'} {why}".rstrip()
    print(line, flush=True)
    with open(os.path.join(OUT, "cells.txt"), "a") as f:
        f.write(line + "\n")
    if not ok:
        FAILS.append(name)


class Server:
    def __init__(self, tag, env, binary=None):
        self.tag, self.log = tag, os.path.join(OUT, f"server-{tag}.log")
        e = dict(os.environ)
        e.update({"MEMRA_COMPAT": "openai", "MEMRA_ADDR": f"127.0.0.1:{PORT}"})
        e.update(env)
        self.p = subprocess.Popen([binary or BIN], env=e, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"server {tag} exited {self.p.returncode}; see {self.log}")
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=3).read()
                return
            except Exception:
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


def chat(model, content, max_tokens=128, temperature=0.0, seed=None, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def text_of(resp):
    m = resp["choices"][0]["message"]
    return (m.get("reasoning") or "") + "\x1f" + (m.get("content") or "")


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def concurrent(model, prompts, max_tokens=128):
    with cf.ThreadPoolExecutor(len(prompts)) as ex:
        futs = [ex.submit(chat, model, p, max_tokens) for p in prompts]
        return [f.result() for f in futs]


def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1)


def mtp_pool():
    s = Server("mtp-pool", {"MEMRA_MODELS": f"q9={Q9}+{Q9D}", "MEMRA_CTX": "8192"})
    try:
        log = s.text()
        cell("spec-cost-boot", bool(re.search(r"\[spec-cost\] model=\"?q9", log)) and "calibration FAILED" not in log,
             "; ".join(l[:160] for l in log.splitlines() if "[spec-cost]" in l)[:400])
        cell("wave-probe-boot", "[wave-probe]" in log and "[wave-probe] failed" not in log,
             "; ".join(l[:160] for l in log.splitlines() if "[wave-probe]" in l)[:300])
        solo = {i: text_of(chat("q9", prompt(i))) for i in range(4)}
        save("mtp-solo.json", {str(i): t for i, t in solo.items()})
        idx = [0, 0, 0, 0, 1, 2, 3, 1]
        resps = concurrent("q9", [prompt(i) for i in idx])
        bad = [k for k, (i, r) in enumerate(zip(idx, resps)) if text_of(r) != solo[i]]
        save("mtp-c8.json", [{"prompt": i, "sha": sha(text_of(r)), "usage": r.get("usage")} for i, r in zip(idx, resps)])
        log = s.text()
        cell("mtp-c8-identity", not bad, f"mismatched rows {bad}" if bad else "8 concurrent greedy rows equal their solo twins")
        cell("mtp-pool-engaged", "pooled MTP cohort engaged" in log, "")
        cell("mtp-cost-verdicts", "source=cost-model" in log,
             "; ".join(sorted({l.strip()[:140] for l in log.splitlines() if "[spec-k]" in l}))[:400])
        long = prompt(5, " " + " ".join(f"Point {k}: keep the answer concrete and use plain words." for k in range(12)))
        r1 = chat("q9", long, max_tokens=1)
        r2 = chat("q9", long, max_tokens=48)
        cached = ((r2.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        save("mtp-drain.json", {"r1": r1.get("usage"), "r2": r2.get("usage"), "r2_sha": sha(text_of(r2))})
        open(os.path.join(OUT, "drain-text.txt"), "w").write(text_of(r2))
        cell("mtp-prime-terminated-publish", cached > 0, f"cached_tokens={cached}")
    finally:
        s.stop()


def plain_ref():
    solo = json.load(open(os.path.join(OUT, "mtp-solo.json")))
    s = Server("plain-ref", {"MEMRA_MODELS": f"q9={Q9}", "MEMRA_CTX": "8192", "MEMRA_SERVE_SPEC": "0"})
    try:
        bad = [i for i in range(4) if text_of(chat("q9", prompt(i))) != solo[str(i)]]
        cell("mtp-vs-plain", not bad, f"spec != plain for prompts {bad}" if bad else "4 spec greedy texts equal plain")
        long = prompt(5, " " + " ".join(f"Point {k}: keep the answer concrete and use plain words." for k in range(12)))
        t = text_of(chat("q9", long, max_tokens=48))
        cell("drain-vs-plain", t == open(os.path.join(OUT, "drain-text.txt")).read(), "restored-boundary text vs a fresh plain boot")
    finally:
        s.stop()


def wave32():
    for tag, extra in (("wave32", {}), ("wave16", {"MEMRA_DECODE_BATCH_CAP": "16"})):
        s = Server(tag, {"MEMRA_MODELS": f"q9={Q9}", "MEMRA_CTX": "4096", "MEMRA_SERVE_SPEC": "0", **extra})
        try:
            solo = [text_of(chat("q9", prompt(i))) for i in range(24)]
            conc = concurrent("q9", [prompt(i) for i in range(24)])
            bad = [i for i, r in enumerate(conc) if text_of(r) != solo[i]]
            log = s.text()
            cell(f"{tag}-c24-identity", not bad, (f"mismatched {bad}; " if bad else "24 rows equal solo; ")
                 + "; ".join(l.strip()[:160] for l in log.splitlines() if "[wave-probe]" in l)[:300])
            save(f"{tag}.json", [sha(t) for t in solo])
        finally:
            s.stop()
    a = json.load(open(os.path.join(OUT, "wave32.json")))
    b = json.load(open(os.path.join(OUT, "wave16.json")))
    cell("wave32-vs-wave16-solo", a == b, "solo texts equal across the two caps")


def dflash_demote():
    texts = []
    # MEMRA_MAX_SESSIONS=4 as in tools/dspark-serve-smoke.sh: with the default cap the 5090's
    # boot calibration charges a 4.8 GB transient floor and refuses every request (floor-ab).
    env = {"MEMRA_MODELS": f"q38={Q38}", "MEMRA_CTX": "8192", "MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": DF2,
           "MEMRA_SPEC_DEMOTE_AT": "40", "MEMRA_ADMIT_YIELD": "0", "MEMRA_MAX_SESSIONS": "4"}
    for boot in ("a", "b"):
        s = Server(f"dflash-demote-{boot}", env)
        try:
            for _ in range(2):
                texts.append(text_of(chat("q38", prompt(7), max_tokens=200, temperature=0.8, seed=4242)))
            lines = [l.strip()[:200] for l in s.text().splitlines() if "demot" in l and "philox" in l]
            cell(f"dflash-demote-line-{boot}", bool(lines), "; ".join(lines[:2]))
        finally:
            s.stop()
    cell("dflash-demote-reproducible", len(set(texts)) == 1, f"{len(set(texts))} distinct texts over 2 boots x 2 requests")


def vision_ab():
    imgs = [f"{REPO}/research/image-url-fetch-20261002/raw/qwen-attempt1/off/red.png",
            f"{REPO}/research/glm5-vision-20260830/fixtures/text448.png"]
    out = {}
    for tag, binary in (("old", BIN_OLD), ("tip", BIN)):
        s = Server(f"vision-{tag}", {"MEMRA_MODELS": f"q38={Q38}", "MEMRA_CTX": "8192", "MEMRA_VISION_DIR": VIS}, binary)
        try:
            out[tag] = []
            for p in imgs:
                b64 = base64.b64encode(open(p, "rb").read()).decode()
                body = {"model": "q38", "max_tokens": 64, "temperature": 0, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
                req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    out[tag].append(text_of(json.loads(r.read())))
        finally:
            s.stop()
    save("vision.json", out)
    cell("vision-old-vs-tip", out["old"] == out["tip"] and all(len(t) > 2 for t in out["tip"]),
         " | ".join(t.replace("\x1f", " ")[:80] for t in out["tip"]))


def mtp_cohort():
    # c=8 is far concurrency (SPEC_COST_FAR_N=7): the cost model serves it plain, so the pooled
    # MTP cohort needs waves under 7 rows. Default env first (cost-model verdicts), then K pinned.
    solo = json.load(open(os.path.join(OUT, "mtp-solo.json")))
    for tag, extra, waves in (("cost", {}, ([0, 1], [0, 1, 2, 3], [0, 1, 2, 3])),
                              ("pinned", {"MEMRA_SPEC_K": "3"}, ([0, 1, 2, 3], [0, 1, 2, 3, 0, 1]))):
        s = Server(f"mtp-cohort-{tag}", {"MEMRA_MODELS": f"q9={Q9}+{Q9D}", "MEMRA_CTX": "8192", **extra})
        try:
            for w, idx in enumerate(waves):
                resps = concurrent("q9", [prompt(i) for i in idx])
                bad = [k for k, (i, r) in enumerate(zip(idx, resps)) if text_of(r) != solo[str(i)]]
                cell(f"mtp-cohort-{tag}-c{len(idx)}-w{w}-identity", not bad,
                     f"mismatched rows {bad}" if bad else f"{len(idx)} concurrent rows equal solo")
            log = s.text()
            eng = sorted({l.strip()[:120] for l in log.splitlines() if "pooled MTP cohort engaged" in l})
            ks = sorted({m.group(0) for m in re.finditer(r"K=\d+ source=[a-z-]+(?= prompt=)", log)})
            cell(f"mtp-cohort-{tag}-engaged", bool(eng), "; ".join(eng)[:240] + " | verdicts " + ",".join(ks))
        finally:
            s.stop()


def floor_ab():
    # Q38 + DFlash2 on the 5090 at the default session cap: boot calibration line and whether a
    # 329-token request admits, on BIN_OLD (a1e9 era) and BIN.
    env = {"MEMRA_MODELS": f"q38={Q38}", "MEMRA_CTX": "8192", "MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": DF2}
    for tag, binary, extra in (("old", BIN_OLD, {}), ("tip", BIN, {}), ("tip-s4", BIN, {"MEMRA_MAX_SESSIONS": "4"})):
        s = Server(f"floor-{tag}", {**env, **extra}, binary)
        try:
            try:
                r = chat("q38", prompt(7), max_tokens=32)
                got = f"admitted ({len(text_of(r))} chars)"
            except urllib.error.HTTPError as e:
                got = f"HTTP {e.code}: {e.read()[:160]!r}"
            log = s.text()
            fl = [m.group(0) for m in re.finditer(r"transient floor \d+MB[^;]*; measured \d+MB[^\]]*\]", log)]
            caps = sorted({l.strip()[:140] for l in log.splitlines() if "max_sessions" in l or "session cap" in l})
            cell(f"floor-{tag}", got.startswith("admitted"), f"{got} | {fl[0][:200] if fl else 'no floor line'} | {'; '.join(caps)[:200]}")
        finally:
            s.stop()


def rider_body(c, j):
    # research/batch-spec-20261101/battery/mixed_cell.py's device-sampled kinds r0..r5
    topics = ["a lighthouse keeper", "a clockmaker's apprentice", "a desert caravan", "a deep-sea diver",
              "a mountain observatory", "a river ferry"]
    b = {"model": "q9", "messages": [{"role": "user", "content":
         f"Wave {c}, writer {j}: write a vivid short story about {topics[j]}, then list three lessons from it."}],
         "max_tokens": 200, "stream": False, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    seed = 2000 + 17 * j + c
    if j in (1, 3):
        b.update(temperature=0.8, seed=seed)
    elif j == 4:
        b.update(temperature=0.7, top_k=40, top_p=0.9, seed=seed)
    elif j == 5:
        b.update(temperature=0.8, presence_penalty=1.5, seed=seed)
    return b


def post_body(b):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def post_json(path, b):
    try:
        req = urllib.request.Request(BASE + path, data=json.dumps(b).encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:  # tokenize is informational
        return {"error": str(e)}


def riders():
    # The unified step (plain device-sampled rows riding a pooled MTP cohort) needs the cost
    # model (its table sets the rider caps; mixed_cell.py's band gate registers none) and plain
    # rows beside spec rows: MEMRA_SPEC_DEMOTE_AT=40 demotes each spec session after 40 tokens,
    # so staggered arrivals put demoted rows next to sessions still in spec. Every greedy row must
    # equal its solo twin; sampled rows are reported (their spec prefix may take another K).
    s = Server("riders", {"MEMRA_MODELS": f"q9={Q9}+{Q9D}", "MEMRA_CTX": "4096", "MEMRA_SPEC_DEMOTE_AT": "40",
                          "MEMRA_ADMIT_YIELD": "0"})
    try:
        waves = (3, 4, 6)
        solo = {(c, j): text_of(post_body(rider_body(c, j))) for c in waves for j in range(c)}
        conc = {}
        for c in waves:
            res, ths = {}, []
            for j in range(c):
                t = threading.Thread(target=lambda j=j: res.__setitem__(j, post_body(rider_body(c, j))))
                t.start()
                ths.append(t)
                time.sleep(0.25)
            for t in ths:
                t.join()
            for j in range(c):
                conc[(c, j)] = text_of(res[j])
        greedy = [(c, j) for c in waves for j in (0, 2) if j < c]
        bad = [k for k in greedy if conc[k] != solo[k]]
        samp = [k for k in solo if k not in greedy]
        same = [k for k in samp if conc[k] == solo[k]]
        save("riders.json", {f"c{c}/r{j}": {"solo": sha(solo[(c, j)]), "conc": sha(conc[(c, j)])} for c, j in solo})
        log = s.text()
        eng = sorted({l.strip()[:150] for l in log.splitlines() if "unified step engaged" in l})
        dem = sum(1 for l in log.splitlines() if "demoted" in l and "FORCED" in l)
        cell("riders-engaged", bool(eng), "; ".join(eng)[:300] + f" | forced demotions {dem}")
        cell("riders-greedy-identity", not bad, f"mismatched {bad}" if bad else f"{len(greedy)} greedy rows equal solo")
        print(f"INFO sampled rows equal to solo: {len(same)}/{len(samp)}", flush=True)
        save("riders-texts.json", {f"c{c}/r{j}": {"solo": solo[(c, j)], "conc": conc[(c, j)]} for c, j in solo})
        for k in samp:
            a, b = solo[k], conc[k]
            n = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            ids = post_json("/v1/tokenize", {"model": "q9", "prompt": a[:n]}) if n else {}
            toks = len(ids.get("tokens", ids.get("ids", []))) if isinstance(ids, dict) else -1
            print(f"INFO c{k[0]}/r{k[1]} first difference at char {n} (about {toks} tokens) of {len(a)}/{len(b)}", flush=True)
    finally:
        s.stop()


MIX_TOPICS = ["a lighthouse keeper", "a clockmaker's apprentice", "a desert caravan", "a deep-sea diver",
              "a mountain observatory", "a river ferry", "a city of bridges", "an orbital greenhouse"]


def mix_body(c, j, max_tokens=200):
    # research/batch-spec-20261101/battery/mixed_cell.py's body(), kinds r0..r6
    b = {"model": "q9", "messages": [{"role": "user", "content":
         f"Wave {c}, writer {j}: write a vivid short story about {MIX_TOPICS[j % 8]}, then list three lessons from it."}],
         "max_tokens": max_tokens, "stream": False, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    seed = 2000 + 17 * j + c
    if j in (1, 3):
        b.update(temperature=0.8, seed=seed)
    elif j == 4:
        b.update(temperature=0.7, top_k=40, top_p=0.9, seed=seed)
    elif j == 5:
        b.update(temperature=0.8, presence_penalty=1.5, seed=seed)
    elif j == 6:
        b.update(frequency_penalty=0.5)
    return b


def mix_text(r):
    m = r["choices"][0]["message"]
    return (m.get("reasoning") or m.get("reasoning_content") or "") + "\0" + (m.get("content") or "")


def staggered(bodies, gaps):
    """Post bodies[i] after gaps[i] seconds from the previous start; return responses in order."""
    res, ths = {}, []
    for i, b in enumerate(bodies):
        if gaps[i]:
            time.sleep(gaps[i])
        t = threading.Thread(target=lambda i=i, b=b: res.__setitem__(i, post_body(b)))
        t.start()
        ths.append(t)
    for t in ths:
        t.join()
    return [res[i] for i in range(len(bodies))]


def lines_with(log, *needles):
    return [l.strip()[:200] for l in log.splitlines() if all(n in l for n in needles)]


Q9_SPEC = {"MEMRA_MODELS": f"{'q9'}={Q9}+{Q9D}"}
Q9_PLAIN = {"MEMRA_MODELS": f"q9={Q9}", "MEMRA_SERVE_SPEC": "0"}


def leftbehind():
    # A6: a host-sampled plain row (frequency penalty) live beside a pooled MTP cohort is left
    # behind: no unified step forms in that boot. A control boot without it does form one.
    band = {**Q9_SPEC, "MEMRA_CTX": "4096", "MEMRA_SPEC_GATE_MODEL": "0", "MEMRA_SPEC_GATE_LOW": "2",
            "MEMRA_SPEC_GATE_HIGH": "16", "MEMRA_ADMIT_YIELD": "0"}
    bodies = {j: mix_body(3, j, 600 if j == 6 else 200) for j in range(7)}
    got = {}
    for tag, order in (("with-r6", [0, 1, 6, 2, 3, 4, 5]), ("control", [0, 1, 2, 3, 4, 5])):
        s = Server(f"leftbehind-{tag}", band)
        try:
            solo = {j: mix_text(post_body(bodies[j])) for j in (0, 2)}
            res = staggered([bodies[j] for j in order], [0] + [0.25] * (len(order) - 1))
            out = {j: mix_text(r) for j, r in zip(order, res)}
            log = s.text()
            eng = lines_with(log, "pooled MTP cohort engaged: 2 sessions")
            uni = lines_with(log, "unified step engaged")
            if tag == "with-r6":
                cell("leftbehind-no-unified", bool(eng) and not uni, f"engaged={len(eng)} unified={len(uni)} {'; '.join(uni)[:160]}")
            else:
                cell("leftbehind-control-unified", bool(uni), "; ".join(uni)[:200])
            bad = [j for j in (0, 2) if out[j] != solo[j]]
            cell(f"leftbehind-{tag}-greedy-solo", not bad, f"mismatched {bad}" if bad else "r0 r2 equal solo")
            got[tag] = out
        finally:
            s.stop()
    s = Server("leftbehind-plain", {**Q9_PLAIN, "MEMRA_CTX": "4096"})
    try:
        for j in (2, 3, 4, 5, 6):
            p = mix_text(post_body(bodies[j]))
            cell(f"leftbehind-r{j}-vs-plain", got["with-r6"][j] == p, f"{sha(got['with-r6'][j])} vs {sha(p)}")
    finally:
        s.stop()


def pinned_sampled():
    # A6(2) + C4(2): under MEMRA_SPEC_K=3 (calibration skipped, so each cohort size prints its arms)
    # sampled cohorts (GraphSampled) and penalised ones (Eager) give rows byte-equal to solo, and the
    # 4-row sampled cohort is reproducible across two boots.
    def body(i, pen=False):
        b = {"model": "q9", "messages": [{"role": "user", "content": prompt(i)}], "max_tokens": 128,
             "stream": False, "temperature": 0.8, "top_p": 0.9, "seed": 11 + i}
        if pen:
            b["presence_penalty"] = 0.5
        return b
    waves = {}
    for boot in ("a", "b"):
        s = Server(f"pinned-sampled-{boot}", {**Q9_SPEC, "MEMRA_CTX": "8192", "MEMRA_SPEC_K": "3", "MEMRA_ADMIT_YIELD": "0"})
        try:
            for wave, n, pen in (("A", 4, False), ("B", 3, True)):
                bodies = [body(i, pen) for i in range(n)]
                solo = [text_of(post_body(b)) for b in bodies]
                conc = [text_of(r) for r in staggered(bodies, [0] * n)]
                bad = [i for i in range(n) if conc[i] != solo[i]]
                cell(f"pinned-sampled-{boot}-{wave}-identity", not bad, f"mismatched {bad}" if bad else f"{n} rows equal solo")
                waves[(boot, wave)] = conc
            log = s.text()
            arms = lines_with(log, "pooled rows draft engaged")
            cell(f"pinned-sampled-{boot}-arms-sampled", any("4 members" in l and "GraphSampled" in l for l in arms), "; ".join(arms)[:240])
            cell(f"pinned-sampled-{boot}-arms-eager", any("3 members" in l and "Eager" in l for l in arms), "")
            cell(f"pinned-sampled-{boot}-cohort4", bool(lines_with(log, "pooled MTP cohort engaged: 4 sessions")), "")
        finally:
            s.stop()
    cell("pinned-sampled-A-reproducible", waves[("a", "A")] == waves[("b", "A")], "4-row sampled cohort equal across two boots")


def bandemote():
    # A8(1): the band's HIGH demotion with a sampled row hands it to plain draws at its philox event.
    env = {**Q9_SPEC, "MEMRA_CTX": "4096", "MEMRA_SPEC_GATE_MODEL": "0", "MEMRA_SPEC_GATE_LOW": "2",
           "MEMRA_SPEC_GATE_HIGH": "3", "MEMRA_ADMIT_YIELD": "0"}
    r0, r2 = mix_body(0, 0), mix_body(0, 2)
    r1 = mix_body(0, 1)  # T=0.8, seed 2017
    s = Server("bandemote", env)
    try:
        solo = {0: mix_text(post_body(r0)), 2: mix_text(post_body(r2))}
        res = staggered([r0, r1, r2], [0, 0.25, 2.0])
        log = s.text()
        dem = lines_with(log, "demoted session to batched decode", "HIGH=3")
        cell("bandemote-greedy-line", any("greedy" in l for l in dem), "; ".join(dem)[:300])
        cell("bandemote-sampled-philox", any("sampled, plain draws from philox event" in l for l in dem), "")
        bad = [j for j, r in ((0, res[0]), (2, res[2])) if mix_text(r) != solo[j]]
        cell("bandemote-greedy-solo", not bad, f"mismatched {bad}" if bad else "r0 r2 equal solo")
    finally:
        s.stop()


def dflash_high():
    # A8(2): DFlash rows demote at HIGH=4 (no cost table for DSpark models) with the philox handoff.
    env = {"MEMRA_MODELS": f"q38={Q38}", "MEMRA_CTX": "8192", "MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": DF2,
           "MEMRA_MAX_SESSIONS": "4", "MEMRA_ADMIT_YIELD": "0"}
    def b(i, temp=0.0, seed=None):
        x = {"model": "q38", "messages": [{"role": "user", "content": prompt(i)}], "max_tokens": 200, "stream": False,
             "temperature": temp}
        if seed is not None:
            x["seed"] = seed
        return x
    bodies = [b(7, 0.8, 4242), b(1), b(2), b(3)]
    s = Server("dflash-high", env)
    try:
        solo = {i: text_of(post_body(bodies[i])) for i in (1, 2, 3)}
        res = staggered(bodies, [0, 0.25, 2.0, 0.25])
        log = s.text()
        dem = lines_with(log, "demoted dspark session to batched decode", "4 active")
        cell("dflash-high-sampled-line", any("sampled, plain draws from philox event" in l for l in dem), "; ".join(dem)[:300])
        cell("dflash-high-greedy-line", any("greedy" in l for l in dem), "")
        bad = [i for i in (1, 2, 3) if text_of(res[i]) != solo[i]]
        cell("dflash-high-greedy-solo", not bad, f"mismatched {bad}" if bad else "r1..r3 equal solo")
    finally:
        s.stop()


def http_status(body, timeout=120):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def shed():
    # B11: the [shed] lines and 429 codes for the choice-group sheds and the body-admission pool.
    import socket
    base = {**Q9_PLAIN, "MEMRA_CTX": "4096"}
    nreq = {"model": "q9", "n": 2, "max_tokens": 16, "temperature": 0.7,
            "messages": [{"role": "user", "content": "Name a color."}]}
    s = Server("shed-queue", {**base, "MEMRA_MAX_QUEUE_DEPTH": "1"})
    try:
        code, text = http_status(nreq)
        cell("shed-queue-429", code == 429 and "shed_queue" in text, f"{code} {text[:120]}")
        cell("shed-queue-line", bool(lines_with(s.text(), "[shed] shed_queue", "choices=2")), "; ".join(lines_with(s.text(), "[shed]"))[:200])
        # body_admission_busy: hold the large body pool (4) with chunked bodies kept open
        socks = []
        for _ in range(4):
            k = socket.create_connection(("127.0.0.1", PORT), timeout=30)
            k.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                      b"Transfer-Encoding: chunked\r\n\r\n1\r\n{\r\n")
            socks.append(k)
        time.sleep(1.0)
        k5 = socket.create_connection(("127.0.0.1", PORT), timeout=30)
        k5.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                   b"Transfer-Encoding: chunked\r\n\r\n1\r\n{\r\n")
        k5.settimeout(15)
        try:
            head = k5.recv(4096).decode(errors="replace")
        except socket.timeout:
            head = "<no response in 15 s>"
        for k in socks + [k5]:
            k.close()
        cell("shed-body-busy-429", " 429 " in head.split("\r\n")[0] + " " and "body_admission_busy" in head, head.split("\r\n")[0][:80])
        time.sleep(1.0)
        cell("shed-body-busy-line", bool(lines_with(s.text(), "[shed] body_admission_busy")), "; ".join(lines_with(s.text(), "body_admission_busy"))[:200])
    finally:
        s.stop()
    for tag, extra, needle in (("wait", {"MEMRA_QUEUE_WAIT_CEILING_S": "1"}, "shed_queue_wait"),
                               ("deadline", {"MEMRA_RL_RESET_S": "120"}, "shed_deadline")):
        s = Server(f"shed-{tag}", {**base, "MEMRA_MAX_SESSIONS": "2", **extra})
        try:
            long = {"model": "q9", "max_tokens": 1500, "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt(3)}]}
            t = threading.Thread(target=lambda: http_status(long, timeout=600))
            t.start()
            time.sleep(3.0)
            code, text = http_status(nreq)
            cell(f"shed-{tag}-429", code == 429, f"{code} {text[:120]}")
            cell(f"shed-{tag}-line", bool(lines_with(s.text(), f"[shed] {needle}", "choices=2")), "; ".join(lines_with(s.text(), "[shed]"))[:200])
            t.join(timeout=600)
        finally:
            s.stop()


def restore_sampled():
    # A17(b): a sampled spec restore is verdicted by the cost model: K>0 solo (restore, same text),
    # K=0 under far concurrency (the guard refuses and the text equals a plain boot's).
    P = prompt(4, " " + " ".join(f"Note {k}: describe one more effect of the moon on daily life." for k in range(16)))
    body = {"model": "q9", "messages": [{"role": "user", "content": P}], "max_tokens": 48, "stream": False,
            "temperature": 0.8, "seed": 7}
    s = Server("restore-sampled", {**Q9_SPEC, "MEMRA_CTX": "8192"})
    try:
        t1 = text_of(post_body(body))
        t2 = text_of(post_body(body))
        log = s.text()
        cell("restore-sampled-solo-restores", bool(lines_with(log, "[prefix-cache] spec restore:")) and t1 == t2,
             f"equal={t1 == t2} {'; '.join(lines_with(log, 'rung=2'))[:160]}")
        hold = [{"model": "q9", "messages": [{"role": "user", "content": prompt(10 + i)}], "max_tokens": 512,
                 "stream": False, "temperature": 0.0} for i in range(8)]
        ths = [threading.Thread(target=lambda b=b: post_body(b)) for b in hold]
        for t in ths:
            t.start()
        time.sleep(4.0)
        t3 = text_of(post_body(body))
        for t in ths:
            t.join()
        guard = lines_with(s.text(), "[spec-restore-guard] sampled restore REFUSED")
        cell("restore-sampled-guard-refuses", bool(guard), "; ".join(guard)[:200])
    finally:
        s.stop()
    s = Server("restore-sampled-plain", {**Q9_PLAIN, "MEMRA_CTX": "8192"})
    try:
        p = text_of(post_body(body))
        cell("restore-sampled-refused-equals-plain", t3 == p, f"{sha(t3)} vs {sha(p)}")
    finally:
        s.stop()


def calib_nopool():
    # C9: the spec-cost calibration's non-pooled arm (pooled=false, pooled_overhead=None).
    for tag, models, extra in (("q9-verify-graph", f"q9={Q9}+{Q9D}", {"MEMRA_SPEC_VERIFY_GRAPH": "1"}),
                               ("ornith", f"orn={ORN}", {})):
        s = Server(f"calib-{tag}", {"MEMRA_MODELS": models, "MEMRA_CTX": "4096", **extra})
        try:
            name = models.split("=")[0]
            post_body({"model": name, "messages": [{"role": "user", "content": prompt(0)}], "max_tokens": 16, "temperature": 0.0})
            log = s.text()
            cal = lines_with(log, "[spec-cost] model=", "depth=")
            cell(f"calib-{tag}-nopool", any("pooled=false" in l and "pooled_overhead=None" in l for l in cal)
                 and "calibration FAILED" not in log, "; ".join(cal)[:240])
            cell(f"calib-{tag}-cost-verdict", bool(lines_with(log, "[spec-k]", "source=cost-model")), "")
        finally:
            s.stop()


def pinned_stagger():
    # The sampled cohort under K=3 with a fixed arrival order (0.5 s apart): row 0 primes with no
    # peer and must equal solo; rows 1.. prime with a peer live (the 1-token prime cut moves one
    # boundary draw by design), so they are checked for boot-to-boot reproducibility instead.
    def body(i):
        return {"model": "q9", "messages": [{"role": "user", "content": prompt(i)}], "max_tokens": 128,
                "stream": False, "temperature": 0.8, "top_p": 0.9, "seed": 11 + i}
    runs = []
    for boot in ("a", "b"):
        s = Server(f"pinned-stagger-{boot}", {**Q9_SPEC, "MEMRA_CTX": "8192", "MEMRA_SPEC_K": "3", "MEMRA_ADMIT_YIELD": "0"})
        try:
            solo0 = text_of(post_body(body(0)))
            conc = [text_of(r) for r in staggered([body(i) for i in range(4)], [0, 0.5, 0.5, 0.5])]
            cell(f"pinned-stagger-{boot}-row0-solo", conc[0] == solo0, f"{sha(conc[0])} vs {sha(solo0)}")
            cell(f"pinned-stagger-{boot}-cohort", bool(lines_with(s.text(), "pooled MTP cohort engaged")), "")
            runs.append(conc)
        finally:
            s.stop()
    diff = [i for i in range(4) if runs[0][i] != runs[1][i]]
    cell("pinned-stagger-reproducible", not diff, f"rows differing across boots {diff}" if diff else "4 rows equal across two boots")


def orn_boot_ab():
    # A default boot of Ornith-1.5 (qwen35_moe with an embedded MTP head, decode cap 8): does the
    # spec-cost calibration ask decode_step_batch for B=16 and panic? Old binary vs tip.
    env = {"MEMRA_MODELS": f"orn={ORN}", "MEMRA_CTX": "4096"}
    for tag, binary in (("old", BIN_OLD), ("tip", BIN)):
        try:
            s = Server(f"orn-boot-{tag}", env, binary)
        except SystemExit as e:
            log = open(os.path.join(OUT, f"server-orn-boot-{tag}.log"), errors="replace").read()
            pan = [l.strip()[:200] for l in log.splitlines() if "panicked" in l or "decode_step_batch: B=" in l]
            cell(f"orn-boot-{tag}", False, f"{e} :: {' | '.join(pan[:2])}")
            continue
        try:
            r = post_body({"model": "orn", "messages": [{"role": "user", "content": prompt(0)}], "max_tokens": 16,
                           "temperature": 0.0})
            cal = lines_with(s.text(), "[spec-cost] model=")
            cell(f"orn-boot-{tag}", True, f"served {len(text_of(r))} chars :: {'; '.join(cal)[:200]}")
        finally:
            s.stop()


def riders_plain_ref():
    # Rows the concurrent riders run admitted at K=0 (plain from the first token, riding the cohort)
    # must equal the same request on a plain boot: same plain program, same Philox stream.
    texts = json.load(open(os.path.join(OUT, "riders-texts.json")))
    keys = os.environ.get("RIDER_KEYS", "c6/r4,c6/r5,c3/r2").split(",")
    s = Server("riders-plain", {"MEMRA_MODELS": f"q9={Q9}", "MEMRA_CTX": "4096", "MEMRA_SERVE_SPEC": "0"})
    try:
        for k in keys:
            c, j = int(k.split("/")[0][1:]), int(k.split("/")[1][1:])
            t = text_of(post_body(rider_body(c, j)))
            cell(f"rider-plain-{k.replace('/', '-')}", t == texts[k]["conc"], f"rider {sha(texts[k]['conc'])} plain {sha(t)}")
    finally:
        s.stop()


def para(i, n):
    # tools/dspark-cold-drafter-gate.py's paragraph text (about 175 tokens per 60-word paragraph)
    words = ("memory bandwidth decode kernel scheduler prefix cache draft tail verify accept "
             "token session window boundary restore prime chunk tensor stream").split()
    return " ".join(f"{words[(i * 7 + k * 3) % len(words)]}{(i * 31 + k) % 97}" for k in range(n))


def completion(b):
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(b).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def restorable():
    # A2/A17(a): the DFlash restorable-hit preference. r1 = P+X served plain (presence_penalty on a
    # greedy request, max_tokens 1) publishes a plain entry near |P+X|; r2 = P greedy cold-primes
    # DFlash and the boundary capture publishes a tail-carrying entry at or below |P|; r3 =
    # P+X+question then has a longer plain hit and a shorter restorable hit, and with
    # decode_budget >= (plain - restorable) / 8 the worker swaps to the restorable one. One fresh
    # boot per mode (greedy, then T=1 seed 1234) for the arm and for a PREFIX_CACHE_MB=0 reference.
    P = "Background notes.\n" + "\n".join(para(i, 60) for i in range(15)) + "\n"
    X = "Session log.\n" + "\n".join(para(1000 + i, 60) for i in range(3)) + "\n"
    q = "\nQuestion: summarize the notes above in three sentences.\nAnswer:"
    # RCTX: 8192 left too little device headroom on a 24 GB card: r3's reclaim-on-defer evicted the
    # restorable tail entry (it spares only the prompt's longest-LCP entry, the plain hit).
    base = {"MEMRA_MODELS": f"q38={Q38}", "MEMRA_CTX": os.environ.get("RCTX", "8192"), "MEMRA_MAX_SESSIONS": "1", "MEMRA_KV_HOST_MB": "0",
            "MEMRA_MTP_SKIP": "1", "MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": DF2,
            "MEMRA_FRSPEC_TRIM": "<models>/q38-gguf/q38-ranks-sxc32768.gguf.txt"}
    arm = dict(base, MEMRA_PREFIX_CACHE_MB="4000", MEMRA_PREFIX_STABLE_BOUNDARY="1", MEMRA_DSPARK_PREFIX_RESTORE="1",
               MEMRA_DSPARK_PARTIAL_RESTORE="1", MEMRA_DSPARK_BOUNDARY_CAPTURE="1")
    ref = dict(base, MEMRA_PREFIX_CACHE_MB="0")
    out = {}
    for mode, temp in (("greedy", 0.0), ("sampled", 1.0)):
        for tag, env in (("arm", arm), ("ref", ref)):
            s = Server(f"restorable-{mode}-{tag}", env)
            try:
                r1 = completion({"model": "q38", "prompt": P + X, "max_tokens": 1, "temperature": 0, "presence_penalty": 1.5})
                r2 = completion({"model": "q38", "prompt": P, "max_tokens": 96, "temperature": 0})
                r3 = completion({"model": "q38", "prompt": P + X + q, "max_tokens": 96, "temperature": temp, "seed": 1234})
                out[f"{mode}/{tag}"] = {"r3": r3["choices"][0]["text"],
                                        "tokens": [(r.get("usage") or {}).get("prompt_tokens") for r in (r1, r2, r3)]}
                log = s.text()
            finally:
                s.stop()
            if tag == "arm":
                pref = [l for l in log.splitlines() if re.search(r"restorable-preferred: prompt=\d+ decode_budget=\d+ plain hit +\d+ -> restorable hit \d+", l)]
                bnd = lines_with(log, "insert (dspark-boundary)")
                rest = lines_with(log, "DSPARK restore:")
                cell(f"restorable-{mode}-boundary", bool(bnd), (bnd[:1] or ["no dspark-boundary insert"])[0][-160:])
                cell(f"restorable-{mode}-preferred", bool(pref), (pref[:1] or ["no restorable-preferred line"])[0][-200:])
                cell(f"restorable-{mode}-restore", bool(pref) and bool(rest), (rest[-1:] or ["no DSPARK restore"])[0][-160:])
                out[f"{mode}/arm-restored"] = bool(pref) and bool(rest)
                ev = lines_with(log, "reclaim-on-defer: evicted")
                if ev:
                    cell(f"restorable-{mode}-reclaim-note", True, ev[-1][-160:])
        a, b = out[f"{mode}/arm"], out[f"{mode}/ref"]
        why = f"tokens {a['tokens']} arm {sha(a['r3'])} ref {sha(b['r3'])} len {len(a['r3'])}/{len(b['r3'])}"
        if mode == "sampled" and not out.get(f"{mode}/arm-restored"):
            # the reference cold-primes DFlash; an arm that served r3 plain draws a different sampled
            # program, so only the greedy rows are comparable then
            cell(f"restorable-{mode}-bytes-not-comparable", True, "arm r3 did not take the DFlash restore; " + why)
        else:
            cell(f"restorable-{mode}-bytes", a["r3"] == b["r3"], why)
    save("restorable.json", out)


{"restorable": restorable, "pinned-stagger": pinned_stagger, "orn-boot-ab": orn_boot_ab, "leftbehind": leftbehind, "pinned-sampled": pinned_sampled, "bandemote": bandemote, "dflash-high": dflash_high,
 "shed": shed, "restore-sampled": restore_sampled, "calib-nopool": calib_nopool,
 "riders-plain-ref": riders_plain_ref, "mtp-pool": mtp_pool, "plain-ref": plain_ref, "wave32": wave32, "dflash-demote": dflash_demote, "vision-ab": vision_ab,
 "mtp-cohort": mtp_cohort, "floor-ab": floor_ab, "riders": riders}[SCEN]()
sys.exit(1 if FAILS else 0)
