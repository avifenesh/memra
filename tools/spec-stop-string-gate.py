#!/usr/bin/env python3
"""Stop-string gate for the speculative routes (memra#1207): a request that ends on a stop
string counts its completion tokens through the token that completes it, as plain decode
does, and its response closes there.

Every request is streamed (SSE, with usage), alone, greedy and sampled (temperature 1, seeded).
Rows:

  free     an answer with no stop string (96 tokens): the reference text, a non-stop row
  eos      a short answer that ends on EOS: a non-stop row
  stop     a one-word stop string taken from the reference answer, five positions
  span     a three-word stop string, three positions: several tokens, so it often begins in
           one round or burst and completes in a later one
  two      two stop strings in one request, the later one listed first: the first to complete
           ends the request
  budget   a stop string that completes on the last token of max_tokens
  turn     a conversation whose first turn ends on a stop string (turn1) and whose second
           turn (turn2) restores the prefix the first one parked
  reject   (MTP boots) a stop string that only a drafted run the verify rejected would
           complete: the request must not stop on it

Boots, one per invocation, so a GPU lease covers one boot:

  boot ref     MEMRA_SPEC_K=0, plain decode. Picks the stop strings from its free answers,
               writes plan.json and runs every row: the reference counts and texts.
  boot NAME    a speculative boot (--route mtp or dspark) with --env NAME=VALUE pins, for
               example `boot small --env MEMRA_SPEC_BURST=2` (a burst ends at nearly every
               round). An MTP boot runs with MEMRA_SPEC_SETUP_TRACE=1 and MEMRA_DEBUG_SPEC=1:
               the burst ends and the per-round drafts are read from its log.
  judge        reads every boot under DIR.

The judge fails a speculative row whose text, finish reason or completion tokens differ from
its plain row (a reject row is compared with the plain free row: its stop string never
completes). The close: per boot and sampler, every row that ends on a stop string must close
within max(3 x, 100 ms more than) the median time from last visible chunk to end of response
of the same boot's non-stop rows, plus twice the time plain decode's row took to close (a
stop string of several tokens is held back from the stream while it may still complete, in plain
decode too). Each stop-ended speculative row must carry one
`[spec-stop]` receipt naming its completion tokens. Coverage, per sampler: the MTP boots must
hold one stop-ended row whose stop string began in an earlier commit slice than it completed
in (`spans_slices=1`), one `two` row whose two stop strings fall in one burst, and one reject
row whose own log shows the rejected draft; else the gate tested nothing and fails. A DSpark
boot must hold the spanning row.

usage: spec-stop-string-gate.py boot CASE SERVER_BINARY MODEL.gguf --out DIR [--route mtp|dspark]
           [--port N] [--env NAME=VALUE ...]
       spec-stop-string-gate.py judge DIR
A GPU lease is the caller's business (gpulock / sparklock around each boot).
Exit 0 on a pass, 1 on any difference, missing coverage or error.
"""
import argparse
import hashlib
import http.client
import json
import os
import pathlib
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

MAX_TOKENS = 96
REGIMES = {
    "t1": lambda seed: {"temperature": 1.0, "seed": seed},
    "greedy": lambda seed: {"temperature": 0},
}
STOP_PROMPT = "Orchard in brief: write about 300 words on how an orchard survey is planned and run."
EOS_PROMPT = "In about 45 words, explain how a tide gauge is read."
TURN_PROMPTS = {
    "t1": "Lagoon field notes. " + " ".join(
        f"Note {i}: the lagoon survey records one measurement per site, and a later visit extends "
        f"the same record with a new reading." for i in range(1, 46)) +
    "\n\nWrite about 300 words on what these notes say about lagoons.",
    "greedy": "Moor field notes. " + " ".join(
        f"Note {i}: the moor survey records one measurement per site, and a later visit extends "
        f"the same record with a new reading." for i in range(1, 46)) +
    "\n\nWrite about 300 words on what these notes say about moors.",
}
TURN2 = "Now give three concrete next steps for the same survey, in about 150 words."
STOP_AT = (14, 24, 34, 44, 54)  # word index in the reference answer; the first free word at or after
SPAN_AT = (18, 38, 58)
TWO_AT = (20, 40, 60)
STOP_KINDS = ("stop", "span", "two", "budget", "turn1")
NONSTOP_KINDS = ("free", "eos", "probe", "reject", "turnfree", "turn2")
ROUND = re.compile(r"\[R(\d+)\] pos=(\d+) out_len=(\d+) last_tok=(\d+) draft=\[([0-9, ]*)\] n_acc=(\d+) "
                   r"bonus=(\d+)")
SETUP = re.compile(r"\[spec-setup\] .* out=(\d+) cont=(true|false)")
PARK = re.compile(r"\[worker\] (?:spec-park|dspark-park|spec-reuse|dspark-reuse|spec-affinity)[^\n]*")
RECEIPT = re.compile(r"\[spec-stop\] route=(\w+) stop string completed: completion_tokens=(\d+) "
                     r"burst_slice=(\d+) slice_pos=(\d+)/(\d+) burst_public=(\d+) spans_slices=([01])")


def seed_of(tag):
    return 800 + int(hashlib.sha256(tag.encode()).hexdigest()[:4], 16) % 97


def user(content):
    return [{"role": "user", "content": content}]


def words_of(text):
    """(stripped word, char offset of its first letter) for every whitespace-split word."""
    out = []
    for m in re.finditer(r"\S+", text):
        raw = m.group(0)
        lead = len(raw) - len(raw.lstrip(".,;:!?()\"'*#-"))
        w = raw.strip(".,;:!?()\"'*#-")
        out.append((w, m.start() + lead))
    return out


def pick_word(text, at):
    """The first word at or after index `at` of five or more letters whose first occurrence in
    the text is this one (the stop string completes there and nowhere earlier)."""
    ws = words_of(text)
    for i in range(at, len(ws)):
        w, off = ws[i]
        if len(w) >= 5 and w.isalpha() and text.find(w) == off:
            return i, w
    return None, None


def pick_span(text, at):
    """Three consecutive words from index `at` on, as written (spaces and inner punctuation),
    first occurring there."""
    ws = words_of(text)
    for i in range(at, len(ws) - 2):
        w0, off = ws[i]
        w2, off2 = ws[i + 2]
        if len(w0) < 2 or not w2:
            continue
        phrase = text[off:off2 + len(w2)]
        if "\n" not in phrase and text.find(phrase) == off:
            return phrase
    return None


class Server:
    def __init__(self, args, case, env):
        self.args, self.case = args, case
        self.log = args.out / f"server-{case}.log"
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", args.port)) == 0:
                raise SystemExit(f"port {args.port} is busy: refusing to measure another server")
        e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        e.update({"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={args.model}", "MEMRA_ADDR": f"127.0.0.1:{args.port}",
                  "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"})
        e.update(env)
        e.update(dict(kv.split("=", 1) for kv in args.env))
        (args.out / f"env-{case}.json").write_text(json.dumps({k: v for k, v in e.items() if k.startswith("MEMRA_")},
                                                              indent=1, sort_keys=True))
        self.proc = subprocess.Popen([args.server], env=e, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{args.port}"
        try:
            t0 = time.time()
            while time.time() - t0 < 900:
                if self.proc.poll() is not None:
                    raise SystemExit(f"server exited {self.proc.returncode} during boot; see {self.log}")
                try:
                    urllib.request.urlopen(self.url + "/readyz", timeout=3).read()
                    return
                except Exception:
                    time.sleep(1)
            raise SystemExit("server not ready in 900 s")
        except BaseException:  # an interrupt during the wait must not leave the server on the GPU
            self.stop()
            raise

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def detok(self, ids):
        req = urllib.request.Request(self.url + "/v1/detokenize", data=json.dumps({"model": "q", "tokens": ids}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["prompt"]

    def stream(self, messages, samp, max_tokens, stop):
        """One streamed chat request. Times are host monotonic seconds at the chunk's arrival."""
        body = {"model": "q", "stream": True, "stream_options": {"include_usage": True}, "messages": messages,
                "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}, **samp}
        if stop:
            body["stop"] = stop
        o = {"text": "", "usage": {}, "finish": None, "error": None, "t_first": None, "t_last": None, "t_done": None}
        parts = []
        t0 = time.monotonic()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.args.port, timeout=900)
            conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                o["error"] = f"HTTP {resp.status}: {resp.read()[:300]!r}"
                return o
            buf = b""
            while True:
                chunk = resp.read1(65536)
                now = time.monotonic() - t0
                if not chunk:
                    break
                buf += chunk.replace(b"\r\n", b"\n")
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            o["t_done"] = now
                            continue
                        obj = json.loads(data)
                        if obj.get("error"):
                            o["error"] = json.dumps(obj["error"])[:300]
                        if obj.get("usage"):
                            o["usage"] = obj["usage"]
                        for ch in obj.get("choices") or []:
                            d = (ch.get("delta") or {}).get("content") or ""
                            if d:
                                parts.append(d)
                                o["t_last"] = now
                                if o["t_first"] is None:
                                    o["t_first"] = now
                            if ch.get("finish_reason"):
                                o["finish"] = ch["finish_reason"]
            conn.close()
        except Exception as exc:  # recorded and counted as a failure, never swallowed
            o["error"] = repr(exc)
        o["text"] = "".join(parts)
        if o["error"] is None and o["t_done"] is None:
            o["error"] = "the stream ended without [DONE]"
        return o

    def row(self, kind, rid, regime, messages, samp, max_tokens, stop=None, want_cached=False):
        mark = self.log.stat().st_size
        o = self.stream(messages, samp, max_tokens, stop)
        time.sleep(0.3)
        with open(self.log, errors="replace") as f:
            f.seek(mark)
            chunk = f.read()
        ends, total = [], 0
        for n, _ in SETUP.findall(chunk):
            total += int(n)
            ends.append(total)
        rounds = [{"round": int(r), "draft": [int(x) for x in d.split(",") if x.strip()], "n_acc": int(a),
                   "bonus": int(b), "last_tok": int(lt)}
                  for r, _pos, _ol, lt, d, a, b in ROUND.findall(chunk)]
        receipts = [{"route": m[0], "completion_tokens": int(m[1]), "burst_slice": int(m[2]), "slice_pos": int(m[3]),
                     "slice_len": int(m[4]), "burst_public": int(m[5]), "spans_slices": m[6] == "1"}
                    for m in RECEIPT.findall(chunk)]
        usage = o["usage"]
        spec = usage.get("spec") or {}
        tail = None if o["t_last"] is None or o["t_done"] is None else round((o["t_done"] - o["t_last"]) * 1e3, 1)
        row = {"case": self.case, "kind": kind, "id": rid, "regime": regime, "max_tokens": max_tokens, "stop": stop,
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "finish": o["finish"], "acceptance": spec.get("acceptance_rate"),
               "ttft_ms": None if o["t_first"] is None else round(o["t_first"] * 1e3, 1),
               "e2e_ms": None if o["t_done"] is None else round(o["t_done"] * 1e3, 1),
               "close_ms": tail, "burst_ends": ends, "receipts": receipts,
               "park": [m[:200] for m in PARK.findall(chunk)],
               "sha": hashlib.sha256(o["text"].encode()).hexdigest()[:16], "error": o["error"]}
        if row["error"] is None:
            if not o["text"] or not row["completion_tokens"]:
                row["error"] = "empty answer"
            elif want_cached and not row["cached"]:
                row["error"] = "the resumed request restored nothing (cached_tokens is 0)"
        (self.args.out / "texts").mkdir(exist_ok=True)
        (self.args.out / "texts" / f"{self.case}-{rid}.txt").write_text(o["text"])
        with open(self.args.out / "runs.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {self.case:>6} {rid:<16} tokens={row['completion_tokens']} finish={row['finish']} "
              f"close={tail}ms cached={row['cached']} receipts={len(receipts)} sha={row['sha']} "
              f"{row['error'] or ''}".rstrip(), flush=True)
        return row, o["text"], rounds


def reject_candidates(srv, rounds, free_text, limit=4):
    """Stop strings that only a rejected drafted run completes: the round's context tail plus the
    first draft token the verify rejected. The real path (the same tail plus the round's bonus)
    must be in the answer, and the stop string nowhere in it."""
    seq, out, seen = [], [], set()
    for r in rounds:
        ctx = seq + r["draft"][:r["n_acc"]]
        if r["n_acc"] < len(r["draft"]) and len(ctx) >= 3 and r["round"] > 0:
            tail, d = ctx[-3:], r["draft"][r["n_acc"]]
            head = srv.detok(tail)
            s = srv.detok(tail + [d])
            real = srv.detok(tail + [r["bonus"]])
            if (s.startswith(head) and s[len(head):].strip() and len(s.strip()) >= 4 and s not in free_text
                    and real in free_text and s not in seen):
                seen.add(s)
                out.append({"stop": s, "tail": tail, "draft_token": d, "n_acc": r["n_acc"]})
        seq = ctx + [r["bonus"]]
    if len(out) > limit:  # spread over the answer
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


def reject_seen(rounds, cand):
    """The row's own log holds the candidate's round: the same context tail, the same accepted
    count and the same rejected draft token."""
    seq = []
    for r in rounds:
        ctx = seq + r["draft"][:r["n_acc"]]
        if (r["n_acc"] == cand["n_acc"] and r["n_acc"] < len(r["draft"]) and ctx[-3:] == cand["tail"]
                and r["draft"][r["n_acc"]] == cand["draft_token"]):
            return True
        seq = ctx + [r["bonus"]]
    return False


def run_boot(args):
    ref = args.case == "ref"
    plan_path = args.out / "plan.json"
    if ref:
        if plan_path.exists():
            raise SystemExit(f"{plan_path} exists: the reference boot runs once per DIR")
        env = {"MEMRA_SPEC_K": "0"}
        plan = {}
    else:
        if not plan_path.exists():
            raise SystemExit("run the ref boot first: no plan.json")
        plan = json.loads(plan_path.read_text())
        env = {"MEMRA_SPEC_SETUP_TRACE": "1", "MEMRA_DEBUG_SPEC": "1"} if args.route == "mtp" else {}
    if any(r.get("case") == args.case for r in load_rows(args.out)):
        raise SystemExit(f"{args.out} already holds boot {args.case}")
    srv = Server(args, args.case, env)
    extra = {}
    try:
        for regime, sampler in REGIMES.items():
            samp = sampler(seed_of(f"stop-{regime}"))
            msgs = user(STOP_PROMPT)
            _, free, _ = srv.row("free", f"free-{regime}", regime, msgs, samp, MAX_TOKENS)
            srv.row("eos", f"eos-{regime}", regime, user(EOS_PROMPT), sampler(seed_of(f"eos-{regime}")), 160)
            if ref:
                p = plan[regime] = {"stop": [], "span": [], "two": []}
                for at in STOP_AT:
                    _, w = pick_word(free, at)
                    if w and w not in p["stop"]:
                        p["stop"].append(w)
                for at in SPAN_AT:
                    s = pick_span(free, at)
                    if s and s not in p["span"]:
                        p["span"].append(s)
                for at in TWO_AT:
                    i, w1 = pick_word(free, at)
                    _, w2 = pick_word(free, i + 1) if i is not None else (None, None)
                    if w1 and w2:
                        p["two"].append([w2, w1])
            p = plan[regime]
            for i, w in enumerate(p["stop"]):
                srv.row("stop", f"stop-{i}-{regime}", regime, msgs, samp, MAX_TOKENS, stop=[w])
            for i, s in enumerate(p["span"]):
                srv.row("span", f"span-{i}-{regime}", regime, msgs, samp, MAX_TOKENS, stop=[s])
            for i, pair in enumerate(p["two"]):
                srv.row("two", f"two-{i}-{regime}", regime, msgs, samp, MAX_TOKENS, stop=pair)
                if ref:  # where each of the pair completes alone
                    srv.row("two-alone", f"two-{i}-b-{regime}", regime, msgs, samp, MAX_TOKENS, stop=[pair[0]])
                    srv.row("two-alone", f"two-{i}-a-{regime}", regime, msgs, samp, MAX_TOKENS, stop=[pair[1]])
            if ref and p["stop"]:
                bi = 1 if len(p["stop"]) > 1 else 0
                n = next((r["completion_tokens"] for r in load_rows(args.out)
                          if r["case"] == "ref" and r["id"] == f"stop-{bi}-{regime}"), None)
                p["budget"] = {"stop": p["stop"][bi], "max_tokens": n}
            if p.get("budget", {}).get("max_tokens"):
                b = p["budget"]
                srv.row("budget", f"budget-{regime}", regime, msgs, samp, b["max_tokens"], stop=[b["stop"]])
            # Turn 2 after a turn that ended on a stop string: the second turn restores what
            # the first one parked.
            tmsgs = user(TURN_PROMPTS[regime])
            tsamp = sampler(seed_of(f"turn-{regime}"))
            _, tfree, _ = srv.row("turnfree", f"turnfree-{regime}", regime, tmsgs, tsamp, MAX_TOKENS)
            if ref:
                _, w = pick_word(tfree, 24)
                p["turn"] = w
            if p.get("turn"):
                _, t1, _ = srv.row("turn1", f"turn1-{regime}", regime, tmsgs, tsamp, MAX_TOKENS, stop=[p["turn"]])
                t2 = tmsgs + [{"role": "assistant", "content": t1}, {"role": "user", "content": TURN2}]
                # The DSpark recipes run with the prefix cache off: only the MTP boots must
                # restore the parked prefix.
                srv.row("turn2", f"turn2-{regime}", regime, t2, sampler(seed_of(f"turn2-{regime}")), 64,
                        want_cached=not ref and args.route == "mtp")
            if not ref and args.route == "mtp":
                # A stop string only a rejected drafted run completes, taken from a warm run of
                # the free request in this boot and confirmed in each row's own log.
                _, _, rounds = srv.row("probe", f"probe-{regime}", regime, msgs, samp, MAX_TOKENS)
                plain_free = (args.out / "texts" / f"ref-free-{regime}.txt").read_text()
                cands = reject_candidates(srv, rounds, plain_free)
                extra[regime] = []
                for i, c in enumerate(cands):
                    row, _, rrounds = srv.row("reject", f"reject-{i}-{regime}", regime, msgs, samp, MAX_TOKENS,
                                              stop=[c["stop"]])
                    c = {**c, "id": row["id"], "confirmed": reject_seen(rrounds, c)}
                    extra[regime].append(c)
                    print(f"         reject-{i}-{regime}: stop={c['stop']!r} rejected draft seen in its own log: "
                          f"{c['confirmed']}", flush=True)
    finally:
        srv.stop()
    if ref:
        plan_path.write_text(json.dumps(plan, indent=1))
    else:
        (args.out / f"reject-{args.case}.json").write_text(json.dumps(extra, indent=1))
    (args.out / "boots.jsonl").open("a").write(json.dumps({"case": args.case, "route": args.route if not ref else "plain",
                                                           "env": args.env}) + "\n")
    errors = [r for r in load_rows(args.out) if r["case"] == args.case and r["error"]]
    for r in errors:
        print(f"ERROR {r['id']}: {r['error']}")
    return 1 if errors else 0


def load_rows(out):
    p = out / "runs.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def judge(out):
    rows = load_rows(out)
    boots = [json.loads(line) for line in (out / "boots.jsonl").read_text().splitlines() if line.strip()]
    route = {b["case"]: b["route"] for b in boots}
    ref = {r["id"]: r for r in rows if r["case"] == "ref"}
    failures = []
    cover = {}
    for r in rows:
        if r["error"]:
            failures.append(f"{r['case']} {r['id']}: {r['error']}")
    for case in [b["case"] for b in boots if b["case"] != "ref"]:
        mine = [r for r in rows if r["case"] == case]
        rejects = json.loads((out / f"reject-{case}.json").read_text()) if (out / f"reject-{case}.json").exists() else {}
        print(f"== {case} ({route[case]}) ==")
        for regime in REGIMES:
            cov = cover.setdefault((route[case], regime), {"spans": 0, "two_one_burst": 0, "reject": 0})
            rr = [r for r in mine if r["regime"] == regime]
            for kind in ("free", "eos", "stop", "span", "two", "budget", "turn1", "turn2"):
                if not any(r["kind"] == kind for r in rr):
                    failures.append(f"{case} {regime}: no {kind} row (the plan or the boot dropped it)")
            for r in rr:
                # A stop-ended session must park or be freed by the pool's own rules, never be
                # refused as an invariant failure (cache rows past its public stream).
                for line in r.get("park", []):
                    if "-park REFUSED" in line:
                        failures.append(f"{case} {r['id']}: {line}")
            stop_equal = stop_total = 0
            for r in rr:
                want = ref.get(f"free-{regime}" if r["kind"] in ("reject", "probe") else r["id"])
                if want is None:
                    failures.append(f"{case} {r['id']}: no plain row")
                    continue
                same = (r["sha"], r["finish"], r["completion_tokens"]) == (want["sha"], want["finish"],
                                                                         want["completion_tokens"])
                if r["kind"] in STOP_KINDS:
                    stop_total += 1
                    stop_equal += same
                if not same:
                    failures.append(f"{case} {r['id']}: tokens {r['completion_tokens']} finish {r['finish']} sha "
                                    f"{r['sha']}, plain decode {want['completion_tokens']} {want['finish']} {want['sha']}")
                if r["kind"] in STOP_KINDS and r["finish"] == "stop":
                    if len(r["receipts"]) != 1 or r["receipts"][0]["completion_tokens"] != r["completion_tokens"]:
                        failures.append(f"{case} {r['id']}: {len(r['receipts'])} [spec-stop] receipts for a "
                                        f"{r['completion_tokens']}-token stop (want one naming it)")
                    elif r["receipts"][0]["spans_slices"]:
                        cov["spans"] += 1
            # Two stop strings in one burst: where each completes alone (plain), against this
            # boot's burst ends for the same request.
            probe = next((r for r in rr if r["kind"] == "probe"), None)
            for r in rr:
                if r["kind"] != "two" or probe is None or not probe["burst_ends"]:
                    continue
                i = r["id"].split("-")[1]
                pa = ref.get(f"two-{i}-a-{regime}", {}).get("completion_tokens")
                pb = ref.get(f"two-{i}-b-{regime}", {}).get("completion_tokens")
                if pa and pb:
                    burst = [sum(1 for e in probe["burst_ends"] if e < p) for p in (pa, pb)]
                    if burst[0] == burst[1]:
                        cov["two_one_burst"] += 1
            for c in rejects.get(regime, []):
                cov["reject"] += bool(c["confirmed"])
            # The close: stop-ended rows against this boot's non-stop rows.
            base = [r["close_ms"] for r in rr if r["kind"] in NONSTOP_KINDS and r["close_ms"] is not None]
            stopped = [r for r in rr if r["kind"] in STOP_KINDS and r["finish"] == "stop" and r["close_ms"] is not None]
            if not base:
                failures.append(f"{case} {regime}: no non-stop row to time the close against")
                continue
            med = statistics.median(base)
            limit = max(3 * med, med + 100)
            worst = max((r["close_ms"] for r in stopped), default=None)
            for r in stopped:
                # A stop string of several tokens is held back from the stream while it may
                # still complete, in plain decode too: its row may also take twice the time plain
                # decode's row took from its last visible chunk to its end.
                held = 2 * (ref.get(r["id"], {}).get("close_ms") or 0)
                if r["close_ms"] > limit + held:
                    failures.append(f"{case} {r['id']}: closed {r['close_ms']} ms after its last visible chunk, "
                                    f"limit {limit + held:.0f} ms (non-stop median {med:.0f} ms, plain decode's "
                                    f"hold-back {held / 2:.0f} ms)")
            print(f"  {regime}: stop rows equal to plain decode {stop_equal}/{stop_total}; close ms: non-stop median "
                  f"{med:.0f} (max {max(base):.0f}, n={len(base)}), stop rows max {worst} (n={len(stopped)}), limit "
                  f"{limit:.0f}")
            print(f"    stop-row completion tokens: {[r['completion_tokens'] for r in rr if r['kind'] in STOP_KINDS]} "
                  f"plain {[ref.get(r['id'], {}).get('completion_tokens') for r in rr if r['kind'] in STOP_KINDS]}")
    if not ref:
        failures.append("no plain (ref) boot in this DIR")
    if not cover:
        failures.append("no speculative boot in this DIR: nothing was compared")
    for (rt, regime), cov in sorted(cover.items()):
        print(f"coverage {rt} {regime}: spanning stops {cov['spans']}, two stops in one burst {cov['two_one_burst']}, "
              f"confirmed rejected-draft stops {cov['reject']}")
        if cov["spans"] == 0:
            failures.append(f"coverage {rt} {regime}: no stop string began in an earlier commit slice than it "
                            "completed in")
        if rt == "mtp" and cov["two_one_burst"] == 0:
            failures.append(f"coverage {rt} {regime}: no request held two stop strings in one burst")
        if rt == "mtp" and cov["reject"] == 0:
            failures.append(f"coverage {rt} {regime}: no stop string that only a rejected draft completes was seen")
    for f in failures:
        print("FAIL", f)
    print("STOP-STRING GATE:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "judge":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("out", type=pathlib.Path)
        return judge(ap.parse_args().out)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("boot",))
    ap.add_argument("case")
    ap.add_argument("server")
    ap.add_argument("model")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--route", choices=("mtp", "dspark"), default="mtp")
    ap.add_argument("--port", type=int, default=18207)
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    return run_boot(args)


if __name__ == "__main__":
    sys.exit(main())
