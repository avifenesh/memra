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
               round). It runs with MEMRA_SPEC_GATE=0 (no demotion to plain decode) and fails a
               row that did not speculate to its end. An MTP boot also runs with
               MEMRA_SPEC_SETUP_TRACE=1 and MEMRA_DEBUG_SPEC=1: the burst ends and the per-round
               drafts are read from its log.
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

Pooled cohorts (memra#1209), `--cohort 2,4` on every boot of its own DIR: requests sent at once
form a pooled cohort (several sessions speculating in one lockstep burst). Member 0 ends early
(a one-word or three-word stop string, EOS, or its max_tokens); the companions, other prompts
with no stop string, run on for 192 tokens. Each boot also runs member 0's requests alone (the
solo rows, whose non-stop rows set the close level) and, per sampler, three throughput cohorts of
companions only (aggregate completion tokens per second, first request sent to last response
ended). The ref boot runs every request alone. The judge fails, per speculative boot, a member 0
whose text, finish reason or completion tokens differ from its plain row (on a route whose
sampled stream is its own, from the same request's solo row in that boot), a companion that
differs from the same companion anywhere else (its other cohorts and every speculative boot of
the DIR: what a request writes may not depend on its cohort), and a member 0 that closes later
than the solo rule above (the non-stop median of the boot's solo rows, max(3x, +100 ms), plus
twice plain decode's own hold-back). Each member 0 that ended on a stop string must carry its
`[spec-stop]` receipt. A boot passed `--settle` (a binary that closes a member at its round)
must also log, for each member 0 that ended, that it finished inside its cohort, and per sampler
at least one of them while another member was still running; else the gate tested nothing.

usage: spec-stop-string-gate.py boot CASE SERVER_BINARY MODEL.gguf --out DIR [--route mtp|dspark]
           [--port N] [--env NAME=VALUE ...] [--cohort 2,4 [--settle] [--tput-only]]
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
import threading
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
# DSpark with the prefix cache on: a cold conversation that ends on a stop string must park
# rewound to its own prime boundary, and its second turn resume from the pool.
COLD_TURN = "Reef field notes. " + " ".join(
    f"Note {i}: the reef survey records one measurement per site, and a later visit extends the "
    f"same record with a new reading." for i in range(1, 46)) + \
    "\n\nWrite about 300 words, in several paragraphs, on what these notes say about reefs."
COLD_STOP = " the "  # inside the first sentences of nearly any English answer
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
# memra#1209: a member of a pooled cohort that finished inside the cohort call.
SETTLE = re.compile(r"\[spec-pool\] member finished inside the cohort: route=(\w+) completion_tokens=(\d+) "
                    r"members_running=(\d+)")
COMPANIONS = (
    "Write about 400 words on how a river gauge network is maintained through a year.",
    "Write about 400 words on how a weather balloon launch is planned and run.",
    "Write about 400 words on how a seed bank catalogues and tests its samples.",
    "Write about 400 words on how a lighthouse keeper's log was kept and read.",
)
COMPANION_TOKENS = 192
TPUT_REPS = 3


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


def make_plan(free):
    """The stop strings of one sampler's rows, taken from an answer without stop strings."""
    p = {"stop": [], "span": [], "two": []}
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
    return p


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

    def log_since(self, mark):
        time.sleep(0.3)
        with open(self.log, errors="replace") as f:
            f.seek(mark)
            return f.read()

    def row(self, kind, rid, regime, messages, samp, max_tokens, stop=None, want_cached=False):
        mark = self.log.stat().st_size
        o = self.stream(messages, samp, max_tokens, stop)
        return self.record(kind, rid, regime, o, max_tokens, stop, self.log_since(mark), want_cached)

    def cohort(self, members, regime):
        """Send every member at once (each its own connection and thread) and record one row per
        member. `members`: (kind, rid, messages, samp, max_tokens, stop). Rows carry the
        member's index, the cohort's size and the host monotonic send and end times."""
        mark = self.log.stat().st_size
        outs = [None] * len(members)
        t0 = time.monotonic()

        def send(i, m):
            start = time.monotonic() - t0
            o = self.stream(m[2], m[3], m[4], m[5])
            o["t_send"] = start
            o["t_end"] = None if o["t_done"] is None else start + o["t_done"]
            outs[i] = o

        threads = [threading.Thread(target=send, args=(i, m)) for i, m in enumerate(members)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        chunk = self.log_since(mark)
        settle = [{"route": s[0], "completion_tokens": int(s[1]), "members_running": int(s[2])}
                  for s in SETTLE.findall(chunk)]
        rows = []
        for i, (m, o) in enumerate(zip(members, outs)):
            extra = {"member": i, "cohort": len(members), "t_send": round(o["t_send"] * 1e3, 1),
                     "t_end": None if o["t_end"] is None else round(o["t_end"] * 1e3, 1)}
            if i == 0:
                # Every member's settle line of this cohort: the judge picks member 0's by count.
                extra["settle"] = settle
            # The stop-string receipts and park lines are member 0's: only it has stop strings.
            row, _, _ = self.record(m[0], m[1], regime, o, m[4], m[5], chunk if i == 0 else "", extra=extra)
            rows.append(row)
        return rows

    def record(self, kind, rid, regime, o, max_tokens, stop, chunk, want_cached=False, extra=None):
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
        row.update(extra or {})
        row["demoted"] = chunk.count("[spec-gate] demoted session")
        speculative = self.case != "ref"
        if row["error"] is None:
            if not o["text"] or not row["completion_tokens"]:
                row["error"] = "empty answer"
            elif speculative and (row["completion_tokens"] or 0) > 1 and row["acceptance"] is None:
                if self.args.route == "dspark" and row["cached"]:
                    # DSpark admits cold sessions only: a request that restored a prefix from the
                    # prefix cache is served by plain decode. Recorded, judged as plain decode.
                    row["plain_by_admission"] = True
                else:
                    row["error"] = "the request did not speculate (no acceptance in its usage): it tested plain decode"
            elif speculative and row["demoted"]:
                row["error"] = "the request was demoted to plain decode mid-answer: its stop tested plain decode"
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


def cohort_members(plan, regime, sampler):
    """Member 0's requests of one sampler: (kind, tag, messages, samp, max_tokens, stop)."""
    samp = sampler(seed_of(f"stop-{regime}"))
    msgs = user(STOP_PROMPT)
    out = [("free", "free", msgs, samp, MAX_TOKENS, None),
           ("eos", "eos", user(EOS_PROMPT), sampler(seed_of(f"eos-{regime}")), 160, None)]
    p = plan.get(regime, {})
    out += [("stop", f"stop-{i}", msgs, samp, MAX_TOKENS, [w]) for i, w in enumerate(p.get("stop", []))]
    out += [("span", f"span-{i}", msgs, samp, MAX_TOKENS, [s]) for i, s in enumerate(p.get("span", []))]
    return out


def companion(k, regime, sampler):
    return ("comp", f"comp{k}", user(COMPANIONS[k]), sampler(seed_of(f"comp-{k}-{regime}")), COMPANION_TOKENS, None)


def run_cohort_boot(args):
    """A pooled-cohort boot (memra#1209): see the module docstring."""
    ref = args.case == "ref"
    sizes = [int(x) for x in args.cohort.split(",")]
    if not all(2 <= n <= len(COMPANIONS) + 1 for n in sizes):
        raise SystemExit(f"--cohort sizes must lie in 2..{len(COMPANIONS) + 1}")
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
        if args.route == "mtp":
            # No demotion to plain decode, so every member speculates to its end.
            env = {"MEMRA_SPEC_GATE": "0"}
        else:
            # With the gate off DSpark admits one session at a time (no cohort ever forms); with
            # it on it admits up to the LOW band. LOW at the largest cohort and HIGH at twice it:
            # every member is admitted to DSpark and none demotes.
            n = max(sizes + [len(COMPANIONS)])
            env = {"MEMRA_SPEC_GATE_LOW": str(n), "MEMRA_SPEC_GATE_HIGH": str(2 * n)}
    env["MEMRA_MAX_SESSIONS"] = str(max(sizes + [len(COMPANIONS)]))
    if any(r.get("case") == args.case for r in load_rows(args.out)):
        raise SystemExit(f"{args.out} already holds boot {args.case}")
    srv = Server(args, args.case, env)
    try:
        for regime, sampler in REGIMES.items():
            if ref:
                # Every request alone: member 0's references (the plan from its free answer) and
                # each companion's.
                _, free, _ = srv.row("free", f"solo-free-{regime}", regime, user(STOP_PROMPT),
                                     sampler(seed_of(f"stop-{regime}")), MAX_TOKENS)
                plan[regime] = {k: v for k, v in make_plan(free).items() if k in ("stop", "span")}
                for kind, tag, msgs, samp, mt, stop in cohort_members(plan, regime, sampler)[1:]:
                    srv.row(kind, f"solo-{tag}-{regime}", regime, msgs, samp, mt, stop=stop)
                for k in range(len(COMPANIONS)):
                    kind, tag, msgs, samp, mt, stop = companion(k, regime, sampler)
                    srv.row(kind, f"solo-{tag}-{regime}", regime, msgs, samp, mt, stop=stop)
                continue
            if not args.tput_only:
                for kind, tag, msgs, samp, mt, stop in cohort_members(plan, regime, sampler):
                    srv.row(kind, f"solo-{tag}-{regime}", regime, msgs, samp, mt, stop=stop)
                for n in sizes:
                    for kind, tag, msgs, samp, mt, stop in cohort_members(plan, regime, sampler):
                        rid = f"c{n}-{tag}-{regime}"
                        comps = [companion(k, regime, sampler) for k in range(n - 1)]
                        srv.cohort([(f"c-{kind}", rid, msgs, samp, mt, stop)]
                                   + [(c[0], f"{rid}-{c[1]}", c[2], c[3], c[4], c[5]) for c in comps], regime)
            n = max(sizes)
            for rep in range(TPUT_REPS):
                comps = [companion(k, regime, sampler) for k in range(min(n, len(COMPANIONS)))]
                srv.cohort([(c[0], f"tput{n}-{rep}-{regime}-{c[1]}", c[2], c[3], c[4], c[5]) for c in comps], regime)
    finally:
        srv.stop()
    if ref:
        plan_path.write_text(json.dumps(plan, indent=1))
    (args.out / "boots.jsonl").open("a").write(json.dumps({
        "case": args.case, "route": args.route if not ref else "plain", "env": args.env, "cohort": sizes,
        "settle": args.settle, "tput_only": args.tput_only}) + "\n")
    errors = [r for r in load_rows(args.out) if r["case"] == args.case and r["error"]]
    for r in errors:
        print(f"ERROR {r['id']}: {r['error']}")
    return 1 if errors else 0


def run_boot(args):
    if args.cohort:
        return run_cohort_boot(args)
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
        # MEMRA_SPEC_GATE=0: no demotion to plain decode, so every row speculates to its end.
        env = {"MEMRA_SPEC_GATE": "0"}
        if args.route == "mtp":
            env.update({"MEMRA_SPEC_SETUP_TRACE": "1", "MEMRA_DEBUG_SPEC": "1"})
    if any(r.get("case") == args.case for r in load_rows(args.out)):
        raise SystemExit(f"{args.out} already holds boot {args.case}")
    srv = Server(args, args.case, env)
    extra = {}
    prefix_cache = "MEMRA_PREFIX_CACHE_MB=0" not in args.env

    def stop_row(kind, rid, regime, msgs, samp, max_tokens, stop):
        """A stop row. On a route whose sampled stream is its own (`self_ref`), the request is
        also sent without its stop strings at max_tokens = its count and one less: the stop
        string must complete on exactly that token of the route's own stream."""
        row, text, _ = srv.row(kind, rid, regime, msgs, samp, max_tokens, stop=stop)
        if self_ref(regime) and row["finish"] == "stop" and row["completion_tokens"]:
            n = row["completion_tokens"]
            for m in (n - 1, n):
                if m > 0:
                    srv.row("selfref", f"{rid}-at-{m}", regime, msgs, samp, m)
        return row, text

    def self_ref(regime):
        return not ref and args.route == "dspark" and regime != "greedy"

    try:
        for regime, sampler in REGIMES.items():
            samp = sampler(seed_of(f"stop-{regime}"))
            msgs = user(STOP_PROMPT)
            _, free, _ = srv.row("free", f"free-{regime}", regime, msgs, samp, MAX_TOKENS)
            srv.row("eos", f"eos-{regime}", regime, user(EOS_PROMPT), sampler(seed_of(f"eos-{regime}")), 160)
            # The plain boot picks the stop strings from its own answer; a route whose sampled
            # stream is its own picks them from that stream, the same way.
            own = ref or self_ref(regime)
            if own:
                p = make_plan(free)
                if ref:
                    plan[regime] = p
                else:
                    extra.setdefault("own_plan", {})[regime] = p
            else:
                p = plan[regime]
            for i, w in enumerate(p["stop"]):
                stop_row("stop", f"stop-{i}-{regime}", regime, msgs, samp, MAX_TOKENS, [w])
            for i, s in enumerate(p["span"]):
                stop_row("span", f"span-{i}-{regime}", regime, msgs, samp, MAX_TOKENS, [s])
            for i, pair in enumerate(p["two"]):
                stop_row("two", f"two-{i}-{regime}", regime, msgs, samp, MAX_TOKENS, pair)
                if ref:  # where each of the pair completes alone
                    srv.row("two-alone", f"two-{i}-b-{regime}", regime, msgs, samp, MAX_TOKENS, stop=[pair[0]])
                    srv.row("two-alone", f"two-{i}-a-{regime}", regime, msgs, samp, MAX_TOKENS, stop=[pair[1]])
            if own and p["stop"]:
                bi = 1 if len(p["stop"]) > 1 else 0
                n = next((r["completion_tokens"] for r in load_rows(args.out)
                          if r["case"] == args.case and r["id"] == f"stop-{bi}-{regime}"), None)
                p["budget"] = {"stop": p["stop"][bi], "max_tokens": n}
            if p.get("budget", {}).get("max_tokens"):
                b = p["budget"]
                stop_row("budget", f"budget-{regime}", regime, msgs, samp, b["max_tokens"], [b["stop"]])
            # Turn 2 after a turn that ended on a stop string: the second turn restores what
            # the first one parked.
            tmsgs = user(TURN_PROMPTS[regime])
            tsamp = sampler(seed_of(f"turn-{regime}"))
            _, tfree, _ = srv.row("turnfree", f"turnfree-{regime}", regime, tmsgs, tsamp, MAX_TOKENS)
            if own:
                _, w = pick_word(tfree, 24)
                p["turn"] = w
            if p.get("turn"):
                _, t1 = stop_row("turn1", f"turn1-{regime}", regime, tmsgs, tsamp, MAX_TOKENS, [p["turn"]])
                t2 = tmsgs + [{"role": "assistant", "content": t1}, {"role": "user", "content": TURN2}]
                # The DSpark recipes run with the prefix cache off: only the MTP boots must
                # restore the parked prefix.
                srv.row("turn2", f"turn2-{regime}", regime, t2, sampler(seed_of(f"turn2-{regime}")), 64,
                        want_cached=not ref and args.route == "mtp")
            if not ref and args.route == "dspark" and regime == "greedy" and prefix_cache:
                cmsgs = user(COLD_TURN)
                _, c1, _ = srv.row("coldturn1", "coldturn1-greedy", regime, cmsgs, sampler(0), MAX_TOKENS,
                                   stop=[COLD_STOP])
                c2 = cmsgs + [{"role": "assistant", "content": c1}, {"role": "user", "content": TURN2}]
                srv.row("coldturn2", "coldturn2-greedy", regime, c2, sampler(0), 64)
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
    (args.out / "boots.jsonl").open("a").write(json.dumps({
        "case": args.case, "route": args.route if not ref else "plain", "env": args.env,
        "self_ref": [r for r in REGIMES if self_ref(r)], "prefix_cache": prefix_cache}) + "\n")
    errors = [r for r in load_rows(args.out) if r["case"] == args.case and r["error"]]
    for r in errors:
        print(f"ERROR {r['id']}: {r['error']}")
    return 1 if errors else 0


def load_rows(out):
    p = out / "runs.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def close_judge(case, regime, rr, ref, failures):
    """The close: each stop-ended row against its boot's non-stop rows of the same sampler, plus
    twice plain decode's own hold-back for that row (a stop string of several tokens is held back
    from the stream while it may still complete, in plain decode too)."""
    base = [r["close_ms"] for r in rr if r["kind"] in NONSTOP_KINDS and r["close_ms"] is not None]
    stopped = [r for r in rr if r["kind"] in STOP_KINDS and r["finish"] == "stop" and r["close_ms"] is not None]
    if not base:
        failures.append(f"{case} {regime}: no non-stop row to time the close against")
        return None, None, None, base, stopped
    med = statistics.median(base)
    limit = max(3 * med, med + 100)
    worst = max((r["close_ms"] for r in stopped), default=None)
    for r in stopped:
        # Without a plain row for this stop string (a route judged on its own stream), a
        # three-word stop string gets a flat 100 ms for its hold-back.
        held = 2 * (ref[r["id"]].get("close_ms") or 0) if r["id"] in ref else (100 if r["kind"] == "span" else 0)
        if r["close_ms"] > limit + held:
            failures.append(f"{case} {r['id']}: closed {r['close_ms']} ms after its last visible chunk, "
                            f"limit {limit + held:.0f} ms (non-stop median {med:.0f} ms, plain decode's "
                            f"hold-back {held / 2:.0f} ms)")
    return med, limit, worst, base, stopped


def self_judge(out, case, rr, failures, cov):
    """A route whose sampled stream is its own: a stop row that stopped must stop on exactly the
    token of that stream where its stop string completes (the request without stop strings at
    max_tokens = its count holds the stop string, at one less it does not), and its text is that
    answer cut at the stop string; a stop row that ran out is its own free answer."""
    text = lambda rid: (out / "texts" / f"{case}-{rid}.txt").read_text()  # noqa: E731
    by_id = {r["id"]: r for r in rr}
    good = total = 0
    counts = []
    untestable = []
    for r in rr:
        if r["kind"] not in STOP_KINDS:
            continue
        total += 1
        n = r["completion_tokens"]
        counts.append(n)
        stops = r["stop"] or []
        if r["finish"] == "stop":
            at, below = by_id.get(f"{r['id']}-at-{n}"), by_id.get(f"{r['id']}-at-{n - 1}")
            if at is None or (n > 1 and below is None):
                failures.append(f"{case} {r['id']}: no self-reference rows for its {n}-token stop")
                continue
            if any(x and x.get("plain_by_admission") for x in (r, at, below)):
                # Its reference rows restored a prefix and ran on plain decode (DSpark admits
                # cold sessions only): another sampled stream, nothing to judge it against.
                untestable.append(r["id"])
                total -= 1
                continue
            full = text(at["id"])
            cut = min((full.find(x) for x in stops if x in full), default=-1)
            ok = cut >= 0 and (n == 1 or not any(x in text(below["id"]) for x in stops)) and text(r["id"]) == full[:cut]
            if len(r["receipts"]) != 1 or r["receipts"][0]["completion_tokens"] != n:
                failures.append(f"{case} {r['id']}: {len(r['receipts'])} [spec-stop] receipts for a {n}-token stop")
            elif r["receipts"][0]["spans_slices"]:
                cov["spans"] += 1
        else:
            free_id = f"turnfree-{r['regime']}" if r["kind"] == "turn1" else f"free-{r['regime']}"
            own = text(free_id) if free_id in by_id else ""
            ok = r["finish"] == "length" and n == r["max_tokens"] and not any(x in text(r["id"]) for x in stops) \
                and (r["kind"] == "budget" or text(r["id"]) == own)
        good += ok
        if not ok:
            failures.append(f"{case} {r['id']}: {n} tokens, finish {r['finish']}: not where its stop string completes "
                            "in this route's own stream")
    print(f"  {rr[0]['regime'] if rr else '?'}: own stream (this route's sampled walk is not plain decode's): stop rows "
          f"that stop on their completing token {good}/{total}; counts {counts}"
          + (f"; not judged (their reference rows ran on plain decode after a prefix restore): {untestable}"
             if untestable else ""))


def cohort_judge(out, rows, boots):
    """The pooled-cohort judge (memra#1209): see the module docstring."""
    failures = []
    route = {b["case"]: b["route"] for b in boots}
    ref = {r["id"]: r for r in rows if r["case"] == "ref"}
    if not ref:
        failures.append("no plain (ref) boot in this DIR")
    key = lambda r: (r["sha"], r["finish"], r["completion_tokens"])  # noqa: E731
    for r in rows:
        if r["error"]:
            failures.append(f"{r['case']} {r['id']}: {r['error']}")
    spec_boots = [b for b in boots if b["case"] != "ref"]
    if not spec_boots:
        failures.append("no speculative boot in this DIR: nothing was compared")
    # A companion writes one text wherever it runs: every cohort and every boot of its route.
    seen = {}
    for r in rows:
        if r["case"] == "ref" or r["kind"] != "comp":
            continue
        comp = r["id"].rsplit("-", 1)[1]
        k = (route[r["case"]], r["regime"], comp)
        seen.setdefault(k, {}).setdefault(key(r), []).append(f"{r['case']}:{r['id']}")
    for (rt, regime, comp), groups in sorted(seen.items()):
        n = sum(len(v) for v in groups.values())
        plain = ref.get(f"solo-{comp}-{regime}")
        same_plain = plain is not None and key(plain) in groups
        print(f"companion {rt} {regime} {comp}: {n} runs, {len(groups)} distinct text(s)"
              f"{'' if plain is None else ', equal to plain decode' if same_plain and len(groups) == 1 else ', not plain decode'}")
        if len(groups) > 1:
            failures.append(f"companion {rt} {regime} {comp}: {len(groups)} distinct texts over {n} runs: "
                            + "; ".join(f"{k}: {v[:3]}" for k, v in groups.items()))
    for b in spec_boots:
        case = b["case"]
        mine = [r for r in rows if r["case"] == case]
        print(f"== {case} ({route[case]}, cohorts {b['cohort']}{', settle' if b.get('settle') else ''}) ==")
        for regime in REGIMES:
            rr = [r for r in mine if r["regime"] == regime]
            tput = {}
            for r in rr:
                if r["id"].startswith("tput") and r.get("t_end") is not None:
                    rep = r["id"].split("-")[1]
                    t = tput.setdefault(rep, {"tokens": 0, "send": None, "end": 0.0})
                    t["tokens"] += r["completion_tokens"] or 0
                    t["send"] = r["t_send"] if t["send"] is None else min(t["send"], r["t_send"])
                    t["end"] = max(t["end"], r["t_end"])
            rates = [t["tokens"] / ((t["end"] - t["send"]) / 1e3) for t in tput.values() if t["end"] > t["send"]]
            if rates:
                print(f"  {regime}: throughput cohorts of {max(b['cohort'])}: aggregate tok/s "
                      f"{[round(x, 1) for x in rates]} (median {statistics.median(rates):.1f})")
            if b.get("tput_only"):
                continue
            solo = {r["id"]: r for r in rr if r["id"].startswith("solo-")}
            base = [r["close_ms"] for r in solo.values() if r["kind"] in ("free", "eos") and r["close_ms"] is not None]
            if not base:
                failures.append(f"{case} {regime}: no solo non-stop row to time the close against")
                continue
            med = statistics.median(base)
            limit = max(3 * med, med + 100)
            own = route[case] == "dspark" and regime != "greedy"
            worst, settled_live = {}, 0
            for r in rr:
                if r.get("member") != 0 or not r["kind"].startswith("c-"):
                    continue
                tag = r["id"].split("-", 1)[1].rsplit("-", 1)[0]
                n = r["cohort"]
                want = solo.get(f"solo-{tag}-{regime}") if own else ref.get(f"solo-{tag}-{regime}")
                if want is None:
                    failures.append(f"{case} {r['id']}: no reference row")
                elif key(r) != key(want):
                    failures.append(f"{case} {r['id']}: tokens {r['completion_tokens']} finish {r['finish']} sha "
                                    f"{r['sha']}, {'its solo row' if own else 'plain decode'} {want['completion_tokens']} "
                                    f"{want['finish']} {want['sha']}")
                for other in spec_boots:
                    o = next((x for x in rows if x["case"] == other["case"] and x["id"] == r["id"]), None)
                    if o is not None and route[other["case"]] == route[case] and key(o) != key(r):
                        failures.append(f"{case} {r['id']}: differs from boot {other['case']}")
                if r["finish"] == "stop" and r["stop"]:
                    if len(r["receipts"]) != 1 or r["receipts"][0]["completion_tokens"] != r["completion_tokens"]:
                        failures.append(f"{case} {r['id']}: {len(r['receipts'])} [spec-stop] receipts for a "
                                        f"{r['completion_tokens']}-token stop (want one naming it)")
                held = 2 * (ref.get(f"solo-{tag}-{regime}", {}).get("close_ms") or 0) if not own else \
                    (100 if r["kind"] == "c-span" else 0)
                if r["close_ms"] is not None:
                    worst[n] = max(worst.get(n, 0), r["close_ms"])
                    if r["close_ms"] > limit + held:
                        failures.append(f"{case} {r['id']}: closed {r['close_ms']} ms after its last visible chunk, "
                                        f"limit {limit + held:.0f} ms (solo non-stop median {med:.0f} ms, plain "
                                        f"decode's hold-back {held / 2:.0f} ms)")
                if b.get("settle"):
                    mine_settle = [s for s in r.get("settle", []) if s["completion_tokens"] == r["completion_tokens"]]
                    if not mine_settle:
                        failures.append(f"{case} {r['id']}: no line says member 0 finished inside its cohort")
                    elif mine_settle[0]["members_running"] > 0:
                        settled_live += 1
            closes = {n: sorted(r["close_ms"] for r in rr if r.get("member") == 0 and r["kind"].startswith("c-")
                                and r["cohort"] == n and r["close_ms"] is not None) for n in b["cohort"]}
            print(f"  {regime}: solo non-stop close median {med:.0f} ms (n={len(base)}), limit {limit:.0f}; member 0 "
                  f"close ms by cohort size: " + "; ".join(
                      f"{n}: median {statistics.median(v):.0f} max {max(v):.0f} (n={len(v)})" for n, v in closes.items() if v))
            if b.get("settle"):
                print(f"  {regime}: member 0 finished while another member ran: {settled_live}")
                if settled_live == 0:
                    failures.append(f"{case} {regime}: no member 0 finished while another member was still running: "
                                    "the cohort close was not tested")
    for f in failures:
        print("FAIL", f)
    print("STOP-STRING COHORT GATE:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


def judge(out):
    rows = load_rows(out)
    boots = [json.loads(line) for line in (out / "boots.jsonl").read_text().splitlines() if line.strip()]
    if any(b.get("cohort") for b in boots):
        return cohort_judge(out, rows, boots)
    route = {b["case"]: b["route"] for b in boots}
    ref = {r["id"]: r for r in rows if r["case"] == "ref"}
    failures = []
    cover = {}
    for r in rows:
        if (route.get(r["case"]) == "dspark" and r["cached"] and r["acceptance"] is None
                and (r["error"] or "").startswith("the request did not speculate")):
            r["plain_by_admission"], r["error"] = True, None  # rows written before the rule
        if r["error"]:
            failures.append(f"{r['case']} {r['id']}: {r['error']}")
        elif (r["case"] != "ref" and (r["completion_tokens"] or 0) > 1 and r["acceptance"] is None
              and not r.get("plain_by_admission")):
            failures.append(f"{r['case']} {r['id']}: the request did not speculate: it tested plain decode")
    for case in [b["case"] for b in boots if b["case"] != "ref"]:
        mine = [r for r in rows if r["case"] == case]
        rejects = json.loads((out / f"reject-{case}.json").read_text()) if (out / f"reject-{case}.json").exists() else {}
        print(f"== {case} ({route[case]}) ==")
        own = next((b.get("self_ref", []) for b in boots if b["case"] == case), [])
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
            if regime in own:
                # This route's sampled verify draws its own stream (DSpark's rejection walk is not
                # plain decode's position-keyed draw), so a row is judged against that stream. Its
                # close is judged against its own non-stop rows only (no plain hold-back: the stop
                # strings are its own).
                self_judge(out, case, rr, failures, cov)
                close_judge(case, regime, rr, {}, failures)
                continue
            stop_equal = stop_total = 0
            for r in rr:
                if r["kind"] in ("coldturn1", "coldturn2"):
                    continue  # judged below on the park, not against plain decode
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
            med, limit, worst, base, stopped = close_judge(case, regime, rr, ref, failures)
            if med is None:
                continue
            print(f"  {regime}: stop rows equal to plain decode {stop_equal}/{stop_total}; close ms: non-stop median "
                  f"{med:.0f} (max {max(base):.0f}, n={len(base)}), stop rows max {worst} (n={len(stopped)}), limit "
                  f"{limit:.0f}")
            print(f"    stop-row completion tokens: {[r['completion_tokens'] for r in rr if r['kind'] in STOP_KINDS]} "
                  f"plain {[ref.get(r['id'], {}).get('completion_tokens') for r in rr if r['kind'] in STOP_KINDS]}")
    for b in boots:
        if b["route"] != "dspark" or not b.get("prefix_cache"):
            continue
        c1 = next((r for r in rows if r["case"] == b["case"] and r["kind"] == "coldturn1"), None)
        c2 = next((r for r in rows if r["case"] == b["case"] and r["kind"] == "coldturn2"), None)
        if c1 is None or c2 is None:
            failures.append(f"{b['case']}: no cold conversation rows (a DSpark boot with the prefix cache on needs them)")
            continue
        parked = any("dspark-park REWOUND" in line for line in c1["park"])
        resumed = any("dspark-reuse:" in line and "resumed" in line for line in c2["park"])
        # With the DSpark restore switch on (the GB10 default) the pool parks nothing by design
        # and the next turn restores from the prefix cache instead.
        restore_switch = any("the restore switch is on" in line for line in c1["park"])
        print(f"== {b['case']}: cold conversation: turn 1 {c1['completion_tokens']} tokens, finish {c1['finish']}, "
              f"parked rewound {parked}; turn 2 resumed from the pool {resumed}, cached {c2['cached']}"
              + ("; the restore switch is on: the prefix cache serves the next turn" if restore_switch else ""))
        if c1["finish"] != "stop":
            failures.append(f"{b['case']} coldturn1: did not end on its stop string (finish {c1['finish']}): nothing tested")
        elif restore_switch:
            if not c2["cached"]:
                failures.append(f"{b['case']} coldturn2: the restore switch is on and the second turn restored nothing")
        elif not parked:
            failures.append(f"{b['case']} coldturn1: the stop-ended session did not park rewound: {c1['park']}")
        if not restore_switch and not resumed:
            failures.append(f"{b['case']} coldturn2: the second turn did not resume the parked session: {c2['park']}")
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
    ap.add_argument("--cohort", help="pooled-cohort boot: member counts, for example 2,4")
    ap.add_argument("--settle", action="store_true",
                    help="the binary closes a cohort member at its round: require its log line")
    ap.add_argument("--tput-only", action="store_true", help="a cohort boot that runs the throughput cohorts only")
    args = ap.parse_args()
    if (args.settle or args.tput_only) and not args.cohort:
        ap.error("--settle and --tput-only need --cohort")
    args.out.mkdir(parents=True, exist_ok=True)
    return run_boot(args)


if __name__ == "__main__":
    sys.exit(main())
