#!/usr/bin/env python3
"""Burst-boundary gate for the MTP speculative route (memra#1186): where a burst ends may
not change what a request writes, and a sampled burst carries its pending token.

A speculative burst ends on a pending token: emitted, not yet in the caches. A greedy burst has
carried it into the next burst's first verify since 2026-08-01; a sampled burst does since
memra#1186, where it used to commit it with a solo pass and feed the boundary token with a
second one. The seeded-invariance gate proves the texts on long answers. This gate puts the
end of a request on the boundary itself:

  eos     the answer ends on EOS: one shorter than a burst, three that cross boundaries
  budget  max_tokens 31, 32, 33, 63, 64, 65 (the burst length and twice it, less and plus one)
  stop    a stop string taken from the reference answer, five positions
  grammar a JSON-object answer (a grammar-constrained request): greedy speculates under the
          grammar, sampled is served by plain decode and must say so by not speculating
  turn    a two-turn conversation whose first turn ends on its budget, second turn resumed
  park    a text completion sent right after the request that parked exactly its prompt
  yield   one long penalised answer while other requests arrive (bursts that run no round)
  ctx     an open answer (no max_tokens) that runs into a small context window (the
          context-full stop)

each sampled and greedy. Boots:

  ref     MEMRA_SPEC_K=0 (plain decode), every request alone and cold. The reference texts.
  spec    speculative, at each (MEMRA_SPEC_K, MEMRA_SPEC_BURST) of --grid. A burst length of
          2 ends a burst at every round that emits two tokens or more, so nearly every
          request ends on a boundary or right after one. (A burst length of 1 never reaches
          a round: each burst emits its boundary token and stops.)
  yield   speculative, MEMRA_MAX_SESSIONS=8, MEMRA_SPEC_GATE=0 (no demotion under load, so the
          long answer keeps speculating while requests arrive).
  load    the same with the demotion gate at its default: under the arrivals the long answer
          may be handed to plain decode with a token still pending.
  refctx, ctx   plain and speculative at MEMRA_CTX=640, three open answers that run into it.

Every speculative text must equal the reference text of its request, with the same finish
reason and the same completion tokens (since memra#1207 a stop string ends a speculative
request on the token that completes it, so a stop-string request counts as plain decode
does). A speculative request stops
short of a full context by its draft length, so a `ctx` text must be a prefix of its plain
text. The speculative boots run with MEMRA_SPEC_SETUP_TRACE=1 and MEMRA_SPEC_BOUNDARY_TRACE=1,
and the gate reads the log:

  placement  per request, the generated positions where its bursts ended. A request "ends on
             the boundary" when its last token is the last of a burst (the pending token has
             no next burst) and "ends after the boundary" when it is the first of the next
             one (round 0 drew it over the carried token). Each end kind must be met at least
             once sampled and once greedy, or the gate tested nothing and fails.
  carry      a sampled burst may not draw a boundary token at its tail
             (`site=burst-tail-commit`), and sampled bursts must enter on a carried token
             (`[spec-carry]`). A binary from before memra#1186 fails here.
  no-round   the yield boot must hold a continuation burst that ran no round (its round loop
             took no time) and that entered on a carried token. Before memra#1186 a sampled
             burst never did (`--carry no` then asks for a no-round burst of any kind).

--compare RUNS.jsonl: every row must also carry the text sha, the finish reason and the
completion tokens a second run (another binary) wrote for the same row. The counts are
printed.
--penalised host: as in seeded-invariance-gate.py, for a model and card where plain decode
samples a penalised request on the host: its speculative texts are compared with each other.

usage: spec-burst-boundary-gate.py SERVER_BINARY MODEL.gguf [--out DIR] [--port N]
           [--grid default,1:2,3:2,3:3] [--compare RUNS.jsonl] [--penalised device|host]
           [--carry yes|no] [--env NAME=VALUE ...]
A GPU lease is the caller's business (gpulock / sparklock around the whole run). Requests are
not streamed, so each must finish inside the server's 90 s non-streaming deadline: on a slow
device raise it for the run with --env MEMRA_TIMEOUT_MS_MAX=600000.
Exit 0 on a full pass, 1 on any difference, missing placement or error.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

BURST = 32


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic, n=45):
    return f"{topic.capitalize()} field notes. {notes(topic, n)}\n\nWrite about 300 words on what these notes say about {topic}."


def short_prompt(topic):
    return f"{topic.capitalize()} in brief: write about 300 words on how a {topic} survey is planned and run."


def user(content):
    return [{"role": "user", "content": content}]


SAMPLERS = {
    "t1": lambda seed: {"temperature": 1.0, "seed": seed},
    "chat": lambda seed: {"temperature": 0.7, "top_p": 0.95, "seed": seed},
    "pen": lambda seed: {"temperature": 1.0, "frequency_penalty": 0.4, "seed": seed},
    "greedy": lambda seed: {"temperature": 0},
}
EOS_PROMPTS = [
    ("short", "Answer in one short sentence: what does a tide gauge measure?", ("t1", "greedy")),
    ("mid", "In about 45 words, explain how a tide gauge is read.", ("t1", "chat", "pen", "greedy")),
    ("mid2", "In about 60 words, say why a river survey repeats its sites.", ("t1", "greedy")),
    ("mid3", "In about 50 words, describe one use of a harbor depth chart.", ("t1", "pen", "greedy")),
]
BUDGETS = (BURST - 1, BURST, BURST + 1, 2 * BURST - 1, 2 * BURST, 2 * BURST + 1)
BUDGET_PROMPT = short_prompt("meadow")
STOP_PROMPT = short_prompt("orchard")
GRAMMAR_PROMPT = ("Return one JSON object that describes a harbor survey site, with the keys name, depth_m, "
                  "visits and notes. Make the notes about forty words.")
STOP_WORDS = (16, 22, 28, 34, 40)  # word index in the reference answer, the first free one at or after
# One conversation and one parking prompt per regime: a shared first turn would let the second
# regime's cold reference restore the first one's prefix.
E1 = {"t1": long_prompt("lagoons"), "greedy": long_prompt("moors")}
E2 = "Now give three concrete next steps for the same survey, in about 150 words."
P1 = {"t1": f"Glacier survey log. {notes('glaciers', 8)}\n\nSummary of the log:",
      "greedy": f"Tundra survey log. {notes('tundra', 8)}\n\nSummary of the log:"}
YIELD_PROMPT = short_prompt("estuary")
CTX_PROMPT = "Write a long report, at least 900 words, on how a coastal survey is planned, run and written up."
PENALISED = "pen"
SETUP = re.compile(r"\[spec-setup\] .* rounds=([0-9.]+)ms .* out=(\d+) cont=(true|false)")


def seed_of(tag):
    return 800 + int(hashlib.sha256(tag.encode()).hexdigest()[:4], 16) % 97


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

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                obj = json.loads(r.read())
            ch = obj["choices"][0]
            text = ch.get("text") if "text" in ch else (ch.get("message") or {}).get("content")
            return {"text": text or "", "usage": obj.get("usage") or {}, "finish": ch.get("finish_reason"), "error": None}
        except Exception as exc:  # recorded and counted as a failure, never swallowed
            return {"text": "", "usage": {}, "finish": None, "error": repr(exc)}

    def ask(self, payload, samp, max_tokens, stop=None, extra=None):
        """payload: chat messages, or a raw text for /v1/completions."""
        body = {"model": "q", "stream": False, **samp, **(extra or {})}
        if max_tokens is not None:  # an open request runs to the server's context window
            body["max_tokens"] = max_tokens
        if stop:
            body["stop"] = [stop]
        if isinstance(payload, str):
            return self.post("/v1/completions", {**body, "prompt": payload})
        return self.post("/v1/chat/completions",
                         {**body, "messages": payload, "chat_template_kwargs": {"enable_thinking": False}})

    def row(self, kind, rid, regime, payload, samp, max_tokens, stop=None, want_cached=None, extra=None,
            plain=False):
        """One request alone. Returns (row, text). The row carries the bursts the server log
        shows for it: the generated positions where each burst ended. plain: the request is
        one the server decodes plainly whatever K is (it must not speculate)."""
        mark = self.log.stat().st_size
        o = self.ask(payload, samp, max_tokens, stop, extra)
        time.sleep(0.3)
        with open(self.log, errors="replace") as f:
            f.seek(mark)
            chunk = f.read()
        ends, total = [], 0
        for _, n, _ in SETUP.findall(chunk):
            total += int(n)
            ends.append(total)
        usage = o["usage"]
        spec = usage.get("spec") or {}
        row = {"case": self.case, "kind": kind, "id": rid, "regime": regime, "max_tokens": max_tokens, "stop": stop,
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "finish": o["finish"], "acceptance": spec.get("acceptance_rate"), "rounds": spec.get("rounds"),
               "burst_ends": ends, "tail_commits": chunk.count("site=burst-tail-commit"),
               "carries": chunk.count("[spec-carry] site=burst-entry"),
               "sha": hashlib.sha256(o["text"].encode()).hexdigest()[:16], "error": o["error"]}
        speculative = self.case not in ("ref", "ref2", "refctx")
        if row["error"] is None:
            if not o["text"] or not row["completion_tokens"]:
                row["error"] = "empty answer"
            elif speculative and not plain and row["acceptance"] is None and (row["completion_tokens"] or 0) > 1:
                row["error"] = "the request did not speculate (no acceptance in its usage)"
            elif speculative and plain and row["acceptance"] is not None:
                row["error"] = "a request the server decodes plainly speculated"
            elif want_cached is True and not row["cached"]:
                row["error"] = "the resumed request restored nothing (cached_tokens is 0)"
            elif want_cached is False and row["cached"]:
                row["error"] = f"the cold request restored {row['cached']} tokens"
        (self.args.out / "texts" / f"{self.case}-{rid}.txt").write_text(o["text"])
        with open(self.args.out / "runs.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {self.case:>8} {rid:<22} tokens={row['completion_tokens']} finish={row['finish']} "
              f"cached={row['cached']} ends={ends[-4:]} sha={row['sha']} {row['error'] or ''}".rstrip(), flush=True)
        return row, o["text"]


def pick_stop(text, at):
    """A stop string from the reference answer: the first word at or after index `at` that is
    five or more letters and does not occur earlier in the text."""
    words = text.split()
    for i in range(at, len(words)):
        w = words[i].strip(".,;:!?()\"'*#-")
        if len(w) >= 5 and w.isalpha() and text.find(w) == text.find(" " + w) + 1 and text.count(w) >= 1:
            if text.find(w) >= len(" ".join(words[:i])) - 1:
                return w
    return None


def plan_requests(srv, stops, turn1, park):
    """The lone requests of one boot, in a fixed order. `stops`, `turn1` and `park` are the
    reference boot's stop strings and first-turn and parking answers (None while it runs)."""
    rows = []
    for name, prompt, regimes in EOS_PROMPTS:
        for regime in regimes:
            rid = f"eos-{name}-{regime}"
            rows.append(srv.row("eos", rid, regime, user(prompt), SAMPLERS[regime](seed_of(rid)), 160))
    for regime in ("t1", "greedy"):
        for n in BUDGETS:
            rid = f"budget-{n}-{regime}"
            rows.append(srv.row("budget", rid, regime, user(BUDGET_PROMPT), SAMPLERS[regime](seed_of(f"budget-{regime}")), n))
    for regime in ("t1", "greedy"):
        rid = f"grammar-{regime}"
        rows.append(srv.row("grammar", rid, regime, user(GRAMMAR_PROMPT), SAMPLERS[regime](seed_of(rid)), 120,
                            extra={"response_format": {"type": "json_object"}}, plain=regime != "greedy"))
    for regime in ("t1", "greedy"):
        samp = SAMPLERS[regime](seed_of(f"stop-{regime}"))
        if srv.case == "ref":
            free = srv.row("stopfree", f"stopfree-{regime}", regime, user(STOP_PROMPT), samp, 96)
            rows.append(free)
            picked = []
            for at in STOP_WORDS:
                w = pick_stop(free[1], at)
                if w and w not in picked:
                    picked.append(w)
            stops[regime] = picked
            (srv.args.out / f"stops-{regime}.json").write_text(json.dumps(picked))
        for i, w in enumerate(stops.get(regime) or []):
            rows.append(srv.row("stop", f"stop-{i}-{regime}", regime, user(STOP_PROMPT), samp, 96, stop=w))
    return rows


def run_ref(args, stops):
    srv = Server(args, "ref", {"MEMRA_SPEC_K": "0"})
    turn1, park = {}, {}
    try:
        rows = plan_requests(srv, stops, None, None)
        for regime in ("t1", "greedy"):
            samp = SAMPLERS[regime]
            t = srv.row("turn1", f"turn1-{regime}", regime, user(E1[regime]), samp(seed_of(f"e1-{regime}")), 2 * BURST)
            p = srv.row("park1", f"park1-{regime}", regime, P1[regime], samp(seed_of(f"p1-{regime}")), 48)
            rows += [t, p]
            turn1[regime], park[regime] = t[1], p[1]
        for regime in ("pen", "t1"):
            rid = f"yield-{regime}"
            rows.append(srv.row("yield", rid, regime, user(YIELD_PROMPT), SAMPLERS[regime](seed_of(rid)), 200))
    finally:
        srv.stop()
    # The second turn and the after-park request cold: their own boot, so nothing of the first
    # turn or of the parking request is resident.
    srv = Server(args, "ref2", {"MEMRA_SPEC_K": "0"})
    try:
        for regime in ("t1", "greedy"):
            samp = SAMPLERS[regime]
            conv = user(E1[regime]) + [{"role": "assistant", "content": turn1[regime]}, {"role": "user", "content": E2}]
            rows.append(srv.row("turn2", f"turn2-{regime}", regime, conv, samp(seed_of(f"e2-{regime}")), 96, want_cached=False))
            rows.append(srv.row("park2", f"park2-{regime}", regime, P1[regime] + park[regime], samp(seed_of(f"p2-{regime}")), 96,
                                want_cached=False))
    finally:
        srv.stop()
    return rows


def run_spec(args, case, env, stops):
    srv = Server(args, case, {"MEMRA_SPEC_SETUP_TRACE": "1", "MEMRA_SPEC_BOUNDARY_TRACE": "1", **env})
    try:
        rows = plan_requests(srv, stops, None, None)
        for regime in ("t1", "greedy"):
            samp = SAMPLERS[regime]
            t = srv.row("turn1", f"turn1-{regime}", regime, user(E1[regime]), samp(seed_of(f"e1-{regime}")), 2 * BURST)
            rows.append(t)
            conv = user(E1[regime]) + [{"role": "assistant", "content": t[1]}, {"role": "user", "content": E2}]
            rows.append(srv.row("turn2", f"turn2-{regime}", regime, conv, samp(seed_of(f"e2-{regime}")), 96, want_cached=True))
            p = srv.row("park1", f"park1-{regime}", regime, P1[regime], samp(seed_of(f"p1-{regime}")), 48)
            rows.append(p)
            rows.append(srv.row("park2", f"park2-{regime}", regime, P1[regime] + p[1], samp(seed_of(f"p2-{regime}")), 96,
                                want_cached=True))
    finally:
        srv.stop()
    return rows


def run_yield(args, case, env):
    """One long answer at a time while other requests arrive: a 3.6k-token prompt that primes
    for a while, and short ones behind it. A request waiting for admission ends a burst at
    its next round boundary, and a burst that begins while one waits runs no round."""
    srv = Server(args, case, {"MEMRA_SPEC_SETUP_TRACE": "1", "MEMRA_SPEC_BOUNDARY_TRACE": "1",
                              "MEMRA_MAX_SESSIONS": "8", **env})
    rows = []
    try:
        wave = 0
        for regime in ("pen", "t1"):
            rid = f"yield-{regime}"
            done = threading.Event()

            def arrivals():
                nonlocal wave
                time.sleep(1.5)
                while not done.is_set() and wave < 24:
                    wave += 1
                    topic = f"{case}quarry{wave}"
                    big = threading.Thread(target=srv.ask, args=(user(long_prompt(topic, 130)), SAMPLERS["greedy"](0), 1))
                    big.start()
                    small = []
                    for gap in (0.15, 0.2, 0.3, 0.4):
                        time.sleep(gap)
                        t = threading.Thread(target=srv.ask, args=(user(f"Name one {topic} tool."), SAMPLERS["greedy"](0), 1))
                        t.start()
                        small.append(t)
                    big.join()
                    for t in small:
                        t.join()
                    time.sleep(0.5)

            th = threading.Thread(target=arrivals)
            th.start()
            try:
                rows.append(srv.row("yield", rid, regime, user(YIELD_PROMPT), SAMPLERS[regime](seed_of(rid)), 200))
            finally:
                done.set()
                th.join()
    finally:
        srv.stop()
    # A burst that ran no round spent no time in its round loop (a round is tens of
    # milliseconds on any device). The other requests of this boot are one-token cold bursts
    # (cont=false), so a continuation burst here is the long answer's. A sampled burst that
    # entered on a carried token says so one line before its own summary line.
    carried = any_kind = demoted = 0
    entered = False
    for line in srv.log.read_text(errors="replace").splitlines():
        if "[spec-carry] site=burst-entry" in line:
            entered = True
        elif "[spec-gate] demoted session" in line:
            demoted += 1
        else:
            m = SETUP.search(line)
            if m:
                if m.group(3) == "true" and float(m.group(1)) < 1.0:
                    any_kind += 1
                    carried += entered and m.group(2) == "0"
                entered = False
    return rows, {"carried": carried, "any": any_kind, "demoted": demoted}


def run_ctx(args, case, env):
    """Three open answers (no max_tokens) that run into a 640-token context window. A finite
    max_tokens sizes the session to the request, so only an open request reaches the edge."""
    srv = Server(args, case, {"MEMRA_CTX": "640", **env})
    rows = []
    try:
        for regime in ("t1", "pen", "greedy"):
            rid = f"ctx-{regime}"
            rows.append(srv.row("ctx", rid, regime, user(CTX_PROMPT), SAMPLERS[regime](seed_of(rid)), None))
    finally:
        srv.stop()
    return rows


def placement(row, ref_tokens):
    """('on' | 'after' | both | neither) for one speculative row: is its last token, at
    generated position ref_tokens, the last of a burst or the first of the next one?"""
    ends = set(row["burst_ends"])
    out = []
    if ref_tokens in ends:
        out.append("on")
    if ref_tokens - 1 in ends:
        out.append("after")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("server")
    ap.add_argument("model")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--port", type=int, default=18198)
    ap.add_argument("--grid", default="default,1:2,3:2,3:3", help="K:BURST per speculative boot; default = no pin")
    ap.add_argument("--compare", type=pathlib.Path)
    ap.add_argument("--penalised", choices=("device", "host"), default="device")
    ap.add_argument("--carry", choices=("yes", "no"), default="yes",
                    help="no: a binary from before memra#1186 (the carry assertions are reported, not judged)")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    args = ap.parse_args()
    if args.out is None:
        args.out = pathlib.Path(tempfile.mkdtemp(prefix="spec-burst-boundary-"))
    if (args.out / "runs.jsonl").exists():
        raise SystemExit(f"{args.out} already holds a run: give a fresh --out")
    (args.out / "texts").mkdir(parents=True, exist_ok=True)
    failures, reported, total = [], [], 0
    reference, ref_tokens, ref_finish, spec_reference = {}, {}, {}, {}
    stops = {}
    seen = {}  # (end kind, sampled|greedy) -> {"on": n, "after": n}
    tail_commits = carries = sampled_spec_rows = 0

    def judge(case, pairs):
        nonlocal total, tail_commits, carries, sampled_spec_rows
        for row, text in pairs:
            total += 1
            rid = row["id"]
            host_drawn = args.penalised == "host" and row["regime"] == PENALISED and case not in ("ref", "ref2", "refctx")
            if row["error"]:
                failures.append(f"{case} {rid}: {row['error']}")
                continue
            if case in ("ref", "ref2", "refctx"):
                reference[rid], ref_tokens[rid], ref_finish[rid] = text, row["completion_tokens"], row["finish"]
                continue
            if host_drawn:
                if case in ("yield", "load", "ctx"):
                    # these requests have one speculative text each, and a demotion hands the
                    # tail to the host sampler: nothing to compare them with
                    reported.append(f"{case} {rid}: sha {row['sha']} (penalised and host-drawn in plain decode, not judged)")
                elif text != spec_reference.setdefault(rid, text):
                    failures.append(f"{case} {rid}: differs from the first speculative text")
            elif row["kind"] == "ctx":
                # Speculation stops short of a full context by its draft length.
                ref = reference.get(rid, "")
                if not ref.startswith(text) or row["finish"] != ref_finish.get(rid):
                    failures.append(f"{case} {rid}: not a prefix of the plain text, or another finish reason "
                                    f"({row['finish']} against {ref_finish.get(rid)})")
            elif text != reference.get(rid):
                ref = reference.get(rid, "")
                same = next((i for i, (x, y) in enumerate(zip(text, ref)) if x != y), min(len(text), len(ref)))
                failures.append(f"{case} {rid}: differs from the reference after {same} characters")
            elif row["finish"] != ref_finish.get(rid):
                failures.append(f"{case} {rid}: finish {row['finish']} against the reference's {ref_finish.get(rid)}")
            elif row["completion_tokens"] != ref_tokens.get(rid):
                failures.append(f"{case} {rid}: {row['completion_tokens']} completion tokens against the "
                                f"reference's {ref_tokens.get(rid)}")
            if row["regime"] != "greedy":
                sampled_spec_rows += 1
                tail_commits += row["tail_commits"]
                carries += row["carries"]
            kind = {"eos": "eos", "budget": "budget", "stop": "stop"}.get(row["kind"])
            if kind == "eos" and row["finish"] != "stop":
                failures.append(f"{case} {rid}: the answer did not end on EOS (finish {row['finish']})")
            elif kind and not host_drawn:
                cell = seen.setdefault((kind, "greedy" if row["regime"] == "greedy" else "sampled"), {"on": 0, "after": 0})
                for where in placement(row, ref_tokens.get(rid) or 0):
                    cell[where] += 1

    print("case ref", flush=True)
    judge("ref", run_ref(args, stops))
    for regime in ("t1", "greedy"):
        if not stops.get(regime):
            failures.append(f"ref: no stop string could be taken from the {regime} reference answer")
    for spec in args.grid.split(","):
        if spec == "default":
            case, env = "spec", {}
        else:
            k, burst = spec.split(":")
            case, env = f"k{k}-b{burst}", {"MEMRA_SPEC_K": k, "MEMRA_SPEC_BURST": burst}
        print(f"case {case}", flush=True)
        judge(case, run_spec(args, case, env, stops))
    print("case yield", flush=True)
    yield_rows, no_round = run_yield(args, "yield", {"MEMRA_SPEC_GATE": "0"})
    judge("yield", yield_rows)
    print("case load", flush=True)
    load_rows, load = run_yield(args, "load", {})
    judge("load", load_rows)
    print("case refctx", flush=True)
    judge("refctx", run_ctx(args, "refctx", {"MEMRA_SPEC_K": "0"}))
    print("case ctx", flush=True)
    judge("ctx", run_ctx(args, "ctx", {"MEMRA_SPEC_SETUP_TRACE": "1", "MEMRA_SPEC_BOUNDARY_TRACE": "1"}))

    print("\nplacement of the request's last token (rows that end on a burst boundary / first after one):")
    for kind in ("eos", "stop", "budget"):
        for regime in ("sampled", "greedy"):
            cell = seen.get((kind, regime), {"on": 0, "after": 0})
            print(f"  {kind:>6} {regime:>7}: on the boundary {cell['on']}, first after it {cell['after']}")
            if cell["on"] + cell["after"] == 0:
                failures.append(f"placement: no {regime} {kind} request ended on a burst boundary (nothing tested)")
    print(f"yield boot: {no_round['any']} continuation bursts ran no round, {no_round['carried']} of them entered on "
          f"a carried token; {no_round['demoted']} demotions")
    print(f"load boot: {load['any']} continuation bursts ran no round, {load['demoted']} demotions to plain decode")
    if (no_round["carried"] if args.carry == "yes" else no_round["any"]) == 0:
        failures.append("no-round: the yield boot held no continuation burst without a round"
                        + (" on a carried token" if args.carry == "yes" else "") + " (nothing tested)")
    line = (f"carry: {tail_commits} boundary draws at a sampled burst's tail, {carries} sampled bursts entered on a "
            f"carried token, over {sampled_spec_rows} sampled speculative rows")
    print(line)
    carry_problems = []
    if tail_commits:
        carry_problems.append(f"carry: {tail_commits} sampled bursts committed their pending token with a tail pass")
    if not carries:
        carry_problems.append("carry: no sampled burst entered on a carried token")
    if args.carry == "yes":
        failures += carry_problems
    else:
        reported += [p + " (expected of a binary from before memra#1186)" for p in carry_problems]
        if not carry_problems:
            failures.append("carry: --carry no, and the binary carried (it is not a pre-memra#1186 binary)")

    if args.compare:
        other = {}
        for raw in args.compare.read_text().splitlines():
            if raw.strip():
                r = json.loads(raw)
                other[(r["case"], r["id"])] = r
        equal = differ = missing = tokens_equal = 0
        for raw in (args.out / "runs.jsonl").read_text().splitlines():
            r = json.loads(raw)
            key = (r["case"], r["id"])
            if key not in other:
                missing += 1
                failures.append(f"compare: {key[0]} {key[1]} is not in {args.compare.name}")
                continue
            o = other[key]
            if o["sha"] == r["sha"] and o["finish"] == r["finish"]:
                equal += 1
            else:
                differ += 1
                failures.append(f"compare: {key[0]} {key[1]} sha {r['sha']} finish {r['finish']} against the other "
                                f"run's {o['sha']} {o['finish']}")
            if o["completion_tokens"] == r["completion_tokens"]:
                tokens_equal += 1
            else:
                failures.append(f"compare: {key[0]} {key[1]} {r['completion_tokens']} completion tokens against the "
                                f"other run's {o['completion_tokens']}")
        print(f"compare with {args.compare}: {equal} rows with the same text and finish, {differ} differ, {missing} "
              f"missing; completion tokens equal on {tokens_equal} rows")
    for r in reported:
        print("  reported: " + r)
    print(f"\nreceipts: {args.out}")
    if failures:
        print(f"SPEC BURST BOUNDARY: FAIL ({len(failures)} problems over {total} rows)")
        for f in failures:
            print("  " + f)
        return 1
    print(f"SPEC BURST BOUNDARY: PASS ({total} rows, grid {args.grid})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
