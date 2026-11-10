#!/usr/bin/env python3
"""Seeded-text invariance gate (memra#1154): one seeded sampled request writes one text.

The MTP speculative route verifies a sampled round by position: the token at generated
position p is the plain sampler's own draw at Philox event p. So what a seeded request writes
may not depend on the draft length, on a companion request, on a resume, or on a demotion from
speculation to plain decode. This gate boots the server once per case and compares texts:

  ref     MEMRA_SPEC_K=0 (plain decode): every prompt alone and cold. The reference texts.
  K=0..3  MEMRA_SPEC_K pinned: the lone prompts again, two prompts sent at once, a two-turn
          conversation whose second turn resumes the first, and a text completion sent right
          after the request that parked exactly its prompt (nothing new to prime).
  demote  no K pin, MEMRA_SPEC_DEMOTE_AT=N: speculation hands the request to plain decode
          after N tokens. The server log must show the handoff for every request.

Prompts: a long and a short one at temperature 1, a long one at temperature 0.7 with
top_p 0.95, a short one with a frequency penalty, the pair, the conversation. Every text must
equal the reference text of its prompt. Exit 0 on a full match, 1 on any difference or error.

--penalised host: for a model and card where plain decode samples a penalised request on the
host (MEMRA_SERVE_DEVPENALTY off: every pair except the dense Qwen35 class on the RTX 5090 and
RTX PRO 6000 classes and the dense Qwen35 and Qwen35Moe classes on GB10). Plain decode and
speculation then draw the penalised prompt from different random streams, so its speculative
texts are compared with each other (the K=1 text) instead of with the plain reference, and its
demotion row is reported, not judged.

usage: seeded-invariance-gate.py SERVER_BINARY MODEL.gguf [--out DIR] [--port N]
           [--max-tokens N] [--cases ref,0,1,2,3,demote] [--demote-at N] [--penalised device|host]
           [--env NAME=VALUE ...]   (extra server environment for every boot)
A GPU lease is the caller's business (gpulock / sparklock around the whole run).
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


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic):
    return f"{topic.capitalize()} field notes. {notes(topic, 45)}\n\nWrite about 300 words on what these notes say about {topic}."


def short_prompt(topic):
    return f"{topic.capitalize()} in brief: write about 300 words on how a {topic} survey is planned and run."


def user(content):
    return [{"role": "user", "content": content}]


def T1(seed):
    return {"temperature": 1.0, "seed": seed}


def CHAT(seed):
    return {"temperature": 0.7, "top_p": 0.95, "seed": seed}


def PEN(seed):
    return {"temperature": 1.0, "frequency_penalty": 0.4, "seed": seed}


A, B = long_prompt("volcanoes"), short_prompt("meadow")
C, D = long_prompt("canyons"), long_prompt("forests")
F, G = long_prompt("harbors"), short_prompt("orchard")
E1 = long_prompt("lagoons")
E2 = "Now give three concrete next steps for the same survey, in about 150 words."
# The parked-prompt pair, on /v1/completions (raw text): P1 answers, and P2's prompt is P1's
# prompt plus that answer, so the session P1 parks holds exactly P2's prompt.
P1 = f"Glacier survey log. {notes('glaciers', 8)}\n\nSummary of the log:"
PENALISED = "g"
LONE = [("a", user(A), T1(701)), ("b", user(B), T1(702)), ("f", user(F), CHAT(707)), (PENALISED, user(G), PEN(708))]


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

    def ask(self, messages, samp, out):
        body = {"model": "q", "messages": messages, "max_tokens": self.args.max_tokens, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}, **samp}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                obj = json.loads(r.read())
            out.update(text=obj["choices"][0]["message"]["content"] or "", usage=obj.get("usage") or {}, error=None)
        except Exception as exc:  # recorded and counted as a failure, never swallowed
            out.update(text="", usage={}, error=repr(exc))

    def complete(self, prompt, samp, out, max_tokens):
        body = {"model": "q", "prompt": prompt, "max_tokens": max_tokens, "stream": False, **samp}
        req = urllib.request.Request(self.url + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                obj = json.loads(r.read())
            out.update(text=obj["choices"][0].get("text") or "", usage=obj.get("usage") or {}, error=None)
        except Exception as exc:  # recorded and counted as a failure, never swallowed
            out.update(text="", usage={}, error=repr(exc))

    def group(self, items):
        """items = [(context, prompt id, messages or raw text, sampler)], sent at once. A raw
        text goes to /v1/completions (48 tokens for the parking request). Returns the rows."""
        outs = [{} for _ in items]

        def send(payload, samp, out, pid):
            if isinstance(payload, str):
                self.complete(payload, samp, out, 48 if pid == "p1" else self.args.max_tokens)
            else:
                self.ask(payload, samp, out)

        threads = [threading.Thread(target=send, args=(m, s, o, pid)) for (_, pid, m, s), o in zip(items, outs)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        rows = []
        for (context, pid, _, _), o in zip(items, outs):
            usage = o["usage"]
            spec = usage.get("spec") or {}
            row = {"case": self.case, "context": context, "prompt": pid,
                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                   "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "acceptance": spec.get("acceptance_rate"),
                   "sha": hashlib.sha256(o["text"].encode()).hexdigest()[:16], "error": o["error"]}
            # A row that tested nothing is a failure, not a match.
            speculative = self.case not in ("ref", "ref2", "0")
            if row["error"] is None:
                if not o["text"] or not row["completion_tokens"]:
                    row["error"] = "empty answer"
                elif speculative and row["acceptance"] is None:
                    row["error"] = "the request did not speculate (no acceptance in its usage)"
                elif context == "resumed" and not row["cached"]:
                    row["error"] = "the resumed turn restored nothing (cached_tokens is 0)"
                elif context == "parked" and speculative and not row["cached"]:
                    row["error"] = "the request after the park restored nothing (cached_tokens is 0)"
                elif context == "cold" and row["cached"]:
                    row["error"] = f"the cold turn restored {row['cached']} tokens"
            (self.args.out / "texts" / f"{self.case}-{context}-{pid}.txt").write_text(o["text"])
            with open(self.args.out / "runs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {self.case:>6} {context:>8} {pid:>2} tokens={row['completion_tokens']} cached={row['cached']} "
                  f"acceptance={row['acceptance']} sha={row['sha']} {row['error'] or ''}".rstrip(), flush=True)
            rows.append((row, o["text"]))
        return rows


def run_case(args, case):
    rows = []
    if case == "ref":
        srv = Server(args, case, {"MEMRA_SPEC_K": "0"})
        try:
            for pid, messages, samp in LONE:
                rows += srv.group([("alone", pid, messages, samp)])
            rows += srv.group([("alone", "c", user(C), T1(703))])
            rows += srv.group([("alone", "d", user(D), T1(704))])
            turn1 = srv.group([("alone", "e1", user(E1), T1(705))])
            rows += turn1
            (args.out / "texts" / "E1.txt").write_text(turn1[0][1])
            park = srv.group([("alone", "p1", P1, T1(709))])
            rows += park
        finally:
            srv.stop()
        # The second turn and the after-park request cold: their own boot, so nothing of
        # turn 1 or of the parking request is resident.
        srv = Server(args, "ref2", {"MEMRA_SPEC_K": "0"})
        try:
            conv = user(E1) + [{"role": "assistant", "content": turn1[0][1]}, {"role": "user", "content": E2}]
            rows += srv.group([("cold", "e2", conv, T1(706))])
            rows += srv.group([("cold", "p2", P1 + park[0][1], T1(710))])
        finally:
            srv.stop()
        return rows
    env = {"MEMRA_SPEC_DEMOTE_AT": str(args.demote_at)} if case == "demote" else {"MEMRA_SPEC_K": case}
    srv = Server(args, case, env)
    try:
        for pid, messages, samp in LONE:
            rows += srv.group([("alone", pid, messages, samp)])
        if case != "demote":
            rows += srv.group([("pair", "c", user(C), T1(703)), ("pair", "d", user(D), T1(704))])
            turn1 = srv.group([("turn1", "e1", user(E1), T1(705))])
            rows += turn1
            conv = user(E1) + [{"role": "assistant", "content": turn1[0][1]}, {"role": "user", "content": E2}]
            rows += srv.group([("resumed", "e2", conv, T1(706))])
            park = srv.group([("park", "p1", P1, T1(709))])
            rows += park
            rows += srv.group([("parked", "p2", P1 + park[0][1], T1(710))])
    finally:
        srv.stop()
    if case == "demote":
        log = srv.log.read_text(errors="replace")
        handoffs = re.findall(r"generated (\d+), sampled, plain draws from philox event (\d+)", log)
        keyed = sum(1 for generated, event in handoffs if int(event) == int(generated) + 1)
        print(f"  demote: {len(handoffs)} sampled handoffs in the server log for {len(rows)} requests, "
              f"{keyed} at event = generated + 1", flush=True)
        problem = None
        if len(handoffs) != len(rows):
            problem = f"{len(handoffs)} demotion handoffs logged for {len(rows)} requests"
        elif keyed != len(handoffs):
            problem = f"{len(handoffs) - keyed} handoffs left the position key (event != generated + 1)"
        if problem:
            rows.append(({"case": case, "context": "handoff", "prompt": "-", "sha": "-", "error": problem}, ""))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("server")
    ap.add_argument("model")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--port", type=int, default=18197)
    ap.add_argument("--max-tokens", type=int, default=240)
    ap.add_argument("--cases", default="ref,0,1,2,3,demote")
    ap.add_argument("--demote-at", type=int, default=60)
    ap.add_argument("--penalised", choices=("device", "host"), default="device")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    args = ap.parse_args()
    if args.out is None:
        args.out = pathlib.Path(tempfile.mkdtemp(prefix="seeded-invariance-"))
    (args.out / "texts").mkdir(parents=True, exist_ok=True)
    cases = args.cases.split(",")
    if cases[0] != "ref":
        raise SystemExit("the first case must be ref: it writes the reference texts")
    if args.penalised == "host" and sum(c in ("1", "2", "3") for c in cases) < 2:
        raise SystemExit("--penalised host compares speculative texts with each other: give two of the cases 1, 2, 3")
    if "demote" in cases and args.max_tokens < args.demote_at + 32:
        raise SystemExit("--max-tokens must exceed --demote-at by 32 tokens or no request reaches the handoff")
    reference, failures, total, reported = {}, [], 0, []
    spec_reference = {}  # --penalised host: the penalised prompt's first speculative text
    for case in cases:
        print(f"case {case}", flush=True)
        for row, text in run_case(args, case):
            total += 1
            host_drawn = args.penalised == "host" and row["prompt"] == PENALISED and case not in ("ref", "0")
            if row["error"]:
                failures.append(f"{row['case']} {row['context']} {row['prompt']}: {row['error']}")
            elif case == "ref":
                reference[row["prompt"]] = text
            elif host_drawn and case == "demote":
                reported.append(f"demote {row['prompt']}: sha {row['sha']} (host-drawn after the handoff, not judged)")
            elif host_drawn:
                if text != spec_reference.setdefault(row["prompt"], text):
                    failures.append(f"{row['case']} {row['context']} {row['prompt']}: differs from the first speculative text")
            elif text != reference.get(row["prompt"]):
                ref = reference.get(row["prompt"], "")
                same = next((i for i, (x, y) in enumerate(zip(text, ref)) if x != y), min(len(text), len(ref)))
                failures.append(f"{row['case']} {row['context']} {row['prompt']}: differs from the reference after {same} characters")
    for line in reported:
        print("  reported: " + line)
    print(f"\nreceipts: {args.out}")
    if failures:
        print(f"SEEDED INVARIANCE: FAIL ({len(failures)} of {total} rows)")
        for f in failures:
            print("  " + f)
        return 1
    print(f"SEEDED INVARIANCE: PASS ({total} rows, {len(reference)} prompts, cases {','.join(cases)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
