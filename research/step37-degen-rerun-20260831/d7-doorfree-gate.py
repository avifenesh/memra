#!/usr/bin/env python3
# PRE-CELL door-free gate (added by the re-run, recorded in PLAN-DIFF.md before any
# generation). The re-run's whole claim is "same cell, uncorrupted binary", so the boot
# has to PROVE uncorrupted text before a single evaluated row runs - a door-free commit is
# not by itself proof that this box's load path is clean. Two independent reasons to gate:
#
#   1. The 2026-08-29 corruption was MARGIN-dependent, not length-dependent: at 25 prompt
#      tokens the first token already diverged, at 613 tokens it still agreed. Every launch
#      gate passed while the server babbled. So the gate prompt must be the SHORT one the
#      incident bisected on, verbatim.
#   2. This box carries a 100 GB `.memra-repack` expert-stack cache written 2026-08-28 by
#      another lane, i.e. in the door era. The cache is keyed by shape + length, not by a
#      content hash, so "a stale cache cannot poison this boot" is an assumption until a
#      decisive output probe says otherwise.
#
# Protocol = the incident's own post-fix battery (darklanes
# research/step37-degen-incident-20260829/INCIDENT.md): the same arithmetic prompt, 8
# greedy + 8 vendor-default sampled. It failed 0/8 greedy and 1/8 sampled with the door
# ON. Greedy is the INSTRUMENT here (byte-deterministic first token = the incident's
# oracle), never the product; the sampled arm is the customer request shape and is what
# the cell itself uses.
import importlib.util, json, os, sys

LANE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d7drive", LANE + "/d7-drive.py")
d7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d7)

PROMPT = "What is 17*23? Reply with the number only."
ANSWER = "391"
N = 8
out = open(LANE + "/receipts/doorfree-gate.jsonl", "w", buffering=1)


def once(greedy, i):
    msgs = [{"role": "user", "content": PROMPT}]
    payload = {"model": "step37", "messages": msgs, "stream": True, "max_tokens": 2048,
               "stream_options": {"include_usage": True}}
    if greedy:
        payload["temperature"] = 0.0
        payload["top_p"] = 1.0
    # Vendor-default arm sends NO sampling params at all: that is the real customer shape
    # (models.toml defaults temp 0.5 / top_p 0.9), and a spec-engagement receipt on THAT
    # shape is what "verified" means per the serving law.
    import time, urllib.request
    req = urllib.request.Request(d7.URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); reason = []; content = []; usage = None; finish = None
    r = urllib.request.urlopen(req, timeout=900)
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        s = line[5:].strip()
        if s == "[DONE]":
            break
        try:
            ch = json.loads(s)
        except Exception:
            continue
        if ch.get("usage"):
            usage = ch["usage"]
        c0 = (ch.get("choices") or [{}])[0]
        if c0.get("finish_reason"):
            finish = c0["finish_reason"]
        d = c0.get("delta") or {}
        if d.get("reasoning"):
            reason.append(d["reasoning"])
        if d.get("content"):
            content.append(d["content"])
    reasoning = "".join(reason); text = "".join(content)
    both = reasoning + text
    row = dict(arm="greedy" if greedy else "vendor_default", rep=i, finish=finish,
               reasoning_chars=len(reasoning), content_chars=len(text),
               first_16=both[:16], answer_present=(ANSWER in both),
               total=round(time.perf_counter() - t0, 3),
               spec=(usage or {}).get("spec"), reasoning=reasoning, content=text)
    out.write(json.dumps(row) + "\n")
    return row


fails = 0
summary = {}
for greedy in (True, False):
    ok = 0
    specs = 0
    for i in range(1, N + 1):
        row = once(greedy, i)
        if row["answer_present"]:
            ok += 1
        sp = row["spec"] or {}
        if (sp.get("rounds") or 0) > 0 or (sp.get("accepted") or 0) > 0:
            specs += 1
        print("GATE %-14s r%d answer=%s finish=%s first16=%r spec_acc=%s"
              % (row["arm"], i, row["answer_present"], row["finish"], row["first_16"],
                 sp.get("acceptance_rate")), flush=True)
    arm = "greedy" if greedy else "vendor_default"
    summary[arm] = dict(correct="%d/%d" % (ok, N), spec_engaged="%d/%d" % (specs, N))
    if ok != N:
        fails += 1
json.dump(summary, open(LANE + "/receipts/doorfree-gate-summary.json", "w"), indent=1)
print("DOORFREE_GATE " + json.dumps(summary), flush=True)
if fails:
    print("DOORFREE_GATE_FAIL: the boot does not produce correct text on the incident's "
          "own short prompt - the cell MUST NOT run", flush=True)
    sys.exit(11)
print("DOORFREE_GATE_PASS", flush=True)
