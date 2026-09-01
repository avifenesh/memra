# lane/step37-affinity driver.
#
# LEG 1 (TTFT win, interleaved x5): a GROWING multi-turn conversation with PRE-BAKED assistant
# texts (never a live output: both binaries must prime byte-identical prompts). Turn k's prompt is
# [U1,A1,...,Uk]; its checkpoint boundary (the last turn marker) is strictly ahead of the resumed
# prefix, so every turn re-arms and every warm turn after the first can rewind. Each warm turn k
# is paired with a COLD twin: the same messages with a unique nonce on U1 and a unique session_id,
# so nothing can match it. Pairs are sent cold-then-warm, k = 2..6 -> 5 interleaved pairs. Prompt
# length grows across turns, so the win is pooled PAIRWISE, never as raw TTFT medians.
#
# LEG 2 (byte identity): one fixed prompt, one session_id, sent twice. Send 1 is cold, send 2
# should rewind. Same prompt, same binary, greedy -> the reused answer must be byte-identical.
#
# An empty completion is a FAILED row: sha256("") is a constant, so two empties compare EQUAL and
# a naive gate reports a false PASS. Empties are rejected here, in the instrument.
import json, hashlib, os, time, urllib.request, urllib.error

port = os.environ["P"]; name = os.environ["NAME"]; logp = os.environ["LOG"]
TURNS = int(os.environ.get("TURNS", "6"))
base = json.load(open("/root/curve-1000.json"))
U1 = base["messages"][0]["content"]
URL = "http://127.0.0.1:%s/v1/chat/completions" % port

# Pre-baked, deterministic. Identical bytes in every arm and both binaries.
ABAKE = ("Here is the digested summary you asked for. The run covered trajectory optimization "
         "with iLQR, DDP and MPC, ran the nightly learning loop end to end, and recorded the "
         "per-stage timings alongside the convergence traces. The main takeaway is that the "
         "line-search schedule dominates wall time on this workload. ") * 3
UFOLLOW = ["Now expand point %d of that summary and explain the trade-offs in detail." % i
           for i in range(1, 12)]

def messages(turn, nonce=""):
    """Conversation through user turn `turn` (1-based)."""
    msgs = [{"role": "user", "content": nonce + U1}]
    for k in range(2, turn + 1):
        msgs.append({"role": "assistant", "content": ABAKE})
        msgs.append({"role": "user", "content": UFOLLOW[k - 2]})
    return msgs

def body(msgs, sid):
    return {"model": "step37", "messages": msgs, "stream": True,
            "max_tokens": 128, "temperature": 0, "session_id": sid}

FIRST_KEYS = []
def stream_once(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; parts = []
    try:
        r = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        return (None, time.perf_counter() - t0, "", e.read().decode("utf-8", "replace")[:300])
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
        d = ch.get("choices", [{}])[0].get("delta", {}) or {}
        if d and not FIRST_KEYS:
            # LOUD INSTRUMENT CHECK: if the server streams thinking under a key this driver does
            # not read, every row would hash the empty string. Print the keys we actually saw.
            FIRST_KEYS.append(sorted(d.keys()))
        piece = (d.get("reasoning") or "") + (d.get("content") or "")
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(piece)
    return (ttft, time.perf_counter() - t0, "".join(parts), None)

logf = open(logp, "r", errors="replace"); logf.seek(0, 2)
rows = []

def send(tag, turn, msgs, sid):
    pos = logf.tell()
    ttft, total, text, err = stream_once(body(msgs, sid))
    logf.seek(pos); sl = logf.read(); logf.seek(0, 2)
    rewound = "plain-affinity: rewound to" in sl
    refused = ("plain-affinity resume failed" in sl) or ("affinity rewind failed" in sl) \
              or ("TP KV kind mismatch" in sl) or ("TP KV restore refused" in sl)
    dropped = "stale distributed KV mirror" in sl
    grew = "plain-affinity: grew parked cache" in sl
    sha = hashlib.sha256(text.encode()).hexdigest()[:16] if text.strip() else None
    r = dict(tag=tag, turn=turn, ttft=ttft, total=total, chars=len(text), sha=sha,
             rewound=rewound, refused=refused, dropped=dropped, grew=grew, err=err)
    rows.append(r)
    rw = [l for l in sl.splitlines() if "plain-affinity: rewound to" in l]
    print("    %-9s turn%d ttft=%s total=%.3f chars=%d sha=%s rewound=%s refused=%s dropped=%s%s"
          % (tag, turn, ("%.3f" % ttft) if ttft is not None else "NONE", total, len(text),
             sha or "EMPTY", rewound, refused, dropped, (" err=%s" % err) if err else ""))
    if rw:
        print("        receipt: %s" % rw[0].strip()[:200])
    return r

print("    --- LEG 1: growing multi-turn, cold twin per turn (interleaved) ---")
send("warm", 1, messages(1), "warm-%s" % name)
for k in range(2, TURNS + 1):
    send("cold", k, messages(k, nonce="[trace %s c%02d] " % (name, k)), "cold-%s-%02d" % (name, k))
    send("warm", k, messages(k), "warm-%s" % name)

print("    --- LEG 2: byte identity, one fixed prompt sent twice on one session ---")
idm = messages(3, nonce="[identity %s] " % name)
send("id-cold", 3, idm, "ident-%s" % name)
send("id-warm", 3, idm, "ident-%s" % name)

print("    delta keys observed on the wire: %s" % (FIRST_KEYS[0] if FIRST_KEYS else "NONE SEEN"))
EMPTY = hashlib.sha256(b"").hexdigest()[:16]
valid = [r for r in rows if r["sha"] is not None and r["ttft"] is not None and not r["err"]]
print("    valid_rows=%d/%d  (empty completions rejected in the instrument; sha256('')=%s)"
      % (len(valid), len(rows), EMPTY))
if not valid:
    print("    ARM INVALID: no valid rows"); raise SystemExit(0)

def stats(v):
    if not v: return "n=0"
    s = sorted(v); m = s[len(s)//2] if len(s) % 2 else (s[len(s)//2-1]+s[len(s)//2])/2
    return "n=%d median=%.3f min=%.3f max=%.3f spread=%.3f" % (len(s), m, s[0], s[-1], s[-1]-s[0])

by = {(r["tag"], r["turn"]): r for r in valid}
print("    --- PAIRWISE TTFT (cold twin vs warm same turn) ---")
wins, ratios = [], []
for k in range(2, TURNS + 1):
    c, w = by.get(("cold", k)), by.get(("warm", k))
    if not c or not w: continue
    print("      turn%d cold=%.3f warm=%.3f delta=%+.3fs ratio=%.2fx warm_rewound=%s"
          % (k, c["ttft"], w["ttft"], c["ttft"]-w["ttft"], c["ttft"]/w["ttft"], w["rewound"]))
    wins.append(c["ttft"] - w["ttft"]); ratios.append(c["ttft"] / w["ttft"])
print("    pairwise TTFT win seconds: %s" % stats(wins))
print("    pairwise TTFT ratio      : %s" % stats(ratios))
print("    warm turns rewound: %s of %s"
      % (sum(1 for r in valid if r["tag"] == "warm" and r["turn"] > 1 and r["rewound"]),
         sum(1 for r in valid if r["tag"] == "warm" and r["turn"] > 1)))

ic, iw = by.get(("id-cold", 3)), by.get(("id-warm", 3))
if ic and iw:
    print("    --- LEG 2 IDENTITY: cold sha=%s chars=%d | reused sha=%s chars=%d rewound=%s -> %s"
          % (ic["sha"], ic["chars"], iw["sha"], iw["chars"], iw["rewound"],
             "MATCH" if ic["sha"] == iw["sha"] else "DIFFER"))
else:
    print("    --- LEG 2 IDENTITY: INVALID (a leg-2 row was empty or errored)")
print("    CROSSARM " + json.dumps({"%s-turn%d" % (r["tag"], r["turn"]):
                                    [r["sha"], r["chars"], r["rewound"]] for r in valid}))
