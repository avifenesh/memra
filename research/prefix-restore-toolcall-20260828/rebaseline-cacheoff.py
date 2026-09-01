#!/usr/bin/env python3
"""Re-baseline the three items the latent-plane defect contaminated. TURNKEY, CACHE OFF.

WHY. research/prefix-restore-toolcall-20260828/FINDINGS.txt establishes that on glm5_next a
prefix-cache restore hands the model an EMPTY attention history. Three findings in the parent
lane (research/glm53-flash-bringup-20260827/rebaseline-and-surface-20260828/) were measured
with that defect live and are therefore attributions, not measurements, until re-run:

  R1  "prompt idx 2 STILL DEGENERATES": rep 1 cold clean, reps 2-4 restored degenerate.
      That is the defect's exact signature. If idx 2 is clean on every rep with the cache
      off, the degeneration was the cache and the "model behaviour, reported separately per
      the greedy law" line has to be withdrawn.
  R2  the OPEN, UNQUANTIFIED "roughly 25-40% of requests degenerate across both arms". Its
      own cell was already confounded by two-entry eviction; it is additionally confounded by
      this. This cell replaces it with a number measured where no restore can occur.
  R3  the COLD-VERSUS-RESTORED BYTE DIVERGENCE that BOX-STATE.txt records as a known engine
      quirk (raw p5 cold 5113588f11c49d5e vs restored d2f40996290fe905, p7 d78355d162c3609c
      vs ef197d4dd6c6cb6a). With the cache off there is no restore, so every rep of one
      prompt must produce one sha. A second sha would mean a SECOND defect and would need
      its own lane.

PRECONDITION, asserted rather than assumed: the server must be running with
MEMRA_PREFIX_CACHE_MB=0 (or the guarded binary). The script refuses to run if any request
comes back with cached_tokens > 0, because a cell that silently measures the contaminated
regime is worse than no cell.

Greedy is the instrument (R3's byte oracle). R1 and R2 are vendor-default SAMPLED, which is
the product shape and the shape the parent lane's claims were made on. reasoning_effort is
PINNED low throughout. A greedy loop is an artifact and never a finding; R2 counts loops in
the SAMPLED arm only and reports them as a rate, not as a defect.

usage: rebaseline-cacheoff.py <outdir> [reps_r1] [reps_r2] [reps_r3]
"""
import hashlib, json, os, sys, time, urllib.request

OUT = sys.argv[1]
R1_REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
R2_REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
R3_REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
EP = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
# The parent lane's own 10-prompt agent pool. Path comes from the environment because the
# cell has already had to run on two boxes with different homes; the default is where the
# parent lane kept it.
POOL = json.load(open(os.environ.get("PROMPTS_JSON", "/home/ubuntu/prompts.json")))["decode"]
os.makedirs(OUT, exist_ok=True)
ALL = {"r1": [], "r2": [], "r3": []}
VIOLATIONS = []


def loopiness(s, w=48):
    """The parent lane's own crude repeat-window score, kept identical so the numbers compare."""
    if len(s) < 4 * w:
        return 0.0
    tail = s[-2000:]
    seen, best = {}, 0
    for i in range(0, len(tail) - w):
        k = tail[i:i + w]
        seen[k] = seen.get(k, 0) + 1
        best = max(best, seen[k])
    return round(best * w / len(tail), 3)


def guard(row, where):
    """A restore in a cache-off cell invalidates the row. Record it; never silently accept."""
    if (row.get("cached_tokens") or 0) > 0:
        VIOLATIONS.append(f"{where}: cached_tokens={row['cached_tokens']} (cache is NOT off)")


def chat(content, name, greedy=False, max_tokens=512):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "reasoning_effort": "low", "stream": True,
            "stream_options": {"include_usage": True}}
    if greedy:
        body["temperature"] = 0.0
    req = urllib.request.Request(EP + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"content-type": "application/json"})
    think, out, fr, usage, sse = [], [], None, None, []
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for rl in r:
                line = rl.decode(); sse.append(line)
                s = line.strip()
                if not s.startswith("data:"):
                    continue
                pay = s[5:].strip()
                if pay == "[DONE]":
                    break
                o = json.loads(pay)
                if o.get("usage"):
                    usage = o["usage"]
                c = (o.get("choices") or [{}])[0]
                if c.get("finish_reason"):
                    fr = c["finish_reason"]
                d = c.get("delta") or {}
                if d.get("reasoning") or d.get("reasoning_content"):
                    think.append(d.get("reasoning") or d.get("reasoning_content"))
                if d.get("content"):
                    out.append(d["content"])
    except Exception as e:
        sse.append(f"{type(e).__name__}: {e}")
    open(f"{OUT}/{name}.sse", "w").write("".join(sse))
    text, thought = "".join(out), "".join(think)
    u = usage or {}
    return {"name": name, "finish": fr,
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "loop_content": loopiness(text), "loop_reasoning": loopiness(thought),
            "thinking_chars": len(thought),
            "wall_s": round(time.time() - t0, 3),
            "out_sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
            "content": text[:200]}


def raw(idx, name, max_tokens=64):
    """/v1/completions: NO chat template. The parent lane's attribution discriminator."""
    prompt = POOL[idx]["text"]
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "stream": False, "temperature": 0.0}
    req = urllib.request.Request(EP + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    r = {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            r = json.loads(resp.read().decode())
    except Exception as e:
        r = {"__error__": f"{type(e).__name__}: {e}"}
    ch = (r.get("choices") or [{}])[0]
    txt = ch.get("text", "") or ""
    u = r.get("usage") or {}
    return {"name": name, "idx": idx,
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "finish": ch.get("finish_reason"),
            "out_sha16": hashlib.sha256(txt.encode()).hexdigest()[:16],
            "head": txt[:120], "error": r.get("__error__")}


print("#" * 78)
print("# R1  prompt idx 2, sequential reps, VENDOR-DEFAULT SAMPLED, cache OFF")
print("#     parent lane saw: rep1 clean (cold), reps 2-4 degenerate (restored)")
print("#" * 78)
for i in range(R1_REPS):
    row = chat(POOL[2]["text"], f"r1-idx2-rep{i}")
    row["rep"] = i
    guard(row, f"R1 rep{i}")
    ALL["r1"].append(row)
    print(f"  rep{i}: finish={str(row['finish']):>8} cached={row['cached_tokens']} "
          f"loop_content={row['loop_content']:<6} loop_reasoning={row['loop_reasoning']:<6} "
          f"think_chars={row['thinking_chars']:<6} content={row['content'][:70]!r}")
print()

print("#" * 78)
print("# R2  degeneration RATE over the 10 real agent prompts, SAMPLED, cache OFF")
print("#     replaces the parent lane's open 'roughly 25-40%, both arms, confounded'")
print("#" * 78)
deg = tot = 0
for idx in range(len(POOL)):
    for i in range(R2_REPS):
        row = chat(POOL[idx]["text"], f"r2-p{idx}-rep{i}")
        row["idx"], row["rep"] = idx, i
        guard(row, f"R2 p{idx} rep{i}")
        # The parent lane's own threshold: a repeat-window score above 0.5 in either channel.
        row["degenerate"] = row["loop_content"] > 0.5 or row["loop_reasoning"] > 0.5
        ALL["r2"].append(row)
        deg += row["degenerate"]; tot += 1
        print(f"  p{idx} rep{i}: degenerate={str(row['degenerate']):>5} "
              f"loop_c={row['loop_content']:<6} loop_r={row['loop_reasoning']:<6} "
              f"finish={str(row['finish']):>8} cached={row['cached_tokens']}")
print(f"  == RATE with no restore possible: {deg}/{tot} = {100.0*deg/max(tot,1):.1f}%")
print()

print("#" * 78)
print("# R3  raw /v1/completions greedy byte oracle, cache OFF, repeated")
print("#     parent BOX-STATE recorded cold != restored: p5 5113588f11c49d5e vs")
print("#     d2f40996290fe905, p7 d78355d162c3609c vs ef197d4dd6c6cb6a")
print("#" * 78)
for idx in (5, 7):
    shas = []
    for i in range(R3_REPS):
        row = raw(idx, f"r3-p{idx}-rep{i}")
        row["rep"] = i
        guard(row, f"R3 p{idx} rep{i}")
        ALL["r3"].append(row)
        shas.append(row["out_sha16"])
        print(f"  p{idx} rep{i}: sha={row['out_sha16']} cached={row['cached_tokens']} "
              f"finish={row['finish']} err={row['error']}")
    print(f"  == p{idx}: {len(set(shas))} distinct sha over {R3_REPS} reps "
          f"({'BYTE-IDENTICAL, the divergence was the cache' if len(set(shas)) == 1 else 'STILL DIVERGES, a second defect'})")
print()

json.dump({"rows": ALL, "violations": VIOLATIONS},
          open(f"{OUT}/rebaseline.json", "w"), indent=1)
if VIOLATIONS:
    print("!! CELL INVALID: the prefix cache was NOT off. " + str(len(VIOLATIONS)) + " rows hit it:")
    for v in VIOLATIONS[:10]:
        print("   " + v)
    sys.exit(2)
print("REBASELINEDONE (no restore occurred in any row)")
