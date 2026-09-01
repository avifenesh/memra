#!/usr/bin/env python3
"""Is the degeneration a PROMPT property or a PREFIX-RESTORE property?

Every degenerate row this lane has produced was a prefix-cache RESTORE, and every cold row was
clean. Three independent sightings, none of them designed to show this:
  * prompt idx 2 re-check: rep1 cold (ttft 7.97s) clean; reps 2,3,4 restored (ttft ~0.019s)
    all degenerate, two of them looping "</think>" into the content;
  * post-deploy p5 probe: rep A cold clean, rep B restored looped the reasoning (score 1.08);
  * the tool-call cell: cold 12/12 emitted the tool call, restored 0/12 did.
It also reframes the earlier decode-attribution lane's exclusion of idx 2: steady.py PRIMES the
prompt and then repeats it, so every row it measured after the prime was a restore.

This cell separates the two. Same prompt, same sampler, same server; the only difference is
whether a unique nonce prefix forces a cold prefill. Arms are INTERLEAVED rep by rep.

usage: degenrate.py <outdir> <reps> <prompt_idx...>
"""
import hashlib, json, os, sys, urllib.request, uuid

OUT = sys.argv[1]
REPS = int(sys.argv[2])
IDXS = [int(x) for x in sys.argv[3:]] or [2, 5, 7, 9]
EP = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
POOL = json.load(open("/home/ubuntu/prompts.json"))["decode"]
os.makedirs(OUT, exist_ok=True)


def loopiness(s, w=48):
    if len(s) < 4 * w:
        return 0.0
    tail = s[-2000:]
    seen, best = {}, 0
    for i in range(0, len(tail) - w):
        k = tail[i:i + w]
        seen[k] = seen.get(k, 0) + 1
        best = max(best, seen[k])
    return round(best * w / len(tail), 3)


def run(content, name):
    """VENDOR-DEFAULT sampled: no temperature, no top_p, no seed. effort PINNED."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 512, "reasoning_effort": "low", "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(EP + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"content-type": "application/json"})
    think, out, fr, usage = [], [], None, None
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            for rl in r:
                s = rl.decode().strip()
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
                if d.get("content"):
                    out.append(d["content"])
                if d.get("reasoning"):
                    think.append(d["reasoning"])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    reasoning, content_txt = "".join(think), "".join(out)
    u = usage or {}
    open(f"{OUT}/{name}.txt", "w").write(
        "=== REASONING ===\n" + reasoning + "\n=== CONTENT ===\n" + content_txt)
    lr, lc = loopiness(reasoning), loopiness(content_txt)
    return {"finish": fr, "prompt_tokens": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "loopR": lr, "loopC": lc,
            "degenerate": max(lr, lc) >= 0.15,
            "think_close_leaks": content_txt.count("</think>"),
            "content_chars": len(content_txt),
            "sha16": hashlib.sha256((reasoning + content_txt).encode()).hexdigest()[:16],
            "head": content_txt[:90] or "(no answer text)"}


ALL = []
tally = {"COLD": [0, 0], "RESTORED": [0, 0]}   # [degenerate, total]
for idx in IDXS:
    base = POOL[idx]["text"]
    print("#" * 76)
    print(f"# prompt idx {idx}  ({POOL[idx]['chars']} chars, sha {POOL[idx]['sha256_16']})")
    print("#" * 76)
    for rep in range(REPS):
        for arm in ("COLD", "RESTORED"):
            content = f"[session {uuid.uuid4().hex}] {base}" if arm == "COLD" else base
            r = run(content, f"p{idx}-{arm}-rep{rep}")
            r.update({"prompt_idx": idx, "arm": arm, "rep": rep})
            ALL.append(r)
            if "error" not in r:
                tally[arm][1] += 1
                tally[arm][0] += r["degenerate"]
                print(f"  {arm:>8} rep{rep}: degenerate={str(r['degenerate']):>5} "
                      f"loopR={r['loopR']:<5} loopC={r['loopC']:<5} "
                      f"</think>-in-content={r['think_close_leaks']:<3} "
                      f"cached={str(r['cached']):>4}/{str(r['prompt_tokens']):>4} "
                      f"{r['head'][:52]!r}")
            else:
                print(f"  {arm:>8} rep{rep}: {r['error']}")
    print()

json.dump(ALL, open(f"{OUT}/degenrate.json", "w"), indent=1)
print("=" * 76)
for arm in ("COLD", "RESTORED"):
    d, n = tally[arm]
    print(f"  {arm:>8}: {d}/{n} degenerate"
          + (f"  ({100*d/n:.0f}%)" if n else ""))
leaks_cold = sum(r.get("think_close_leaks", 0) for r in ALL if r.get("arm") == "COLD")
leaks_rest = sum(r.get("think_close_leaks", 0) for r in ALL if r.get("arm") == "RESTORED")
print(f"  stray '</think>' emitted into CONTENT: cold={leaks_cold} restored={leaks_rest}")
print("DEGENRATEDONE")
