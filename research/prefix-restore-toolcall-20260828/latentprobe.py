#!/usr/bin/env python3
"""Is a prefix-cache RESTORE on glm5_next context-blind?

MECHANISM UNDER TEST (read from the source, then measured here). `prefix_snapshot` /
`prefix_restore_at` in crates/memra-server/src/worker.rs handle exactly two cache state
planes: `cache.kv` (ordinary full-attention K/V) and `cache.recur` (conv/ssm). glm5_next's
ModelPlan declares its 11 full-attention layers as `StatePlan::LatentKvCache`, which the
allocator puts in a THIRD plane, `cache.latent` — a plane the whole prefix-cache path never
reads, writes or validates. So a "whole-entry hit" restores the 34 KDA recurrent states,
sets `cache.pos = N`, and leaves every MLA layer at `len = 0`. Decode then attends over an
EMPTY attention history while `cached_tokens` reports N of N.

PREDICTION: a restored request keeps a diffuse recurrent gist and loses every retrievable
fact — the tool definitions AND anything stated in the prompt.

THREE PROBES, each designed so a fluent guess cannot pass:
  tool    an UNGUESSABLE function name + arbitrary required arg names + an enum value that
          no prior produces. Emitting it proves the tools block was visible.
  recall  an unguessable passphrase stated ONLY in the user prompt, asked straight back.
          Tools are present but irrelevant. Separates "tools block lost" from "prompt lost".
  bare    the same recall probe with NO tools at all. If this fails too, the mechanism is
          not about tools or the tool-call wire at all.

ARM DESIGN: rep 0 of every cell is a COLD prefill of the EXACT bytes reps 1..N-1 restore.
Same boot, same prompt bytes, same tools, same sampler; the ONLY variable is the rep index,
i.e. whether an entry existed. This removes the nonce confound in the parent lane's cell
(its cold arm carried a unique nonce prefix, so cold and restored were not byte-identical).
Each (probe, mode) pair gets its own marker word so its rep 0 is genuinely cold.

Greedy is the instrument (byte-deterministic). Vendor-default sampled (NO temperature, NO
top_p, NO seed) is the product shape. reasoning_effort is PINNED in every request.

usage: latentprobe.py <outdir> [reps]
"""
import json, os, sys, urllib.request, hashlib, time

OUT = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
EP = "http://127.0.0.1:18400"
MODEL = "zai/glm-5.3-flash"
os.makedirs(OUT, exist_ok=True)

# Unguessable by construction: the name, both argument names and the enum member appear
# nowhere outside this block, so a model that cannot see the block cannot fake the call.
TOOLS = [{"type": "function", "function": {
    "name": "zqx_fetch_glimb_status",
    "description": ("Retrieve the current glimb status of a vault. This is the ONLY way to "
                    "obtain a glimb status; never guess or invent one."),
    "parameters": {"type": "object", "properties": {
        "vault_sigil": {"type": "string", "description": "Vault sigil, e.g. VS-7"},
        "glimb_mode": {"type": "string", "enum": ["thrum", "quon"],
                       "description": "Read mode for the glimb sensor"}},
        "required": ["vault_sigil", "glimb_mode"]}}}]

PASSPHRASE = "QUARTZ-77-NIMBUS-4"

PAD = ("You are an operations assistant for a vault-monitoring service. Answer precisely, "
       "keep replies to one short sentence, and never invent sensor readings. If a tool is "
       "declared for a task, call it rather than answering from memory. ")


def prompt_for(probe, mode):
    # marker keeps each (probe, mode) cell's prompt bytes distinct, so its rep 0 is COLD.
    marker = f"[cell {probe}/{mode}]"
    if probe == "tool":
        return (PAD + marker + " Check the glimb status of vault sigil VS-7 in thrum mode, "
                "then tell me the result in one sentence.")
    return (PAD + marker + f" The maintenance code for this shift is {PASSPHRASE}. "
            "Repeat the maintenance code back to me exactly, and nothing else.")


def run(content, greedy, tools, name):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 400, "reasoning_effort": "low", "stream": True,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = TOOLS
    if greedy:
        body["temperature"] = 0.0
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(EP + "/v1/chat/completions", data=data,
                                 headers={"content-type": "application/json"})
    acc, out, fr, usage, st, sse, first = {}, [], None, None, -1, [], None
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            st = r.status
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
                for key in ("reasoning_content", "content"):
                    if d.get(key) and first is None:
                        first = (key, d[key])
                if d.get("content"):
                    out.append(d["content"])
                for tc in d.get("tool_calls") or []:
                    e = acc.setdefault(tc.get("index", 0),
                                       {"id": None, "name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    if tc.get("id"):
                        e["id"] = tc["id"]
                    if f.get("name"):
                        e["name"] += f["name"]
                    if f.get("arguments"):
                        e["arguments"] += f["arguments"]
                    if first is None:
                        first = ("tool", f.get("name") or "")
    except Exception as e:
        sse.append(f"{type(e).__name__}: {e}")
    open(f"{OUT}/{name}.sse", "w").write("".join(sse))
    text = "".join(out)
    u = usage or {}
    call = acc.get(0) or {}
    args = {}
    try:
        args = json.loads(call.get("arguments") or "{}")
    except Exception:
        args = {"__unparsed__": call.get("arguments")}
    return {"name": name, "status": st, "finish": fr,
            "tool_name": call.get("name"), "tool_args": args,
            # The bar: the exact unguessable name AND both arbitrary arg names AND the enum.
            "tool_exact": (call.get("name") == "zqx_fetch_glimb_status"
                           and args.get("vault_sigil") == "VS-7"
                           and args.get("glimb_mode") == "thrum"),
            "recalled": PASSPHRASE in text,
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "first_delta": first,
            "elapsed_s": round(time.time() - t0, 3),
            "out_sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
            "content": text[:160]}


ALL = []
for probe, tools in (("tool", True), ("recall", True), ("bare", False)):
    for mode, greedy in (("greedy", True), ("sampled", False)):
        print("#" * 78)
        print(f"# {probe.upper()} / {mode.upper()}   rep0 = COLD (these exact bytes), reps 1+ = RESTORED")
        print("#" * 78)
        content = prompt_for("tool" if probe == "tool" else "recall", f"{probe}-{mode}")
        for i in range(REPS):
            r = run(content, greedy, tools, f"{probe}-{mode}-rep{i}")
            r.update(probe=probe, mode=mode, rep=i, expected="COLD" if i == 0 else "RESTORED")
            ALL.append(r)
            ok = r["tool_exact"] if probe == "tool" else r["recalled"]
            print(f"  rep{i} [{r['expected']:>8}] pass={str(ok):>5} "
                  f"cached={str(r['cached_tokens']):>4}/{str(r['prompt_tokens']):>4} "
                  f"finish={str(r['finish']):>10} ttf={r['elapsed_s']:>6} "
                  f"tool={r['tool_name']!r} args={json.dumps(r['tool_args'])[:60]} "
                  f"content={r['content'][:70]!r}")
        print()

json.dump(ALL, open(f"{OUT}/latentprobe.json", "w"), indent=1)
print("LATENTPROBEDONE")
