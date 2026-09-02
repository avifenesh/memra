#!/usr/bin/env python3
"""qwen4_exp (Qwen3.8-Flash-Next) FR-Spec ranks mint: corpus extraction from the owner SXC pools.

CPU-only, and that is the point: mtp9/mtp10 priced a 32,768-id discovery set on this
248,320-row vocab at ~4M own-generated tokens (~28 GPU-hours). The GLM-5.3 mint
(darklanes research/glm53-ranks-mint-20260830) got corpus scale for zero GPU-hours by
ranking the ASSISTANT-SIDE emissions of the same owner session pools under the model's
own tokenizer. This is that extractor, re-rendered for THIS family's emission shape.

Extracts the text the model would actually EMIT (assistant turns) from
/home/avifenesh/projects/colbert-2/data/sessions/{claude,codex,eigen,hermes} per the
per-source schemas in memory/sxc-corpora-for-rank-mints.md, into two traffic classes:

  corpus/agentic/<pool>.txt  the full emission shape of this checkpoint's own
                             chat_template.jinja assistant branch, from the first token
                             AFTER the generation prompt:
                               {reasoning}\n</think>\n\n{content}
                               [\n\n]<tool_call>\n<function={name}>\n
                                 <parameter={k}>\n{value}\n</parameter>\n
                               </function>\n</tool_call>
                               <|im_end|>
                             (the prompt supplies the opening `<think>\n`, so a corpus
                             line starts with the reasoning text, never with `<think>`;
                             string parameter values raw, non-string tojson
                             ensure_ascii=False, per the template expression).
  corpus/prose/<pool>.txt    plain natural-language content parts (no markers, no tool
                             parameters): >=200 chars, alpha ratio >=0.6, no ``` fence.
                             The thinkoff/writing emission class, where the prompt
                             supplies `<think>\n\n</think>\n\n` and the model emits
                             content only.

mixed is NOT a corpus: it is the 50/50 normalised count blend of the two streams
(same rank law), computed at rank time by rank_ranks.py.

Determinism: pools walked in sorted path order; full-string dedup per class;
per-pool extracted-byte caps for the two large pools (claude, codex 48 MiB each)
so the class stays pool-balanced without discarding the small pools.
"""
import json
import os
import sys

ROOT = "/home/avifenesh/projects/colbert-2/data/sessions"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
CAPS = {"claude": 48 * 1024 * 1024, "codex": 48 * 1024 * 1024,
        "eigen": None, "hermes": None}
PROSE_MIN_CHARS = 200
PROSE_MAX_CHARS = 20000
PROSE_MIN_ALPHA = 0.6


def arg_value(v):
    # template: {{ v | tojson(ensure_ascii=False) if v is not string else v }}
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def render_turn(reasoning, content, tool_calls):
    """The assistant branch of chat_template.jinja, EMISSION side (what the model writes
    after the generation prompt's `<|im_start|>assistant\n<think>\n`)."""
    c = (content or "").strip()
    # Template: '<think>\n' + reasoning|trim + '\n</think>\n\n' + content. The opening
    # marker belongs to the prompt; emission starts inside the reasoning block.
    out = [(reasoning or "").strip(), "\n</think>\n\n", c]
    for i, (name, args) in enumerate(tool_calls):
        if i == 0:
            out.append("\n\n<tool_call>\n<function=" if c else "<tool_call>\n<function=")
        else:
            out.append("\n<tool_call>\n<function=")
        out.append((name or "") + ">\n")
        if isinstance(args, str):
            # codex/hermes carry arguments as a JSON string on the wire; the template
            # iterates the DICT. Unparseable stays raw (still real emitted bytes).
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                pass
        if isinstance(args, dict):
            for k, v in args.items():
                out.append("<parameter=" + str(k) + ">\n" + arg_value(v) + "\n</parameter>\n")
        elif args:
            out.append("<parameter=arguments>\n" + arg_value(args) + "\n</parameter>\n")
        out.append("</function>\n</tool_call>")
    out.append("<|im_end|>")
    return "".join(out)


def prose_ok(text):
    t = text.strip()
    if not (PROSE_MIN_CHARS <= len(t) <= PROSE_MAX_CHARS):
        return False
    if "```" in t:
        return False
    if t[0] in "<{[":
        return False
    alpha = sum(ch.isalpha() or ch.isspace() for ch in t) / len(t)
    return alpha >= PROSE_MIN_ALPHA


def iter_claude(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if '"assistant"' not in line:
                continue
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if d.get("type") != "assistant" or d.get("isSidechain"):
                continue
            m = d.get("message") or {}
            c = m.get("content")
            reasoning, content, tcs = "", "", []
            if isinstance(c, str):
                content = c
            elif isinstance(c, list):
                texts = []
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "text":
                        texts.append(b.get("text") or "")
                    elif bt == "thinking":
                        reasoning += b.get("thinking") or ""
                    elif bt == "tool_use":
                        tcs.append((b.get("name") or "", b.get("input")))
                content = "\n".join(t for t in texts if t)
            yield reasoning, content, tcs


def iter_codex(path):
    """Rollout items are fragments; group message + following function_calls
    into one emission turn (reasoning is encrypted upstream: absent)."""
    turn = None  # (content_parts, tool_calls)
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if '"response_item"' not in line:
                continue
            if '"message"' not in line and '"function_call"' not in line:
                continue
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if d.get("type") != "response_item":
                continue
            p = d.get("payload") or {}
            pt = p.get("type")
            if pt == "message":
                if p.get("role") == "assistant":
                    if turn:
                        yield "", "\n".join(turn[0]), turn[1]
                    texts = [b.get("text") or "" for b in p.get("content") or []
                             if isinstance(b, dict) and b.get("type") == "output_text"]
                    turn = ([t for t in texts if t], [])
                elif turn:
                    yield "", "\n".join(turn[0]), turn[1]
                    turn = None
            elif pt == "function_call":
                if turn is None:
                    turn = ([], [])
                turn[1].append((p.get("name") or "", p.get("arguments")))
    if turn:
        yield "", "\n".join(turn[0]), turn[1]


def iter_eigen(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if '"assistant"' not in line:
                continue
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if d.get("Role") != "assistant":
                continue
            tcs = [(tc.get("Name") or "", tc.get("Arguments"))
                   for tc in d.get("ToolCalls") or [] if isinstance(tc, dict)]
            yield d.get("Reasoning") or "", d.get("Text") or "", tcs


def iter_hermes(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if '"assistant"' not in line:
                continue
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if d.get("role") != "assistant":
                continue
            tcs = []
            for tc in d.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    tcs.append((fn.get("name") or "", fn.get("arguments")))
            reasoning = d.get("reasoning_content") or d.get("reasoning") or ""
            yield reasoning, d.get("content") or "", tcs


POOLS = {"claude": iter_claude, "codex": iter_codex,
         "eigen": iter_eigen, "hermes": iter_hermes}


def main():
    os.makedirs(os.path.join(OUT, "agentic"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "prose"), exist_ok=True)
    stats = {}
    seen_agentic, seen_prose = set(), set()
    for pool in sorted(POOLS):
        it = POOLS[pool]
        cap = CAPS[pool]
        files = []
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, pool)):
            dirnames.sort()
            for fn in sorted(filenames):
                if fn.endswith(".jsonl"):
                    files.append(os.path.join(dirpath, fn))
        files.sort()
        a_bytes = p_bytes = turns = p_parts = nfiles = 0
        capped = False
        with open(os.path.join(OUT, "agentic", pool + ".txt"), "w", encoding="utf-8") as fa, \
             open(os.path.join(OUT, "prose", pool + ".txt"), "w", encoding="utf-8") as fp:
            for path in files:
                if cap is not None and a_bytes >= cap:
                    capped = True
                    break
                nfiles += 1
                for reasoning, content, tcs in it(path):
                    turn = render_turn(reasoning, content, tcs)
                    if len(turn) < 30:  # empty-think empty-content stub
                        continue
                    if turn not in seen_agentic:
                        seen_agentic.add(turn)
                        fa.write(turn + "\n")
                        a_bytes += len(turn) + 1
                        turns += 1
                    c = (content or "").strip()
                    if c and prose_ok(c) and c not in seen_prose:
                        seen_prose.add(c)
                        fp.write(c + "\n\n")
                        p_bytes += len(c) + 2
                        p_parts += 1
        stats[pool] = {"files_read": nfiles, "files_total": len(files),
                       "capped": capped, "turns": turns,
                       "agentic_bytes": a_bytes, "prose_parts": p_parts,
                       "prose_bytes": p_bytes}
        print(pool, stats[pool], flush=True)
    with open(os.path.join(OUT, "extract-stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
