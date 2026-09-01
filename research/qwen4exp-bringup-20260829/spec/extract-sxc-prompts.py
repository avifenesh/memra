#!/usr/bin/env python3
"""Extract OWNER-BLESSED prompts from the SXC agent-session pools (rig-only).

Owner directive (2026-08-14, memory sxc-corpora-for-rank-mints): FR-Spec rank corpora
prompts come from the SXC corpora + owner agent-session pools. This runs on the RIG (the
pools are not on the fleet box) and emits plain prompt text, one per line, for
make-corpus-prompts.py to render through the chat template.

Per-pool user-message schemas (from the same memory):
  hermes  normalized {"role":"user","content":str}  — the OR-user class, weight highest
  claude  raw Claude Code transcripts: type=="user", message.content str or blocks;
          skip isSidechain
  codex   rollout: type=="response_item", payload.type=="message", payload.role=="user",
          content blocks input_text
  eigen   {"Role":"user","Text":str}

Filters that worked (same memory): 40-6000 chars, >=8 words, >=0.6 alpha ratio, skip
lines starting < { [, skip system-reminder / command wrappers, full-string dedup,
whitespace-normalize to one line. Pools are round-robin interleaved so ANY cutoff of the
output stays pool-balanced (a truncated corpus must not become one pool's distribution).

Usage: python3 extract-sxc-prompts.py <sessions_root> <out.tsv> [limit]
Output: `pool<TAB>prompt-text`, one per line.
"""
import json
import os
import sys

root, out = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0

WRAPPER_MARKERS = (
    "system-reminder",
    "<command-name>",
    "<command-message>",
    "<local-command",
    "Caveat: The messages below",
    "This session is being continued from a previous",
    "<user-prompt-submit-hook>",
)


def keep(text):
    if not text:
        return None
    t = " ".join(text.split())
    if not (40 <= len(t) <= 6000):
        return None
    if len(t.split()) < 8:
        return None
    if t[0] in "<{[":
        return None
    if any(m in t for m in WRAPPER_MARKERS):
        return None
    alpha = sum(c.isalpha() or c.isspace() for c in t)
    if alpha / len(t) < 0.6:
        return None
    return t


def blocks_text(content):
    """A string, or a list of content blocks — return the concatenated text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") in ("text", "input_text") and isinstance(
                    b.get("text"), str
                ):
                    parts.append(b["text"])
        return "\n".join(parts)
    return ""


def pool_files(pool):
    """Every .jsonl under a pool, RECURSIVELY: the claude and codex pools are keyed by
    project directory, so a flat listdir finds nothing and reports zero prompts without
    failing — the exact silent-zero this returns loudly instead (see `missing` below)."""
    d = os.path.join(root, pool)
    if not os.path.isdir(d):
        return []
    found = []
    for dirpath, _, names in os.walk(d):
        for name in names:
            if name.endswith(".jsonl"):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def read_pool(pool):
    """Yield candidate prompt strings from one pool, oldest file first."""
    for path in pool_files(pool):
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(o, dict):
                        continue
                    text = None
                    if pool == "hermes":
                        if o.get("role") == "user":
                            text = blocks_text(o.get("content"))
                    elif pool == "claude":
                        if o.get("type") == "user" and not o.get("isSidechain"):
                            text = blocks_text((o.get("message") or {}).get("content"))
                    elif pool == "codex":
                        p = o.get("payload") or {}
                        if (
                            o.get("type") == "response_item"
                            and p.get("type") == "message"
                            and p.get("role") == "user"
                        ):
                            text = blocks_text(p.get("content"))
                    elif pool == "eigen":
                        if o.get("Role") == "user":
                            text = blocks_text(o.get("Text"))
                    t = keep(text)
                    if t:
                        yield t
        except OSError:
            continue


# Round-robin across pools so any truncation of the output stays pool-balanced.
POOLS = ["hermes", "claude", "codex", "eigen"]
gens = {p: read_pool(p) for p in POOLS}
seen = set()
picked = []
per_pool = {p: 0 for p in POOLS}
while gens and (limit == 0 or len(picked) < limit):
    progressed = False
    for p in list(POOLS):
        if p not in gens:
            continue
        if limit and len(picked) >= limit:
            break
        for t in gens[p]:
            if t in seen:
                continue
            seen.add(t)
            picked.append((p, t))
            per_pool[p] += 1
            progressed = True
            break
        else:
            del gens[p]
    if not progressed:
        break

# `pool<TAB>text`: the pool travels with the prompt so the coverage table can report a
# per-pool class instead of guessing one from file order.
with open(out, "w") as f:
    for pool, t in picked:
        f.write(f"{pool}\t{t}\n")

print(f"{len(picked)} owner prompts -> {out}")
for p in POOLS:
    print(f"  {p:8s} files={len(pool_files(p)):5d}  prompts={per_pool[p]}")
lens = [len(t) for _, t in picked]
if lens:
    print(f"  chars min={min(lens)} max={max(lens)} mean={sum(lens) // len(lens)}")

# A pool that HAS files but contributed nothing is a schema mismatch, not an empty pool —
# fail loudly rather than shipping a corpus that silently lost a class (the first run of
# this script reported "48 owner prompts" with claude and codex at zero because their
# transcripts live in per-project subdirectories).
missing = [p for p in POOLS if pool_files(p) and per_pool[p] == 0]
if missing:
    sys.exit(
        f"pools with files but ZERO extracted prompts (schema mismatch?): {missing} — "
        "a silent zero here would drop a whole prompt class from the rank corpus"
    )

