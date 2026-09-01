#!/usr/bin/env python3
"""v2: multi-turn extraction — each sample = conversation history + user turn.
Labels still come from target regeneration (train_only_last_turn); history context
is the real serving shape. Caps: history 6000 chars, 12 samples/session, 40K total."""
import argparse, hashlib, json, re
from pathlib import Path

SKIP_PAT = re.compile(
    r"<system-reminder>|<task-notification>|tool_result|<local-command|Caveat: The messages below|"
    r"^\s*\[Request interrupted", re.IGNORECASE)


def turns(f: Path):
    """Yield (role, text) in order for claude-session-format jsonl."""
    try:
        for line in f.open(errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            if t not in ("user", "assistant"):
                continue
            c = (d.get("message") or {}).get("content")
            texts = [c] if isinstance(c, str) else \
                [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"] if isinstance(c, list) else []
            txt = "\n".join(x.strip() for x in texts if x.strip())
            if txt:
                yield t, txt
    except Exception:
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, default=Path("/scratch/corpus/sessions"))
    ap.add_argument("--tiers", nargs="+", default=["claude", "codex", "eigen", "hermes"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-total", type=int, default=40_000)
    ap.add_argument("--max-per-session", type=int, default=12)
    ap.add_argument("--max-history-chars", type=int, default=6_000)
    args = ap.parse_args()

    seen, n_out = set(), 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for tier in args.tiers:
            root = args.corpus_root / tier
            if not root.is_dir():
                continue
            for f in sorted(root.rglob("*.jsonl")):
                if n_out >= args.max_total:
                    break
                hist, n_sess = [], 0
                for role, txt in turns(f):
                    if role == "user" and 40 <= len(txt) <= 8000 and not SKIP_PAT.search(txt) \
                            and n_sess < args.max_per_session:
                        # build sample: trimmed history + this user turn
                        conv, budget = [], args.max_history_chars
                        for r, t in reversed(hist):
                            t = t[:2000]
                            if budget - len(t) < 0:
                                break
                            conv.append({"role": r, "content": t})
                            budget -= len(t)
                        conv.reverse()
                        conv.append({"role": "user", "content": txt})
                        key = hashlib.sha256(json.dumps(conv).encode()).hexdigest()[:16]
                        if key not in seen:
                            seen.add(key)
                            out.write(json.dumps({"id": f"own2-{tier}-{key}", "conversations": conv}) + "\n")
                            n_out += 1
                            n_sess += 1
                    if not SKIP_PAT.search(txt):
                        hist.append((role if role == "user" else "assistant", txt))
                        if len(hist) > 24:
                            hist = hist[-24:]
            print(f"{tier}: cumulative {n_out}")
    print(f"total: {n_out} -> {args.out}")


if __name__ == "__main__":
    main()
