#!/usr/bin/env python3
"""Own-session acceptance bench: real owner turns from Claude/codex session corpus
against a running SGLang DSPARK server. Receipts: per-request JSONL + summary."""
import argparse, json, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

SKIP_PAT = re.compile(
    r"<system-reminder>|<task-notification>|tool_result|<local-command|Caveat: The messages below|"
    r"^\s*\[Request interrupted", re.IGNORECASE)


def extract_prompts(root: Path, tiers=("claude", "codex")) -> list[dict]:
    out = []
    for tier in tiers:
        for f in sorted((root / tier).rglob("*.jsonl")):
            try:
                for line in f.open(errors="ignore"):
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("type") != "user":
                        continue
                    msg = d.get("message") or {}
                    c = msg.get("content")
                    texts = []
                    if isinstance(c, str):
                        texts = [c]
                    elif isinstance(c, list):
                        texts = [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
                    for t in texts:
                        t = t.strip()
                        if not (80 <= len(t) <= 4000):
                            continue
                        if SKIP_PAT.search(t):
                            continue
                        out.append({"tier": tier, "text": t})
            except Exception:
                continue
    return out


_TOKENIZER = None


def get_tokenizer(model_path):
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return _TOKENIZER


def run_one(base_url, prompt, args):
    # spec stats (spec_verify_ct) only surface on the native /generate endpoint —
    # template client-side, same as dflash.benchmark's sglang path
    tok = get_tokenizer(args.model_path)
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt["text"]}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(args.enable_thinking))
    body = {"text": text, "sampling_params": {
        "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "max_new_tokens": args.max_new_tokens}}
    t0 = time.time()
    try:
        r = requests.post(base_url + "/generate", json=body, timeout=args.timeout_s)
        r.raise_for_status()
        d = r.json()
        if not isinstance(d, dict):
            d = d[0]
    except Exception as e:
        return {"ok": False, "err": str(e)[:200], **prompt}
    meta = d.get("meta_info") or {}
    comp = int(meta.get("completion_tokens") or 0)
    verify = int(meta.get("spec_verify_ct") or 0)
    return {"ok": True, "tier": prompt["tier"], "chars": len(prompt["text"]),
            "completion_tokens": comp, "spec_verify_ct": verify,
            "accept_len": (comp / verify) if verify else None,
            "wall_s": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, default=Path("/scratch/corpus/sessions"))
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--model-path", default="/scratch/models/qwen38-27b-fp8")
    ap.add_argument("--per-bucket", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    get_tokenizer(args.model_path)  # preload before thread pool (transformers v5 lazy-import is racy)
    pool = extract_prompts(args.corpus_root)
    print(f"extracted {len(pool)} candidate turns", flush=True)
    buckets = {
        "chat-short": [p for p in pool if len(p["text"]) < 600],
        "agentic-brief": [p for p in pool if len(p["text"]) >= 600],
    }
    picked = []
    for name, cand in buckets.items():
        random.shuffle(cand)
        sel = cand[: args.per_bucket]
        for p in sel:
            p["bucket"] = name
        picked += sel
        print(f"bucket {name}: {len(sel)} prompts", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for res in ex.map(lambda p: run_one(args.base_url, p, args), picked):
            results.append(res)
            if len(results) % 16 == 0:
                print(f"{len(results)}/{len(picked)} done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p, r in zip(picked, results):
            f.write(json.dumps({"bucket": p.get("bucket"), **r}) + "\n")

    for name in buckets:
        rs = [r for p, r in zip(picked, results) if p.get("bucket") == name and r.get("ok") and r.get("accept_len")]
        if rs:
            al = [r["accept_len"] for r in rs]
            toks = sum(r["completion_tokens"] for r in rs)
            print(f"[{name}] n={len(rs)} accept_len mean={sum(al)/len(al):.3f} "
                  f"min={min(al):.2f} max={max(al):.2f} total_tokens={toks}")
        fails = [r for p, r in zip(picked, results) if p.get("bucket") == name and not r.get("ok")]
        if fails:
            print(f"[{name}] FAILURES {len(fails)}: {fails[0].get('err')}")


if __name__ == "__main__":
    main()
