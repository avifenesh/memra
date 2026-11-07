"""Public copy of an agentic cell's rows: numbers per turn, sha256 of every generated text.

The workload replays private coding-agent sessions; the generated texts answer them, so the public
receipt keeps the per-turn numbers and a digest of each output (byte identity stays checkable)
and drops the texts. The session file name is reduced to its first 8 characters.

  python3 sanitize_agentic.py ROWS.json ROWS_PUBLIC.json
"""
import hashlib, json, sys

src, dst = sys.argv[1], sys.argv[2]
j = json.load(open(src))
h = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
rows = []
for r in j["rows"]:
    rows.append({"session": r["session"][:8], "turn": r["turn"],
                 "prime": {k: r["prime"].get(k) for k in ("ttft_s", "cached", "prompt_tokens", "wall_s")},
                 "ttft_s": r["ttft_s"], "wall_s": r["wall_s"], "prompt_tokens": r["prompt_tokens"],
                 "cached": r["cached"], "completion_tokens": r["completion_tokens"],
                 "generation_tps": r["generation_tps"], "finish": r["finish"],
                 "output_sha256": h([r["content"], r["reasoning"], r["tool_calls"]])})
json.dump({"cell": j["cell"], "rows": rows}, open(dst, "w"), indent=1)
