#!/usr/bin/env bash
# ST rung 2b/3 round 2. Round 1 taught: run-gen's raw-id branch is gate-only by
# design (single verify step — and its argmax DID match the oracle's first token
# on all probes it printed), so the 48-token compare needs the --prompt text
# path; and box3 keeps cargo out of the non-interactive PATH.
set -uo pipefail
cd "$HOME/models/ornith15"
export CUDA_VISIBLE_DEVICES=1
export PATH="$HOME/.cargo/bin:$PATH"
ST="$HOME/models/ornith15/nvfp4-official"

echo "== rung 2b round 2: 48-tok greedy via --prompt (chat-templated) =="
python3 - <<'PYEOF'
import json, re, subprocess, os
probes = json.load(open("gates/oracle-cpu.json"))
st = os.path.expanduser("~/models/ornith15/nvfp4-official")
bin_ = os.path.expanduser("~/memra-src/target/release/run-gen")
results = []
for i, p in enumerate(probes):
    env = dict(os.environ, MEMRA_NGEN="48", MEMRA_CHAT="1")
    out = subprocess.run([bin_, st, "--prompt", p["prompt"]],
                         capture_output=True, text=True, env=env, timeout=1800)
    open(f"st-gates/run-gen2-p{i}.log", "w").write(out.stdout + out.stderr)
    m_prompt = re.search(r"prompt tokens: \[([0-9, ]+)\]", out.stdout)
    prompt_ok = None
    if m_prompt:
        got_p = [int(x) for x in m_prompt.group(1).split(",") if x.strip()]
        prompt_ok = got_p == p["prompt_ids"]
    arrays = re.findall(r"\ntokens: \[([0-9, ]+)\]", out.stdout)
    if not arrays:
        results.append({"probe": i, "verdict": "NO-OUTPUT", "prompt_ids_match": prompt_ok})
        continue
    got = [int(x) for x in arrays[-1].split(",") if x.strip()]
    if got[: len(p["prompt_ids"])] == p["prompt_ids"]:
        got = got[len(p["prompt_ids"]):]
    want = p["gen_ids"]
    n = min(len(got), len(want), 48)
    div = next((j for j in range(n) if got[j] != want[j]), None)
    results.append({"probe": i, "prompt_ids_match": prompt_ok,
                    "verdict": "MATCH" if div is None else f"DIVERGE@{div}",
                    "n_compared": n, "got_head": got[:8], "want_head": want[:8]})
json.dump(results, open("st-gates/argmax-vs-oracle-round2.json", "w"), indent=1)
print(json.dumps(results, indent=1))
PYEOF

echo "== rung 3 round 2: serve-st-gate with cargo on PATH =="
(cd "$HOME/memra-src" && bash tools/serve-st-gate.sh "$ST") > st-gates/serve-st-gate2.log 2>&1
echo "serve-st-gate rc=$?" >> st-gates/serve-st-gate2.log
tail -8 st-gates/serve-st-gate2.log
echo "ST-GATES-R2 DONE"
