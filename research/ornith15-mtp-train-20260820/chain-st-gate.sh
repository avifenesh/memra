#!/usr/bin/env bash
# ST rung gates for the OFFICIAL modelopt NVFP4 checkpoint, on GPU1 once the
# hidden capture releases it. Non-timed pass/fail cells only (house rule for the
# second card while GPU0 carries the training campaign).
set -uo pipefail
cd "$HOME/models/ornith15"
mkdir -p st-gates

while ! grep -q "CAPTURE DONE" mtp-train/capture.log 2>/dev/null; do sleep 120; done
while ! grep -q "DOWNLOAD DONE" nvfp4-download.log 2>/dev/null; do sleep 60; done
sleep 15

export CUDA_VISIBLE_DEVICES=1
ST="$HOME/models/ornith15/nvfp4-official"
BIN="$HOME/memra-src/target/release"

echo "== rung 2a: load + fixed-token forward ==" | tee st-gates/load.log
"$BIN/run-safetensors" "$ST" 1 2 3 4 >> st-gates/load.log 2>&1
echo "load rc=$?" >> st-gates/load.log

echo "== rung 2b: 48-tok greedy argmax vs BF16 oracle (3 probes, raw ids) =="
python3 - <<'PYEOF'
import json, re, subprocess, os
probes = json.load(open("gates/oracle-cpu.json"))
st = os.path.expanduser("~/models/ornith15/nvfp4-official")
bin_ = os.path.expanduser("~/memra-src/target/release/run-gen")
results = []
for i, p in enumerate(probes):
    ids = [str(t) for t in p["prompt_ids"]]
    env = dict(os.environ, MEMRA_NGEN="48")
    out = subprocess.run([bin_, st] + ids, capture_output=True, text=True, env=env, timeout=1200)
    open(f"st-gates/run-gen-p{i}.log", "w").write(out.stdout + out.stderr)
    arrays = re.findall(r"\ntokens: \[([0-9, ]+)\]", out.stdout)
    if not arrays:
        results.append({"probe": i, "verdict": "NO-OUTPUT"})
        continue
    got = [int(x) for x in arrays[-1].split(",") if x.strip()]
    # drop the echoed prompt if the stream includes it
    if got[: len(p["prompt_ids"])] == p["prompt_ids"]:
        got = got[len(p["prompt_ids"]):]
    want = p["gen_ids"]
    n = min(len(got), len(want), 48)
    div = next((j for j in range(n) if got[j] != want[j]), None)
    results.append({"probe": i, "verdict": "MATCH" if div is None else f"DIVERGE@{div}",
                    "n_compared": n, "got_head": got[:8], "want_head": want[:8]})
json.dump(results, open("st-gates/argmax-vs-oracle.json", "w"), indent=1)
print(results)
PYEOF

echo "== rung 3: serve-st-gate (CLI-vs-server exactness + template + spec prefix) =="
if [ -d "$HOME/memra-src/tools" ]; then
  (cd "$HOME/memra-src" && bash tools/serve-st-gate.sh "$ST") > st-gates/serve-st-gate.log 2>&1
  echo "serve-st-gate rc=$?" >> st-gates/serve-st-gate.log
  tail -5 st-gates/serve-st-gate.log
fi
echo "ST-GATES DONE"
