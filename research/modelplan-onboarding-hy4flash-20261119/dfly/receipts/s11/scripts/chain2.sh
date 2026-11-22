#!/usr/bin/env bash
# GB10 box, step 11 chain 2: build h2 (hy4f-spec replay + cost), price the step shapes, and
# replay the six chat turns and the long probe through the three drafter arms (teacher-forced,
# sampled at temperature 0.9 seed 20261121). Log: logs/s11-chain2.log.
set -u
D=$HOME/hy4flash/s11
L=$HOME/hy4flash/logs
cd $D
echo "CHAIN2 START $(date '+%F %T') $(cat src/HEAD.txt)"
./build-gpu.sh > $L/s11-build-h2.log 2>&1
grep -q BUILD_EXIT=0 $L/s11-build-h2.log || { echo BUILD-FAILED; exit 1; }
mkdir -p bins-h2 && cp target/release/memra-server target/release/hy4f-spec target/release/hy4f-gpu bins-h2/
sha256sum bins-h2/* | tee bins-h2.sha256
S=$D/bins-h2/hy4f-spec
R=$HOME/hy4flash/pr7/requests
python3 - <<'PY'
import json
t = json.load(open("turns-prompts.json"))
r = json.load(open("/home/avifenesh/hy4flash/pr7/requests/long.json"))
ids = r["token_ids"][:r["decode_from"]]
t.append({"name": "long", "prompt_ids": ids, "n": 64, "temperature": 0.9, "seed": 20261121, "eos": None})
json.dump(t, open("replay-prompts.json", "w"))
print(len(t), "prompts")
PY
mkdir -p replay
memok() { waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null; }
memok; $S cost $HOME/hy4flash/hf-stage $R/long.json --reps 8 --temperature 0.9 > replay/cost-nv-sampled.json 2> replay/cost-nv-sampled.log; echo "cost nv sampled rc=$?"
memok; $S cost $HOME/hy4flash/hf-stage $R/long.json --reps 8 > replay/cost-nv-greedy.json 2> replay/cost-nv-greedy.log; echo "cost nv greedy rc=$?"
memok; $S cost $D/arm-bf16 $R/long.json --reps 8 --temperature 0.9 > replay/cost-bf16-sampled.json 2> replay/cost-bf16-sampled.log; echo "cost bf16 sampled rc=$?"
for arm in nv:$HOME/hy4flash/hf-stage bf16:$D/arm-bf16 bf16mlp:$D/arm-bf16-mlp; do
  memok; $S replay ${arm#*:} replay-prompts.json replay/replay-${arm%%:*}.jsonl > replay/replay-${arm%%:*}.log 2>&1; echo "replay ${arm%%:*} rc=$?"
done
echo "CHAIN2-DONE $(date '+%F %T')"
