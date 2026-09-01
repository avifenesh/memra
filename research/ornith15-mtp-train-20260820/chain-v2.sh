#!/usr/bin/env bash
# v2 sequence: multi-depth (D=3) chain-rollout training -> patch -> 3-arm A/B
# (vendor / v1 depth-1 head / v2 chain head) in one window. Box idle at launch.
set -uo pipefail
cd "$HOME/models/ornith15"

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FATAL: gpu lock busy"; exit 1; }
export CUDA_VISIBLE_DEVICES=0

echo "== v2 smoke (3 shards, 1 epoch, D=3) =="
mkdir -p mtp-train/smoke-hiddens
cp mtp-train/hiddens/shard-0000{0,1,2}.pt mtp-train/smoke-hiddens/ 2>/dev/null || true
RC=0
python3 mtp-train/train_mtp.py --bf16-dir bf16 --hiddens-dir mtp-train/smoke-hiddens \
  --corpus mtp-train/corpus.jsonl --out-dir mtp-train/smoke-v2-out \
  --epochs 1 --batch-tokens 4096 > mtp-train/smoke-v2.log 2>&1 || RC=$?
grep -q "TRAIN DONE" mtp-train/smoke-v2.log || { echo "FATAL: v2 smoke failed rc=$RC"; tail -20 mtp-train/smoke-v2.log; exit 1; }
rm -rf mtp-train/smoke-hiddens mtp-train/smoke-v2-out
echo "smoke PASS"

echo "== v2 full train =="
RC=0
python3 mtp-train/train_mtp.py --bf16-dir bf16 --hiddens-dir mtp-train/hiddens \
  --corpus mtp-train/corpus.jsonl --out-dir mtp-train/train-v2-out \
  > mtp-train/train-v2.log 2>&1 || RC=$?
grep -q "TRAIN DONE" mtp-train/train-v2.log || { echo "FATAL: v2 train failed rc=$RC"; tail -20 mtp-train/train-v2.log; exit 1; }

BEST=$(python3 - <<'PYEOF'
import json
best_epoch, best = None, -1.0
for line in open("mtp-train/train-v2-out/metrics.jsonl"):
    r = json.loads(line)
    if r.get("event") == "eval" and "epoch" in r:
        # rank by mean top1 across depths — the chain is what serves
        d = r["by_depth"]
        score = sum(d[k]["top1"] for k in d) / len(d)
        if score > best:
            best, best_epoch = score, r["epoch"]
print(best_epoch if best_epoch is not None else "")
PYEOF
)
[ -n "$BEST" ] || { echo "FATAL: no epoch evals"; exit 1; }
echo "v2 best epoch by mean-depth top1: $BEST"

python3 mtp-train/patch_st_mtp.py --src-dir nvfp4-official \
  --mtp "mtp-train/train-v2-out/mtp-trained-epoch${BEST}.safetensors" \
  --out-dir nvfp4-patched-v2 > mtp-train/patch-v2.log 2>&1 || { echo "FATAL: patch failed"; exit 1; }

echo "== 3-arm A/B: vendor / v1 / v2 =="
RC=0
python3 mtp-train/ab_head.py \
  --arms "vendor=$HOME/models/ornith15/nvfp4-official,v1=$HOME/models/ornith15/nvfp4-patched,v2=$HOME/models/ornith15/nvfp4-patched-v2" \
  --out mtp-train/ab-v2.jsonl > mtp-train/ab-v2.log 2>&1 || RC=$?
tail -3 mtp-train/ab-v2.log
echo "V2-CHAIN DONE rc=$RC"
exit "$RC"
