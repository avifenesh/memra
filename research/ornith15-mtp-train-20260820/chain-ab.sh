#!/usr/bin/env bash
# Final chain link: after training (GPU0) and the ST gates (GPU1) both finish,
# pick the best epoch export by heldout top1, patch it into a copy of the
# official NVFP4 ST dir, and run the vendor-vs-trained serve A/B on GPU0 under
# the box GPU lock (timed cells — the box must be otherwise quiet).
set -uo pipefail
cd "$HOME/models/ornith15"

while ! grep -q "CHAIN DONE" mtp-train/chain.out 2>/dev/null; do sleep 180; done
while ! grep -q "ST-GATES DONE" mtp-train/chain-st.out 2>/dev/null; do sleep 60; done
sleep 15

BEST=$(python3 - <<'PYEOF'
import json
best_epoch, best_top1 = None, -1.0
for line in open("mtp-train/train-out/metrics.jsonl"):
    r = json.loads(line)
    if r.get("event") == "eval" and "epoch" in r and r["top1"] > best_top1:
        best_top1, best_epoch = r["top1"], r["epoch"]
print(best_epoch if best_epoch is not None else "")
PYEOF
)
[ -n "$BEST" ] || { echo "FATAL: no epoch eval rows in metrics.jsonl"; exit 1; }
echo "best epoch by heldout top1: $BEST"

python3 mtp-train/patch_st_mtp.py \
  --src-dir nvfp4-official \
  --mtp "mtp-train/train-out/mtp-trained-epoch${BEST}.safetensors" \
  --out-dir nvfp4-patched > mtp-train/patch.log 2>&1 || { echo "FATAL: patch failed"; exit 1; }

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FATAL: gpu lock busy"; exit 1; }
export CUDA_VISIBLE_DEVICES=0

RC=0
python3 mtp-train/ab_head.py \
  --vendor-dir "$HOME/models/ornith15/nvfp4-official" \
  --trained-dir "$HOME/models/ornith15/nvfp4-patched" \
  --out mtp-train/ab-head.jsonl > mtp-train/ab-head.log 2>&1 || RC=$?
tail -3 mtp-train/ab-head.log
echo "AB-CHAIN DONE rc=$RC"
exit "$RC"
