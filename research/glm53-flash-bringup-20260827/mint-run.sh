#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 mint launcher (bench box). Runs mint-nvfp4.py under nohup,
# logs everything, and shouts MINT-DONE / MINT-FAILED. No calibration, no forwards;
# IO-bound: reads 656 GB bf16, writes ~204 GB NVFP4. GPU 0 is used (if visible)
# only to speed the per-tensor quantize math — CPU-only also works.
#
# Usage:   ./mint-run.sh            # background via nohup, prints log path
#          MINT_FG=1 ./mint-run.sh  # foreground (debugging)
# Env:     MINT_SRC (default ~/models/glm53-bf16)
#          MINT_OUT (default ~/models/glm53-nvfp4)
#          MINT_SPOT_EVERY (default 500) — cross-impl dequant gate cadence
set -euo pipefail

PY="${MINT_PY:-$HOME/modelopt-env/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINT="$SCRIPT_DIR/mint-nvfp4.py"
SRC="${MINT_SRC:-$HOME/models/glm53-bf16}"
OUT="${MINT_OUT:-$HOME/models/glm53-nvfp4}"
LOG="${MINT_LOG:-$HOME/mint-nvfp4-$(date +%Y%m%d-%H%M%S).log}"

[ -x "$PY" ] || { echo "MINT-FAILED: python not found at $PY (create ~/modelopt-env first)"; exit 1; }
[ -f "$MINT" ] || { echo "MINT-FAILED: $MINT missing"; exit 1; }
[ -d "$SRC" ] || { echo "MINT-FAILED: source model dir $SRC missing"; exit 1; }

# Hard dependency preflight — modelopt 0.46.0 needs all of these importable
# (requests + huggingface_hub are pulled in by `import modelopt.torch`).
"$PY" - <<'EOF' || { echo "MINT-FAILED: env deps missing. Fix with:
  ~/modelopt-env/bin/pip install 'nvidia-modelopt==0.46.0' torch safetensors requests huggingface_hub"; exit 1; }
import modelopt, torch, safetensors, requests, huggingface_hub
v = tuple(int(x) for x in modelopt.__version__.split(".")[:2])
assert v >= (0, 45), f"nvidia-modelopt {modelopt.__version__} < 0.45 (need W4A16_NVFP4)"
print(f"deps ok: modelopt {modelopt.__version__}, torch {torch.__version__}, "
      f"cuda={torch.cuda.is_available()}")
EOF

# Free-disk sanity: output ~204 GB + slack.
avail_gb=$(df --output=avail -B G "$(dirname "$OUT")" | tail -1 | tr -dc '0-9')
if [ "${avail_gb:-0}" -lt 230 ]; then
  echo "MINT-FAILED: only ${avail_gb} GB free at $(dirname "$OUT"); need >= 230 GB"
  exit 1
fi

export MINT_SRC="$SRC" MINT_OUT="$OUT"

if [ "${MINT_FG:-0}" = "1" ]; then
  exec "$PY" -u "$MINT" 2>&1 | tee "$LOG"
fi

nohup "$PY" -u "$MINT" >"$LOG" 2>&1 &
pid=$!
echo "mint launched: pid=$pid"
echo "log:  tail -f $LOG"
echo "done: grep -E 'MINT-DONE|MINT-FAILED' $LOG"
disown "$pid"
