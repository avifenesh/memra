#!/usr/bin/env bash
set -euo pipefail

if [[ ${HY3_ACCEL_NONPROD:-0} != 1 ]]; then
  echo "refusing: set HY3_ACCEL_NONPROD=1 only on a verified non-production accelerator pod" >&2
  exit 2
fi

PROFILE=${1:?usage: run-modelopt.sh experts}
case "$PROFILE" in
  experts)
    PACK=hy3_nvfp4
    ;;
  *)
    echo "unsupported profile: $PROFILE" >&2
    exit 2
    ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${HY3_RUN_ROOT:-/workspace/hy3-modelopt}
SOURCE_DIR=$RUN_ROOT/source-bf16
OUTPUT_DIR=$RUN_ROOT/output-$PROFILE
RECEIPT_DIR=$RUN_ROOT/receipts-$PROFILE
MODELOPT_DIR=$RUN_ROOT/modelopt
VENV_DIR=$RUN_ROOT/modelopt-venv
SOURCE_REPO=tencent/Hy3
SOURCE_REV=a960ebc3da325ba167f069f76c41eb62c9280d22
MODELOPT_SHA=43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a
CONFIG_SHA=0c9daab42bff9cce1b6f058b10d7b730f76d583e583e28ad56e92b36373246f0
INDEX_SHA=9594f1a9419e62ca7afca51bb644f38ef19039374f7812449381ccf42f0ef79b
MINT_SCRIPT=$REPO_ROOT/research/modelplan-onboarding-hy3-20260830/mint-nvfp4.py
MINT_SPOT_EVERY=${HY3_MINT_SPOT_EVERY:-500}
CARGO_BIN=${MEMRA_CARGO:-}
if [[ -z $CARGO_BIN ]]; then
  CARGO_BIN=$(command -v cargo || true)
fi
if [[ -z $CARGO_BIN ]]; then
  for candidate in /root/.cargo/bin/cargo /usr/local/cargo/bin/cargo /usr/bin/cargo; do
    if [[ -x $candidate ]]; then
      CARGO_BIN=$candidate
      break
    fi
  done
fi
if [[ ! -x $CARGO_BIN ]]; then
  echo "refusing: cargo is not executable; set MEMRA_CARGO to its absolute path" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$RECEIPT_DIR"
AVAILABLE=$(df -PB1 "$RUN_ROOT" | awk 'NR==2 {print $4}')
if [[ -f $SOURCE_DIR/model.safetensors.index.json ]]; then
  REQUIRED_FREE=220000000000
else
  REQUIRED_FREE=850000000000
fi
if (( AVAILABLE < REQUIRED_FREE )); then
  echo "refusing: HY3 source/candidate needs at least $REQUIRED_FREE bytes free; have $AVAILABLE" >&2
  exit 2
fi
if [[ -e /etc/tiyuvta || -e /var/lib/tiyuvta ]]; then
  echo "refusing: host carries Tiyuvta production identity" >&2
  exit 2
fi
exec 9>/tmp/memra-gpu.lock
flock -n 9 || {
  echo "refusing: /tmp/memra-gpu.lock is held by another accelerator campaign" >&2
  exit 2
}
MINT_DEVICES=${HY3_MINT_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd,)}
if [[ -z $MINT_DEVICES ]]; then
  echo "refusing: no CUDA devices available for the streaming mint" >&2
  exit 2
fi
nvidia-smi -L | tee "$RECEIPT_DIR/gpus.txt"
nvidia-smi -q > "$RECEIPT_DIR/nvidia-smi-q.txt"

if [[ ! -d $MODELOPT_DIR/.git ]]; then
  git clone https://github.com/NVIDIA/Model-Optimizer.git "$MODELOPT_DIR"
fi
git -C "$MODELOPT_DIR" fetch origin "$MODELOPT_SHA"
git -C "$MODELOPT_DIR" checkout --detach "$MODELOPT_SHA"
[[ $(git -C "$MODELOPT_DIR" rev-parse HEAD) == "$MODELOPT_SHA" ]]

if [[ ! -x $VENV_DIR/bin/python ]]; then
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -e "${MODELOPT_DIR}[hf]"
  "$VENV_DIR/bin/pip" install 'transformers==5.6.0'
fi
"$VENV_DIR/bin/pip" freeze > "$RECEIPT_DIR/pip-freeze.txt"

if [[ ! -f $SOURCE_DIR/model.safetensors.index.json ]]; then
  export HF_HUB_DISABLE_XET=1
  "$VENV_DIR/bin/hf" download "$SOURCE_REPO" --revision "$SOURCE_REV" --local-dir "$SOURCE_DIR"
fi
echo "$CONFIG_SHA  $SOURCE_DIR/config.json" | sha256sum -c -
echo "$INDEX_SHA  $SOURCE_DIR/model.safetensors.index.json" | sha256sum -c -

if [[ -e $OUTPUT_DIR ]]; then
  echo "refusing: output already exists: $OUTPUT_DIR" >&2
  exit 2
fi

{
  echo "profile=$PROFILE"
  echo "source=$SOURCE_REPO@$SOURCE_REV"
  echo "modelopt=$MODELOPT_SHA"
  echo "mint_script_sha256=$(sha256sum "$MINT_SCRIPT" | awk '{print $1}')"
  echo "quantizer=NVIDIA_ModelOpt_NVFP4QTensor.quantize:block16:fused-gate-up-shared-scale2"
  echo "deployment_quant_algo=W4A16_NVFP4"
  echo "calibration=none:weight-only-source-faithful-stream"
  echo "devices=$MINT_DEVICES"
  echo "spot_check_every=$MINT_SPOT_EVERY"
  echo "cargo=$CARGO_BIN"
  echo "memra=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RECEIPT_DIR/run.lock"

MINT_SRC=$SOURCE_DIR \
MINT_OUT=$OUTPUT_DIR \
MINT_MODELOPT_REPO=$MODELOPT_DIR \
MINT_DEVICES=$MINT_DEVICES \
MINT_SPOT_EVERY=$MINT_SPOT_EVERY \
  "$VENV_DIR/bin/python" -u "$MINT_SCRIPT" 2>&1 | tee "$RECEIPT_DIR/modelopt.log"

"$CARGO_BIN" build --manifest-path "$REPO_ROOT/Cargo.toml" -p memra-cli --bin memra --release
"$REPO_ROOT/target/release/memra" model inspect "$OUTPUT_DIR" \
  --against "$PACK" --out "$RECEIPT_DIR/inspect"

(
  cd "$OUTPUT_DIR"
  find . -type f -printf '%P\0' | sort -z | xargs -0 sha256sum
) > "$RECEIPT_DIR/artifact.sha256"
du -sb "$OUTPUT_DIR" > "$RECEIPT_DIR/artifact-bytes.txt"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RECEIPT_DIR/run.lock"
