#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: the q45k MMQ_X=64 probe (round 4's last candidate).
#   build: the lever 8b tree (efbfeb8b10) with MEMRA_MMQ_X_Q45K=64 in its own target dir, banked as
#          bins/q45x64-efbfeb8b10, plus kernel-check's nvfp4-gemm section (the q45k cells) on both
#   NN-B / NN-T: one cold 131k served prefill with the timestamped log (B = tft-efbfeb8b10, T = x64)
# usage: q45_cell.sh CAMPAIGN NAME
set -u
D=$HOME/.cache/memra-gb10-r4
CAMP=$1; NAME=$2
CD=$D/runs/$CAMP; mkdir -p "$CD"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
B=$D/bins/tft-efbfeb8b10; T=$D/bins/q45x64-efbfeb8b10
case "$NAME" in
  build)
    git -C "$D/tree" checkout -q --detach efbfeb8b1072eae07c6f49f3af7130727f187785 || { echo Q45-CELL-FAIL >> "$CD/campaign.log"; exit 1; }
    (cd "$D/tree" && MEMRA_MMQ_X_Q45K=64 CARGO_TARGET_DIR=$D/target-x64 nice -n 10 cargo build --release --locked -j 16 \
      -p memra-engine --bin kernel-check -p memra-server --bin memra-server) > "$D/logs/build-x64.log" 2>&1 \
      || { echo Q45-CELL-FAIL >> "$CD/campaign.log"; exit 1; }
    mkdir -p "$T" && cp -a "$D/target-x64/release/memra-server" "$D/target-x64/release/kernel-check" "$T/"
    for arm in B T; do
      bin=$B; [ "$arm" = T ] && bin=$T
      (cd "$D/tree" && unset MEMRA_KC_FAST; MEMRA_KC_ONLY=nvfp4-gemm "$bin/kernel-check" "$G8") > "$CD/kernel-check-nvfp4-gemm-$arm.log" 2>&1
      echo "kernel-check-$arm rc=$? $(tail -n 1 "$CD/kernel-check-nvfp4-gemm-$arm.log")" >> "$CD/campaign.log"
    done
    ;;
  *)
    ARM=${NAME#*-}; bin=$B; [ "$ARM" = T ] && bin=$T
    OUT=$CD/$NAME; mkdir -p "$OUT"
    OUT=$OUT BIN=$bin EXTRA="MEMRA_TICK_TRACE=1" WRAP="python3 $D/box/tsrun.py" bash "$D/box/serve.sh" &
    sp=$!
    sleep 2
    spid=$(cat "$OUT/server.pid")
    waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1 || { kill -TERM "$spid"; wait $sp; echo Q45-CELL-FAIL >> "$CD/campaign.log"; exit 1; }
    python3 "$D/box/pf_long_memra.py" "$OUT/pf.jsonl" --targets 131072 --reps 1 --tag "$NAME" >> "$OUT/run.log" 2>&1
    kill -TERM "$spid"
    waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
    wait $sp
    grep '^{' "$OUT/pf.jsonl" | sed "s/^/$ARM /" >> "$CD/campaign.log"
    ;;
esac
echo "Q45-CELL-DONE $NAME" >> "$CD/campaign.log"
