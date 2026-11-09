#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one step of lever 8 (the boundary tail fold).
#   build: build B = main BS and T = the fold head TS (plus prime-fold-gate at TS)
#   gate:  prime-fold-gate on the 27B (split against fold, bit for bit)
#   NN-B / NN-T: one served cell of the A/B on that arm, 131k and 260k with the per-call trace
#                (MEMRA_TICK_TRACE=1: `[prime-chunk] phase=mtp-trunk rows=N wall_ms=W`); 32k keeps
#                the split on both arms (its stop is a call boundary)
# usage: tf_cell.sh CAMPAIGN NAME B_SHA T_SHA
set -u
D=$HOME/.cache/memra-gb10-r4
CAMP=$1; NAME=$2; BS=$3; TS=$4
CD=$D/runs/$CAMP; mkdir -p "$CD"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
B=$(ls -d "$D"/bins/tfb-"${BS:0:10}" 2>/dev/null | tail -n 1); T=$(ls -d "$D"/bins/tft-"${TS:0:10}" 2>/dev/null | tail -n 1)
case "$NAME" in
  build)
    git -C "$D/tree" fetch -q origin main perf/gb10-tail-fold || { echo TF-CELL-BUILD-FAIL >> "$CD/campaign.log"; exit 1; }
    if [ -z "$B" ]; then git -C "$D/tree" checkout -q --detach "$BS" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" tfb > "$D/logs/build-tfb.out" 2>&1; fi
    if [ -z "$T" ]; then
      git -C "$D/tree" checkout -q --detach "$TS" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" tft > "$D/logs/build-tft.out" 2>&1
      T=$(ls -d "$D"/bins/tft-"${TS:0:10}" 2>/dev/null | tail -n 1)
      [ -n "$T" ] && (cd "$D/tree" && CARGO_TARGET_DIR=$D/target nice -n 10 cargo build --release --locked -j 16 -p memra-engine --bin prime-fold-gate) > "$D/logs/build-pfg.log" 2>&1 \
        && cp -a "$D/target/release/prime-fold-gate" "$T/"
    fi
    B=$(ls -d "$D"/bins/tfb-"${BS:0:10}" 2>/dev/null | tail -n 1)
    if [ -n "$B" ] && [ -n "$T" ] && [ -x "$T/prime-fold-gate" ]; then echo "$(date -Is) built B=$B T=$T" >> "$CD/campaign.log"; else echo "TF-CELL-BUILD-FAIL" >> "$CD/campaign.log"; exit 1; fi
    ;;
  gate)
    (cd "$D/tree" && "$T/prime-fold-gate" "$G8") > "$CD/prime-fold-gate.log" 2>&1
    echo "prime-fold-gate rc=$? $(tail -n 1 "$CD/prime-fold-gate.log")" >> "$CD/campaign.log"
    grep -E "PRIME-FOLD|FAIL" "$CD/prime-fold-gate.log" | head -n 20 >> "$CD/campaign.log"
    ;;
  *)
    ARM=${NAME#*-}
    [ -n "$B" ] && [ -n "$T" ] || { echo "TF-CELL-BUILD-FAIL" >> "$CD/campaign.log"; exit 1; }
    bin=$B; [ "$ARM" = T ] && bin=$T
    echo "$(date -Is) cell $NAME bin=$bin" >> "$CD/campaign.log"
    bash "$D/box/run_arm.sh" "$CD/$NAME" "$bin" "MEMRA_TICK_TRACE=1" "131072 260000" "1 1" >> "$CD/campaign.log" 2>&1
    grep '^{' "$CD/$NAME/run.log" | sed "s/^/$ARM /" >> "$CD/campaign.log"
    grep -h 'stable-boundary capture\|tail fold\|insert (spec-boundary)' "$CD/$NAME/server.log" | sed "s/^/$ARM $NAME: /" >> "$CD/campaign.log"
    ;;
esac
echo "TF-CELL-DONE $NAME" >> "$CD/campaign.log"
