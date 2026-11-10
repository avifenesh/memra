#!/usr/bin/env bash
# Runs ON a GB10 unit under the caller's lease: lever 1's missing margin cell, the calibrated argmax
# margin gate on Qwen3.8-27B, real text (the first 16,390 tokens of research/e2e/prompts/p4-16k.txt,
# window rows 16,378-16,389 straddling the 16,384 floor), the SERVED BATCH decode arm.
# Arms: T = the landing bundle's recorded build 12c3f554c4, B = main 031f146690's recorded build.
# The probe binaries are copies of the landing agent's recorded binaries, sha256 checked before and
# after; nothing is built and the landing's tree is not touched. usage: rt_cell.sh T|B
set -u
R=$HOME/.cache/memra-split-rung
case $1 in
  T) C=12c3f554c4fdc3992fea68acf3a1eb6a2f943e6b; WANT=22293a1dac13; OUT=am-27b-p4-16390-b-12c3f554c4 ;;
  B) C=031f14669046c5ec1e9f0e0d7f5a143b7d28aa17; WANT=6e6650071f22; OUT=am-27b-p4-16390-b-main-031f146690 ;;
  *) echo "usage: rt_cell.sh T|B"; exit 2 ;;
esac
KT=$R/rt/kt-$C
D=$R/cells/$OUT
mkdir -p "$D"
got=$(sha256sum "$KT/release/argmax-margin-probe" | cut -c1-12)
echo "$(date -Is) arm=$1 commit=$C probe sha256 before: $(sha256sum "$KT/release/argmax-margin-probe" | cut -d' ' -f1)" > "$D/log"
[ "$got" = "$WANT" ] || { echo "RT-FAIL probe digest $got, want $WANT" | tee -a "$D/log"; exit 1; }
sha256sum "$R/prompts/p4-16390.txt" >> "$D/log"
CARGO_TARGET_DIR=$KT bash "$R/src-1a4f769861/tools/argmax-margin-gate.sh" \
  "$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
  --prompt "$R/prompts/p4-16390.txt" --decode batch --logdir "$D/gate" >> "$D/log" 2>&1
rc=$?
echo "$rc" > "$D/rc"
echo "$(date -Is) rc=$rc probe sha256 after: $(sha256sum "$KT/release/argmax-margin-probe" | cut -d' ' -f1)" >> "$D/log"
echo "RT-DONE $1 rc=$rc"
