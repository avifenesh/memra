#!/usr/bin/env bash
# Runs ON cb33 under the caller's sparklock lease (no timing runs): build the follow-up head in
# ~/.cache/memra-f8f4v3/src, kernel-check in the default mode (every MMQ-F8F4-V2* cell, the gate/up cell included,
# under both mainloops against v1), then gates-v3b.sh (16k argmax-margin with v3 and V3=0, pp3855
# logits with v3, V3=0 and V2=0). Stops before a step that cannot end by 05:46.
# usage: fup-gates.sh SHA BRANCH
set -u
SHA=$1; BR=$2
D=$HOME/.cache/memra-f8f4v3
S=${SHA:0:10}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH" TMPDIR=$D/tmp
log(){ echo "$(date +%T) $*"; }
late(){ [ "$(date +%H%M)" -ge "$1" ] && [ "$(date +%H%M)" -lt 0700 ]; }
T=$D/src   # rsynced from the follow-up worktree by content, so the build is incremental over $D/target
cd "$T" || exit 1
CARGO_TARGET_DIR=$D/target nice -n 10 cargo build --release --locked -j 16 \
  -p memra-engine --bin kernel-check --bin run-gen --bin run-spec --bin argmax-margin-probe \
  -p memra-server --bin memra-server > "$D/logs-build-$S.log" 2>&1
rc=$?; log "cargo rc=$rc"
[ $rc = 0 ] || { tail -n 30 "$D/logs-build-$S.log"; exit 1; }
B=$D/bins/v3-$S; mkdir -p "$B"
cp -a "$D"/target/release/{memra-server,run-gen,run-spec,kernel-check,argmax-margin-probe} "$B"/
echo "$SHA" > "$B/COMMIT"; (cd "$B" && sha256sum memra-server run-gen run-spec kernel-check argmax-margin-probe > SHA256SUMS)
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
O=$D/runs/fup-$S; mkdir -p "$O"
if late 0535; then log "past 05:35, stopping before kernel-check"; echo FUP-STOPPED; exit 0; fi
(unset MEMRA_KC_FAST MEMRA_KC_ONLY; "$B/kernel-check" "$G8" --require-manifest "$T/tools/kernel-check-27b.cells" \
   --require-manifest "$T/tools/kernel-check-step35.cells") > "$O/kernel-check-default.log" 2>&1
log "kernel-check rc=$? $(tail -n 1 "$O/kernel-check-default.log")"
grep -E "MMQ-F8F4-V2.*(mainloop|SUMMARY|GATEUP)" "$O/kernel-check-default.log" | head -n 20
if late 0541; then log "past 05:41, stopping before the logits gates"; echo FUP-STOPPED; exit 0; fi
bash "$D/gates-v3b.sh" "$B" "$O/gates" "$T" > "$O/gates.out" 2>&1
log "gates done"; cat "$O/gates/gates.log"
echo FUP-GATES-DONE
