#!/usr/bin/env bash
# Runs ON u1 (CPU only, under the caller's lease): more sm_121a memra-server arms from the
# 782dc471b7 tree in src2 (spark_build_probe.sh unpacked it), each the tree with the given
# worker.rs and spec.rs from arms/. Stops at CAP_MIN (default 25); the next call resumes.
# usage: spark_build_arms.sh CAP_MIN OUT:WORKER_FILE:SPEC_FILE...   -> bins/OUT/memra-server
set -u
R=$HOME/.cache/memra-1221; CAP=$1; shift
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target2
L=$R/logs/build-arms.log
end=$(( $(date +%s) + CAP * 60 ))
cd "$R/src2" || exit 1
for a in "$@"; do
  IFS=: read -r OUT WF SF <<< "$a"
  [ -x "$R/bins/$OUT/memra-server" ] && continue
  left=$(( end - $(date +%s) )); [ "$left" -gt 60 ] || { echo "$(date -Is) cap reached before $OUT" >> "$L"; exit 1; }
  cp "$R/arms/$WF" crates/memra-server/src/worker.rs; touch crates/memra-server/src/worker.rs
  cp "$R/arms/$SF" crates/memra-engine/src/spec.rs; touch crates/memra-engine/src/spec.rs
  echo "== $OUT ($WF, $SF) $(date -Is)" >> "$L"; sha256sum "$R/arms/$WF" "$R/arms/$SF" | sed "s|$HOME|~|" >> "$L"
  timeout -s INT "$left" nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 \
    || { echo "$(date -Is) BUILD-STOP $OUT" >> "$L"; exit 1; }
  mkdir -p "$R/bins/$OUT"; cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$OUT/"
  sha256sum "$R/bins/$OUT/memra-server" | sed "s|$HOME|~|" >> "$L"
done
echo "$(date -Is) BUILD-DONE" >> "$L"
