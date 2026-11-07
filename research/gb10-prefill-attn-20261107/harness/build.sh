#!/usr/bin/env bash
# Runs ON a GB10 host: build the gate/bench/server binaries from a tree into the private target
# dir and bank them under bins/NAME-SHA10. usage: build.sh NAME
# Touches every tracked file first so no crate is reused from another tree state.
set -u
NAME=$1
D=$HOME/.cache/memra-attn-gb10
T=$D/tree
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CARGO_TARGET_DIR=$D/target
sha=$(git -C "$T" rev-parse --short=10 HEAD)
dirty=$(git -C "$T" status --porcelain --untracked-files=no | wc -l)
log=$D/logs/build-$NAME.log
{
  git -C "$T" log --oneline -1; echo "dirty_files=$dirty"; git -C "$T" status --porcelain --untracked-files=no
  date -Is
  [ "${NOTOUCH:-0}" = 1 ] || git -C "$T" ls-files -z | (cd "$T" && xargs -0 touch -c)
  cd "$T" && flock "$HOME/.cache/memra-rig-build.lock" nice -n 10 cargo build --release --locked -j "${JOBS:-12}" \
    -p memra-engine --bin kernel-check --bin run-gen --bin run-spec --bin argmax-margin-probe \
    -p memra-server --bin memra-server 2>&1
  rc=$?
  date -Is
  echo "cargo rc=$rc"
} > "$log" 2>&1
if grep -q "^cargo rc=0" "$log"; then
  B=$D/bins/$NAME-$sha; mkdir -p "$B"
  cp -a "$D"/target/release/{memra-server,run-gen,run-spec,kernel-check,argmax-margin-probe} "$B"/
  { git -C "$T" log --oneline -1; echo "dirty_files=$dirty"; } > "$B/COMMIT"
  git -C "$T" diff > "$B/dirty.patch"
  (cd "$B" && sha256sum memra-server run-gen run-spec kernel-check argmax-margin-probe > SHA256SUMS)
  n=$(grep -c "Compiling memra" "$log"); bad=$(grep "Compiling memra" "$log" | grep -vc "$T/crates/")
  echo "BANK $NAME $sha $B compiled_memra_crates=$n foreign=$bad"
else
  echo "BUILD-FAIL $NAME $sha"; grep -E "^(error|warning: unused)" -A5 "$log" | head -n 60
fi
