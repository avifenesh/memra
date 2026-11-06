#!/usr/bin/env bash
# Runs ON the GB10 host: build each named branch in the lane tree and bank its binaries.
# usage: build_queue.sh name=branch ...   -> bins/<name>-<sha10>/ ; prints BANK lines
set -u
D=$HOME/.cache/memra-prefill
T=$D/tree
for spec in "$@"; do
  name=${spec%%=*}; br=${spec#*=}
  git -C "$T" checkout -q --detach "$br" || { echo "checkout $br failed"; continue; }
  sha=$(git -C "$T" rev-parse --short=10 HEAD)
  # touch tracked files so no crate is reused from another tree state (shared-target trap)
  git -C "$T" ls-files -z | (cd "$T" && xargs -0 touch -c)
  log=$D/build-$name.log
  if bash "$D/build.sh" "$T" "$D/target" > "$log" 2>&1; then
    B=$D/bins/$name-$sha; mkdir -p "$B"
    cp -a "$D"/target/release/{memra-server,run-gen,run-spec,kernel-check,argmax-margin-probe} "$B"/
    git -C "$T" log --oneline -1 > "$B/COMMIT"
    (cd "$B" && sha256sum memra-server run-gen run-spec kernel-check > SHA256SUMS)
    n=$(grep -c "Compiling memra" "$log"); bad=$(grep "Compiling memra" "$log" | grep -vc "$T/crates/")
    echo "BANK $name $sha $B compiled_memra_crates=$n foreign=$bad"
  else
    echo "BUILD-FAIL $name $sha (see $log)"; tail -30 "$log"
  fi
done
echo QUEUE-DONE
