#!/usr/bin/env bash
# Compile env_audit.rs's real test module standalone against a generated registry and run it.
#   A  new test, old registry, old sources   (fail-before: expect a_retired_door... to FAIL)
#   B  new test, new registry, new sources   (expect every test to pass)
#   C  old test, old registry, old sources   (harness sanity: expect pass)
#   D  new test, new registry, old sources   (the old env read: expect retired_names... to FAIL)
set -u
FB=$HOME/.cache/memra-gb10-r5/seam/failbefore
WT=$HOME/.cache/memra-gb10-r5/wt-seam
NEWT=$WT/crates/memra-engine/src/env_audit.rs
OLDT=$FB/old/crates/memra-engine/src/env_audit.rs
cell() { # NAME TESTSRC REGISTRY_TREE SOURCE_TREE
  local name=$1 src=$2 reg=$3 tree=$4
  OUT_DIR=$FB/$reg/out CARGO_MANIFEST_DIR=$FB/$tree/crates/memra-engine \
    rustc --edition 2024 --test --cfg memra_env_registry_present -A warnings -o "$FB/test-$name" "$src" \
    > "$FB/cell-$name.log" 2>&1 || { echo "$name: compile failed"; return; }
  "$FB/test-$name" --test-threads 1 >> "$FB/cell-$name.log" 2>&1
  echo "$name: rc=$? $(grep -E '^test result' "$FB/cell-$name.log")"
  grep -E '^test .* FAILED$|panicked at|refusals|still read' "$FB/cell-$name.log" | head -n 6
}
cell A "$NEWT" old old
cell B "$NEWT" new new
cell C "$OLDT" old old
cell D "$NEWT" new old
