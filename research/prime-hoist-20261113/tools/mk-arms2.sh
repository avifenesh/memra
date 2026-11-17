#!/usr/bin/env bash
# memra#1321 fold: the three diagnostic arms on main 0db086f87a, as file sets for the dev desk's
# one-tree incremental build (dd-cap2-build.sh). Each arm = the five files the PR touches, with
# the [diag1255] lines (diag-worker.py) and the allocation ring (diag-ring.py).
#   dmain2 = main 0db086f87a
#   ds2h2  = the PR head before the fold (7b014d604a: steps 1 and 2 and the park floor)
#   ds2i   = the fold (wt-s2r's working tree, or the commit given as $1)
# usage: mk-arms2.sh [FOLD_COMMIT]
set -eu
W=$HOME/.cache/memra-1258/wt-s2r
A=$HOME/.cache/memra-1258/cap-arms2
F="crates/memra-engine/src/hybrid_forward.rs crates/memra-engine/src/lib.rs crates/memra-engine/src/mmq_ffi.rs crates/memra-engine/src/qwen_prime_graph.rs crates/memra-server/src/worker.rs"
mkdir -p "$A/dmain2" "$A/ds2h2" "$A/ds2i"
for f in $F; do
  git -C "$W" show "0db086f87a:$f" > "$A/dmain2/$(basename "$f")"
  git -C "$W" show "7b014d604a:$f" > "$A/ds2h2/$(basename "$f")"
  if [ -n "${1:-}" ]; then git -C "$W" show "$1:$f" > "$A/ds2i/$(basename "$f")"; else cp "$W/$f" "$A/ds2i/$(basename "$f")"; fi
done
for arm in dmain2 ds2h2 ds2i; do
  python3 "$HOME/.cache/memra-1258/diag-worker.py" "$A/$arm/worker.rs"
  python3 "$HOME/.cache/memra-1258/fold/diag-ring.py" "$A/$arm/lib.rs" "$A/$arm/worker.rs"
  rustfmt --edition 2024 --check "$A/$arm/lib.rs" "$A/$arm/worker.rs" > /dev/null 2>&1 || echo "note: $arm not rustfmt-clean (diagnostic only)"
  ( cd "$A/$arm" && sha256sum ./*.rs > SHA256 )
done
echo "${1:-worktree}" > "$A/ds2i/SOURCE"
echo "arms ready under $A"
