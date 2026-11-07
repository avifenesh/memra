#!/usr/bin/env bash
# On a Spark, under a sparklock lease the caller holds: sm_121a cell evidence for one commit of
# this lane, built on the Spark first (the recipe the other memra lanes use).
#   gb10-evidence.sh COMMIT
set -u
export PATH=$HOME/.cargo/bin:$PATH
COMMIT=$1
REPO=$HOME/projects/memra-cells-repo
S=$HOME/.cache/memra-prefix-budget
mkdir -p "$S"
git -C "$REPO" fetch -q origin main fix/discrete-prefix-budget || { echo "fetch failed"; exit 2; }
if [ ! -d "$S/gb10-src" ]; then
  git -C "$REPO" worktree add -q --detach "$S/gb10-src" "$COMMIT" || { echo "worktree failed"; exit 2; }
fi
git -C "$S/gb10-src" checkout -q --detach "$COMMIT" || { echo "checkout failed"; exit 2; }
python3 -B "$S/gb10-src/tools/cell_evidence.py" run --repo "$REPO" --device gb10 --commit "$COMMIT" \
  --kinds battery,perf,serving --base origin/main --out "$S/gb10-out-${COMMIT:0:10}" --work "$S/gb10-work" --no-lock
echo "EVIDENCE EXIT $?"
