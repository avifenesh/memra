#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's sparklock lease ($LEASE, unit $UNIT): the lane's sm_121a
# cell evidence. A detached worktree of REPO at HEAD runs tools/cell_evidence.py with its own work
# dir, --no-lock because the caller holds the unit's lease.
# usage: LEASE=<id> UNIT=<unit2|unit1> cells_run.sh REPO HEAD BASE
set -u
REPO=$1; HEAD=$2; BASE=$3
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export SPARKLOCK_LEASE=${LEASE:?lease} MEMRA_SPARK_UNIT=${UNIT:?unit}
D=$HOME/.cache/memra-attn-gb10
S=$D/cells-src
mkdir -p "$D/cells" "$D/logs"
git -C "$REPO" fetch -q origin main research/gb10-prefill-attn || { echo "fetch failed"; exit 2; }
git -C "$REPO" cat-file -e "$HEAD^{commit}" || { echo "head $HEAD missing"; exit 2; }
if [ -e "$S/.git" ]; then
  git -C "$S" checkout -q --detach --force "$HEAD" || exit 2
else
  git -C "$REPO" worktree add -q --detach "$S" "$HEAD" || exit 2
fi
cd "$S" || exit 2
date -Is
python3 tools/cell_evidence.py run --repo "$S" --device gb10 --commit "$HEAD" --kinds battery,perf,serving \
  --base "$BASE" --out "$D/cells/gb10-${HEAD:0:10}" --work "$D/cells-work" --no-lock
echo "cell rc=$?"
date -Is
echo CELLS-DONE
