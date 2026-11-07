#!/usr/bin/env bash
# Runs ON a GB10 host under this session's lease: the sm_121a cell record (battery, perf, serving)
# for the PR's final commit against BASE. usage: cells_gb10d.sh SHA BRANCH UNIT BASE
set -u
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
SHA=$1; BR=$2; UNIT=$3; BASE=$4
R=$HOME/projects/memra-gate
D=$HOME/.cache/memra-gb10-next
mkdir -p "$D/logs" "$D/cells"
git -C "$R" fetch -q origin main "$BR" fix/grid-law-tests-and-self-yield || { echo FETCH-FAIL; exit 1; }
git -C "$R" worktree remove --force "$D/cells-src" > /dev/null 2>&1
git -C "$R" worktree prune
git -C "$R" worktree add -q --detach "$D/cells-src" "$SHA" || { echo WORKTREE-FAIL; exit 1; }
cd "$D/cells-src" || exit 1
git log --oneline -1
MEMRA_SPARK_UNIT=$UNIT python3 tools/cell_evidence.py run --device gb10 --commit "$SHA" \
  --kinds battery,perf,serving --base "$BASE" --out "$D/cells/gb10" --work "$D/cells-work/gb10" --no-lock
echo "CELLS-GB10-DONE rc=$?"
