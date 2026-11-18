#!/usr/bin/env bash
# the PRO 6000 box, landing 27, CPU only (no card): dsv41-gate of the landing head for #1315's one-card PRO 6000 items (test list
# 5, 6a, 6b, 6e and the red arm), in its own full worktree ($L/src1315) and target dir ($L/target1315), so the cell
# record's tree and its six hashed binaries are not touched. Green first; then the red arm: the PR's red.patch
# (Group::moe back to main's check) applied, the gate rebuilt, the source restored. Not LTO (a gate binary: pass/fail).
# Stop: kill -TERM -- -$(cat $L/build1315.pgid)
# usage: build1315.sh HEAD_SHA   -> $L/build1315-<head10>.done holds "rc=N"; bins in $L/bins1315/{green,red}/dsv41-gate
set -u
. <land27>/kit/env27.sh
SHA=$1; T=${SHA:0:10}
S=$L/src1315; TG=$L/target1315; B=$L/bins1315
RED=research/modelplan-onboarding-dsv41-20261109/row-faults/receipts/gb10-f2efe6a3c9/redarm/red.patch
F=crates/memra-engine/src/dsv41_gpu/tp.rs
DN=$L/build1315-$T.done
rm -f "$DN"
echo "$$" > "$L/build1315.pgid"
fail() { say "build1315 $T FAILED: $1"; echo "rc=1 $1" > "$DN"; exit 1; }
mkdir -p "$B/green" "$B/red" "$L/logs"
git -C "$L/repo" cat-file -e "$SHA^{commit}" || git -C "$L/repo" fetch -q origin land/1308 || fail fetch
git -C "$L/repo" worktree remove --force "$S" > /dev/null 2>&1
git -C "$L/repo" worktree prune
nice -n 19 git -C "$L/repo" worktree add -q --detach "$S" "$SHA" > "$L/logs/src1315-$T.log" 2>&1 || fail "worktree"
cd "$S" || fail cd
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
export CARGO_TARGET_DIR=$TG MEMRA_CUDA_ARCH=120a CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
build() { taskset -c "$CPUS" nice -n 19 cargo build --release --locked -j 8 -p memra-engine --bin dsv41-gate > "$L/logs/build1315-$T-$1.log" 2>&1; }
say "build1315 $T start; load $(cut -d' ' -f1-3 /proc/loadavg)"
t0=$SECONDS
build green || fail "green build (logs/build1315-$T-green.log)"
cp "$TG/release/dsv41-gate" "$B/green/dsv41-gate"
say "build1315 $T: green dsv41-gate built in $((SECONDS - t0)) s"
git apply "$RED" || fail "red.patch does not apply"
git diff --stat > "$B/red/red.diffstat" 2>&1
git diff > "$B/red/red-as-applied.diff" 2>&1
t0=$SECONDS
build red; rc=$?
[ $rc = 0 ] && cp "$TG/release/dsv41-gate" "$B/red/dsv41-gate"
git checkout -q -- "$F"
[ -z "$(git status --porcelain)" ] || say "build1315 $T: WARNING src1315 not clean after the restore"
[ $rc = 0 ] || fail "red build (logs/build1315-$T-red.log)"
say "build1315 $T: red dsv41-gate built in $((SECONDS - t0)) s"
echo "$SHA" > "$B/SOURCE"
( cd "$B" && sha256sum green/dsv41-gate red/dsv41-gate ) > "$B/bins.sha256"
cmp -s "$B/green/dsv41-gate" "$B/red/dsv41-gate" && fail "red and green binaries are identical"
say "build1315 $T DONE"
echo "rc=0" > "$DN"
