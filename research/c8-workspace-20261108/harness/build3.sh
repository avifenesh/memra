#!/usr/bin/env bash
# The three binaries of the scored campaign, built one after another in one worktree and one
# target dir (incremental): S = the fix-stack tip, C1 = + items 1 and 3, C2 = + item 2 (the 5090
# batch OOM key). Leaves the branch at C1 and item 2 on c8r2/oom-5090.
#   build3.sh S_SHA C1_SHA ITEM2_COMMIT
set -u
S=$1; C1=$2; I2=$3
WT=<tree>
H=<harness>
export TARGET=<lane>/target-stack
cd "$WT" || exit 2
git checkout -q --detach "$S" && bash "$H/build.sh" "$WT" "S-$(git rev-parse --short=10 HEAD)" || exit 3
git checkout -q c8r2/prefill-held && [ "$(git rev-parse HEAD)" = "$(git rev-parse "$C1")" ] || exit 4
bash "$H/build.sh" "$WT" "C1-$(git rev-parse --short=10 HEAD)" || exit 5
git checkout -q -B c8r2/oom-5090 && git cherry-pick "$I2" > /dev/null || exit 6
bash "$H/build.sh" "$WT" "C2-$(git rev-parse --short=10 HEAD)" || exit 7
git checkout -q c8r2/prefill-held
echo BUILD3_DONE
