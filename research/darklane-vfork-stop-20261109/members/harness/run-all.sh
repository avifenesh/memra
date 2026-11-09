#!/bin/bash
# The red control, then the verification on the restored fix.
set -u
D=$WORK/darklane/round3
rm -f "$D/all.DONE"
bash "$D/red.sh" > "$D/red.out" 2>&1
bash "$D/verify.sh" > "$D/verify.out" 2>&1
echo done > "$D/all.DONE"
