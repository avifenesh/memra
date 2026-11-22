#!/bin/bash
# The script before the change is tools/serve-smoke.sh at 8c0b39b40e, written to
# tools/serve-smoke-base.sh for the run (git show 8c0b39b40e:tools/serve-smoke.sh) and removed after.
# serve-smoke on the inkling2 pack directory, then the default GGUF battery with this branch's
# script and with the script before the change, on the same binary (run from the worktree root).
set -uo pipefail
export CARGO_TARGET_DIR=$HOME/.cache/inkling2mini/target-serve
O=research/modelplan-onboarding-inkling2mini-20261121/serve/smoke
mkdir -p $O/inkling2 $O/gguf-new $O/gguf-base
SMOKE_OUT=$O/inkling2 bash tools/serve-smoke.sh $HOME/models/inkling2-mini > $O/inkling2.txt 2>&1; echo "inkling2 rc=$?"
SMOKE_OUT=$O/gguf-new bash tools/serve-smoke.sh > $O/gguf-new.txt 2>&1; echo "gguf-new rc=$?"
SMOKE_OUT=$O/gguf-base bash tools/serve-smoke-base.sh > $O/gguf-base.txt 2>&1; echo "gguf-base rc=$?"
echo RUN_SMOKE_DONE
