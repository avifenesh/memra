#!/usr/bin/env bash
# Build the final candidate and its flush variant, after the given landing build ends.
set -u
WAIT_PID=$1
kill -0 "$WAIT_PID" 2>/dev/null && waitfor --gone "$WAIT_PID" --timeout 90m --quiet
cd /home/avifenesh/.cache/memra-prefix-budget/wt || exit 2
L=cand-$(git rev-parse --short=10 HEAD)
bash /home/avifenesh/.cache/memra-prefix-budget/build.sh "$L" || exit 3
bash /home/avifenesh/.cache/memra-prefix-budget/harness/build-flush-variant.sh || exit 4
echo FINAL_BUILDS_DONE "$L"
