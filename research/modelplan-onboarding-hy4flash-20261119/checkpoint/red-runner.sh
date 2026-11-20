#!/usr/bin/env bash
# The checkpoint gate's red arms: hy4f-run with a wrong program the gate must refuse.
# usage: HY4F_RUN=path/to/hy4f-run HY4F_RED=engram-off|ties-reversed|dequant-bf16 \
#        memra model verify checkpoint ... --native-runner red-runner.sh
exec "${HY4F_RUN:?HY4F_RUN names the hy4f-run binary}" "$@" --red "${HY4F_RED:?HY4F_RED names the arm}"
