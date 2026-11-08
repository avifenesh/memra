#!/usr/bin/env bash
# Phase X (exactness): greedy agentic cells (T=0, 256 new tokens, prime then measure).
#   phaseX.sh LABEL BIN USERS [VAR=VALUE ...]
set -u
H=<harness>
O=<runs>/phaseX
mkdir -p "$O"
L=$1; B=$2; U=$3; shift 3
bash $H/arm2.sh "$L" "$B" "$O" "agentic:$U:0.0:256" "$@"
