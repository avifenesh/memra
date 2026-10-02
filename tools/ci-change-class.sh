#!/usr/bin/env bash
# Compatibility entry point: changed-input and dependency-aware CI selection.
# Missing Python, a broken planner, or malformed output must never suppress checks.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "${1:-}" = census ]; then
  shift
  python3 "$HERE/validation_plan.py" census "$@" || printf '?\n'
  exit 0
fi
if out=$(python3 "$HERE/validation_plan.py" ci "$@"); then
  printf '%s\n' "$out"
else
  printf 'code=true\nreason=validation-planner-unavailable\n'
  for job in build clippy server engine portable core lanes arch publish; do
    printf '%s=true\n' "$job"
  done
  printf 'packages=\nrequires_cuda=true\nmode=full\ncontracts=\n'
fi
