#!/usr/bin/env bash
set -euo pipefail
printf 'DRY_RUN_STUB: %s\n' "$1"
if [[ ${1:-} == fail ]]; then
  echo 'injected stub failure, not a GPU failure' >&2
  exit 7
fi
