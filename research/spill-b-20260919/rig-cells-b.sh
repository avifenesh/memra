#!/usr/bin/env bash
# Non-serving Linux RTX 5090 fitting cells ONLY; PRO-pair long contexts are not here.
# v2 order: baseline build → prefix teeth → patched build → prefix teeth → 8k/32k.
# The Python driver preserves raw stdout/stderr before emitting any parsed JSONL.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/rig-cells-b.py" "$@"
