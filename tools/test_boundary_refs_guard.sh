#!/usr/bin/env bash
set -euo pipefail
workflow=$(cd "$(dirname "$0")/.." && pwd)/.github/workflows/boundary-refs.yml
python3 - "$workflow" <<'PY'
import pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text()
required = [
    "remote_pull=$(git ls-remote origin 'refs/pull/*/head' | wc -l)",
    'if [ "$remote_pull" -gt 0 ]; then',
    'if [ "$local_pull" -lt "$remote_pull" ]; then',
    'if [ "$total" -gt 5000 ]; then',
]
for needle in required:
    assert needle in text, f"missing boundary-ref guard: {needle}"
assert text.index('if [ "$remote_pull" -gt 0 ]; then') < text.index('pull-refs: remote namespace is empty')
PY
echo "boundary-ref workflow guard: fail-closed PR namespace and ref budget present"
