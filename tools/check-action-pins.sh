#!/usr/bin/env bash
# Every external GitHub Action is executable supply-chain code. Only an immutable full commit SHA
# is accepted; keep the human version after a YAML comment for update readability.
set -euo pipefail

bad=0
while IFS=: read -r file line rest; do
  ref=${rest#*uses: }
  ref=${ref%% *}
  case "$ref" in
    ./*) continue ;;
  esac
  if [[ ! "$ref" =~ ^[^@[:space:]]+@[0-9a-f]{40}$ ]]; then
    printf '%s:%s: external action is not pinned to a full commit SHA: %s\n' "$file" "$line" "$ref" >&2
    bad=1
  fi
done < <(rg -n '^[[:space:]]*- uses:' .github/workflows)
exit "$bad"
