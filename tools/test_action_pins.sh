#!/usr/bin/env bash
# Teeth for check-action-pins.sh: named steps and reusable-workflow jobs must not
# escape the immutable-SHA census merely because `uses:` is not the list key.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d /tmp/action-pins-test.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

fail() { echo "FAIL: $1" >&2; exit 1; }
mkdir -p "$tmp/.github/workflows" "$tmp/tools"
cp "$here/tools/check-action-pins.sh" "$tmp/tools/"

cat > "$tmp/.github/workflows/test.yml" <<'EOF'
name: fixture
jobs:
  named-step:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
  reusable-job:
    uses: owner/workflow/.github/workflows/reuse.yml@main
EOF

if out=$(cd "$tmp" && tools/check-action-pins.sh 2>&1); then
  fail "census accepted named-step and reusable-job mutable refs"
fi
echo "$out" | grep -q 'actions/checkout@v4' \
  || fail "named-step mutable ref was not reported: $out"
echo "$out" | grep -q 'owner/workflow/.github/workflows/reuse.yml@main' \
  || fail "reusable-job mutable ref was not reported: $out"

sed -i \
  -e 's#actions/checkout@v4#actions/checkout@11d5960a326750d5838078e36cf38b85af677262#' \
  -e 's#owner/workflow/.github/workflows/reuse.yml@main#owner/workflow/.github/workflows/reuse.yml@1111111111111111111111111111111111111111#' \
  "$tmp/.github/workflows/test.yml"
(cd "$tmp" && tools/check-action-pins.sh) \
  || fail "census refused full-SHA named-step and reusable-job refs"

echo "action-pin census fixture: PASS"
