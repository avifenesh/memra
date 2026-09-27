#!/usr/bin/env bash
# Teeth for tools/check-conflict-markers.sh: each marker kind refused in a tracked doc, a clean
# tree passes, a marker inside an excluded receipt log or raw dir is not a refusal.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
chk=$here/tools/check-conflict-markers.sh
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
repo=$tmp/repo; git init -q "$repo"
g() { git -C "$repo" "$@"; }
g config user.email t@t; g config user.name t
pass=0; ok() { pass=$((pass+1)); echo "ok   $*"; }; bad() { echo "FAIL $*" >&2; exit 1; }
mkdir -p "$repo/research/lane/raw" "$repo/crates/x/src"
printf 'a | row\nb | row\n' > "$repo/research/INDEX.md"
printf 'fn main() {}\n' > "$repo/crates/x/src/main.rs"
g add -A; g commit -q -m base
"$chk" "$repo" >/dev/null || bad "arm1 clean tree refused"; ok "arm1 clean tree passes"
i=2
for marker in '<<<<<<< HEAD' '=======' '>>>>>>> theirs' '||||||| parent of abc (msg)'; do
  printf 'a | row\n%s\nb | row\n' "$marker" > "$repo/research/INDEX.md"; g add -A; g commit -q -m "m$i"
  if "$chk" "$repo" >/dev/null 2>&1; then bad "arm$i marker '$marker' not refused"; fi
  ok "arm$i marker '$marker' refused"; i=$((i+1))
done
printf 'a | row\nb | row\n' > "$repo/research/INDEX.md"
printf 'diff capture\n<<<<<<< HEAD\n=======\n>>>>>>> x\n' > "$repo/research/lane/raw/capture.txt"
printf '<<<<<<< HEAD\n' > "$repo/research/lane/cell.log"
g add -A; g commit -q -m receipts
"$chk" "$repo" >/dev/null || bad "arm$i receipt log or raw dir refused"; ok "arm$i receipt log and raw dir excluded"; i=$((i+1))
printf 'x\n<<<<<<< HEAD\n' > "$repo/crates/x/src/main.rs"; g add -A; g commit -q -m src
if "$chk" "$repo" >/dev/null 2>&1; then bad "arm$i marker in source not refused"; fi
ok "arm$i marker in a source file refused"
echo "test_conflict_markers: $pass arms PASS"
