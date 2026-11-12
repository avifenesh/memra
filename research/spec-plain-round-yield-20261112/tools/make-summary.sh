#!/usr/bin/env bash
# Rebuilds SUMMARY.md in the receipts dir: one section per (summary tool, receipts dir), each the
# tool's output over that dir. usage: tools/make-summary.sh (from anywhere)
set -u
cd "$(dirname "$0")/.." || exit 1
first=1
while read -r tool dir; do
  compgen -G "$dir/*.jsonl" > /dev/null || continue
  [ "$first" = 1 ] || echo
  first=0
  printf '## %s\n\n`python3 tools/%s %s`\n\n' "$dir" "$tool" "$dir"
  python3 "tools/$tool" "$dir"
done > SUMMARY.md <<'EOF'
summ_plain.py gb10/plain
summ_plain.py gb10/t1yield
summ_plain.py gb10/plain2
summ_plain.py gb10/plain3
summ_plain.py gb10/plain5
summ_plain.py gb10/longprime
summ_plain.py rtx5090/plain
summ_plain.py rtx5090/plain2
summ_plain.py rtx5090/plain5
summ_riders.py gb10/riders
summ_riders.py gb10/riders9
summ_riders.py gb10/riders10
summ_riders.py rtx5090/riders5
summ_riders.py rtx5090/riders6
summ_riders.py rtx5090/riders7
summ_newcomer.py gb10/new
summ_newcomer.py gb10/new2
summ_newcomer.py rtx5090/new
summ_newcomer.py rtx5090/new5
EOF
