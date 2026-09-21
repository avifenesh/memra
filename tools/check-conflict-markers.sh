#!/usr/bin/env bash
# check-conflict-markers.sh: refuse a tree whose tracked source or docs carry a merge-conflict
# marker line. WHY (2026-09-21, twice in one day): a merge resolved by hand left a diff3 base
# marker (`||||||| parent of ...`) in research/INDEX.md on main (#604, then #614). The next
# lane's union resolve read that line as a hunk base and silently dropped the neighbouring row
# (#607, restored in #609); the second time `git diff --check` on a moved base was the first
# thing to notice. `git diff --check` only sees a marker inside a diff; this census sees the
# tree. All four marker kinds are refused: `<<<<<<< `, `=======`, `>>>>>>> `, `||||||| `.
# Scope: tracked files with source/doc suffixes; receipt logs and raw dirs are excluded because
# a receipt may legitimately capture a diff. No skip switch: a tree with a marker in a source or
# registry file has no emergency in which pushing it is right. Teeth: tools/test_conflict_markers.sh.
# usage: tools/check-conflict-markers.sh [repo_dir]   (exit 0 clean, 1 markers found)
set -u
cd "${1:-.}" || { echo "check-conflict-markers: cannot cd to ${1:-.}" >&2; exit 2; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "check-conflict-markers: not a git repo" >&2; exit 2; }
pattern='^(<<<<<<< |=======$|>>>>>>> |\|\|\|\|\|\|\| )'
hits=$(git ls-files -z -- '*.md' '*.rs' '*.cu' '*.cuh' '*.h' '*.hpp' '*.cpp' '*.c' '*.py' '*.sh' \
        '*.toml' '*.yml' '*.yaml' '*.jinja' '*.json' '*.jsonl' '*.txt' \
      | grep -zvE '(^|/)(raw|receipts?)/|\.log$' \
      | xargs -0 grep -nE "$pattern" 2>/dev/null)
if [ -n "$hits" ]; then
  echo "check-conflict-markers: REFUSED, marker lines in tracked files:" >&2
  printf '%s\n' "$hits" | cut -c1-160 >&2
  exit 1
fi
echo "check-conflict-markers: OK (no conflict marker line in tracked source or docs)"
