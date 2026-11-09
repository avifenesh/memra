#!/usr/bin/env bash
# Run on a Spark (under the unit's sparklock lease): a private clone, then the census check with
# main's tools and with the branch's tools, then a cell tree built by the branch's tree_at.
set -uo pipefail
W=$HOME/.cache/memra-c8r2-gb10
C=${1:?commit}
mkdir -p "$W"
cd "$W" || exit 2
if [ ! -d repo ]; then
  if [ -d "$HOME/projects/memra-cells-repo" ]; then
    nice -n 10 git clone -q --no-local --no-checkout "$HOME/projects/memra-cells-repo" repo || exit 3
  else
    nice -n 10 git clone -q --no-checkout https://github.com/avifenesh/memra.git repo || exit 3
  fi
fi
nice -n 10 git -C repo fetch -q https://github.com/avifenesh/memra.git main || exit 4
git -C repo config --get-regexp 'promisor|partialclone'; echo "partial-clone keys rc=$? (1 = none)"
for v in main branch; do
  rm -rf "tools-$v"; mkdir -p "tools-$v"; tar -xzf "tools-$v.tgz" -C "tools-$v"
done
echo "== main tools"
python3 -I census_check.py repo "$C" tools-main 2>&1 | tail -n 8
echo "== branch tools, with a cell tree"
python3 -I census_check.py repo "$C" tools-branch "$W/tree-121a" 2>&1 | tail -n 10
echo "git status of the tree: $(git -C "$W/tree-121a" status --porcelain | wc -l) lines"
