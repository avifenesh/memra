#!/usr/bin/env bash
# Laptop (memra#1233): the sellgate replay's tail-witness check, green on the tree and red with
# the check removed. The red arm runs the same unit tests against a copy of tools/ and research/
# whose sellgate_replay.py no longer compares whole-prompt restores with prefix_cache_tail_hits.
set -u
S=$HOME/.cache/memra-1233
OUT=$S/red-witness
mkdir -p "$OUT"
cd "$S/wt" || exit 1
run() { # arm root
  { date -Is; sha256sum "$2/research/sellgate-20260812/sellgate_replay.py"; } > "$OUT/$1.log"
  (cd "$2/tools" && python3 -m unittest -v test_q35_cold_mixed_gate 2>&1) >> "$OUT/$1.log"
  echo "$1 rc=$? $(tail -n 1 "$OUT/$1.log")" | tee -a "$OUT/summary.txt"
}
: > "$OUT/summary.txt"
git rev-parse HEAD >> "$OUT/summary.txt"
run green "$S/wt"
RED=$(mktemp -d "$S/red-witness-tree.XXXX")
mkdir -p "$RED/research"
cp -a tools "$RED/tools"
cp -a research/sellgate-20260812 "$RED/research/"
for d in $(grep -o 'research/[a-z0-9-]*' tools/test_q35_cold_mixed_gate.py tools/q35-cold-mixed-gate.py | cut -d: -f2 | sort -u); do
  [ -e "$RED/$d" ] || cp -a "$d" "$RED/research/" 2>/dev/null
done
python3 - "$RED/research/sellgate-20260812/sellgate_replay.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    if whole_n != tail_hits:\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "    if False and whole_n != tail_hits:\n"))
PY
diff -u research/sellgate-20260812/sellgate_replay.py "$RED/research/sellgate-20260812/sellgate_replay.py" > "$OUT/red.patch"
run red "$RED"
echo "$RED" > "$OUT/red-tree-path.txt"
