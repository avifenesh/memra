#!/usr/bin/env bash
# Reader self-check on the Workstation receipt (DAY40.md 2.1, pro-single-day40/box/): re-run the per-boot parser and
# the three readers on the receipt's raw boots and compare every output with the receipt's. CPU only.
# The parser writes rows.jsonl into its boot dir, so each boot is a dir of symlinks to the receipt's raw inputs under
# <scratch> (small; the caller deletes <scratch>). The receipt itself is never written.
# usage: selfcheck.sh <ws box dir> <scratch dir>     (<ws box dir> holds boots/, SUMMARY.txt, READINGS.txt, FAULTS-local.txt)
set -uo pipefail
WS=$(cd "${1:?ws box dir}" && pwd -P); S=${2:?scratch dir}
K=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
D=$K/research/spill-b-20260919
[ -e "$S" ] && { echo "scratch $S exists; refusing"; exit 2; }
mkdir -p "$S/boots" "$S/out"
ok=1
for b in O1-off O1-on2048 O1-on8192 O1-on32768 O2-off O2-on2048 O2-on8192 O2-on32768; do
  mkdir "$S/boots/$b"
  for f in server.log client.jsonl samples.csv marks.json shape.txt; do ln -s "$WS/boots/$b/$f" "$S/boots/$b/$f"; done
  python3 "$D/day31-parse.py" "$S/boots/$b" > "$S/boots/$b/REPORT.txt" 2>&1
  prc=$?
  cmp -s "$S/boots/$b/REPORT.txt" "$WS/boots/$b/REPORT.txt" && rep=same || { rep=DIFFERS; ok=0; }
  cmp -s "$S/boots/$b/rows.jsonl" "$WS/boots/$b/rows.jsonl" && rows=same || { rows=DIFFERS; ok=0; }
  echo "SELFCHECK parse boot=$b exit=$prc REPORT.txt=$rep rows.jsonl=$rows"
done
bash "$K/readers.sh" "$S/boots" "$S/out" > "$S/out/readers.out" 2>&1
echo "SELFCHECK readers exit=$?"
for pair in SUMMARY.txt:SUMMARY.txt READINGS.txt:READINGS.txt FAULTS.txt:FAULTS-local.txt; do
  mine=${pair%%:*}; theirs=${pair#*:}
  if cmp -s "$S/out/$mine" "$WS/$theirs"; then echo "SELFCHECK $mine = receipt $theirs (byte-identical)"
  else ok=0; echo "SELFCHECK $mine DIFFERS from receipt $theirs:"; diff "$WS/$theirs" "$S/out/$mine" | head -20; fi
done
want=$(grep -m1 '^DAY34 V-DOOR ' "$WS/SUMMARY.txt")
got=$(head -1 "$S/out/VERDICT.txt")
echo "SELFCHECK receipt verdict: $want"
echo "SELFCHECK reader verdict:  $got"
[ -n "$want" ] && [ "$want" = "$got" ] || ok=0
[ $ok = 1 ] && echo "SELFCHECK PASS: the kit's parser and readers reproduce the Workstation receipt" || echo "SELFCHECK FAIL"
[ $ok = 1 ]
