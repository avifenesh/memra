#!/usr/bin/env bash
# The DAY40 target-card readers, as research/spill-b-20260919/pro-single-day40/chain.sh:37-42 ran them, with the
# DAY40.md 2.1 fault-lister fix (each boot dir, not the boots root). CPU only; never touches a GPU.
# usage: readers.sh <boots dir> <dest dir>
#   writes <dest>/SUMMARY.txt (day34-compare.py, the registered command line), FAULTS.txt (day31-faults.py),
#   READINGS.txt (day40-read.py) and VERDICT.txt (its first line is the DAY34 V-DOOR line verbatim, or NO-VERDICT).
set -uo pipefail
B=${1:?boots dir}; DEST=${2:?dest dir}
K=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
D=$K/research/spill-b-20260919
mkdir -p "$DEST"
BOOTS="$B/O1-off $B/O1-on2048 $B/O1-on8192 $B/O1-on32768 $B/O2-off $B/O2-on2048 $B/O2-on8192 $B/O2-on32768"
python3 "$D/day34-compare.py" --card pro6000 --served-ctx 262144 --model-ctx 262144 --registry-value 32768 \
  --survey context O1:$B/O1-off O1:$B/O1-on2048 O1:$B/O1-on8192 O1:$B/O1-on32768 O2:$B/O2-off O2:$B/O2-on2048 \
  O2:$B/O2-on8192 O2:$B/O2-on32768 > "$DEST/SUMMARY.txt" 2>&1
crc=$?
# shellcheck disable=SC2086  # BOOTS is the fixed list of eight dirs above
python3 "$D/day31-faults.py" $BOOTS > "$DEST/FAULTS.txt" 2>&1
frc=$?
python3 "$D/day40-read.py" pro6000 "$B" > "$DEST/READINGS.txt" 2>&1
rrc=$?
line=$(grep -m1 '^DAY34 V-DOOR ' "$DEST/SUMMARY.txt")
{
  if [ -n "$line" ]; then echo "$line"; else echo "NO-VERDICT: day34-compare.py printed no DAY34 V-DOOR line (exit $crc); see SUMMARY.txt"; fi
  echo "readers: day34-compare exit=$crc day31-faults exit=$frc day40-read exit=$rrc"
} > "$DEST/VERDICT.txt"
cat "$DEST/VERDICT.txt"
[ -n "$line" ] && [ $crc = 0 ]
