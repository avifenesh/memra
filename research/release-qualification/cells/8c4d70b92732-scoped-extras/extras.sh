#!/usr/bin/env bash
# Scoped extras of landing 7 on the landing's own build (receipts beside the records, not gate records),
# run inside the caller's device lease after the record:
#  (1) #1230's pooled cohort gate (tools/spec-stop-string-gate.py --cohort 2,4; every boot runs t1 and greedy):
#      a plain ref boot, then the landing binary on the MTP route with --settle, then the judge.
#  (2) #1224's end-of-request cells at temperature 1 (close_cell.py MODE=close and MODE=demote), judged against
#      the #1224 lane's candidate rows (close_summary.py).
# usage: extras.sh DEVICE(rtx5090|gb10) TREE SERVER MODEL OUT
set -u
DEV=$1; TREE=$2; SRV=$3; MODEL=$4; OUT=$5
for v in $(compgen -e | grep '^MEMRA_'); do unset "$v"; done
mkdir -p "$OUT"
cd "$TREE" || exit 1
R=research/spec-sampled-pending-carry-20261110
S="$OUT/summary.txt"
: > "$S"
echo "tree $(git log --oneline -1), server $(sha256sum "$SRV" | cut -c1-16), model $MODEL, $(date -Is)" >> "$S"
# (1) cohort gate
C="$OUT/cohort"
mkdir -p "$C"
python3 tools/spec-stop-string-gate.py boot ref "$SRV" "$MODEL" --out "$C" --cohort 2,4 --port 18217 > "$OUT/cohort-ref.log" 2>&1
echo "cohort ref boot rc=$?" >> "$S"
python3 tools/spec-stop-string-gate.py boot land "$SRV" "$MODEL" --out "$C" --cohort 2,4 --settle --port 18217 > "$OUT/cohort-land.log" 2>&1
echo "cohort land boot rc=$?" >> "$S"
python3 tools/spec-stop-string-gate.py judge "$C" --case land > "$OUT/cohort-judge.log" 2>&1
echo "cohort judge rc=$?: $(grep -E 'GATE: (PASS|FAIL)' "$OUT/cohort-judge.log" | tail -n 1)" >> "$S"
# (2) end-of-request cells
E="$OUT/close"
mkdir -p "$E"
cp "$R/$DEV/close/stops.json" "$E/"
for m in close demote; do
  env CELL_MODEL="$MODEL" CELL_PORT=18216 BOOT_START=1 MODE=$m python3 "$R/tools/close_cell.py" "$E" "$SRV" land > "$OUT/close-$m.log" 2>&1
  echo "close cell $m rc=$?: $(tail -n 1 "$OUT/close-$m.log")" >> "$S"
  python3 "$R/tools/close_summary.py" "$R/$DEV/close/$m.jsonl" "$E/$m.jsonl" cand land > "$OUT/close-$m-judge.log" 2>&1
  echo "close judge $m rc=$?: $(grep -E 'rows: [0-9]+, with a judged difference' "$OUT/close-$m-judge.log" | tail -n 1)" >> "$S"
done
echo "EXTRAS-DONE $(date -Is)" >> "$S"
