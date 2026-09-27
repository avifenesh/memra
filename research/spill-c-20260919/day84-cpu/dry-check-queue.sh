#!/usr/bin/env bash
# DAY84: queue v18's control flow and the reader under stubs, no card: stub run-gen-i20 and run-gen-i21 print one
# BOX39 I20C run's log (dispatch clock) plus, when --expert-bank-stages is passed, DAY64b's first I15S run's stage
# lines (the reading means nothing); the idle wait skipped, a scratch lock, no systemd scope.
set -uo pipefail
L=/home/avifenesh/projects/wt-spill-c/research/spill-c-20260919
D=/home/avifenesh/projects/wt-spill-c/target/c84-dry
rm -rf "$D"; mkdir -p "$D/bins"
cut -f2- "$L/pro-single-day82/i20/ev/o1-i20c-r1.log" > "$D/base.log"
grep -E 'stages phase=' "$L/pro-single-day64b/i15b/ev/o1-i15s-r1.log" | cut -f2- > "$D/stages.log"
for b in run-gen-i20 run-gen-i21; do
    printf '#!/usr/bin/env bash\necho "%s $*" >> %s/calls.log\ncat %s/base.log\ncase " $* " in *" --expert-bank-stages "*) cat %s/stages.log ;; esac\nexit 0\n' \
        "$b" "$D" "$D" "$D" > "$D/bins/$b"
    chmod +x "$D/bins/$b"
done
D84_BINS=$D/bins D84_R=$D/r D84_LOCK=$D/lock D84_ART=$D/base.log D84_WRAP=env D84_IDLE=skip \
    bash "$L/rtx5090-queue-v18-20260926.sh" > "$D/queue.out" 2>&1
echo "queue rc=$?"
cat "$D/queue.out"
echo "== calls: $(wc -l < "$D/calls.log") lines; per binary and clocks:"
sed -E 's/ \/.*55 88 13//' "$D/calls.log" | sort | uniq -c
echo "== order (marks):"; awk -F'\t' '/start/ {printf "%s ", $2} END {print ""}' "$D/r/split21/ev/marks.tsv" | sed 's/ start//g'
echo "== check reading"; cut -c1-300 "$D/r/check/reading.log"; echo "== split reading"; grep -E "CHECKS|LARGEST|CHANGE" "$D/r/split21/reading.log" | cut -c1-300
rm -rf "$D"
