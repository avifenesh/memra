#!/usr/bin/env bash
# DAY89: queue v21's control flow and readers under stubs, no card: stub run-gen-p88, run-gen-i23 and run-gen-i24 print
# one split22 I22 run's log (both clocks), without its stage lines when --expert-bank-stages is absent (the reading
# means nothing); the idle wait skipped, a scratch lock in the scratch dir, no systemd scope.
set -uo pipefail
L=/home/avifenesh/projects/wt-spill-c/research/spill-c-20260919
D=/home/avifenesh/projects/wt-spill-c/target/c89-dry
rm -rf "$D"; mkdir -p "$D/bins"
cut -f2- "$L/rtx5090-day85/split22/ev/o1-i22s-r1.log" > "$D/both.log"
grep -v 'stages phase=' "$D/both.log" > "$D/base.log"
for b in run-gen-p88 run-gen-i23 run-gen-i24; do
    printf '#!/usr/bin/env bash\necho "%s $*" >> %s/calls.log\ncase " $* " in *" --expert-bank-stages "*) cat %s/both.log ;; *) cat %s/base.log ;; esac\nexit 0\n' \
        "$b" "$D" "$D" "$D" > "$D/bins/$b"
    chmod +x "$D/bins/$b"
done
D89_BINS=$D/bins D89_R=$D/r D89_LOCK=$D/lock D89_ART=$D/base.log D89_WRAP=env D89_IDLE=skip \
    bash "$L/rtx5090-queue-v21-20260927.sh" > "$D/queue.out" 2>&1
echo "queue rc=$?"
cat "$D/queue.out"
echo "== calls: $(wc -l < "$D/calls.log") lines; per binary and clocks:"
sed -E 's/ \/.*55 88 13//' "$D/calls.log" | sort | uniq -c
echo "== order (marks):"; awk -F'\t' '/start/ {printf "%s ", $2} END {print ""}' "$D/r/split24/ev/marks.tsv" | sed 's/ start//g'
echo "== check reading"; cut -c1-200 "$D/r/check/reading.log"; echo "== split reading"; grep -E "CHECKS|CHANGE|SIZING|BESIDE" "$D/r/split24/reading.log" | cut -c1-220
rm -rf "$D"
