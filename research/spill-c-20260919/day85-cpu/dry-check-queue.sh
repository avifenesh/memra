#!/usr/bin/env bash
# DAY85: queue v19's control flow and the reader under stubs, no card: stub run-gen-i21 and run-gen-i22 print one
# BOX39 I20C run's log (dispatch clock) plus, when --expert-bank-stages is passed, DAY64b's first I15S run's stage
# lines (the reading means nothing); the idle wait skipped, a scratch lock, no systemd scope.
set -uo pipefail
L=/home/avifenesh/projects/wt-spill-c/research/spill-c-20260919
D=/home/avifenesh/projects/wt-spill-c/target/c85-dry
rm -rf "$D"; mkdir -p "$D/bins"
cut -f2- "$L/pro-single-day82/i20/ev/o1-i20c-r1.log" > "$D/base.log"
grep -E 'stages phase=' "$L/pro-single-day64b/i15b/ev/o1-i15s-r1.log" | cut -f2- > "$D/stages.log"
for b in run-gen-i21 run-gen-i22; do
    printf '#!/usr/bin/env bash\necho "%s $*" >> %s/calls.log\ncat %s/base.log\ncase " $* " in *" --expert-bank-stages "*) cat %s/stages.log ;; esac\nexit 0\n' \
        "$b" "$D" "$D" "$D" > "$D/bins/$b"
    chmod +x "$D/bins/$b"
done
D85_BINS=$D/bins D85_R=$D/r D85_LOCK=$D/lock D85_ART=$D/base.log D85_WRAP=env D85_IDLE=skip \
    bash "$L/rtx5090-queue-v19-20260926.sh" > "$D/queue.out" 2>&1
echo "queue rc=$?"
cat "$D/queue.out"
echo "== calls: $(wc -l < "$D/calls.log") lines; per binary and clocks:"
sed -E 's/ \/.*55 88 13//' "$D/calls.log" | sort | uniq -c
echo "== order (marks):"; awk -F'\t' '/start/ {printf "%s ", $2} END {print ""}' "$D/r/split22/ev/marks.tsv" | sed 's/ start//g'
echo "== check reading"; cut -c1-300 "$D/r/check/reading.log"; echo "== split reading"; grep -E "CHECKS|LARGEST|CHANGE" "$D/r/split22/reading.log" | cut -c1-300
rm -rf "$D"
