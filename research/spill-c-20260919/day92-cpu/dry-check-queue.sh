#!/usr/bin/env bash
# DAY92: queue v22's control flow and readers under stubs, no card: stub run-gen-p88, run-gen-i24 and run-gen-i25
# print one split22 I22 run's log (both clocks), without its stage lines when --expert-bank-stages is absent, and
# run-gen-i25 without its trace lines unless --expert-bank-trace is passed (the reading means nothing); the idle wait
# skipped, a scratch lock in the scratch dir, no systemd scope.
set -uo pipefail
L=/home/avifenesh/projects/wt-spill-c/research/spill-c-20260919
D=/home/avifenesh/projects/wt-spill-c/target/c92-dry
rm -rf "$D"; mkdir -p "$D/bins"
cut -f2- "$L/rtx5090-day85/split22/ev/o1-i22s-r1.log" > "$D/both.log"
for b in run-gen-p88 run-gen-i24 run-gen-i25; do
    untraced=0; [ "$b" = run-gen-i25 ] && untraced=1
    printf '#!/usr/bin/env bash\necho "%s $*" >> %s/calls.log\nf=%s/both.log\ncase " $* " in *" --expert-bank-stages "*) cat "$f" ;; *) grep -v "stages phase=" "$f" ;; esac | { if [ %s = 1 ] && [[ " $* " != *" --expert-bank-trace "* ]]; then grep -v "expert-host-slru\\] key="; else cat; fi; }\nexit 0\n' \
        "$b" "$D" "$D" "$untraced" > "$D/bins/$b"
    chmod +x "$D/bins/$b"
done
D92_BINS=$D/bins D92_R=$D/r D92_LOCK=$D/lock D92_ART=$D/both.log D92_WRAP=env D92_IDLE=skip \
    bash "$L/rtx5090-queue-v22-20260927.sh" > "$D/queue.out" 2>&1
echo "queue rc=$?"
cat "$D/queue.out"
echo "== calls: $(wc -l < "$D/calls.log") lines; per binary and flags:"
sed -E 's/ \/.*55 88 13//' "$D/calls.log" | sort | uniq -c
echo "== order (marks):"; awk -F'\t' '/start/ {printf "%s ", $2} END {print ""}' "$D/r/split25/ev/marks.tsv" | sed 's/ start//g'
echo "== check reading"; cut -c1-200 "$D/r/check/reading.log"; echo "== split reading"; grep -E "CHECKS|CHANGE|SIZING" "$D/r/split25/reading.log" | cut -c1-220
rm -rf "$D"
