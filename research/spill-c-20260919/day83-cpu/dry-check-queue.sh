#!/usr/bin/env bash
# DAY83: queue v17's control flow and the reader under stubs, no card: stub run-gen-i15 and run-gen-i20 print one
# BOX39 I20C run's log (dispatch clock) plus, when --expert-bank-stages is passed, DAY64b's first I15S run's stage
# lines (the reading means nothing); the idle wait skipped, a scratch lock, no systemd scope.
set -uo pipefail
L=/home/avifenesh/projects/wt-spill-c/research/spill-c-20260919
D=/home/avifenesh/projects/wt-spill-c/target/c83-dry
rm -rf "$D"; mkdir -p "$D/bins"
cut -f2- "$L/pro-single-day82/i20/ev/o1-i20c-r1.log" > "$D/base.log"
grep -E 'stages phase=' "$L/pro-single-day64b/i15b/ev/o1-i15s-r1.log" | cut -f2- > "$D/stages.log"
for b in run-gen-i15 run-gen-i20; do
    printf '#!/usr/bin/env bash\necho "%s $*" >> %s/calls.log\ncat %s/base.log\ncase " $* " in *" --expert-bank-stages "*) cat %s/stages.log ;; esac\nexit 0\n' \
        "$b" "$D" "$D" "$D" > "$D/bins/$b"
    chmod +x "$D/bins/$b"
done
D83_BINS=$D/bins D83_R=$D/r D83_LOCK=$D/lock D83_ART=$D/base.log D83_WRAP=env D83_IDLE=skip \
    bash "$L/rtx5090-queue-v17-20260926.sh" > "$D/queue.out" 2>&1
echo "queue rc=$?"
cat "$D/queue.out"
echo "== calls: $(wc -l < "$D/calls.log") lines; per binary and clocks:"
sed -E 's/ \/.*55 88 13//' "$D/calls.log" | sort | uniq -c
echo "== order (marks):"; awk -F'\t' '/start/ {printf "%s ", $2} END {print ""}' "$D/r/split20/ev/marks.tsv" | sed 's/ start//g'
echo "== reading"; cut -c1-400 "$D/r/split20/reading.log"
rm -rf "$D"
