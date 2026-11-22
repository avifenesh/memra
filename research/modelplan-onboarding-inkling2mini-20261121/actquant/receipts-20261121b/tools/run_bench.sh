#!/bin/bash
# timed cells for #1396 under a --perf gpulock lease: clamp check, 250 ms telemetry, bw at the bench's
# sizes before and after, the bench arms interleaved. Usage: run_bench.sh OUTDIR GATEBIN_A GATEBIN_B
set -u
O=$1; GA=$2; GB=$3; mkdir -p $O
C=/home/avifenesh/.cache/inkling2mini/ceiling
{ date -Is; powerprofilesctl get; cat /sys/firmware/acpi/platform_profile; nvidia-smi --query-gpu=name,driver_version,power.limit,enforced.power.limit,clocks.max.sm,clocks.max.mem,temperature.gpu --format=csv; cat /proc/loadavg;
  for f in /sys/class/power_supply/A*/online; do echo "$f $(cat $f)"; done; } > $O/conditions.txt 2>&1
bash $C/clampcheck.sh > $O/clampcheck.txt 2>&1; rc=$?
echo "clampcheck rc=$rc" >> $O/conditions.txt
if [ $rc -eq 1 ]; then
  sudo -n systemctl restart nvidia-powerd >> $O/conditions.txt 2>&1
  waitfor --until 'bash /home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh' --timeout 30s >> $O/clampcheck.txt 2>&1
  bash $C/clampcheck.sh >> $O/clampcheck.txt 2>&1; rc=$?
  echo "after powerd restart clampcheck rc=$rc" >> $O/conditions.txt
  if [ $rc -eq 1 ]; then echo "STILL CLAMPED, not timing" >> $O/conditions.txt; exit 3; fi
fi
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu,clocks_event_reasons.active,pstate,enforced.power.limit --format=csv,nounits -lms 250 > $O/smi.csv &
SMI=$!
echo "#MARK $(date +%s.%N | cut -c1-14) bw-pre start" > $O/bw-pre.txt
/home/avifenesh/.cache/inkling2mini/actq-fix/bench/bwslice 23887872 47775744 95551488 191102976 382205952 11943936 23887872 47775744 95551488 191102976 43794432 87588864 175177728 350355456 700710912 21897216 43794432 87588864 175177728 350355456 >> $O/bw-pre.txt 2>&1
echo "#MARK $(date +%s.%N | cut -c1-14) bw-pre end" >> $O/bw-pre.txt
for arm in old pad swz nosw nosw swz pad old; do
  echo "#MARK $(date +%s.%N | cut -c1-14) quant-$arm start" >> $O/quant-ab.txt
  /home/avifenesh/.cache/inkling2mini/actq-fix/qbench/qb_$arm >> $O/quant-ab.txt 2>&1
  echo "#MARK $(date +%s.%N | cut -c1-14) quant-$arm end" >> $O/quant-ab.txt
done
i=0
for G in $GA $GB $GB $GA; do
  i=$((i+1)); echo "# arm $G" > $O/bench-$i.txt
  $G --bench 128 200 >> $O/bench-$i.txt 2>&1; echo "bench $i rc=$?" >> $O/conditions.txt
done
echo "#MARK $(date +%s.%N | cut -c1-14) bw-post start" > $O/bw-post.txt
/home/avifenesh/.cache/inkling2mini/actq-fix/bench/bwslice 23887872 47775744 95551488 191102976 382205952 11943936 23887872 47775744 95551488 191102976 43794432 87588864 175177728 350355456 700710912 21897216 43794432 87588864 175177728 350355456 >> $O/bw-post.txt 2>&1
echo "#MARK $(date +%s.%N | cut -c1-14) bw-post end" >> $O/bw-post.txt
kill $SMI
{ date -Is; cat /proc/loadavg; for f in /sys/class/power_supply/A*/online; do echo "$f $(cat $f)"; done; journalctl --since -30min 2>/dev/null | grep -c 'set-epp power'; } >> $O/conditions.txt
