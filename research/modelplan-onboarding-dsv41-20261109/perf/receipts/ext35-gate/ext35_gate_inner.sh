#!/bin/bash
# ext35 order gate, the engine side, inside ONE card's memra-gpu-run lease (ext35_gate.sh starts it): each job
# BUILD:GROUP runs perf-bins/BUILD teacher-forced on GROUP's vendor tapes (dsv41-run --cases, as ordertapes-k1 and
# ordertapes-cutlass ran them: MEMRA_FULL_PREC=1, no route log; the gate reads top-32, lse and argmax).
#   GROUP ext35            dec/ext35/ext35-forced.txt (35 cases, 3011 steps)
#   GROUP chat|long|long257 dec/GROUP-forced.txt (the decode lane's 18 tapes: 13 + 4 + 1)
# -> receipts-perf/ext35-BUILD/GROUP/ID/{top_ids.i32,top_vals.f32,lse.f64,argmax.i32,meta.json}, GROUP.log,
#    GROUP.sha256 (the four gate files of every case), runs.txt; one line a job in receipts-perf/ext35-gate/jobs-TAG.txt.
# Memory guard of ordertapes-k1: not started under 400 GB available, stopped under 150 GB. A job starts only when
# receipts-perf/ext35-gate/STOP is absent.
# usage (inside the lease): JOBS=k3:ext35,cfix:ext35,... TAG=... bash ext35_gate_inner.sh
set -u
D=/home/access/dsv41
G=$D/receipts-perf/ext35-gate
C=$D/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
J=$G/jobs-${TAG:-x}.txt
export MEMRA_FULL_PREC=1
avail() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
for job in $(echo "$JOBS" | tr , ' '); do
  b=${job%%:*}; g=${job##*:}
  if [ -e $G/STOP ]; then echo "$(date +%T) $job not started: STOP" >> $J; break; fi
  case $g in ext35) L=$D/dec/ext35/ext35-forced.txt ;; chat|long|long257) L=$D/dec/$g-forced.txt ;; *) echo "$(date +%T) $job: unknown group" >> $J; continue ;; esac
  B=$D/perf-bins/$b
  [ -x $B/dsv41-run ] || { echo "$(date +%T) $job: no $B/dsv41-run" >> $J; continue; }
  if [ "$(avail)" -lt 400 ]; then echo "$(date +%T) $job not started: $(avail) GB available" >> $J; break; fi
  R=$D/receipts-perf/ext35-$b
  mkdir -p $R
  sha256sum $B/dsv41-run > $R/bins.sha256
  rm -rf $R/$g
  t0=$(date +%s)
  $B/dsv41-run $C --token-map $TM --cases $L --out $R/$g > $R/$g.log 2>&1 &
  P=$!
  ( while kill -0 $P 2> /dev/null; do
      a=$(avail)
      if [ "$a" -lt 150 ]; then echo "STOPPED: $a GB available" >> $R/$g.log; kill $P; break; fi
      /bin/sleep 2
    done ) &
  W=$!
  wait $P; rc=$?
  # the guard ends by itself within 2 s of the run (killing it would leave its sleep in the lease's group)
  wait $W 2> /dev/null
  n=0
  for c in $(cut -d' ' -f1 $L); do
    for f in top_ids.i32 top_vals.f32 lse.f64 argmax.i32; do [ -s $R/$g/$c/$f ] && n=$((n + 1)); done
  done
  (cd $R/$g && sha256sum */top_ids.i32 */top_vals.f32 */lse.f64 */argmax.i32 2> /dev/null | sort -k2) > $R/$g.sha256
  line="$job rc $rc $(( $(date +%s) - t0 )) s, $n of $(( $(wc -l < $L) * 4 )) files, list sha $(cut -c1-64 $R/$g.sha256 | sha256sum | cut -c1-16); $(grep -h 'dsv41-run: [0-9]* forwards' $R/$g.log | tail -n 1)"
  echo "$line" >> $R/runs.txt
  echo "$(date +%T) $line" >> $J
done
echo "$(date +%T) INNER-END" >> $J
