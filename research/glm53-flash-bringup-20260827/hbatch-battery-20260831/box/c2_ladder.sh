#!/usr/bin/env bash
# CELL 2 — THE LADDER (TIMED, marker up): c in {1,2,4,8,12}, hyper-batch OFF vs ON,
# interleaved x3 per arm (owner protocol 2026-08-30: x3 default, escalate to x5 on
# (a) within-arm aggregate rel spread >0.5% at any rung or (b) arms within 2x pooled spread).
# c=1 = the flip-battery timed shape (decode-pool median MUST reproduce the 35.4 baseline).
# Round 1 adds the vendor-default sampled ladder twin; round 3 adds the 8-turn cache twin.
set -uo pipefail
OUT=/root/out-hbatch/c2
RP="python3 /root/out-hbatch/run_pool.py"
mkdir -p "$OUT"

boot_ladder() {  # round, arm, extras...
  local r="$1" arm="$2"; shift 2
  local d="$OUT/l$r-$arm"
  mkdir -p "$d"
  echo "######## LADDER round=$r arm=$arm ########"
  /root/out-hbatch/serve.sh start "l$r-$arm" "$@" || { echo "LADDER_${r}_${arm}=BOOTFAIL"; return 1; }
  $RP sample --out "$d" || return 1
  echo "--- c=1 (flip-battery timed shape) ---"
  $RP timed --out "$d/c1"
  for c in 2 4 8 12; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$d/vram-before-c$c.txt"
    echo "--- c=$c greedy ---"
    $RP conc --n "$c" --out "$d/c$c" || echo "WARN: conc c=$c had errors (see json)"
  done
  if [ "$r" = "1" ]; then
    for c in 2 4 8 12; do
      echo "--- c=$c vendor-default twin ---"
      $RP conc --n "$c" --mode vendor --out "$d/c$c-vendor" || echo "WARN: vendor c=$c had errors"
    done
  fi
  if [ "$r" = "3" ]; then
    echo "--- 8-turn cache twin (vendor mode; glm5 prefix cache is receipted dead: cached_tokens must read 0) ---"
    $RP twin --out "$d/twin"
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$d/vram-after.txt"
  local log=/root/out-hbatch/logs/boot-l$r-$arm.log
  {
    echo "batched_on_lines=$(grep -c 'BATCHED DECODE (mHC hyper arm' "$log")"
    echo "eager_only_lines=$(grep -c 'EAGER-ONLY serving' "$log")"
    echo "admit_oom_lines=$(grep -c '\[admit-oom\]' "$log")"
    echo "overloaded_lines=$(grep -c 'Overloaded' "$log")"
    echo "panic_lines=$(grep -cE 'panicked|FATAL' "$log")"
  } > "$d/log-receipts.txt"
  cat "$d/log-receipts.txt"
  echo "LADDER_${r}_${arm}=DONE"
}

date -u +%Y-%m-%dT%H:%M:%SZ > /root/TIMING-IN-FLIGHT
echo "hbatch-battery cell 2 ladder (owner: hbatch-battery agent)" >> /root/TIMING-IN-FLIGHT

rc=0
for r in 1 2 3; do
  boot_ladder "$r" off || rc=1
  boot_ladder "$r" on MEMRA_HYPER_BATCH=1 || rc=1
done

/root/out-hbatch/serve.sh stop
rm -f /root/TIMING-IN-FLIGHT
echo "=== LOOP-LAW SCREEN (all cell-2 tapes) ==="
python3 /root/out-hbatch/looplaw_screen.py "$OUT"/l*-off "$OUT"/l*-on
echo "=== LADDER TABLE + ESCALATION CHECK ==="
python3 /root/out-hbatch/ladder_table.py "$OUT"
echo "C2_EXIT=$rc"
exit "$rc"
