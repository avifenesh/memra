#!/usr/bin/env bash
# Rig, the argmax fix (memra#1273): the sm_120a records under gpulock (landing 16's pattern).
#   rec5090.sh battery SHA BASE   old-code GPU tests (own lease, balanced), then the battery+serving
#                                 record (--perf lease)
#   rec5090.sh perf SHA BASE      the perf-only record (--perf lease): 10 min rest, card under 70 C,
#                                 then takes under the owner's heat rule
set -u
export PATH=/usr/bin:$PATH
D=$HOME/.cache/engineagent/argmax/rec
REPO=$HOME/.cache/engineagent/wt-argmax
WORK=$HOME/.cache/engineagent/cells-work/rtx5090
MODE=$1; SHA=$2; BASE=$3; T=${SHA:0:10}
mkdir -p "$D/cells" "$D/logs"
case "$MODE" in
battery)
  # diagnostics on the old code: a short lease of their own on the balanced profile (SKIP_OLD=1: banked)
  [ "${SKIP_OLD:-0}" = 1 ] || gpulock run --ttl 15m --wait 3h --for "memra landing (argmax): old-code GPU tests, round 2 (~2 min, no model)" -- \
    bash -c '
      D=$1; OLD=$HOME/.cache/engineagent/argmax/bins/argmax_no_finite_gpu-old2
      O=$D/old-code-round2; mkdir -p "$O"
      cat /sys/firmware/acpi/platform_profile > "$O/platform_profile.txt"
      for t in sampled_spec_gather_gives_an_id_past_the_row_probability_zero residual_sample_of_a_row_without_a_finite_value_is_token_none; do
        timeout 120 "$OLD" --exact "$t" --test-threads=1 --nocapture > "$O/old-$t.log" 2>&1
        echo "old  $t rc=$?" | tee -a "$O/summary.txt"
      done
    ' _ "$D" > "$D/logs/old-code-$T.out" 2>&1
  echo "rc=$?" >> "$D/logs/old-code-$T.out"
  # the record: --perf (performance profile) for the lease's whole run
  gpulock run --ttl 60m --wait 3h --perf --for "memra landing (argmax): sm_120a battery+serving record of $T (~25 min)" -- \
    bash -c '
      set -u
      D=$1; REPO=$2; WORK=$3; SHA=$4; BASE=$5; T=${SHA:0:10}
      bash "$D/platform_snap.sh" > "$D/logs/platform-battery-before-$T.txt" 2>&1
      cd "$REPO" || exit 1
      python3 tools/cell_evidence.py run --device rtx5090 --commit "$SHA" --kinds battery,serving \
        --base "$BASE" --out "$D/cells/rtx5090-$T-battery-serving" --work "$WORK" --no-lock
      rc=$?
      echo "CELLS-5090-BATTERY rc=$rc"
      bash "$D/platform_snap.sh" > "$D/logs/platform-battery-after-$T.txt" 2>&1
      exit $rc
    ' _ "$D" "$REPO" "$WORK" "$SHA" "$BASE" > "$D/logs/battery-$T.out" 2>&1
  echo "rc=$?" >> "$D/logs/battery-$T.out"
  ;;
perf)
  # A timed cell under the owner's heat rule (2026-11-13): a take is bankable when the card starts below
  # 86 C and no arm has sw thermal slowdown on more than 5% of its 1 Hz samples. Otherwise a 5 min break
  # with the lease held, the card back under 70 C, and a retake: up to 3 retries, then the best take is
  # banked with its throttle share in the conditions. The arms alternate inside a take (A B B A A B).
  # Rig load is logged (vmstat), not gated.
  gpulock run --ttl 100m --wait 3h --perf --for "memra landing (argmax): sm_120a perf-only record of $T (10 min rest, card under 70 C, timed cell, heat retries)" -- \
    bash -c '
      set -u
      D=$1; REPO=$2; WORK=$3; SHA=$4; BASE=$5; T=${SHA:0:10}
      L=$D/logs
      bash "$D/platform_snap.sh" > "$L/platform-before-$T.txt" 2>&1
      /usr/bin/python3 "$D/ac_log.py" > "$L/ac-$T.tsv" 2>&1 &
      A=$!
      cd "$REPO" || exit 1
      exec 9> "$HOME/.cache/memra-rig-build.lock"
      flock -w 1200 9 || { echo "build lock not obtained in 20 min"; kill "$A"; exit 1; }
      REST_UNTIL=$(( $(date +%s) + 600 ))
      waitfor --quiet --timeout 30m --every 10s --until "bash $D/quiet_ok.sh $REST_UNTIL"
      rc=$?
      echo "rest/cool wait rc=$rc: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C, load $(cut -d" " -f1-4 /proc/loadavg), $(date -Is)" > "$L/cool-$T.txt"
      [ $rc = 0 ] || { echo "card not under 70 C in 30 min after the rest; nothing ran"; kill "$A"; exit 1; }
      Q=timestamp,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_power_cap,temperature.gpu,fan.speed,power.draw,utilization.gpu,clocks.sm
      chosen=""
      for take in 1 2 3 4; do
        O=$D/cells/rtx5090-$T-perf-take$take
        temp0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
        nvidia-smi --query-gpu=$Q --format=csv > "$L/throttle-before-$T-take$take.csv"
        nvidia-smi --query-gpu=$Q --format=csv -l 1 > "$L/throttle-1hz-$T-take$take.csv" 2>&1 9>&- &
        S=$!
        vmstat -t 5 > "$L/load-$T-take$take.log" 2>&1 9>&- &
        V=$!
        start=$(date -Is); echo "$start" > "$L/perf-start-$T-take$take.txt"
        PATH=$D/shim:$PATH python3 tools/cell_evidence.py run --device rtx5090 --commit "$SHA" --kinds perf \
          --base "$BASE" --out "$O" --work "$WORK" --no-lock
        crc=$?
        kill "$S" "$V" 2> /dev/null
        nvidia-smi --query-gpu=$Q --format=csv > "$L/throttle-after-$T-take$take.csv"
        echo "take $take: cell rc=$crc, start ${temp0} C"
        python3 "$D/arm_conditions.py" "$L/throttle-1hz-$T-take$take.csv" "$O/perf" "$start" "$L/arms-$T-take$take.tsv" > /dev/null 2>&1
        python3 "$D/throttle_gate.py" "$L/throttle-1hz-$T-take$take.csv" "$O/perf" "$start" "$temp0" "$L/gate-$T-take$take.json" "$L/ac-$T.tsv" performance
        grc=$?
        if [ $crc = 0 ] && [ $grc = 0 ]; then chosen=$take; echo "take $take bankable"; break; fi
        # off AC or off the asked profile: a power event, not heat; the profile stays wrong for the rest
        # of the lease (the mains-return hook sets balanced), so stop and report instead of retrying
        if python3 -c "import json,sys; g=json.load(open(sys.argv[1])); sys.exit(0 if (g.get(\"power\") or {}).get(\"off_ac_or_profile_seconds\") else 1)" "$L/gate-$T-take$take.json"; then
          echo "take $take void: off AC or off profile inside the window; stopping, report it"; break
        fi
        [ $take = 4 ] && break
        echo "take $take not bankable: 5 min break with the lease held, then the card under 70 C"
        UNTIL=$(( $(date +%s) + 300 ))
        waitfor --quiet --timeout 25m --every 10s --until "bash $D/cool_ok.sh $UNTIL 70"
      done
      if [ -z "$chosen" ]; then
        chosen=$(python3 "$D/best_take.py" "$L" "$T")
        echo "no bankable take in 4: best is take ${chosen:-none} (its throttle share is in its gate json); tell Avi"
        echo "${chosen:-none}" > "$L/perf-best-not-clean-$T.txt"
      fi
      echo "${chosen:-none}" > "$L/perf-chosen-$T.txt"
      kill "$A" 2> /dev/null
      bash "$D/platform_snap.sh" > "$L/platform-after-$T.txt" 2>&1
      echo "CELLS-5090-PERF chosen take ${chosen:-none}"
      [ -n "$chosen" ]
    ' _ "$D" "$REPO" "$WORK" "$SHA" "$BASE" > "$D/logs/perf-$T.out" 2>&1
  echo "rc=$?" >> "$D/logs/perf-$T.out"
  ;;
*) echo "usage: rec5090.sh battery|perf SHA BASE"; exit 2 ;;
esac
