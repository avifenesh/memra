#!/usr/bin/env bash
# moeu lane cell: the MoE routed-union PAYOFF CURVE on the box card, timed.
#
# WHAT IT MEASURES. moe_union_probe runs the SHIPPED sel kernels
# (qmatvec_nvfp4_modelopt_sel_gu_silu_f32 + qmatvec_nvfp4_modelopt_sel_f32_v3) at a FIXED
# 60-slot verify-chunk shape while sweeping only the number of DISTINCT experts those slots
# name. That is the one quantity a union-of-experts gather changes, so the `union=U` row IS
# the idealised union gather's cost -- priced before any kernel exists
# (LAW:price-the-dispatch-first).
#
# WHY IT IS A ~30 SECOND CELL. No checkpoint: synthetic banks of the serving geometry,
# ~1.3 GiB device. It therefore interleaves between the 262k queue's multi-hour cells
# instead of waiting for the whole queue to drain.
#
# LOCK AND CAPACITY DISCIPLINE (the vfuse lesson, stated in its VFUSE.md): the capacity
# guard sits INSIDE the lock hold. Checking the cards and THEN blocking on flock is a race
# -- the sibling queue refills card 0 while this cell waits, and the load then OOMs a
# minute into an exclusive hold. Acquiring the lock first and waiting for the cards second
# also covers the VRAM release lag (nvidia-smi free is not driver free). This cell also
# waits for card 0 to be IDLE, not merely free, because it quotes microsecond kernel times.
#
# Never kills, never reorders: it parks on flock -x behind whatever holds the lock and
# releases in seconds.
set -u
BIN=$HOME/realgate/bin/moe_union_probe.moeu
OUT=$HOME/realgate/moeu
LK=/tmp/q48fn-measure.lock
Q=$OUT/QUEUE.log
mkdir -p "$OUT"
say(){ echo "[$(date -u +%FT%TZ)] moeu: $*" >> "$Q"; }

[ -x "$BIN" ] || { say "ABORT: $BIN missing or not executable"; exit 1; }

# Rebuild-attribution law: name the binary's identity in the receipt, from the binary.
SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
SRC=$(cd "$HOME/moeu-wt" && git log -1 --format=%H)
say "cell start bin_sha256=$SHA src=$SRC"

# run_cell <label> <t> <reps>. The probe reads its shape from the environment, so the two
# knobs are passed through `env` EXPLICITLY rather than as a `VAR=x run_cell` prefix: a
# function-call assignment surviving into a nested `flock ... bash -c` is exactly the kind
# of implicit plumbing that silently measures the default shape instead of the asked one.
run_cell() {
  local label="$1" mt="$2" mreps="$3"
  local log="$OUT/$label.tsv"
  say "cell $label acquiring -x (t=$mt reps=$mreps)"
  # -x around the WHOLE invocation. The capacity+idle wait is INSIDE the hold.
  flock -x "$LK" env MEMRA_MOEU_T="$mt" MEMRA_MOEU_REPS="$mreps" bash -c '
    label="$1"; log="$2"; bin="$3"; q="$4"; shift 4
    say(){ echo "[$(date -u +%FT%TZ)] moeu: $*" >> "$q"; }
    waited=0
    while :; do
      u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d " ")
      g0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d " ")
      if [ -n "$u0" ] && [ "$u0" -lt 2000 ] && [ -n "$g0" ] && [ "$g0" -lt 5 ]; then
        [ "$waited" -gt 0 ] && say "card 0 free+idle after ${waited}s (in-hold), starting $label"
        break
      fi
      # 20 min ceiling: if the card never goes idle inside the hold, release rather than
      # sit on the lock the sibling queue needs.
      if [ "$waited" -ge 1200 ]; then
        say "cell $label ABANDONED: card 0 not free+idle within 1200s in-hold (used=${u0:-?} util=${g0:-?})"
        exit 75
      fi
      sleep 15; waited=$((waited+15))
    done
    "$bin" > "$log" 2>&1
    rc=$?
    say "cell $label rc=$rc rows=$(grep -c "^[0-9]" "$log" 2>/dev/null) t=$MEMRA_MOEU_T"
    exit $rc
  ' _ "$label" "$log" "$BIN" "$Q"
  say "cell $label released -x (rc=$?)"
}

# Interleaved x3 per LAW:interleaved-ab -- the sweep is one process, so the arms
# (union sizes) are already interleaved within a single residency by construction; the
# three passes bound the per-boot spread and are reported separately, never averaged
# across passes without their spread.
for pass in 1 2 3; do
  run_cell "moeu-t6-pass$pass" 6 25
done

# Larger verify chunk, to check the payoff curve is not a t=6-only artifact.
run_cell "moeu-t9-pass1" 9 25

say "cell done"
