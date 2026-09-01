#!/usr/bin/env bash
# Assert, from the SERVER LOG, that the program this arm claims to be measuring actually ran --
# and REFUSE the arm otherwise.
#
# Why this is a script and not a grep somebody remembers to run
# (LAW:engagement-receipt-before-any-perf-row): milestone 4 of this lane produced a whole
# pricing rotation -- boot 1 at OFF 106.20, SM 107.08, SM+GU 106.37 decode tok/s -- before anyone
# noticed there was no evidence any door had reached the code. A ~0.8% spread where the bundle was
# priced at +23.7% cannot distinguish "the programs are worth ~1%" from "the doors never ran", and
# the MEMRA_BF16_MMV lane banked the same defect from the other side (its engagement grep returned
# 0 in BOTH arms, which was a missing-line bug read as a no-engagement result). So engagement is
# asserted mechanically, per arm, before the arm's rows are allowed to mean anything.
#
# THE DEFAULT FLIP MAKES THIS SHARPER, not softer. In the flip arms NO door env is set, so the
# environment census -- the instrument milestone 4 trusted -- is byte-identical between `flip-on`
# and a pre-flip `v3-off`. The ONLY evidence that the default armed the program is the announce
# line's `source=`/`door_source=` field. That is why the engine prints the SOURCE and not just the
# boolean, and why this script matches on the source string and not merely on `true`.
#
# Usage: assert-engagement.sh <arm-tag> <mode>
set -u
ARM=${1:?arm tag}
MODE=${2:?mode}
D=/home/ubuntu/bankv3/lane
LOG=$D/logs/server-$ARM.log
R=$D/receipts/boot-$ARM.receipt

[ -f "$LOG" ] || { echo "ENGAGE_FAIL: no server log $LOG"; exit 1; }

# Per-mode REQUIRED announce substrings. Every entry is a full fact including the SOURCE, so a
# line that says the right boolean for the wrong reason still fails.
#
# NOTE ON THE TWO BINARY GENERATIONS: the pre-flip binary prints `source=default` /
# `source=MEMRA_NVFP4_BANK_SM` and has no `door_source=` field at all; the flipped binary prints
# `source=default-on` / `source=env=0 (rollback seam)` and carries `door_source=`. The
# expectations below are therefore ALSO a binary-generation check -- a pre-flip binary cannot
# satisfy a `flip-*` expectation and vice versa, which is the same both-directions discipline
# launch.sh's FLIPMARK applies to the strings in the file.
NEED=()
FORBID=()
case "$MODE" in
  # ---- pre-flip binary, env is the axis ----
  # THE INERTNESS RECEIPT. `door=true` proves the env reached the code; `down8=false` proves the
  # program did NOT run anyway. Both halves are required: `down8=false` alone would also be
  # produced by a door that was never read, which is the opposite finding.
  v3-d8only)
    NEED+=("[nvfp4-bank] layout=block-nvfp4-v1")
    NEED+=("down8=false door=true")
    NEED+=("slot_major=false") ;;
  v3-sm-d8)
    NEED+=("[nvfp4-bank] layout=slot-major source=MEMRA_NVFP4_BANK_SM")
    NEED+=("down8=true door=true")
    NEED+=("slot_major=true") ;;
  v3-off|gate-off)
    NEED+=("[nvfp4-bank] layout=block-nvfp4-v1")
    NEED+=("down8=false door=false") ;;
  # gate-main is the PRE-restore binary: every announce above was introduced with the
  # restore (01ed43c1a), so this arm can never print them — requiring them made the
  # control arm unrunnable (revuto finding on PR #76). Its engagement contract is the
  # inverse: those announce lines must be ABSENT, and launch.sh has already pinned the
  # binary as pre-milestone-3 by the door-name byte test. FORBID is checked below.
  gate-main)
    FORBID+=("[nvfp4-bank]")
    FORBID+=("[nvfp4-sweep]")
    FORBID+=("[nvfp4-door]") ;;
  # ---- flipped binary, the DEFAULT is the axis ----
  flip-on|gflip-on)
    NEED+=("[nvfp4-bank] layout=slot-major source=default-on")
    NEED+=("down8=true door=true door_source=default-on")
    NEED+=("slot_major=true") ;;
  flip-off|gflip-off)
    NEED+=("[nvfp4-bank] layout=block-nvfp4-v1 source=env=0 (rollback seam)")
    NEED+=("down8=false door=false door_source=env=0 (rollback seam)")
    NEED+=("slot_major=false") ;;
  # The surgical seam: the DEFAULT keeps the slot-major layout while =0 rolls back only the
  # down8 program. layout source stays default-on, the door source names the seam, and
  # slot_major stays true -- the exact composition the FLAGS row promises an operator.
  flip-d8off)
    NEED+=("[nvfp4-bank] layout=slot-major source=default-on")
    NEED+=("down8=false door=false door_source=env=0 (rollback seam)")
    NEED+=("slot_major=true") ;;
  *) echo "ENGAGE_FAIL: no engagement expectation for mode $MODE (add one; a mode with no expectation is an unmeasured arm)"; exit 6 ;;
esac

# The gate/up [nvfp4-sel] line DISAPPEARS when the _gu fusion takes over the launch path, which is
# how milestone 4 proved the fusion rather than the flag. GU stays OFF in every arm of this
# battery, so that line must be PRESENT everywhere -- its absence would mean the fusion armed
# itself, which is the auto-arming defect the three-door split exists to prevent.
# gate-main is exempt: the announce itself postdates that binary (see its FORBID contract).
[ "$MODE" = gate-main ] || NEED+=("[nvfp4-sel]")

echo "--- engagement ($MODE) ---" >> "$R"
grep -a -e '\[nvfp4-bank\]' -e '\[nvfp4-sel\]' -e '\[nvfp4-sweep\]' -e '\[nvfp4-door\]' "$LOG" \
  | sort -u >> "$R"

FAIL=0
for n in "${NEED[@]}"; do
  if grep -aqF -- "$n" "$LOG"; then
    echo "ENGAGE_OK   $MODE :: $n" >> "$R"
  else
    echo "ENGAGE_FAIL $MODE :: MISSING $n" | tee -a "$R"
    FAIL=1
  fi
done

for f in "${FORBID[@]:-}"; do
  [ -n "$f" ] || continue
  if grep -aqF -- "$f" "$LOG"; then
    echo "ENGAGE_FAIL $MODE :: FORBIDDEN announce present: $f (this arm must run a binary that predates it)" | tee -a "$R"
    FAIL=1
  else
    echo "ENGAGE_OK   $MODE :: absent as required: $f" >> "$R"
  fi
done

# A door WARN line means an operator typo'd a value. It never invalidates the default (that is the
# point of door_default_on keeping the default), but a battery that silently tolerated one would
# be banking rows against an env nobody meant to set.
if grep -aq '\[nvfp4-door\] WARN' "$LOG"; then
  echo "ENGAGE_FAIL $MODE :: a door value was UNRECOGNIZED (see [nvfp4-door] WARN above)" | tee -a "$R"
  FAIL=1
fi

[ "$FAIL" = 0 ] || { echo "ENGAGEMENT REFUSED for arm=$ARM mode=$MODE — no row from this arm may be aggregated"; exit 7; }
echo "ENGAGEMENT VERIFIED arm=$ARM mode=$MODE (${#NEED[@]} required facts)" | tee -a "$R"
