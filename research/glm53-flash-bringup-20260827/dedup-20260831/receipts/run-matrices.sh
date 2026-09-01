#!/usr/bin/env bash
# lane/glm5-dedup split-gate matrices (rig 5090, exactness only — rig law).
#
# The §6 row this pays: "split matrices (glm5-spec-ppn-gate, glm5-hyper-ppn-gate,
# glm5-hyper-batch-gate) + a dedup compose arm — DEFERRED per owner order 2026-09-01".
# Run 2026-08-31 by the end-of-day debt lane after the owner released the rig.
#
# Arm shapes copied verbatim from ../../moe-loc-20260831/receipts/run-matrices.sh so the receipts
# are comparable arm-for-arm with the door-D/H lane that this lane's schedule composes onto:
#   glm5-spec-ppn-gate    stages P N : 2/24/20 (even, split1, split3, streams0, overlap0)
#                                      3/24/20 (even, asym 1,3, streams0)
#   glm5-hyper-ppn-gate   stages P N : 2/6/8 (even, split1, split3, streams0, overlap0, shard0),
#                                      3/6/8 asym, 4/6/8 (even, streams0), 2/16/24 longer
#   glm5-hyper-batch-gate   B P N ppn: the banked 10-arm ladder (B <= 15; the engine's own
#                                      hyper_batch_cap() = PRIME_MIN_T - 1 = 15)
#
# THREE compose classes, each pinned, never merely unset (the ep-place lesson: doors T/X/K passed
# vacuously because their reference arms were unset-shaped):
#   E    = this lane's two doors ON alone
#   E+DH = this lane's two doors ON composed with door D + door H, the serving shape the box
#          will price, since D is a live 1.0154x win
#   base = every non-compose arm above, with this lane's two doors and door M all pinned =0
#
# WHAT ENGAGEMENT TO EXPECT, stated up front so a silent log is not read as a pass:
# MEMRA_MOE_VROWS_* engages ONLY in the SPEC verify walk (moe-loc LANE.md), so the doors are
# STRUCTURALLY SILENT in glm5-hyper-ppn-gate and glm5-hyper-batch-gate. Their compose arms are
# therefore no-perturbation arms, not engagement arms; the engagement receipt lives in the
# spec-ppn compose arms and in the standing battery's compose phases (run-battery.sh phases 2-4).
#
# PASS-LINE COUNTS ARE ASSERTED, not merely printed. The moe-loc runner greps `gate PASS` and
# echoes the count; a count of 0 with exit 0 would have read as green. Per-gate expected counts:
# spec-ppn 23, hyper-ppn 6, hyper-batch 3.
set -u
cd "$(dirname "$0")/../../../.."
BASE=research/glm53-flash-bringup-20260827/dedup-20260831/receipts
fails=0
STARTED=""

# Pre-warm sccache OUTSIDE every flock: a `cargo run` under the lock spawns an sccache daemon that
# inherits the lock fd, and the exclusive lock then outlives the flock'd command (~80 min of rig
# deadlock measured by the extract-general lane, /proc/locks naming a dead pid). The bins are also
# pre-built by the caller for the same reason.
sccache --start-server >/dev/null 2>&1 || true

run() { # run <outdir> <log> <bin> <expected-pass-lines> <env-or-"-"> <args...>
  local dir="$1" log="$2" bin="$3" want="$4" envs="$5"; shift 5
  mkdir -p "$BASE/$dir"
  # TRUNCATE the summary on the first arm of a matrix (moe-loc: a `tee -a` across re-runs left a
  # stale FAIL line in a banked receipt, reading exactly like a live failure).
  case " $STARTED " in *" $dir "*) ;; *) : >"$BASE/$dir/matrix.out"; STARTED="$STARTED $dir";; esac
  echo "########## $log :: ${envs} $* ##########" | tee -a "$BASE/$dir/matrix.out"
  # shellcheck disable=SC2086
  local envlist=""
  [ "$envs" != "-" ] && envlist="$envs"
  # CAPTURE-THEN-GATE: redirect to a file, take rc, then judge. No pipe on the failable step.
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 $envlist \
    timeout 3600 nice -n 5 ./target/debug/"$bin" "$@" \
    >"$BASE/$dir/$log" 2>&1
  local rc=$?
  echo "exit=$rc" >>"$BASE/$dir/$log"
  local got
  got="$(grep -cE 'gate PASS' "$BASE/$dir/$log")"
  echo "pass_lines=$got want=$want" >>"$BASE/$dir/$log"
  if [ "$rc" -ne 0 ]; then
    fails=$((fails+1)); echo "FAIL: $dir/$log (exit=$rc, pass_lines=$got)" | tee -a "$BASE/$dir/matrix.out"
  elif [ "$got" -ne "$want" ]; then
    fails=$((fails+1))
    echo "FAIL: $dir/$log WRONG PASS-LINE COUNT (exit=0, pass_lines=$got, want=$want)" | tee -a "$BASE/$dir/matrix.out"
  else
    echo "PASS: $dir/$log pass_lines=$got" | tee -a "$BASE/$dir/matrix.out"
  fi
  tail -3 "$BASE/$dir/$log" >>"$BASE/$dir/matrix.out"
}

# Doors pinned =0 on every non-compose arm — the SHIP arm for this lane.
OFF="MEMRA_MOE_VROWS_DEDUP_ORDER=0 MEMRA_MOE_VROWS_DOWN_TMAJ=0 MEMRA_MOE_VROWS_PACK=0"
# This lane's two doors alone.
E="MEMRA_MOE_VROWS_DEDUP_ORDER=1 MEMRA_MOE_VROWS_DOWN_TMAJ=1 MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=0"
# Composed with door D (device-built tables) + door H (HtoD diet) — the box's serving shape.
EDH="MEMRA_MOE_VROWS_DEDUP_ORDER=1 MEMRA_MOE_VROWS_DOWN_TMAJ=1 MEMRA_MOE_VROWS_PACK=0 MEMRA_MOE_VROWS_DEV_TABLES=1 MEMRA_GLM5_HTOD_DIET=1"

# ---- glm5-spec-ppn-gate (23 PASS lines/arm) ----
run ppn-gate 10-n2-even.log     glm5-spec-ppn-gate 23 "$OFF"                       2 24 20
run ppn-gate 11-n2-split1.log   glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_SPLITS=1"     2 24 20
run ppn-gate 12-n2-split3.log   glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_SPLITS=3"     2 24 20
run ppn-gate 13-n2-streams0.log glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_STREAMS=0"    2 24 20
run ppn-gate 14-n2-overlap0.log glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_OVERLAP=0"    2 24 20
run ppn-gate 16-n3-even.log     glm5-spec-ppn-gate 23 "$OFF"                       3 24 20
run ppn-gate 17-n3-asym.log     glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_SPLITS=1,3"   3 24 20
run ppn-gate 18-n3-streams0.log glm5-spec-ppn-gate 23 "$OFF MEMRA_PP_STREAMS=0"    3 24 20
# The dedup compose arms — this is where the transposed schedules actually run inside a SPLIT
# verify walk, which the identity gates cannot substitute for.
run ppn-gate compose-n2-even-doors-E.log    glm5-spec-ppn-gate 23 "$E"                     2 24 20
run ppn-gate compose-n3-even-doors-E.log    glm5-spec-ppn-gate 23 "$E"                     3 24 20
run ppn-gate compose-n3-asym-doors-E.log    glm5-spec-ppn-gate 23 "$E MEMRA_PP_SPLITS=1,3" 3 24 20
run ppn-gate compose-n3-even-doors-EDH.log  glm5-spec-ppn-gate 23 "$EDH"                   3 24 20
run ppn-gate compose-n2-even-doors-EDH.log  glm5-spec-ppn-gate 23 "$EDH"                   2 24 20

# ---- glm5-hyper-ppn-gate (6 PASS lines/arm) ----
run hppn-gate 10-n2-even.log     glm5-hyper-ppn-gate 6 "$OFF"                      2 6 8
run hppn-gate 11-n2-split1.log   glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_SPLITS=1"    2 6 8
run hppn-gate 12-n2-split3.log   glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_SPLITS=3"    2 6 8
run hppn-gate 13-n2-streams0.log glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_STREAMS=0"   2 6 8
run hppn-gate 14-n2-overlap0.log glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_OVERLAP=0"   2 6 8
run hppn-gate 15-n2-shard0.log   glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_SHARD=0"     2 6 8
run hppn-gate 16-n3-asym.log     glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_SPLITS=1,3"  3 6 8
run hppn-gate 17-n4-even.log     glm5-hyper-ppn-gate 6 "$OFF"                      4 6 8
run hppn-gate 18-n4-streams0.log glm5-hyper-ppn-gate 6 "$OFF MEMRA_PP_STREAMS=0"   4 6 8
run hppn-gate 19-n2-longer.log   glm5-hyper-ppn-gate 6 "$OFF"                      2 16 24
run hppn-gate compose-n2-even-doors-E.log   glm5-hyper-ppn-gate 6 "$E"             2 6 8
run hppn-gate compose-n2-even-doors-EDH.log glm5-hyper-ppn-gate 6 "$EDH"           2 6 8

# ---- glm5-hyper-batch-gate (3 PASS lines/arm), B P N stages ----
run hbatch-gate 10-b3-default.log       glm5-hyper-batch-gate 3 "$OFF"                     3 5 8 1
run hbatch-gate 11-b8-wide.log          glm5-hyper-batch-gate 3 "$OFF"                     8 5 8 1
run hbatch-gate 12-b2-longer.log        glm5-hyper-batch-gate 3 "$OFF"                     2 12 24 1
run hbatch-gate 13-b3-ppn2.log          glm5-hyper-batch-gate 3 "$OFF"                     3 5 8 2
run hbatch-gate 14-b3-ppn2-streams0.log glm5-hyper-batch-gate 3 "$OFF MEMRA_PP_STREAMS=0"  3 5 8 2
run hbatch-gate 15-b3-ppn4.log          glm5-hyper-batch-gate 3 "$OFF"                     3 5 8 4
run hbatch-gate 16-b8-ppn2.log          glm5-hyper-batch-gate 3 "$OFF"                     8 5 8 2
run hbatch-gate 17-b12.log              glm5-hyper-batch-gate 3 "$OFF"                     12 5 8 1
run hbatch-gate 18-b15-cap.log          glm5-hyper-batch-gate 3 "$OFF"                     15 5 8 1
run hbatch-gate 19-b15-ppn2.log         glm5-hyper-batch-gate 3 "$OFF"                     15 5 8 2
run hbatch-gate compose-b3-doors-E.log   glm5-hyper-batch-gate 3 "$E"                      3 5 8 1
run hbatch-gate compose-b3-doors-EDH.log glm5-hyper-batch-gate 3 "$EDH"                    3 5 8 1

echo "=========================================================="
if [ "$fails" -eq 0 ]; then echo "dedup matrices: ALL ARMS PASS"; else echo "dedup matrices: $fails ARM(S) FAILED"; fi
exit "$fails"
