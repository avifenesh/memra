#!/usr/bin/env bash
# lane/glm5-moe-loc split-gate matrices (rig 5090, exactness only — rig law).
#
# Arm shapes copied verbatim from the matvec lane's banked matrices so the receipts are
# comparable arm-for-arm:
#   glm5-spec-ppn-gate   stages P N : 2/24/20 (even, split1, split3, streams0, overlap0)
#                                     3/24/20 (even, asym 1,3, streams0)
#   glm5-hyper-ppn-gate  stages P N : 2/6/8 (even, split1, split3, streams0, overlap0, shard0),
#                                     3/6/8 asym, 4/6/8 (even, streams0), 2/16/24 longer
#   glm5-hyper-batch-gate  B P N ppn: the banked 10-arm ladder
# Plus a COMPOSE arm per matrix with doors D + H ON (door M pinned =0, never unset).
set -u
cd "$(dirname "$0")/../../../.."
BASE=research/glm53-flash-bringup-20260827/moe-loc-20260831/receipts
fails=0
STARTED=""

run() { # run <outdir> <log> <bin> <env-or-"-"> <args...>
  local dir="$1" log="$2" bin="$3" envs="$4"; shift 4
  mkdir -p "$BASE/$dir"
  # TRUNCATE the summary on the first arm of a matrix: `tee -a` across re-runs left a stale FAIL
  # line from an earlier (typo'd) invocation sitting in a banked receipt, which reads exactly like
  # a live failure. A summary that accumulates across runs is not a receipt.
  case " $STARTED " in *" $dir "*) ;; *) : >"$BASE/$dir/matrix.out"; STARTED="$STARTED $dir";; esac
  echo "########## $log :: ${envs} $* ##########" | tee -a "$BASE/$dir/matrix.out"
  # shellcheck disable=SC2086
  local envlist=""
  [ "$envs" != "-" ] && envlist="$envs"
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 $envlist \
    timeout 3600 nice -n 5 cargo run -q -p memra-engine --bin "$bin" -- "$@" \
    >"$BASE/$dir/$log" 2>&1
  local rc=$?
  echo "exit=$rc" >>"$BASE/$dir/$log"
  if [ "$rc" -ne 0 ]; then fails=$((fails+1)); echo "FAIL: $dir/$log" | tee -a "$BASE/$dir/matrix.out";
  else grep -cE 'gate PASS' "$BASE/$dir/$log" | sed "s/^/PASS lines: /" | tee -a "$BASE/$dir/matrix.out"; fi
  tail -3 "$BASE/$dir/$log" >>"$BASE/$dir/matrix.out"
}

DOORS="MEMRA_MOE_VROWS_DEV_TABLES=1 MEMRA_GLM5_HTOD_DIET=1 MEMRA_MOE_VROWS_PACK=0"

# ---- glm5-spec-ppn-gate ----
run ppn-gate 10-n2-even.log      glm5-spec-ppn-gate -                     2 24 20
run ppn-gate 11-n2-split1.log    glm5-spec-ppn-gate MEMRA_PP_SPLITS=1     2 24 20
run ppn-gate 12-n2-split3.log    glm5-spec-ppn-gate MEMRA_PP_SPLITS=3     2 24 20
run ppn-gate 13-n2-streams0.log  glm5-spec-ppn-gate MEMRA_PP_STREAMS=0    2 24 20
run ppn-gate 14-n2-overlap0.log  glm5-spec-ppn-gate MEMRA_PP_OVERLAP=0    2 24 20
run ppn-gate 16-n3-even.log      glm5-spec-ppn-gate -                     3 24 20
run ppn-gate 17-n3-asym.log      glm5-spec-ppn-gate MEMRA_PP_SPLITS=1,3   3 24 20
run ppn-gate 18-n3-streams0.log  glm5-spec-ppn-gate MEMRA_PP_STREAMS=0    3 24 20
run ppn-gate compose-n3-even-doors-DH.log glm5-spec-ppn-gate "$DOORS"     3 24 20

# ---- glm5-hyper-ppn-gate ----
run hppn-gate 10-n2-even.log     glm5-hyper-ppn-gate -                    2 6 8
run hppn-gate 11-n2-split1.log   glm5-hyper-ppn-gate MEMRA_PP_SPLITS=1    2 6 8
run hppn-gate 12-n2-split3.log   glm5-hyper-ppn-gate MEMRA_PP_SPLITS=3    2 6 8
run hppn-gate 13-n2-streams0.log glm5-hyper-ppn-gate MEMRA_PP_STREAMS=0   2 6 8
run hppn-gate 14-n2-overlap0.log glm5-hyper-ppn-gate MEMRA_PP_OVERLAP=0   2 6 8
run hppn-gate 15-n2-shard0.log   glm5-hyper-ppn-gate MEMRA_PP_SHARD=0     2 6 8
run hppn-gate 16-n3-asym.log     glm5-hyper-ppn-gate MEMRA_PP_SPLITS=1,3  3 6 8
run hppn-gate 17-n4-even.log     glm5-hyper-ppn-gate -                    4 6 8
run hppn-gate 18-n4-streams0.log glm5-hyper-ppn-gate MEMRA_PP_STREAMS=0   4 6 8
run hppn-gate 19-n2-longer.log   glm5-hyper-ppn-gate -                    2 16 24
run hppn-gate compose-n2-even-doors-DH.log glm5-hyper-ppn-gate "$DOORS"   2 6 8

# ---- glm5-hyper-batch-gate ----
run hbatch-gate 10-b3-default.log       glm5-hyper-batch-gate -                  3 5 8 1
run hbatch-gate 11-b8-wide.log          glm5-hyper-batch-gate -                  8 5 8 1
run hbatch-gate 12-b2-longer.log        glm5-hyper-batch-gate -                  2 12 24 1
run hbatch-gate 13-b3-ppn2.log          glm5-hyper-batch-gate -                  3 5 8 2
run hbatch-gate 14-b3-ppn2-streams0.log glm5-hyper-batch-gate MEMRA_PP_STREAMS=0 3 5 8 2
run hbatch-gate 15-b3-ppn4.log          glm5-hyper-batch-gate -                  3 5 8 4
run hbatch-gate 16-b8-ppn2.log          glm5-hyper-batch-gate -                  8 5 8 2
run hbatch-gate 17-b12.log              glm5-hyper-batch-gate -                  12 5 8 1
run hbatch-gate 18-b15-cap.log          glm5-hyper-batch-gate -                  15 5 8 1
run hbatch-gate 19-b15-ppn2.log         glm5-hyper-batch-gate -                  15 5 8 2
run hbatch-gate compose-b3-doors-DH.log glm5-hyper-batch-gate "$DOORS"           3 5 8 1

# ---- instrument S: the dedup counters must SPEAK (the [moe-vrows-dedup] line is the
# receipt a box window greps; door D pinned =0 because S counts the HOST selection) ----
run stat-gate dedup-n3-even.log glm5-spec-ppn-gate "MEMRA_MOE_VROWS_DEDUP_STAT=1 MEMRA_MOE_VROWS_DEV_TABLES=0 MEMRA_MOE_VROWS_PACK=0" 3 24 20

echo "=========================================================="
if [ "$fails" -eq 0 ]; then echo "moe-loc matrices: ALL ARMS PASS"; else echo "moe-loc matrices: $fails ARM(S) FAILED"; fi
exit "$fails"
