#!/usr/bin/env bash
# tpd-battery engine-probe launcher (TP-2 DIET RE-PRICE window). Cards 0,1 (TP pair) —
# port NONE, engine-level; the served PP-3 calibration boot uses serve.sh.
# One binary, env-selected arms (comparability requirement): this file owns the arm env
# tables so every boot of an arm is byte-identical env. Derived VERBATIM from
# tp2-battery-20260831/box/probe_arm.sh (the banked v1 arm tables are reused unchanged so
# the diet arms differ from v1 by the DOOR ALONE) + struct-battery's FIXED
# announce/counter-receipting tail.
#
# TRAP (banked, struct-battery cell 5): this script SCRUBS inherited BOXP_*, so probe
# extras (BOXP_SAMPLED / BOXP_MAX_NEW / BOXP_FORCE_DIR) MUST ride as TRAILING ARGS, never
# as env-prefix assignments. The tp2 RUNBOOK's env-prefix spelling carries the same trap.
#
# DOOR SPELLING (owner law, ep-diet LANE §2): both diet doors default OFF at this head, and
# an OFF arm is spelled by PINNING `=0` — never by leaving the var unset (a pin does not
# inherit a future default flip). Every arm below therefore carries an explicit value.
set -uo pipefail
OUT=${OUT:-/root/out-tpd}
BIN=${BIN:-/root/memra-tpd/target/release/glm5-tp2-box-probe}
MODEL=/root/models/glm53-nvfp4
MAP=${MAP:-/root/out-tpd/maps/agentic-t1-coactivation.json}
mkdir -p "$OUT/logs"

arm=$1; mode=$2; prompts=$3; run=$4; shift 4   # extra env as "$@"

# scrub inherited MEMRA_*/BOXP_* (flip-battery discipline), then the arm table
unsets=()
while IFS='=' read -r k _; do case "$k" in MEMRA_*|BOXP_*) unsets+=(-u "$k");; esac; done < <(env)

common=(NVIDIA_TF32_OVERRIDE=0 MEMRA_BF16_MMV=1)
# the v1 TP-2 pair env, banked verbatim (tp2-battery cell 4 arm `tp2`)
tp2base=(CUDA_VISIBLE_DEVICES=0,1 MEMRA_GLM5_TP=all@0,1 MEMRA_RP=0 \
         MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=16)
want_diet=0; want_gp=0; want_map=0
case "$arm" in
  # ---- pricing arms -------------------------------------------------------------------
  v1)     armenv=("${tp2base[@]}" MEMRA_GLM5_EP_DIET=0 MEMRA_GLM5_EP_GROUPED_PRIME=0) ;;
  diet)   armenv=("${tp2base[@]}" MEMRA_GLM5_EP_DIET=1 MEMRA_GLM5_EP_GROUPED_PRIME=0)
          want_diet=1 ;;
  dietmap) armenv=("${tp2base[@]}" MEMRA_GLM5_EP_DIET=1 MEMRA_GLM5_EP_GROUPED_PRIME=0 \
                   MEMRA_GLM5_EP_MAP="$MAP")
          want_diet=1; want_map=1 ;;
  dietgp) armenv=("${tp2base[@]}" MEMRA_GLM5_EP_DIET=1 MEMRA_GLM5_EP_GROUPED_PRIME=1)
          want_diet=1; want_gp=1 ;;
  # ---- class-gate arms (diet ON; the sub-trunk shapes of tp2-battery cell 1) ----------
  dietred) armenv=("${tp2base[@]}" MEMRA_GLM5_EP_DIET=1 MEMRA_GLM5_EP_GROUPED_PRIME=0 \
                   MEMRA_GLM5_TP_GATE_RED=swap-wo)
          want_diet=1 ;;
  # single-layer / sub-shard arms take their TP spec + slot env as trailing extras, exactly
  # as the banked v1 runs did (MEMRA_GLM5_TP=0-2@0,1 / 3@0,1 / 4@0,1 [+ MOE_SLOTS=12000]).
  dietsub) armenv=(CUDA_VISIBLE_DEVICES=0,1 MEMRA_RP=0 MEMRA_MOE_RESIDENT=0 \
                   MEMRA_MOE_SLOTS=16 MEMRA_GLM5_EP_DIET=1 MEMRA_GLM5_EP_GROUPED_PRIME=0)
          want_diet=1 ;;
  v1sub)  armenv=(CUDA_VISIBLE_DEVICES=0,1 MEMRA_RP=0 MEMRA_MOE_RESIDENT=0 \
                   MEMRA_MOE_SLOTS=16 MEMRA_GLM5_EP_DIET=0 MEMRA_GLM5_EP_GROUPED_PRIME=0) ;;
  # ---- controls ------------------------------------------------------------------------
  plain1) armenv=(CUDA_VISIBLE_DEVICES=0 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=12000 \
                  MEMRA_ST_PINNED=1) ;;
  pp3)    armenv=(CUDA_VISIBLE_DEVICES=0,1,2 MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 \
                  MEMRA_PP_DEVICES=0,1,2 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 \
                  MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16) ;;
  *) echo "unknown arm $arm (v1|diet|dietmap|dietgp|dietred|dietsub|v1sub|plain1|pp3)"; exit 2 ;;
esac

name="$arm-$mode-$run"
log="$OUT/logs/probe-$name.log"
echo "[tpd] arm=$arm mode=$mode prompts=$prompts run=$run extras=$* bin=$BIN map=$MAP" | tee "$log"
sha256sum "$BIN" | tee -a "$log"
[ "$want_map" = 1 ] && sha256sum "$MAP" | tee -a "$log"
env "${unsets[@]}" "${common[@]}" "${armenv[@]}" BOXP_MODE="$mode" "$@" \
  "$BIN" "$MODEL" "$prompts" "$OUT/$name" 2>>"$log" | tee -a "$log"
rc=${PIPESTATUS[0]}
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$log"

# ---- arm-identity receipts (per-boot, demanded/forbidden by arm) -----------------------
ndiet=$(grep -c "\[glm5-ep-diet\] engaged" "$log" || true)
ngp=$(grep -c "\[glm5-ep-grouped-prime\] execute" "$log" || true)
ngpflag=$(grep -c "\[glm5-ep-grouped-prime\] flag" "$log" || true)
nmap=$(grep -c "ep-map armed" "$log" || true)
peer=$(grep -o "ep-peer-slot-dispatches=[0-9]*" "$log" | tail -1)
ctr=$(grep -o "ep-diet-counters .*" "$log" | tail -1)
# The diet is an EP-walk door: it can only engage on a boot that actually shards MoE layers.
# The preflight line self-receipts that (`moe_ep=<n>`), so the demand is conditional on it —
# a KDA-only sub-shard arm (MEMRA_GLM5_TP=0-2@0,1 -> moe_ep=0) legitimately shows the door
# armed with ZERO dieted layer-calls, and that is a receipt, not a failure. Found in this
# window on the tinykda arm; the arm's tapes were unaffected.
moe_ep=$(grep -o "moe_ep=[0-9]*" "$log" | head -1 | cut -d= -f2)
moe_ep=${moe_ep:-0}
if [ "$want_diet" = 1 ] && [ "$moe_ep" -gt 0 ]; then
  [ "${ndiet:-0}" -ge 1 ] || { echo "[tpd] ARM-IDENTITY FAIL: diet arm with moe_ep=$moe_ep but no '[glm5-ep-diet] engaged'"; rc=1; }
elif [ "$want_diet" = 1 ]; then
  echo "[tpd] diet arm with moe_ep=0 (no EP layer sharded): diet announce not applicable, counters must be 0"
  [ "${ndiet:-0}" -eq 0 ] || { echo "[tpd] ARM-IDENTITY FAIL: moe_ep=0 yet the diet announced"; rc=1; }
else
  [ "${ndiet:-0}" -eq 0 ] || { echo "[tpd] ARM-IDENTITY FAIL: pinned-0 arm carries '[glm5-ep-diet] engaged'"; rc=1; }
fi
if [ "$want_gp" = 1 ]; then
  [ "${ngpflag:-0}" -ge 1 ] || { echo "[tpd] ARM-IDENTITY FAIL: gp arm without '[glm5-ep-grouped-prime] flag'"; rc=1; }
else
  [ "${ngp:-0}" -eq 0 ] || { echo "[tpd] ARM-IDENTITY FAIL: non-gp arm carries a grouped-prime execute line"; rc=1; }
fi
if [ "$want_map" = 1 ]; then
  [ "${nmap:-0}" -ge 1 ] || { echo "[tpd] ARM-IDENTITY FAIL: map arm without 'ep-map armed'"; rc=1; }
  grep -m1 "ep-map armed" "$log"
else
  [ "${nmap:-0}" -eq 0 ] || { echo "[tpd] ARM-IDENTITY FAIL: non-map arm carries 'ep-map armed'"; rc=1; }
fi
{
  echo "arm=$arm mode=$mode run=$run rc=$rc"
  echo "announce diet_engaged=$ndiet grouped_prime_flag=$ngpflag grouped_prime_execute=$ngp ep_map_armed=$nmap"
  echo "${peer:-ep-peer-slot-dispatches=<missing>}"
  echo "${ctr:-ep-diet-counters=<missing>}"
} | tee "$OUT/logs/probe-$name.identity" | tee -a "$log"
echo "[tpd] rc=$rc arm=$arm mode=$mode run=$run" | tee -a "$log"
exit $rc
