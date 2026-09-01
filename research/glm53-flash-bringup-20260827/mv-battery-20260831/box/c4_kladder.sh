#!/usr/bin/env bash
# MV-DOORS CELL 4 — K-LADDER RE-PIN on the composed winner shape (all five doors ON +
# DFlash2 + PMIN0.7): the drafter-head fix (door T) removes the 15x head re-read that
# was PER-ROUND, and doors X/M cheapen the verify row — both move the K economics, so
# the banked K5 re-pin (+1.8% on the vrest head) must be re-measured, not inherited.
# Rows: pinned K3 (pin-vs-policy control) / K5 / K7, single boots (diet v3k precedent:
# ladder rows are attribution; escalate the winner to interleaved rounds via
#   c4_kladder.sh confirm <K> <round>   if a pin beats the c2 don median).
# Caller holds /root/TIMING-IN-FLIGHT.
set -uo pipefail
OUT=/root/out-mv/c4
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7)
DOORS=(MEMRA_BF16_TCOLS_WIDE=1 MEMRA_BF16_TCOLS_X1=1 MEMRA_MOE_VROWS_PACK=1 MEMRA_TOPK_SHARDS=1 MEMRA_GLM5_VERIFY_WS=1)
mkdir -p "$OUT"

boot_k() {  # name, K
  local name="$1" k="$2"
  echo "######## C4 BOOT $name (K=$k) ########"
  /root/out-mv/serve.sh start "c4-$name" "${DFL[@]}" "${DOORS[@]}" "MEMRA_SPEC_K=$k" \
    || { echo "C4_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-mv/run_pool.py sample --out "$OUT/$name" || { echo "C4_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-mv/serve.sh doors "c4-$name" "${DFL[@]}" "${DOORS[@]}" || { echo "C4_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-mv/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-mv/logs/boot-c4-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  echo "C4_${name}_EXIT=0"
}

rc=0
case "${1:-ladder}" in
  ladder)
    boot_k k3p-1 3 || rc=1
    boot_k k5p-1 5 || rc=1
    boot_k k7p-1 7 || rc=1
    ;;
  confirm)
    K="${2:?confirm needs K}"; R="${3:?confirm needs round}"
    boot_k "k${K}p-$R" "$K" || rc=1
    ;;
  *) echo "usage: c4_kladder.sh [ladder | confirm <K> <round>]"; exit 2 ;;
esac
/root/out-mv/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c4 tapes) ==="
python3 /root/out-mv/looplaw_screen.py "$OUT"/*/
echo "=== K-LADDER TABLE (baseline = c2 don auto-K rows) ==="
for d in /root/out-mv/c2/don-*; do ln -sfn "$d" "$OUT/$(basename "$d")"; done
python3 /root/out-mv/mv_check.py --base "$OUT" --baseline don --arms k3p,k5p,k7p || true
echo "C4_DONE rc=$rc"
exit "$rc"
