#!/usr/bin/env bash
# memra#1275 Step-3.7 check (2x RTX PRO 6000 Blackwell, PP-2, the IQ4_XS GGUF route of
# docs/models/step37-flash.md). Run on the box under a lease on the pair; this script
# takes no lock itself.
#
# Arms (sm_120a builds of the two commits, identical except memra#1275). Which pair a run used is
# in its guard.log (the binaries' sha256) and in the receipt's README.
#   - run-191529 (prime half): base 224d374573, head 2e526edfae.
#   - bins3: base = main d6ae4e525d, head = 9ba7342be8 (PR #1297 on it).
#   - bins4, the landing pair after main moved: base = main 63c351a2f4, head = eb648a2150 (PR #1297
#     rebased onto it). hybrid_forward.rs on main is byte-identical from d6ae4e525d to 63c351a2f4.
# Both arms of bins3 and bins4 carry the print-only patches (widths-patch.py: the ENGAGED line
# lists every call's width; bins4 also walk-widths-patch.py).
#   base: the step35 GEMM prime folds a tail of 1..15 rows into the last call: 4,100 tokens prime
#         as one 4,100-row call, 8,200 as 4,096 + 4,104.
#   head: fixed_prime_chunk_ranges at the cap under the model's ring contract: 4,100 -> 4,084 + 16,
#         8,200 -> 4,096 + 4,088 + 16.
# Prompts: exact 4,100- and 8,200-token prefixes of research/e2e/prompts/p4-16k.txt, tokenized
# with the Step-3.7-Flash tokenizer (stepfun-ai/Step-3.7-Flash-FP8@b3d7916f tokenizer.json, BOS 0
# prepended). The same tokenizer plus BOS reproduces the 1,942 ids of the 2026-09-20 record's
# run-gen log exactly. The ids go in on the command line, so both arms prime the same tokens.
#
# Per arm and length, two processes:
#   1. run-gen MEMRA_PP_ONLY=1 MEMRA_PP_PRIME=1 MEMRA_PP_LOGITS: the serving prime's last-row
#      logits (prime_cache on a fresh cache; the GEMM entry, receipt `[gemm-prime] ENGAGED`);
#   2. run-spec MEMRA_GEN_ONLY=1 MEMRA_NGEN=16 with MEMRA_MTP_DRAFT=$STEP_MTP (the route's external
#      Q8_0 MTP head; the trunk GGUF reads nextn=0): the serving prime plus 16 plain greedy tokens.
#      run-191529 ran it without the head (all four exited 2); run-192323 tried run-gen's plain
#      path, which cannot generate on PP-2 (memra#1295).
#
# PASS, per length, written down before the run (step-compare.py applies it):
#   - the argmax of the last-row logits is equal across arms;
#   - the 16 greedy tokens are equal across arms;
#   - the full-vocab last-row maxdiff, base vs head, is within the documented near-tie band:
#     BAND = 2.4. Source: crates/memra-engine/src/bin/argmax_margin_probe.rs, header
#     (lane/q8-argmax, 2026-08-06): maxdiff "is the max over a 248k-wide vocab ..., mostly deep
#     in the tail, and it is routinely 0.3-2.4 on runs the same gate calls MATCH"; the same law
#     in concat_prime_probe.rs (primepath: "the deep-tail maxdiff is 0.3-2.4 even between
#     MATCHING configs"). On this route, the 2026-09-20 record measured the m-dependent
#     decompositions at 1,942 tokens: batched-prime vs tokenwise 1.126e0, prefill vs decode
#     2.052e0 (research/modelplan-onboarding-537-20260919/native-20260920, raw-receipts.tar.gz,
#     phases/replacement-step-gguf-a1/run-gen.log lines 28 and 31), both inside the band.
#   - both arms show the GEMM prime engaged at the full length, and every process exits 0.
# Reported, not judged: the top-2 margin of each arm, and the cross-arm delta at base's two
# contending ids (the flip-possibility quantity of the argmax-margin-probe law).
#
# usage: step-check.sh GGUF_SHARD1 BIN_DIR OUT_DIR [DEVICES] [PP_FROM]
#   GGUF_SHARD1  .../IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf (the pinned GGUF revision)
#   BIN_DIR      holds run-gen-{base,head} and run-spec-{base,head}
#   DEVICES      CUDA_VISIBLE_DEVICES for the pair (default 2,3; step-lease.sh passes the UUIDs)
#   STEP_LENS    (env) the prompts to run, by name: p<name>.ids, default "4100 8200"; 4100b is a
#                second 4,100-token prompt (BOS + the 4,099 source tokens after p8200's prefix,
#                mkids-b.py), for when run-gen's own prefill-vs-tokenwise gate trips a near-tie
#   PP_FROM      a finished run dir whose prime half (pp-only logits and logs, all rc=0) this run
#                copies instead of running it again; that dir is left as it is
set -uo pipefail
GG=${1:?GGUF shard 1}; BIN=${2:?bin dir}; OUT=${3:?out dir}; DEV=${4:-2,3}; PP_FROM=${5:-}
MTP=${STEP_MTP:-$(dirname "$GG")/../Step3.7-flash-mtp-Q8_0.gguf}
HERE=$(cd "$(dirname "$0")" && pwd)
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
export CUDA_VISIBLE_DEVICES=$DEV NVIDIA_TF32_OVERRIDE=0 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PARALLEL=off
mkdir -p "$OUT"
G=$OUT/guard.log
LENS=${STEP_LENS:-4100 8200}
{ date -Is; hostname; echo "CUDA_VISIBLE_DEVICES=$DEV prompts: $LENS MTP=$MTP"; sha256sum "$BIN"/run-gen-base "$BIN"/run-gen-head "$BIN"/run-spec-base "$BIN"/run-spec-head "$HERE"/p*.ids "$MTP"
  ls -la "$(dirname "$GG")"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,temperature.gpu,power.limit,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; } > "$G" 2>&1
[ -f "$GG" ] || { echo "GGUF missing: $GG" | tee -a "$G"; exit 2; }
[ -f "$MTP" ] || { echo "MTP head missing: $MTP (set STEP_MTP)" | tee -a "$G"; exit 2; }
rc=0
for n in $LENS; do
  ids=$(cat "$HERE/p$n.ids")
  for arm in base head; do
    if [ -n "$PP_FROM" ]; then
      cp -p "$PP_FROM/$n-$arm.logits" "$PP_FROM/$n-$arm-pp.log" "$OUT/" || rc=1
      echo "$(date -Is) $n $arm run-gen pp-only copied from $PP_FROM: $(grep "^[^ ]* $n $arm run-gen pp-only" "$PP_FROM/guard.log" | cut -d' ' -f2-)" | tee -a "$G"
      grep -q "^[^ ]* $n $arm run-gen pp-only rc=0 " "$PP_FROM/guard.log" || rc=1
    else
      # shellcheck disable=SC2086
      MEMRA_PP_ONLY=1 MEMRA_PP_PRIME=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 MEMRA_PP_LOGITS=$OUT/$n-$arm.logits \
        timeout 10m "$BIN/run-gen-$arm" "$GG" $ids > "$OUT/$n-$arm-pp.log" 2>&1
      r=$?; [ $r = 0 ] || rc=1
      echo "$(date -Is) $n $arm run-gen pp-only rc=$r $(grep -m1 -o 'first concat prime: B=1 tokens=[0-9]*' "$OUT/$n-$arm-pp.log") $(grep -m1 -o 'ENGAGED t=[0-9]* base=0[^(]*' "$OUT/$n-$arm-pp.log")" | tee -a "$G"
    fi
    # shellcheck disable=SC2086
    MEMRA_MTP_DRAFT=$MTP MEMRA_GEN_ONLY=1 MEMRA_NGEN=16 timeout 10m "$BIN/run-spec-$arm" "$GG" $ids > "$OUT/$n-$arm-gen.log" 2>&1
    r=$?; [ $r = 0 ] || rc=1
    echo "$(date -Is) $n $arm run-spec gen-only rc=$r $(grep -m1 -o 'ENGAGED t=[0-9]* base=0[^(]*' "$OUT/$n-$arm-gen.log")" | tee -a "$G"
  done
done
# shellcheck disable=SC2086
python3 "$HERE/step-compare.py" "$OUT" $LENS | tee "$OUT/verdict.txt"
v=${PIPESTATUS[0]}
[ "$rc" = 0 ] || { echo "a process failed (rc above): the check is not valid" | tee -a "$OUT/verdict.txt"; exit 1; }
exit "$v"
