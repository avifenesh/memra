#!/usr/bin/env bash
# tpd-battery CELL 1 — REAL-ARTIFACT CLASS GATE WITH THE DIET DOORS ON (untimed, exactness
# only; NO timing number leaves this cell). Shape = tp2-battery cell 1 verbatim, re-run with
# MEMRA_GLM5_EP_DIET=1, so every row has a banked v1 twin to be compared against:
#
#   banked v1 (tp2-battery RESULTS.md cell 1, worst norm_rel vs the plain reference)
#     tinykda   TP layers 0-2, tiny prime : 0.0  BYTE-IDENTICAL          <- non-MoE class bar
#     tinymoe4  TP layer 4 (KDA+EP), tiny : 4.8e-2                       <- EP band
#     tinymla3  TP layer 3 (MLA+EP), tiny : 2.9e-2                       <- EP band
#     tinytp    TP all@0,1, tiny          : 5.2e-2  (saturating, 32/32 tape holds)
#     c1tp      TP all@0,1, deep prime    : 1.2e-1  (forks only at margins <= 0.23)
#     tinyred   RED swap-wo               : 0.93-1.05, argmax rank 1.5e4-1.4e5
#
# STOP RULE (window-level): any diet row that DIVERGES from its banked v1 class verdict —
# a byte class that stops being byte-exact, an EP-band row outside the banked band class, a
# red that stops biting orders above green, a missing announce, or an own-vs-forced tape fork
# — STOPS the window with verdict SILENT-WRONG-SUSPECT and nothing timed runs.
#
# Extra rows this cell adds beyond the v1 shape, because the diet is a TRANSPORT change:
#   dietvsv1  diet-ON vs diet-OFF on the SAME build, teacher-forced on the same reference:
#             the rig arm B2 bar is decode BYTE-IDENTICAL, so this is the strongest
#             available real-artifact statement about the door (identity, not band).
#   mapspot   diet+map teacher-forced on the diet-even tape: cell-4 phase A (the placement
#             A/B prices a correctness-free lever only if this is byte-identical, per the
#             struct-battery 56/56 result on the v1 walk).
set -uo pipefail
OUT=/root/out-tpd
A=$OUT/tpd_arm.sh
cd "$OUT"
mkdir -p "$OUT/analysis"
rc=0
run() { echo "######## C1 $* ########"; bash "$A" "$@" || rc=1; }

# ---- (a) plain single-card reference tapes (diet-agnostic; the class-gate reference) -----
run plain1 tape "$OUT/prompts-tiny" tinyref BOXP_MAX_NEW=32
run plain1 tape "$OUT/prompts-c1"   c1ref

echo "=== C1 CROSS-BUILD REFERENCE DETERMINISM (banked tp2-battery f32 manifest) ==="
# The banked manifest names ./plain1-tape-c1ref/<tag>.<step>.f32 sha256s from build 4a680d0ca.
# Reproducing them on 25537ca8e proves same artifact + same plain program => every diet row
# below is compared against a reference that is itself byte-equal to the banked one.
grep -E 'plain1-tape-(c1ref|tinyref)/' "$OUT/f32-manifest.sha256" > "$OUT/f32-ref.sha256" || true
echo "reference lines checked: $(wc -l < "$OUT/f32-ref.sha256")" | tee "$OUT/analysis/c1-refcheck.txt"
( cd "$OUT" && sha256sum -c f32-ref.sha256 2>&1 | grep -v ': OK$' | head -20 ) \
  | tee -a "$OUT/analysis/c1-refcheck.txt"
( cd "$OUT" && sha256sum -c --quiet f32-ref.sha256 >/dev/null 2>&1 \
  && echo "C1_REFCHECK=ALL_MATCH_BANKED" || echo "C1_REFCHECK=MISMATCH_OR_MISSING (see above)" ) \
  | tee -a "$OUT/analysis/c1-refcheck.txt"

echo "=== C1 output-sample + loop-law screen on the reference tapes ==="
python3 "$OUT/looplaw_screen.py" "$OUT/plain1-tape-tinyref" "$OUT/plain1-tape-c1ref"

# ---- (b) the diet class arms, teacher-forced on the reference tapes ----------------------
run dietsub tape "$OUT/prompts-tiny" tinykda \
  BOXP_FORCE_DIR="$OUT/plain1-tape-tinyref" MEMRA_GLM5_TP=0-2@0,1 BOXP_MAX_NEW=32
run dietsub tape "$OUT/prompts-tiny" tinymla3 \
  BOXP_FORCE_DIR="$OUT/plain1-tape-tinyref" MEMRA_GLM5_TP=3@0,1 MEMRA_MOE_SLOTS=12000 \
  MEMRA_ST_PINNED=1 BOXP_MAX_NEW=32
run dietsub tape "$OUT/prompts-tiny" tinymoe4 \
  BOXP_FORCE_DIR="$OUT/plain1-tape-tinyref" MEMRA_GLM5_TP=4@0,1 MEMRA_MOE_SLOTS=12000 \
  MEMRA_ST_PINNED=1 BOXP_MAX_NEW=32
run diet tape "$OUT/prompts-tiny" tinytp  BOXP_FORCE_DIR="$OUT/plain1-tape-tinyref" BOXP_MAX_NEW=32
run diet tape "$OUT/prompts-c1"   c1tp    BOXP_FORCE_DIR="$OUT/plain1-tape-c1ref"
# the v1 twin of the tiny full-trunk arm on THIS build: the diet-vs-v1 identity row
run v1   tape "$OUT/prompts-tiny" tinyv1  BOXP_FORCE_DIR="$OUT/plain1-tape-tinyref" BOXP_MAX_NEW=32
# RED through the dieted walk
run dietred tape "$OUT/prompts-tiny" tinyred BOXP_FORCE_DIR="$OUT/plain1-tape-tinyref" BOXP_MAX_NEW=32
# cell-4 phase A: map identity on the dieted walk (forced on the DIET-even tape)
run dietmap tape "$OUT/prompts-tiny" tinymap BOXP_FORCE_DIR="$OUT/diet-tape-tinytp" BOXP_MAX_NEW=32
run dietmap tape "$OUT/prompts-c1"   c1map   BOXP_FORCE_DIR="$OUT/diet-tape-c1tp"

# ---- (c) the compare table --------------------------------------------------------------
cmp_pair() { # label a b
  echo "===== $1 ====="
  python3 "$OUT/compare.py" "$OUT/$2" "$OUT/$3" > "$OUT/analysis/cmp-$1.txt" 2>&1
  echo "compare_rc=$? (band $(printenv BAND 2>/dev/null || echo 2e-4) — the CALLER owns the bar)" \
    >> "$OUT/analysis/cmp-$1.txt"
  tail -3 "$OUT/analysis/cmp-$1.txt"
  python3 "$OUT/logit_stats.py" "$OUT/$2" "$OUT/$3" > "$OUT/analysis/stats-$1.txt" 2>&1
  tail -2 "$OUT/analysis/stats-$1.txt"
}
cmp_pair tinykda-vs-plain  plain1-tape-tinyref dietsub-tape-tinykda
cmp_pair tinymla3-vs-plain plain1-tape-tinyref dietsub-tape-tinymla3
cmp_pair tinymoe4-vs-plain plain1-tape-tinyref dietsub-tape-tinymoe4
cmp_pair tinytp-vs-plain   plain1-tape-tinyref diet-tape-tinytp
cmp_pair c1tp-vs-plain     plain1-tape-c1ref   diet-tape-c1tp
cmp_pair tinyred-vs-plain  plain1-tape-tinyref dietred-tape-tinyred
cmp_pair diet-vs-v1-tiny   v1-tape-tinyv1      diet-tape-tinytp
cmp_pair map-vs-diet-tiny  diet-tape-tinytp    dietmap-tape-tinymap
cmp_pair map-vs-diet-c1    diet-tape-c1tp      dietmap-tape-c1map

echo "=== C1 ENGAGEMENT / COUNTER RECEIPTS PER ARM ==="
for f in "$OUT"/logs/probe-*-tape-*.identity; do echo "--- $f"; cat "$f"; done \
  | tee "$OUT/analysis/c1-engagement.txt"
echo "C1_DONE rc=$rc"
