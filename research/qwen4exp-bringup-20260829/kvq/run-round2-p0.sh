#!/usr/bin/env bash
# qwen4_exp round 2, PHASE 0 — the BOX-BASELINE receipt.
#
# Why this phase exists at all: the round-1 box was lost to two preemptions, and this is a
# DIFFERENT machine (a fresh preemptible box; provider, region and instance ids are fleet
# state and stay in darklanes). It is the
# same card class — 2x RTX PRO 6000 Blackwell **Server Edition** 600 W, 97,887 MiB — so
# the round-1 numbers ARE the comparison and any delta is a finding, not a new baseline.
# Nothing downstream (TP2 band calibration, the 1M ladder, spec at depth) may be read
# until this phase reproduces round 1.
#
# Every arm runs at the FLIPPED DEFAULTS with NO seam env (the §7 doctrine: a default is
# a claim about what runs when nobody passes a flag, so it is verified by running with
# nothing armed and asserting an OUTCOME). Note the binary's reference-parity pin: the
# --goldens / --prompts comparisons force the f32 exactness-instrument cache arms even
# under the flipped serving defaults, so the no-env hidden/greedy arms are the kvq0 rows
# and the `kvq` arm below is the armed twin.
set -uo pipefail
cd ~/memra
git fetch origin -q && git checkout -q qwen4exp-bringup-20260829 && git pull -q --ff-only origin qwen4exp-bringup-20260829
echo "== tip: $(git log -1 --oneline)"        # rebuild-after-checkout-attribution law
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release -p memra-engine --bin qwen4exp_real_gate --bin qwen4exp_gpu_gate 2>&1 | tail -3
BIN=./target/release/qwen4exp_real_gate
GGATE=./target/release/qwen4exp_gpu_gate
CKPT=$HOME/data/q48fn-nvfp4
OUT=$HOME/realgate/kvq2
DUMP=$HOME/realgate/dump
SHAPES=$HOME/realgate/shapes
SHIP="--mtp --mtp-dev1 --spec-pmin 0.3 --spec-adapt 1 --spec-k 5"
mkdir -p "$OUT"
echo "== nvidia-smi =="; nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv

run () { local tag="$1"; shift; echo; echo "########## $tag ##########"; "$@" 2>&1 | tail -60; echo "## rc=$? tag=$tag"; }

# ---- 0a: tiny gate, no seam env (the 263-row / 0-failure round-1 receipt) -------------
run tiny "$GGATE" "$OUT/tiny-gate-r2base-box.tsv"

# ---- 0b: real-checkpoint hidden goldens (f32 instrument arm by the pin) ---------------
run hidden-defaults $BIN "$CKPT" "$OUT" --label r2base-defaults --goldens "$DUMP"

# ---- 0c: greedy first-divergence vs the banked goldens, BOTH cache arms --------------
#   no-env  -> expect the kvq0 pattern -1/8/-1/48
#   kvq     -> expect the kvq1 pattern -1/8/-1/26 (one extra near-tie fork; mint class)
# The thinkon greedy arm is DELIBERATELY NOT RUN: round 1 found it a BROKEN INSTRUMENT
# (fork at step 0 on both arms incl. f32 — the thinkon goldens do not match the gate's
# thinkon render). Re-running it would only re-bank a known-bad instrument.
run greedy-defaults $BIN "$CKPT" "$OUT" --label r2base-raw --prompts "$DUMP/prompts.tsv" --max-new 64
run greedy-kvq env MEMRA_Q4E_SEAMS=kvq $BIN "$CKPT" "$OUT" --label r2base-raw-kvq --prompts "$DUMP/prompts.tsv" --max-new 64

# ---- 0d: the three rule gates at ship admission --------------------------------------
run vbit $BIN "$CKPT" "$OUT" --label r2base-vbit $SHIP --goldens "$DUMP" --verify-bit-gate 24
run spec-raw $BIN "$CKPT" "$OUT" --label r2base-raw $SHIP --prompts "$DUMP/prompts.tsv" --spec-gate 256
run spec-thinkon $BIN "$CKPT" "$OUT" --label r2base-thinkon $SHIP --prompts "$SHAPES/thinkon-prompts.tsv" --spec-gate 256

# ---- 0e: tp2-gate (24 decode rows, single-card vs TP2) -------------------------------
run tp2 $BIN "$CKPT" "$OUT" --label r2base-tp2 --tp2 --goldens "$DUMP" --tp2-gate 24

# ---- 0f: the live router audit proves the flipped default ENGAGES (rows>0) -----------
run audit env MEMRA_Q4E_ROUTER_AUDIT=1 $BIN "$CKPT" "$OUT" --label r2base-audit $SHIP \
  --goldens "$DUMP" --verify-bit-gate 8

echo; echo "== round-2 phase 0 complete; receipts in $OUT"
ls -la "$OUT"
