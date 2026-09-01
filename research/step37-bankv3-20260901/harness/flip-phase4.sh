#!/usr/bin/env bash
# PHASE 4: the device-side oracle and its LOCALIZING TEETH ARM, re-run on the FLIPPED tree.
#
# Why re-run it when the flip changes no kernel: because "the flip changes no kernel" is a claim
# about the tree, and the whole point of this lane is that claims about trees get gated. The
# oracle is cheap (no model, seeded in-process bank, ~30 s per arm) so there is no reason to
# inherit a green light from a different checkout.
#
# THE TEETH ARM IS THE POINT, and it is BEHAVIOURAL rather than md5-based on purpose. This lane
# measured that the build is NOT byte-reproducible from identical source (clean1 != clean2 over two
# forced builds), so md5 inequality is necessary and NOT sufficient: it proves two binaries are
# different builds, never that they contain different programs. The decisive evidence is a
# deliberately corrupted binary FAILING the gate and naming the right kernel. A coordinator
# requirement from another rig makes this non-optional: there, three different .cu states produced
# a byte-identical oracle binary that reported PASS -- a green light proving the build system
# works, not the kernel.
set -u
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
D=/home/ubuntu/bankv3/lane
OUT=/home/ubuntu/bankv3/bin-teeth

echo "##################### PHASE 4a: forced build flow (clean / teeth / clean) #####################"
/home/ubuntu/bankv3/teeth.sh 2>&1 | tail -30

echo "##################### PHASE 4b: CLEAN oracle, all four prefill tile forms #####################"
# All four forms, because the 2026-08-29 defect lived in ONE of them (moe_kq_sktail_kernel) and the
# other three passed on the broken binary. A single-form oracle would have been green then too.
for form in "hybrid:" "SK128:MEMRA_F16G_SK=128" "SK32:MEMRA_F16G_SK=32" "TAIL0:MEMRA_F16G_TAIL=0"; do
  L=${form%%:*}; E=${form#*:}
  echo "########## ORACLE clean1 form=$L env=${E:-default} ##########"
  # capture-then-gate: $? after a pipe is grep's status, not the oracle's — the exact
  # pipe-swallows-gate-exits trap this lane already hit once (RESULTS.md); the exit code
  # is the gate, the grep is only display.
  set +e
  CLEAN_OUT=$(env $E "$OUT/oracle-clean1" 2>&1)
  CLEAN_RC=$?
  set -e
  printf '%s\n' "$CLEAN_OUT" | grep -E "cell |decode |PASS|FAIL|arm policy"
  echo "EXIT_clean1_$L=$CLEAN_RC"
  # The gate, not just the print: a clean arm that does not PASS ends the phase.
  [ "$CLEAN_RC" -eq 0 ] || { echo "PHASE4 GATE FAIL: clean oracle form=$L exited $CLEAN_RC"; exit 1; }
done

echo "##################### PHASE 4c: TEETH oracle — must FAIL and LOCALIZE #####################"
# Expected shape: the prefill QT_NVFP4_V2 GEMM cells stay DIFF=0 (untouched kernel), the P1
# `_sel_v2` cells go 100% differing at exactly x1.5000, and P2/P3 stay DIFF=0 because neither
# calls `_sel_v2`. That is resolution PER PROGRAM, not merely per binary -- a gate that failed
# everywhere would prove far less.
# capture-then-gate (see 4b): this is the one arm whose whole job is a NONZERO exit,
# so reading grep's $? here asserted nothing at all.
set +e
TEETH_OUT=$("$OUT/oracle-teethA" 2>&1)
TEETH_RC=$?
set -e
printf '%s\n' "$TEETH_OUT" | grep -E "cell |decode |PASS|FAIL|oracle=|=> "
echo "EXIT_teethA=$TEETH_RC  (a ZERO here is the alarm: the corruption control must FAIL)"
# The gate: teeth MUST fail (nonzero) AND localize (only the two P1 sel_v2 cells deviate).
[ "$TEETH_RC" -ne 0 ] || { echo "PHASE4 GATE FAIL: the corruption-control arm PASSED — the oracle is toothless (void-gate class)"; exit 1; }
TEETH_DEV=$(printf '%s\n' "$TEETH_OUT" | grep -c "=> DEVIATION" || true)
[ "$TEETH_DEV" -eq 2 ] || { echo "PHASE4 GATE FAIL: teeth deviated in $TEETH_DEV cells, expected exactly 2 (the P1 sel_v2 pair) — the failure did not localize"; exit 1; }

echo "##################### PHASE 4d: md5 census + tree restored #####################"
md5sum "$OUT"/oracle-clean1 "$OUT"/oracle-teethA "$OUT"/oracle-clean2
cd /home/ubuntu/bankv3/src && git status --short && echo "(empty above = the injection was reverted)"
git log --oneline -1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "===== PHASE 4 DONE ====="
