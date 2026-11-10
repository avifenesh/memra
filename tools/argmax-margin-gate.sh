#!/usr/bin/env bash
# argmax-margin-gate — the run-gen prefill-vs-decode argmax gate, CALIBRATED.
# Self-gating (`kind=cmd` in tools/fast-gate/models.tsv): exit 0 = PASS.
#
#   tools/argmax-margin-gate.sh [<model.gguf|hf_dir>] [--prompt <file>] [--window N]
#                              [--max-flips N] [--decode eager|batch] [--floors FILE]
#                              [--device NAME] [--require-floor] [--canary] [--logdir DIR]
#   tools/argmax-margin-gate.sh <model> [--prompt <file>] [--decode eager|batch] --calibrate
#                              [--receipt PATH] [--logdir DIR]
#
# Raw logs land in $MEMRA_GATE_LOGDIR (default $TMPDIR/memra-argmax-margin-gate), NOT in the
# tracked research dir — this runs every battery. Use --logdir research/<lane>/ to keep one.
#
# WHY THIS GATE EXISTS (research/q8-argmax-20260806/VERDICT.md)
# ------------------------------------------------------------
# run-gen carries a hard assert: `forward_last` (batched prefill) and the `decode_step`
# tokenwise loop must produce the same last-position argmax, else "cache threading bug"
# (run_gen.rs:896). That assert is right to exist and its wording is a landmine: on the 27B
# Q8_0 + board-2048 pair it has been red on the 188-SM pod since at least v0.69.0, and there
# is no cache threading bug. The two configs are two legitimate arithmetics; the last
# position of that prompt happens to sit on a THREE-WAY near-tie (top-3 inside 0.05 on a
# logit scale whose median top-2 gap is 1.25), so the two configs pick differently. Proven by
# the f32 oracle (MEMRA_FAST=0) flipping the same pair — no fast kernel is involved — and by
# the same {332,485} pair firing on 3 silicon classes and 3 model sizes.
#
# THE COVERAGE BUG THIS GATE FIXES
# --------------------------------
# The assert inspects ONE position: the last. So which prompts "pass" is decided by whether
# their final token happens to land on a near-tie — luck of prompt length, not a property of
# the engine. Measured (research/q8-argmax-20260806/RESULTS.jsonl, arm P3): the 27B NVFP4 arm
# the release battery calls GREEN carries a config flip at position 2042 whose margin (0.0184)
# is SMALLER than the failing arm's (0.0307). Same mechanism, invisible, shipping green.
# Conversely a MATCH at maxdiff 1.165 (arm L4) sits beside a MISMATCH at 0.4659 — so the
# `logit maxdiff` the gate prints is NOT a severity signal and must never be read as one.
#
# WHAT THIS GATE ASSERTS INSTEAD (the rule since memra#1199,
# docs/decisions/ARGMAX-GATE-GAP-SHIFT-FLOOR.md)
# ------------------------------------------------------------
# Per row the probe gives each program's top-1 id and top-2 margin. A flip (prefill picks i,
# decode picks j) needs p_i - p_j >= margin_p and d_j - d_i >= margin_d, so the two programs
# disagree about i against j by at least margin_p + margin_d. At an agreeing row they disagree
# about the top-1 against the runner-up by |margin_p - margin_d|. That number is the row's GAP
# SHIFT. The FLOOR of a (model, prompt, decode program, device) is the largest gap shift at an
# agreeing row over the whole prompt, measured once by --calibrate and kept in
# tools/argmax-margin-floors.tsv with its row count, the row that set it, the p50, p99 and
# p99.9 of the agreeing rows' gap shift (how loose the maximum is) and its known rows.
#
#   near tie    a flip whose gap shift is inside the floor. Counted against --max-flips.
#   WIDE-FLIP   a flip whose gap shift is above the floor. FAILS at any count: one program
#               holds its token, or both do, by more than the two programs ever differ where
#               they agree. This is the real defect class.
#   WIDE-ROW    an agreeing row whose gap shift is above the floor. FAILS: the programs moved
#               a decision by more than their calibrated difference without crossing it.
#   TOO-MANY-NEAR-TIES   more near ties than the budget. FAILS: a near tie is isolated by
#               nature, a broken kernel disagrees repeatedly.
#   KNOWN-ROW   a calibration can hold flips above its own floor: rows where main's two
#               programs already disagree by more than at any agreeing row of the prompt
#               (memra#1203). The floor row lists them with their recorded shift. A known row
#               inside a judged window is printed and named, and does not fail the cell by
#               itself. It is not counted against the budget.
#   KNOWN-ROW-WIDER   a candidate whose gap shift at a known row exceeds main's recorded
#               shift at that row FAILS.
#   UNCALIBRATED  no floor row for this model, prompt (by sha256), decode program and device.
#               A flip then cannot be classified and FAILS; a table with no flip passes and
#               says UNCALIBRATED, and its agreeing rows are not judged. --require-floor (the
#               release battery passes it) refuses instead. There is no fallback floor and no
#               floor borrowed from another prompt, model or device.
#
# For every flip and every wide row the gate prints the clause, both top-1 ids and margins,
# the spread, the gap shift and the floor, and under it the probe's `detail` line: both
# programs' top-2 ids with their text and logits and each program's logit at the other's
# top-1. A reader examines the row from the log alone.
#
# The verdict is one function, memra_engine::argmax_margin::judge. This wrapper calls it
# through `argmax-margin-probe --judge <log>` on the measured log and on the canary's copy.
#
# The rule this replaced asked that the spread at the contending ids exceed the smaller of the
# two margins ("margin-explained"). The spread at a flip is at least the mean of the two
# margins, so that held for every flip and could not fail: the only working check was the
# count. A wide flip at one row in twelve passed.
#
# CALIBRATION. `--calibrate` runs the probe over every row of the prompt (window `all`) and
# prints the floors-table row; it judges nothing. Bank its log under research/<lane>/ and add
# the row with that path as its receipt. A lane that gates on its own prompt calibrates the
# base binary (main's) on that prompt first and passes its own table with --floors. When a
# decision record accepts a change of numeric class, the same PR recalibrates and shows the
# old and the new floor rows (docs/TESTING.md).
#
# THE BATCHED SERVING PRIME. The probe also runs `prime_cache` (what seeds generation in
# generate and serving) over the whole prompt and prints one `PRIME-GATE:` line, the class
# `forward::prime_gate_verdict` gives its last-position logits against the tokenwise
# reference: MATCH, FLIP-NEARTIE (a flip at a tokenwise margin inside the calibrated bound,
# reported and non-fatal, as in run-gen) or MISMATCH-STRUCTURED (drift past the calibrated
# bounds, or a wide-margin flip), which FAILS here. `SKIP reason=...` passes and is printed.
# A missing or malformed line fails like a malformed table: the release battery runs this
# gate and not run-gen, and run-spec primes reference and candidate through the same
# prime_cache, so this line is the battery's only prime-vs-tokenwise check. The canary also
# injects a MISMATCH-STRUCTURED line and requires the prime reader to reject it.
#
# THE SERVED BATCHED DECODE CELL (--decode batch). The default cell's decode side is
# `decode_step` (decode_step_h), the eager program. The server's batched scheduler decodes every
# non-speculative row of a batch-served model through decode_step_batch_sampled_lean_masked at
# every width, B=1 included, and that body carries a decode-config FP-composition gap against
# decode_step_h. So the default cell can stay green while a change to the batched body or its
# epilogue moves served tokens. (Speculative rows, the default MEMRA_SERVE_SPEC route for
# spec-eligible requests on a model with a draft head, decode through the spec path, which
# run-spec gates.) With --decode batch the probe makes the server's own call for one
# unconstrained greedy row (greedy device sample, lean logits read from the cache's device park)
# at every prompt position. The table, the prefill reference, the flip budget and the margin
# rules are the default cell's. Instead of the PRIME-GATE line (the default cell owns the prime)
# the probe prints one `SERVED-ARGMAX: rows=N device_mismatch=M` line: M counts window steps
# whose device-sampled token differs from the host argmax of the same served row. M > 0, or a
# missing or malformed line, FAILS: a probe that ignores --decode runs the eager program and
# prints no such line. The canary also injects an M=1 line and requires the served reader to
# reject it. A model the scheduler decodes through the per-session eager body
# (HybridModel::served_decode_is_eager, the worker's own routing read) has no served batched
# program: the probe then steps decode_step, the program that serves it, and prints
# `SERVED-ARGMAX: n/a reason=eager-routed`. The wrapper prints
# `SERVED-SUMMARY served=<MATCH|MISMATCH|EAGER-ROUTED>`; EAGER-ROUTED passes on the table's
# verdict alone.
#
# TEETH: --canary INJECTS A BREAK rather than relabelling an expectation (the trap documented
# in tools/chunk-invariance-gate.sh's header, learned the hard way on lane/chunkinv-flip).
# It runs the real probe, then FAULT-INJECTS into a copy of the measured table one row of the
# class this gate exists to catch: a flip where both programs hold their token by 1000 logits
# (gap shift 2000, above any floor a calibration can set) and requires the judge to name that
# row WIDE-FLIP, or UNCALIBRATED-FLIP when the cell has no floor row. This is a mutation test
# of the judge on its own parse path, so it fires whether or not the model under test
# produces a natural flip. The synthetic near tie, the budget overrun, the wide row and the
# old rule's tautology are pinned on CPU: `cargo test -p memra-engine --lib argmax_margin` and
# `cargo test -p memra-engine --bin argmax-margin-probe`.
#   Rejected canary designs, and why:
#   (a) flip only the EXPECTED verdict -> re-runs the identical data, perfectly correlated
#       with the default gate, proves nothing (the chunkinv trap).
#   (b) raise a margin floor above the prompt's median -> only touches rows that ALREADY
#       flipped, so on a clean model (0 flips) there is nothing to reject.
#   (c) the row the canary injected until memra#1199 (margins 5.0, spread 0.1): no real flip
#       can have a spread below its mean margin, so it proved only that the comparator
#       rejected a row that cannot occur.
set -uo pipefail
cd "$(dirname "$0")/.."
PROBE=${CARGO_TARGET_DIR:-./target}/release/argmax-margin-probe

MODEL="${1:-}"
if [ -n "$MODEL" ] && [ "${MODEL#--}" != "$MODEL" ]; then MODEL=""; else shift 2>/dev/null || true; fi
EXPLICIT_MODEL=0
[ -z "$MODEL" ] || EXPLICIT_MODEL=1
PROMPT=research/e2e/prompts/board-2048.txt
WINDOW=12
MAX_FLIPS=""    # unset = model-calibrated default (resolved after MODEL below)
FLOORS=tools/argmax-margin-floors.tsv   # the calibrated floor rows
DEVICE=""            # floor key; unset = the name nvidia-smi reports for the probe's device
REQUIRE_FLOOR=0      # 1 = a cell with no floor row is refused (the release battery)
CALIBRATE=0          # 1 = measure the whole prompt and print its floors row; no verdict
RECEIPT=""           # --calibrate: the receipt path written into the row (default: the log)
DECODE=eager         # decode side of the table: eager (decode_step) or batch (the served program)
CANARY=0
PRIME=""; SERVED=""
while [ $# -gt 0 ]; do
    case "$1" in
        --prompt|--window|--max-flips|--decode|--logdir|--floors|--device|--receipt)
            [ $# -ge 2 ] && [[ "$2" != --* ]] || { echo "FAIL: $1 requires a value"; exit 2; } ;;
    esac
    case "$1" in
        --prompt)       PROMPT="$2"; shift 2 ;;
        --window)       WINDOW="$2"; shift 2 ;;
        --max-flips)    MAX_FLIPS="$2"; shift 2 ;;
        --decode)       DECODE="$2"; shift 2 ;;
        --floors)       FLOORS="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
        --receipt)      RECEIPT="$2"; shift 2 ;;
        --logdir)       MEMRA_GATE_LOGDIR="$2"; shift 2 ;;
        --require-floor) REQUIRE_FLOOR=1; shift ;;
        --calibrate)    CALIBRATE=1; shift ;;
        --canary)       CANARY=1; shift ;;
        *)              echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ "$WINDOW" =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: window must be a positive integer"; exit 2; }
case "$DECODE" in eager|batch) ;; *) echo "FAIL: --decode takes eager or batch"; exit 2 ;; esac
if [ $CALIBRATE = 1 ]; then
    [ $CANARY = 0 ] || { echo "FAIL: --calibrate measures a floor; it takes no --canary"; exit 2; }
    WINDOW=all
fi
if [ "$EXPLICIT_MODEL" = 1 ] && [ ! -f "$MODEL" ] && [ ! -d "$MODEL" ]; then
    echo "FAIL: requested model does not exist: $MODEL"; exit 1
fi

# Default model: the fast-gate's Q8_0 row if present, else the first probe model available.
if [ -z "$MODEL" ]; then
    for cand in \
        /root/models/Qwen3.6-27B-Q8_0.gguf \
        /data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf \
        /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
    do [ -f "$cand" ] && { MODEL="$cand"; break; }; done
fi
# SKIP (not PASS, not FAIL) when the inputs are absent: exiting 0 silently on a missing
# artifact is indistinguishable from a real pass by exit code alone — the hole that reported
# chunkinv/chunkinvc as "PASS (0s)" on the 188-SM pod during the v0.70.0 battery. The
# fast-gate cmd runner greps "^<name>: *SKIP", so emit exactly that shape.
[ -n "$MODEL" ] && { [ -f "$MODEL" ] || [ -d "$MODEL" ]; } || { echo "argmax-margin-gate: SKIP (no probe model on this rig)"; exit 0; }
# PER-MODEL FLIP-BUDGET CALIBRATION ROWS (explicit --max-flips always wins).
# gemma-4-31B (added 2026-08-17, ship-lane merge diligence): the banked board-2048
# margin measurement (evidence/gemma-ship-20260817/zoofusion/gate-*.log) shows decode
# margins p10 0.293 / p50 2.28 with config spreads at the contending ids up to 3.75 —
# ~5 of 12 tail positions are near-tie coins, expectation ~2.5 flips per 12-window.
# Budget 3 per 12-window, CALIBRATED FROM MEASURED MARGINS (not loosened-to-green).
# All other models keep the original 1. (The "flippable" test of these rows, a margin
# below the spread, holds at every flip and many agreeing rows; since memra#1199 the
# budget counts near ties, flips inside the cell's floor, and a flip above it fails.)
# gemma-4-26B-A4B (added 2026-09-20, lane/exec-p1a-gemma-prime, research/exec-p1a-gemma-prime-
# 20260919/raw/26b-a4b + raw/26b-routerfix): the same board-2048 window read 6 of 12 and 5 of 12
# positions arithmetically flippable (min(margin_p, margin_d) < config spread) on two binaries
# whose DECODE margins were identical (min 0.014 / p10 0.342 / p50 0.695, spreads to 3.02) — the
# 31B's coin distribution on a 128-expert MoE; expectation ~2.5 flips per 12-window. The runs
# landed 1 and 2 flips; the decision position agreed both times.
# Budget 3 per 12-window from those margins, same rule as the 31B row above.
# PER-PROMPT ROWS: a longer prompt is a different tail. Qwen3.8-27B NVFP4 on the 16k prompt (p4-16k,
# 23,476 tokens; added 2026-11-07, research/gb10-prefill-20261106/gemm-v2-gates-main-unit1/
# argmax16k-*): 5 of the 12 tail positions are arithmetically flippable (min(margin_p,
# margin_d) < config spread: 0.0121 < 0.0713, 0.0246 < 0.0742, 0.0305 < 0.1382,
# 0.0146 < 0.1187, 0.0960 < 0.2410), decode margins p10 0.0146 / p50 0.7378, spreads to
# 0.3621: an expectation of ~2.5 flips per 12-window. main c2e04719b6 and the F8F4 v2 head land 3
# flips each with byte-identical probe tables, the decision position agreeing. Budget 3 per
# 12-window from those margins, the gemma rows' rule. The default 2,048-token prompt keeps the
# model's row. Measured on the NVFP4-Q5K-mtp artifact only, so the row matches the NVFP4
# artifacts. The wrapper prints the row it used.
BUDGET_ROW="--max-flips"
if [ -z "$MAX_FLIPS" ]; then
    case "$(basename "$MODEL")|$(basename "$PROMPT")" in
        Qwen3.8-27B-NVFP4*"|p4-16k.txt") MAX_FLIPS=$(( (3 * WINDOW + 11) / 12 )); BUDGET_ROW="Qwen3.8-27B NVFP4 on p4-16k (3/12-window)" ;;
    esac
fi
if [ -z "$MAX_FLIPS" ]; then
    case "$(basename "$MODEL")" in
        gemma-4-31B*) MAX_FLIPS=$(( (3 * WINDOW + 11) / 12 )); BUDGET_ROW="gemma-4-31B (3/12-window)" ;;
        gemma-4-26B*) MAX_FLIPS=$(( (3 * WINDOW + 11) / 12 )); BUDGET_ROW="gemma-4-26B (3/12-window)" ;;
        *)            MAX_FLIPS=1; BUDGET_ROW="default" ;;
    esac
fi
[[ "$MAX_FLIPS" =~ ^[0-9]+$ ]] || { echo "FAIL: max-flips must be a nonnegative integer"; exit 2; }
[ -f "$PROMPT" ] || { echo "FAIL: prompt $PROMPT missing"; exit 1; }
[ -f "$FLOORS" ] || { echo "FAIL: floors table $FLOORS missing"; exit 1; }
if [ ! -x "$PROBE" ]; then
    if [ "$EXPLICIT_MODEL" = 1 ]; then
        echo "FAIL: build argmax-margin-probe (cargo build --release -p memra-engine --bin argmax-margin-probe) before running the requested gate"; exit 1
    fi
    echo "argmax-margin-gate: SKIP (build argmax-margin-probe to enable)"; exit 0
fi

# Logs go to a scratch dir, NOT into the tracked research dir: this gate runs on every
# fast-gate invocation, and writing under research/ made each battery run dirty the worktree
# (and committed the canary's .injected scratch table once already). Point --logdir at the
# lane dir when you want a receipt to keep. Same convention as chunk-invariance-gate.sh.
D="${MEMRA_GATE_LOGDIR:-${TMPDIR:-/tmp}/memra-argmax-margin-gate}"; mkdir -p "$D"
LOG="$D/gate-$(basename "${MODEL%.gguf}")-$(basename "${PROMPT%.txt}")$([ "$DECODE" = batch ] && echo .batch)$([ $CANARY = 1 ] && echo .canary)$([ $CALIBRATE = 1 ] && echo .calibrate).log"
# The printed raw-log path is relative to MEMRA_GATE_LOG_ROOT when the log lies under it: the
# release battery sets it to its evidence dir, so banked evidence names no user or mount.
LOG_SHOWN="$LOG"
if [ -n "${MEMRA_GATE_LOG_ROOT:-}" ]; then
    case "$LOG" in "${MEMRA_GATE_LOG_ROOT%/}"/*) LOG_SHOWN="${LOG#"${MEMRA_GATE_LOG_ROOT%/}"/}" ;; esac
fi

# The floor key: the model and prompt names, the prompt's bytes, the decode program and the
# device the probe runs on (its first visible one), as nvidia-smi names it.
MODEL_NAME=$(basename "${MODEL%.gguf}")
PROMPT_NAME=$(basename "${PROMPT%.txt}")
PROMPT_SHA=$(sha256sum "$PROMPT" | cut -d' ' -f1)
if [ -z "$DEVICE" ]; then
    DEVICE=$(nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "${CUDA_VISIBLE_DEVICES%%,*}"} \
        --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
fi
[ -n "$DEVICE" ] || DEVICE=unknown
KEY=(--model "$MODEL_NAME" --prompt "$PROMPT_NAME" --prompt-sha256 "$PROMPT_SHA" --decode "$DECODE" --device "$DEVICE")

echo "argmax-margin-gate: model=$(basename "$MODEL") prompt=$(basename "$PROMPT") window=$WINDOW decode=$DECODE canary=$CANARY"
echo "  device: $DEVICE"
[ $CALIBRATE = 1 ] || echo "  flip budget: $MAX_FLIPS (row: $BUDGET_ROW)"
PROBE_ARGS=("$MODEL" "$PROMPT" "$WINDOW")
[ "$DECODE" = eager ] || PROBE_ARGS+=(--decode "$DECODE")
# Raw log FIRST, parse second (never let a pipe swallow the failure text).
"$PROBE" "${PROBE_ARGS[@]}" > "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    # A VRAM shortage says nothing about argmax margins. On a shared rig another lane can hold
    # the card (seen live: "Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, ...)" with 21.6 GB held
    # by a concurrent job), and failing the exactness battery for that is a flake, not a find.
    # SKIP with the cause QUOTED — an inferred cause is not a cause (AGENTS.md evidence rules).
    if grep -qaE 'CUDA_ERROR_OUT_OF_MEMORY|out of memory' "$LOG"; then
        echo "argmax-margin-gate: SKIP (GPU out of memory — not an exactness signal)"
        grep -aE 'CUDA_ERROR_OUT_OF_MEMORY|out of memory' "$LOG" | head -2 | sed 's/^/  quoted: /'
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
            | sed 's/^/  concurrent-GPU: /'
        exit 0
    fi
    echo "  FAIL: probe exited $rc"; tail -20 "$LOG"; exit 1
fi

# Validate the measured table before fault injection: empty or malformed output cannot
# become PASS, including in canary mode. This does not relax any numerical threshold.
if ! awk -v expected="$([ $CALIBRATE = 1 ] && echo 0 || echo "$WINDOW")" '
    /^[0-9]+[ \t]+[0-9]+/ {
        rows++
        if (seen[$1]++ || $4 !~ /^[0-9]+$/ || ($7 != "yes" && $7 != "NO")) bad++
        for (i=3; i<=6; i++) {
            if (i == 4) continue
            if ($i !~ /^[+-]?[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$/ || $i+0 < 0) bad++
            if (sprintf("%g", $i+0) ~ /[iI][nN][fF]|[nN][aA][nN]/) bad++
        }
    }
    END { exit ((expected ? rows != expected : rows == 0) || bad != 0) }
' "$LOG"; then
    echo "  FAIL: probe table is missing, malformed, non-finite, or has the wrong row count"
    echo "  raw: $LOG_SHOWN"
    exit 1
fi

# The batched-prime line: exactly one, class read from it. Prints MATCH, FLIP-NEARTIE,
# MISMATCH-STRUCTURED, SKIP:<reason>, MISSING or MALFORMED.
prime_judge() {
    awk '
        /^PRIME-GATE: / { n++; line = $0 }
        END {
            if (n != 1) { print "MISSING"; exit }
            if (line ~ /^PRIME-GATE: SKIP reason=[a-z-]+$/) {
                sub(/^PRIME-GATE: SKIP reason=/, "", line); print "SKIP:" line; exit
            }
            if (!match(line, /^PRIME-GATE: class=[A-Z-]+ /)) { print "MALFORMED"; exit }
            c = substr(line, 19, RLENGTH - 19)
            if (c == "MATCH" || c == "FLIP-NEARTIE" || c == "MISMATCH-STRUCTURED") print c
            else print "MALFORMED"
        }' "$1"
}
# The served-batch line (--decode batch): exactly one, with the measured window. Prints MATCH,
# MISMATCH (a device-sampled token differs from the host argmax of its row), EAGER-ROUTED (the
# scheduler decodes this model eagerly, so the table measured decode_step), MISSING or MALFORMED.
served_judge() {
    awk -v rows="$([ $CALIBRATE = 1 ] && echo 0 || echo "$WINDOW")" '
        /^SERVED-ARGMAX: / { n++; line = $0 }
        END {
            if (n != 1) { print "MISSING"; exit }
            if (line == "SERVED-ARGMAX: n/a reason=eager-routed") { print "EAGER-ROUTED"; exit }
            if (line !~ /^SERVED-ARGMAX: rows=[0-9]+ device_mismatch=[0-9]+$/) { print "MALFORMED"; exit }
            split(line, f, "[ =]")
            if (rows + 0 && f[3] + 0 != rows + 0) { print "MALFORMED"; exit }
            print (f[5] + 0 == 0 ? "MATCH" : "MISMATCH")
        }' "$1"
}
if [ "$DECODE" = batch ]; then
    SERVED=$(served_judge "$LOG")
    case "$SERVED" in
        MISSING|MALFORMED)
            echo "  FAIL: probe SERVED-ARGMAX line is $SERVED (the served batch arm did not report;"
            echo "        a probe that ignores --decode runs the eager program)"
            echo "  raw: $LOG_SHOWN"
            exit 1 ;;
    esac
else
    PRIME=$(prime_judge "$LOG")
    case "$PRIME" in
        MISSING|MALFORMED)
            echo "  FAIL: probe PRIME-GATE line is $PRIME (the batched-prime arm did not report)"
            echo "  raw: $LOG_SHOWN"
            exit 1 ;;
    esac
fi

# --calibrate ends here: the floors row of this table, and no verdict.
if [ $CALIBRATE = 1 ]; then
    "$PROBE" --judge "$LOG" --calibrate --receipt "${RECEIPT:-$LOG_SHOWN}" "${KEY[@]}" | sed 's/^/  /'
    rc=${PIPESTATUS[0]}
    [ "$DECODE" = batch ] && echo "  SERVED-SUMMARY served=$SERVED" || echo "  PRIME-SUMMARY prime=$PRIME"
    echo "  raw: $LOG_SHOWN"
    if [ "$rc" -ne 0 ]; then echo "  FAIL: the table gives no floor"; exit 1; fi
    echo "  CALIBRATED: add the FLOOR-ROW line to $FLOORS with its banked log as the receipt"
    exit 0
fi

# The canary appends ONE synthetic row of the defect class to the table the judge parses: a
# flip where both programs hold their token by 1000 logits. No floor covers it, so a judge
# that reads the margins MUST name it WIDE-FLIP (UNCALIBRATED-FLIP on a cell with no floor).
PARSE="$LOG"
if [ $CANARY = 1 ]; then
    PARSE="$LOG.injected"
    cp "$LOG" "$PARSE"
    printf '%-8s %-13s %-16s %-12s %-16s %-16s %s\n' \
        9999 111111 1.000000000e3 222222 1.000000000e3 1.000000000e3 "NO <-- FLIP" >> "$PARSE"
    echo "  [canary] injected a WIDE flip (pos 9999, margins 1000 and 1000, gap shift 2000)"
    echo "           that no floor covers; the judge must reject that row"
    if [ "$DECODE" = batch ]; then
        grep -v '^SERVED-ARGMAX: ' "$LOG" > "$LOG.served-injected"
        echo "SERVED-ARGMAX: rows=$WINDOW device_mismatch=1" >> "$LOG.served-injected"
        CANARY_SERVED=$(served_judge "$LOG.served-injected")
        echo "  [canary] injected a served-token mismatch line; the reader saw: $CANARY_SERVED"
    else
        grep -v '^PRIME-GATE: ' "$LOG" > "$LOG.prime-injected"
        echo "PRIME-GATE: class=MISMATCH-STRUCTURED bp_argmax=1 tw_argmax=2 tw_margin=5.0e0 bp_margin=5.0e0 maxdiff=9.0e1" \
            >> "$LOG.prime-injected"
        CANARY_PRIME=$(prime_judge "$LOG.prime-injected")
        echo "  [canary] injected a MISMATCH-STRUCTURED batched-prime line; the reader saw: $CANARY_PRIME"
    fi
fi

# The verdict: memra_engine::argmax_margin::judge on the table, through the probe's --judge.
# Exit 0 passes, 1 fails, anything else is a table or floors file it could not read.
verdict=$("$PROBE" --judge "$PARSE" --max-flips "$MAX_FLIPS" --floors "$FLOORS" "${KEY[@]}" 2>&1)
jrc=$?
echo "$verdict" | sed 's/^/  /'
if [ $jrc -ne 0 ] && [ $jrc -ne 1 ]; then
    echo "  FAIL: the judge could not read the table or the floors table (exit $jrc)"
    echo "  raw: $LOG_SHOWN"
    exit 1
fi
if ! echo "$verdict" | grep -q '^SUMMARY flips=[0-9]* bad=[0-9]* '; then
    echo "  FAIL: the judge printed no SUMMARY line"
    echo "  raw: $LOG_SHOWN"
    exit 1
fi
UNCALIBRATED=0
echo "$verdict" | grep -q '^FLOOR: none ' && UNCALIBRATED=1
if [ "$DECODE" = batch ]; then
    echo "  SERVED-SUMMARY served=$SERVED"
    grep -m1 '^SERVED-ARGMAX: ' "$LOG" | sed 's/^/    /'
else
    echo "  PRIME-SUMMARY prime=$PRIME"
    grep -m1 '^PRIME-GATE: ' "$LOG" | sed 's/^/    /'
fi

if [ $CANARY = 1 ] && [ "$DECODE" = eager ] && [ "${CANARY_PRIME:-}" != MISMATCH-STRUCTURED ]; then
    echo "  CANARY FAIL: the injected MISMATCH-STRUCTURED prime line was read as ${CANARY_PRIME:-nothing};"
    echo "               the batched-prime reader has no teeth."
    exit 1
fi
if [ $CANARY = 1 ] && [ "$DECODE" = batch ] && [ "${CANARY_SERVED:-}" != MISMATCH ]; then
    echo "  CANARY FAIL: the injected served-token mismatch line was read as ${CANARY_SERVED:-nothing};"
    echo "               the served-batch reader has no teeth."
    exit 1
fi
if [ $CANARY = 1 ]; then
    if [ $jrc -eq 1 ] && echo "$verdict" | grep -qE '^(WIDE-FLIP|UNCALIBRATED-FLIP) pos=9999 '; then
        echo "  CANARY PASS: the judge rejected the injected wide flip as required (teeth)"
        echo "  raw: $LOG_SHOWN"
        exit 0
    fi
    echo "  CANARY FAIL: the injected wide flip did NOT fail the verdict: the judge is not"
    echo "               reading its own margins; this gate has no teeth."
    exit 1
fi
if [ $UNCALIBRATED = 1 ] && [ $REQUIRE_FLOOR = 1 ]; then
    echo "argmax-margin-gate: UNCALIBRATED (no floor row in $FLOORS for model=$MODEL_NAME prompt=$PROMPT_NAME sha256=$PROMPT_SHA decode=$DECODE device=$DEVICE; calibrate with --calibrate)"
    echo "  raw: $LOG_SHOWN"
    exit 1
fi
if [ "$PRIME" = MISMATCH-STRUCTURED ]; then
    echo "  FAIL: the batched serving prime (prime_cache) diverges structurally from the tokenwise reference"
    echo "  raw: $LOG_SHOWN"
    exit 1
fi
if [ "$SERVED" = MISMATCH ]; then
    echo "  FAIL: the served batch program's device-sampled token differs from the host argmax of its own row"
    echo "  raw: $LOG_SHOWN"
    exit 1
fi
if [ $jrc -eq 0 ]; then
    if [ $UNCALIBRATED = 1 ]; then
        echo "  PASS: UNCALIBRATED, no flip. No floor row for this model, prompt, decode program and"
        echo "        device, so agreeing rows were not judged (calibrate with --calibrate)"
    else
        echo "  PASS: no wide flip, no wide row, no known row wider than recorded, near ties within the budget"
    fi
    echo "  raw: $LOG_SHOWN"
    exit 0
fi
echo "  FAIL: a wide flip, a wide row, a known row wider than recorded, a flip with no floor to judge it by, or too many near ties (the lines above name each row)"
echo "  raw: $LOG_SHOWN"
exit 1
