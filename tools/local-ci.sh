#!/usr/bin/env bash
# memra local CI — the real gate (GitHub CI is compile-only; the rig is the test machine).
#
#   tools/local-ci.sh                correctness stage only (~3 min)
#   tools/local-ci.sh --perf         correctness + full perf battery (~15 min)
#   tools/local-ci.sh --perf-quick   correctness + gemma-31B cells only (~6 min)
#
# Correctness stage: kernel-check, run-gen argmax gate, run-spec K=1..8 self-consistency,
# Gemma stream agreement, and VERIFY-GATE logit maxdiff at depth — the standing exactness
# battery, one command.
#
# Perf stage: the cell battery from research/tune-data/perf-cells.json. Every spec cell
# records tok/s + ACCEPTANCE + tok/round — the drift class that silently cost the spec
# board 2026-07-13..15 (acceptance 1.000 -> 0.669 across ~40 green-gated commits).
# Rows append to research/tune-data/perf-ci.jsonl; each cell is verdicted against the
# rolling median of its last N rows: FAIL on >3% tok/s drop or >0.05 acceptance drop,
# WARN on >1.5%. A FAIL exits non-zero — treat it like a red test.
#
# Contributor machines: cells whose model file is absent are SKIPPED cleanly; the
# correctness stage runs wherever a GPU + at least one model exists. Set
# MEMRA_MODELS_DIR to your model root (default /data/ai-ml/hf-models).
#
# MEMRA_CI_DIRTY_WAIT (default 600s): how long a cell waits for a co-resident GPU process
# to leave before recording its row as window_clean=false. Latched after the first cell
# that outwaits it — a permanently-co-resident process (an owner service holding an idle
# CUDA context) must not turn the perf stage into an unbounded hang.
#
# Window discipline (recorded per row, enforced where it can be): no other compute
# process on the GPU (co-resident engines spill experts and read 10x low), host load
# sane, power profile noted (pin it with gpu-full-power on|off — profiles pair fairly
# only against themselves).
#
# GPU LOCK (lane/localci-lock-20260821): a bare invocation is safe on a shared rig. This
# script takes ${MEMRA_CI_LOCK:-/tmp/memra-5090.lock} ONCE, before the first GPU-touching
# step, and holds it on an fd for the rest of the run — so prime-gate, run-spec, gemma-gate,
# run-gen, kernel-check, decode-batch-gate and every serving gate run inside one exclusive
# window (previously prime-gate and friends ran LOCK-LESS: a foreign 10GB co-resident OOMed
# a lane's battery on 2026-08-21). If another holder is active we WAIT loudly (default 2h),
# never run lock-less, and a timeout is a loud non-zero exit. Inner steps do not re-take the
# whole-run lock (MEMRA_CI_LOCK_HELD=1 is exported), and spec-on-cache-hit-gate's own
# per-boot flock is redirected to a distinct inner file (MEMRA_GPU_LOCK) — wrapping this
# script in an outer `flock /tmp/memra-5090.lock` used to SELF-DEADLOCK on exactly that
# gate. The fd-held lock releases on ANY exit, SIGKILL included — no stale lock possible.
# Mechanics tested CPU-only by tools/test_local_ci_lock.sh (wired below, fatal).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MEMRA_MODELS_DIR:-/data/ai-ml/hf-models}"
MANIFEST=research/tune-data/perf-cells.json
OUT=research/tune-data/perf-ci.jsonl
MODE="${1:---correctness}"

command -v jq >/dev/null || { echo "local-ci: jq required"; exit 2; }

# ---- whole-run GPU lock (see header) ----
CI_LOCK="${MEMRA_CI_LOCK:-/tmp/memra-5090.lock}"
CI_LOCK_WAIT="${MEMRA_CI_LOCK_WAIT:-7200}"
acquire_gpu_lock() {
    if [ "${MEMRA_CI_LOCK_HELD:-0}" = "1" ]; then
        # A wrapper already holds the rig lock (the lanes' documented outer-lock pattern).
        # Do NOT flock again: the same file is not re-entrant across processes and a second
        # take here is the self-deadlock this lane exists to kill.
        echo "local-ci: GPU lock externally held (MEMRA_CI_LOCK_HELD=1) — not re-acquiring"
    else
        # fd-based, NOT `flock <file> <cmd>`: the fd (inherited by every child) is the lock
        # lifetime. Kernel drops it when the last holder fd closes — a mid-run kill releases
        # it, and an orphaned server that is still on the GPU correctly keeps it held.
        # TRAP:sccache-inherits-gpu-flock (two deadlocks, 2026-08-31/09-01): a build under
        # this lock SPAWNS the sccache daemon, which inherits fd 9 and outlives any kill of
        # the run — /proc/locks then names a dead pid and the rig wedges. Pre-warm the
        # daemon BEFORE taking the fd so it never inherits it; the intentional inheritance
        # (orphaned GPU-holding servers keep the lock) stays.
        command -v sccache >/dev/null 2>&1 && sccache --start-server >/dev/null 2>&1 || true
        exec 9>"$CI_LOCK"
        if ! flock -n 9; then
            echo "local-ci: GPU lock $CI_LOCK is HELD by another run — WAITING (up to ${CI_LOCK_WAIT}s; set MEMRA_CI_LOCK_WAIT to change). Never running lock-less."
            flock -w "$CI_LOCK_WAIT" 9 || {
                echo "local-ci: FAIL — $CI_LOCK still held after ${CI_LOCK_WAIT}s; refusing to run GPU steps lock-less" >&2
                exit 1
            }
        fi
        echo "local-ci: GPU lock acquired ($CI_LOCK) — held for the whole run"
        export MEMRA_CI_LOCK_HELD=1
    fi
    # Inner gates that flock per server boot (spec-on-cache-hit-gate) must not contend the
    # lock this run already holds — that contention is the receipted self-deadlock (their
    # flock -w 300 times out against our own hold -> "server died during boot", empty log).
    # Redirect them to a distinct inner file. This is NOT a third rig lock name (CLAUDE.md
    # lock-names law): it grants no rig-wide exclusion and is only ever exported INSIDE an
    # exclusive hold of the canonical lock; standalone gate runs keep the canonical default.
    # A caller's explicit private MEMRA_GPU_LOCK (the lanes' green pattern) is respected.
    if [ -z "${MEMRA_GPU_LOCK:-}" ] || [ "${MEMRA_GPU_LOCK:-}" = "$CI_LOCK" ]; then
        export MEMRA_GPU_LOCK="${CI_LOCK}.inner"
    fi
    echo "local-ci: inner gate lock seam MEMRA_GPU_LOCK=$MEMRA_GPU_LOCK"
}

# LOCK-SMOKE self-test door (tools/test_local_ci_lock.sh): exercise the REAL acquisition
# path above — wait, hold, env exports, release-on-kill — without building or touching the
# GPU. Guarded by an env only the harness sets; a bare run never enters this branch.
if [ "${MEMRA_CI_LOCK_SMOKE:-0}" = "1" ]; then
    acquire_gpu_lock
    echo "local-ci: LOCK-SMOKE holding (MEMRA_GPU_LOCK=$MEMRA_GPU_LOCK)"
    sleep "${MEMRA_CI_LOCK_SMOKE_HOLD:-3}"
    echo "local-ci: LOCK-SMOKE done"
    exit 0
fi
# Build UNCONDITIONALLY (cargo incremental = no-op when fresh). The old
# `[ -x BIN ] || cargo build` idiom silently ran STALE binaries whenever they merely
# existed — the battery is a gate, and a gate that can run a week-old binary is a rotted
# gate (H100 lane law 3). This builds the full release set so serve-smoke's memra-server,
# the engine bins, and the gate tools all match HEAD before a single check runs.
cargo build --release || { echo "local-ci: build FAILED"; exit 1; }

# CLIPPY GATE (lane/clippy-zero-restore-20260901, wired on the PR #87 review's finding).
# CPU-only, pre-lock, on the target dir the build above just warmed. Mirrors ci.yml's gate
# exactly so "local-ci green" keeps implying "pushable": the workspace is clippy-zero and
# -D warnings holds it there; without this line a local-ci-green tree bounces off CI on a
# lint 20 minutes after the push. Steady-state cost is small (cargo replays cached
# diagnostics on an unchanged tree); first run after an edit pays the lint pass. -j8 per
# the rig CPU cap. MEMRA_CI_CLIPPY=0 skips, announced — same door pattern as the stages
# below, and like them it is not an engine flag (check-flags censuses runtime .rs only).
# VERSION AGREEMENT: ci.yml pins its clippy toolchain to the workspace rust-version
# (Cargo.toml), and this line lints with the rig's default toolchain — keep the rig on
# that same version, and bump rust-version + the ci.yml pin + the rig together in one
# lane (the gate's first CI run redded on a stable that moved overnight; receipt in the
# ci.yml toolchain-pin comment).
cpu_chain() {
    if [ "${MEMRA_CI_CLIPPY:-1}" = "1" ]; then
        echo "== local-ci: clippy gate (-D warnings) =="
        cargo clippy --release --all-targets -j8 -- -D warnings \
            || { echo "local-ci: clippy gate FAILED — the workspace stays clippy-zero (fix or #[allow] with a stated reason)"; return 1; }
    else
        echo "local-ci: clippy gate SKIPPED (MEMRA_CI_CLIPPY=0)" >&2
    fi
    echo "== local-ci: memra-server HTTP-surface unit suite =="
    if ! cargo test --release -p memra-server -j8; then
        echo "local-ci: memra-server unit suite FAILED"; return 1
    fi
    # ENGINE LIB SUITE (memra#18, ci-diet lane 2026-09-02). vision::tests and every other
    # memra-engine lib test ran NOWHERE: ci.yml ran `cargo test -p memra-engine cpu_experts
    # --lib`, a NAME FILTER. The CPU-safe part of the suite runs here (358 tests, measured with
    # CUDA_VISIBLE_DEVICES empty on 2026-09-02: 358 passed, 0 failed, 3 ignored); the three
    # `#[ignore]` tests that need a CUDA device run below, on the GPU chain, under the lock.
    echo "== local-ci: memra-engine lib suite (CPU-safe part) =="
    if ! cargo test --release -p memra-engine --lib -j8; then
        echo "local-ci: memra-engine lib suite FAILED"; return 1
    fi
    # GGUF ARTIFACT-PRESENT SKIP CENSUS (rehomed 2026-09-02 from tools/validate-h100.sh,
    # deleted with the Hopper CI lane; PR #73 review caught the orphaning — the deleted
    # battery was the only executor that ran the artifact-gated memra-gguf tests WITH their
    # artifacts present, and ci.yml's budget-12 arm runs where every one of them skips by
    # design). Budget 10 is MEASURED on this rig (2026-09-02: 212 passed, 10 skipped —
    # ckpt/twin, Hy3-repack and /tmp/iq3s_raw.bin artifacts not staged here), stated out
    # loud per the ci.yml precedent, and every skip still prints and still counts. An 11th
    # skip means an artifact regressed further: refuse it. MEMRA_CI_GGUF=0 skips.
    if [ "${MEMRA_CI_GGUF:-1}" = "1" ]; then
        echo "== local-ci: memra-gguf artifact-present skip census =="
        if ! MEMRA_CI_GGUF_SKIP_BUDGET=10 python3 tools/skip-census.py run \
                --budget-var MEMRA_CI_GGUF_SKIP_BUDGET --min-passed 200 \
                -- cargo test --release -j8 -p memra-gguf --lib; then
            echo "local-ci: gguf artifact-present skip census FAILED"; return 1
        fi
    else
        echo "local-ci: gguf skip census SKIPPED (MEMRA_CI_GGUF=0)" >&2
    fi
}
# OVERLAP (ci-diet lane 2026-09-02). The three steps above are CPU-bound and touch no GPU;
# every gate below the lock is GPU-bound and leaves most cores idle. Serial, the correctness
# stage paid their sum; overlapped, it pays the larger of the two chains. Only in the
# correctness mode: a perf mode serializes everything, because a compile sharing the box with
# a timing cell is exactly the co-resident noise the perf rows refuse (window_clean). The chain
# is forked BEFORE acquire_gpu_lock so it never inherits fd 9 (the sccache trap's shape).
# MEMRA_CI_OVERLAP=0 restores the serial order, announced. Joined by join_cpu_chain below;
# a chain failure fails the run with the chain's own log.
CPU_LOG=""
CPU_PID=""
# A GPU gate that exits 1 must not leave the chain compiling on an idle rig: kill the chain's
# children (cargo) and the chain, keep its log for the post-mortem. A joined chain has CPU_PID
# cleared, so a normal exit does nothing here.
trap 'if [ -n "${CPU_PID:-}" ] && kill -0 "$CPU_PID" 2>/dev/null; then pkill -TERM -P "$CPU_PID" 2>/dev/null || true; kill "$CPU_PID" 2>/dev/null || true; echo "local-ci: CPU chain killed on exit; its log is kept at $CPU_LOG" >&2; fi' EXIT
if [ "$MODE" = "--correctness" ] && [ "${MEMRA_CI_OVERLAP:-1}" = "1" ]; then
    CPU_LOG=$(mktemp "${TMPDIR:-/tmp}/local-ci-cpu-chain.XXXXXX")
    echo "local-ci: CPU chain (clippy, memra-server suite, memra-engine lib suite) running alongside the GPU gates; log $CPU_LOG"
    cpu_chain > "$CPU_LOG" 2>&1 &
    CPU_PID=$!
else
    [ "$MODE" = "--correctness" ] && echo "local-ci: CPU chain overlap SKIPPED (MEMRA_CI_OVERLAP=0), running serially" >&2
    cpu_chain || exit 1
fi
join_cpu_chain() {
    [ -n "$CPU_PID" ] || return 0
    if wait "$CPU_PID"; then
        echo "== local-ci: CPU chain joined: PASS =="
        grep -E '^(== local-ci|test result)' "$CPU_LOG" | sed 's/^/    /' || true
    else
        echo "== local-ci: CPU chain FAILED (full log follows) =="
        cat "$CPU_LOG"
        rm -f "$CPU_LOG"
        exit 1
    fi
    rm -f "$CPU_LOG"
    CPU_PID=""
}

if ! tools/check-flags.sh; then
    echo "local-ci: WARNING — new MEMRA_* flag drift detected (non-fatal; correctness battery continues)"
fi
# The census's own coverage. Drift is non-fatal above; the census going BLIND is not the same
# thing as the census being clean, and v0.94.0 shipped a load-deciding flag past a green run.
if ! tools/test_check_flags.sh >/dev/null; then
    echo "local-ci: WARNING — check-flags self-test FAILED; the flags census may be blind" >&2
    tools/test_check_flags.sh || true
fi

# Gate lint: prevent vacuous gates caused by remove_var on door flags (memra#136)
if ! tools/check-no-remove-var-gates.sh; then
    echo "local-ci: WARNING — un-allowlisted remove_var on door flag detected (memra#136)"
fi
if ! tools/test_check_no_remove_var_gates.sh >/dev/null; then
    echo "local-ci: WARNING — check-no-remove-var-gates self-test FAILED" >&2
    tools/test_check_no_remove_var_gates.sh || true
fi

# DRAFTER-ATTACH WIRING (2026-08-19). FATAL, and it runs before any GPU cell because it needs
# none. tools/assert-drafter-attached.sh was wired into FIVE gates and had never executed once;
# its first run failed because the gemma arm's assertion was unsatisfiable by construction
# (fda04083f4). Only ONE of those five gates is in this battery, so the other four would rot the
# same way. This gate asserts the wiring itself — every call site registered, each asserting the
# same variable it hands the engine, each seam's engine log line still interpolating its path,
# and the assertion actually EXECUTED in both directions. A gate that exists but never executes
# reads as coverage while providing none, which is worse than no gate at all.
if [ -x tools/drafter-attach-wiring-gate.sh ]; then
    tools/drafter-attach-wiring-gate.sh \
        || { echo "drafter-attach wiring gate FAIL"; exit 1; }
else
    echo "local-ci: WARNING — tools/drafter-attach-wiring-gate.sh missing; drafter-attach"
    echo "          coverage is unverified and the four out-of-battery gates can rot" >&2
fi

# LOCK-DISCIPLINE SELF-TEST (lane/localci-lock-20260821). CPU-only, ~20s, private lock
# files, and FATAL: a battery whose lock harness is broken runs its GPU steps fail-open,
# which is how a foreign co-resident OOMed prime-gate and how an outer-flock wrapper
# self-deadlocked the hit-gate — both on 2026-08-21, both invisible to every other gate
# here. In-battery per the H100 lane law: gates outside the battery rot silently.
if ! tools/test_local_ci_lock.sh; then
    echo "local-ci: lock-discipline self-test FAILED — the GPU lock harness is not trustworthy"
    exit 1
fi

# ---- whole-run GPU lock: everything below this line may touch the GPU ----
# The prelude above (build, unit suite, flags census, wiring gate, lock self-test) is
# CPU-only by construction, so it runs OUTSIDE the lock and overlaps another lane's GPU
# window instead of blocking it. The window-state sample below happens INSIDE the lock so
# a cooperating lane can never read as a dirty window.
acquire_gpu_lock

# ---- window state ----
# allowed co-residents: embedding servers (tiny, CPU-bound; identified by --embedding in cmdline)
apps=""
while IFS=, read -r pid _name; do
    pid=$(echo "$pid" | tr -d ' '); [ -n "$pid" ] || continue
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--embedding"; then
        apps+="$pid $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)\n"
    fi
done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
apps=$(printf "%b" "$apps")
if [ -n "$apps" ]; then
    echo "local-ci: WARNING — other GPU compute apps present (numbers not window-valid):"
    echo "$apps"
    WINDOW_CLEAN=false
else
    WINDOW_CLEAN=true
fi
# Per-cell recheck (2026-07-26): the entry-only check let a co-agent job that joined
# MID-battery silently poison later cells (26b-spec-d1736 read accept 0.656 in a battery
# whose entry was clean; 7 windowed re-runs read 0.846). Cells re-verify the window after
# their reps and retry once instead of recording a contended row as evidence.
window_free_now() {
    local n=0 pid
    while IFS=, read -r pid _; do
        pid=$(echo "$pid" | tr -d ' '); [ -n "$pid" ] || continue
        tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qE -- "--embedding|llama-server" \
            || n=$((n+1))
    done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
    [ "$n" -eq 0 ]
}
LOAD=$(awk '{print $1}' /proc/loadavg)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)

echo "== local-ci: correctness stage =="
python3 tools/test_perf_acceptance_baseline.py
if ! kc_out=$(target/release/kernel-check \
        --require-manifest tools/kernel-check-27b.cells \
        --require-manifest tools/kernel-check-step35.cells 2>&1); then
    echo "$kc_out"
    echo "kernel-check FAIL"
    exit 1
fi
echo "$kc_out" | grep '^SKIP ' || true
out=$(echo "$kc_out" | tail -1)
echo "$out" | grep -q '^ALL GREEN ([0-9][0-9]* cells, [0-9][0-9]* skipped)$' \
    || { echo "kernel-check FAIL"; exit 1; }
# SKIP ACCOUNTING (rehomed 2026-09-02 from tools/validate-h100.sh, deleted with the Hopper CI
# lane; PR #73 review caught the enforcement gap). Budget 11 is MEASURED, not fudged (rig,
# 2026-09-02, default model lookup): 10 missing-model cells (q35 IQ4_XS/dtype5 family,
# gemma-4-12B/26B q4_0, ornith-35B, KAT-Coder/Step-3.7 IQ4_XS, 27B NVFP4 twins — the flat
# MEMRA_KC_MODELS_DIR candidates do not cover this rig's subdirectory layout) plus the
# sigrouter-served-replay env-capture cell. A 12th skip means an artifact regressed further:
# refuse it. Tighten the budget when models get staged, never widen it silently.
kc_skipped=$(echo "$out" | sed -n 's/^ALL GREEN ([0-9][0-9]* cells, \([0-9][0-9]*\) skipped)$/\1/p')
if [ "${kc_skipped:-99}" -gt "${MEMRA_CI_KC_SKIP_BUDGET:-11}" ]; then
    echo "kernel-check FAIL — ${kc_skipped} cell(s) skipped, budget ${MEMRA_CI_KC_SKIP_BUDGET:-11}"
    echo "  (set MEMRA_CI_KC_SKIP_BUDGET only to account for a deliberate new skip, out loud)"
    exit 1
fi
echo "kernel-check: $out"

# SAMPLED-SPEC DISTRIBUTIONAL ORACLE (lane/sampledspec, wired in by lane/sampled-hit-spec):
# sample_check is the ONLY test in the tree that proves the rejection-sampling accept walk
# emits x ~ p (Leviathan/Chen) rather than merely running — 20k-draw empirical PMF vs a CPU
# f64 reference, maxabs/TV thresholds, with banked negative controls that FAILED it (inverted
# accept: tv 0.018 -> 0.883; missing residual: tv 0.018 -> 0.088 at an UNCHANGED acceptance
# rate, i.e. no acceptance-rate check could have caught it). It also carries its own gate
# teeth (the old cross-row bonus rule must fail).
#
# It sat outside the battery until now (its own lane flagged the gap at
# research/sampledspec-20260804/RESULTS.md:430) and nothing else here asks a sampled question
# at all: the run-spec sweep below pins MEMRA_SPEC_TEMP=0, accept-gate is temperature 0 by
# premise, and the perf cells are greedy. That was tolerable while sampled traffic only rode
# the cold spec path; since lane/sampled-hit-spec it also rides every prefix-cache HIT, so
# this oracle is load-bearing for the serving default (temperature 1.0). ~15s, GPU-locked.
# MEMRA_CI_SAMPCHECK=0 skips.
if [ "${MEMRA_CI_SAMPCHECK:-1}" = "1" ]; then
    # Built explicitly (the run-spec/decode-batch-gate precedent below): sample_check has no
    # [[bin]] row in memra-engine/Cargo.toml, so a silent "binary absent" skip would quietly
    # un-wire the gate again.
    [ -x target/release/sample_check ] \
        || cargo build --release -p memra-engine --bin sample_check >/dev/null 2>&1 \
        || { echo "sample-check FAIL (binary would not build)"; exit 1; }
    # Same lock discipline as the perf stage's run_gpu_locked (defined later in this file):
    # one GPU gate at a time on this rig, honoring an already-held lock.
    run_sampcheck() {
        if [ "${MEMRA_CI_LOCK_HELD:-0}" = "1" ]; then
            "$@"
        else
            flock -w "${MEMRA_CI_LOCK_WAIT:-7200}" \
                "${MEMRA_CI_LOCK:-/tmp/memra-5090.lock}" "$@"
        fi
    }
    if sc_out=$(run_sampcheck target/release/sample_check 2>&1); then
        echo "$sc_out" | grep -q '=== sample-check ALL GREEN ===' \
            || { echo "$sc_out" | tail -20; echo "sample-check FAIL (no ALL GREEN line)"; exit 1; }
        echo "sample-check: $(echo "$sc_out" | grep 'composed accept-walk output' | head -1)"
    else
        echo "$sc_out" | tail -20
        echo "sample-check FAIL"
        exit 1
    fi
fi

# prime-gate (#46): batched-prime vs tokenwise first-token agreement on the mixed prompt
# set — near-tie flips report, structured divergence or non-determinism exits non-zero.
Q35="$MODELS/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
Q35_DRAFT="$MODELS/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf"
if [ -f "$Q35" ]; then
    if ! target/release/prime-gate "$Q35" \
            --prompts-file research/prime-gate-coverage-20260802/prompts-mixed.txt \
            --steps 0 > /tmp/prime-gate-ci.log 2>&1; then
        echo "prime-gate FAIL (q35)"; tail -3 /tmp/prime-gate-ci.log; exit 1
    fi
    grep "prime-gate" /tmp/prime-gate-ci.log | tail -2
else
    echo "prime-gate: SKIP (no q35 model at $Q35)"
fi

# The standing MTP exactness gate. A naked run-spec invocation sweeps K=1..8; explicitly clear
# single-K and alternate-mode env so a caller cannot silently narrow or change the gate.
# The Gemma-4 31B target below uses a separate assistant-drafter API, so its independent
# stream-agreement check remains on gemma-gate. MEMRA_CI_RUNSPEC=0 skips this sweep.
if [ "${MEMRA_CI_RUNSPEC:-1}" = "1" ]; then
    if [ -f "$Q35" ] && [ -f "$Q35_DRAFT" ]; then
        [ -x target/release/run-spec ] \
            || cargo build --release -p memra-engine --bin run-spec >/dev/null
        RUNSPEC_LOG=/tmp/local-ci-run-spec.log
        runspec_rc=0
        (
            unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY
            MEMRA_SPEC_TEMP=0 MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=32 \
                MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt \
                timeout 900 target/release/run-spec "$Q35"
        ) 2>&1 | tee "$RUNSPEC_LOG" >/dev/null || runspec_rc=$?
        runspec_passes=$(grep -c "self-consistency: PASS" "$RUNSPEC_LOG" || true)
        runspec_ks=$(grep -cE '^\[generate_spec K=[1-8]\]' "$RUNSPEC_LOG" || true)
        if [ "$runspec_rc" -ne 0 ] || [ "$runspec_passes" -ne 8 ] \
                || [ "$runspec_ks" -ne 8 ] \
                || ! grep -q "=== SELF-CONSISTENCY PASS ===" "$RUNSPEC_LOG"; then
            fail_detail=$(awk '
                /^\[generate_spec K=[0-9]+\]/ {
                    k = $0
                    sub(/^.*K=/, "", k)
                    sub(/\].*$/, "", k)
                }
                /self-consistency: FAIL/ { failed_k = k }
                /FIRST DIVERGENCE at index [0-9]+:/ && failed_k != "" {
                    pos = $0
                    sub(/^.*FIRST DIVERGENCE at index /, "", pos)
                    sub(/:.*/, "", pos)
                    print failed_k " " pos
                    exit
                }
            ' "$RUNSPEC_LOG")
            if [ -n "$fail_detail" ]; then
                echo "run-spec self-consistency FAIL (K=${fail_detail%% *}, FIRST DIVERGENCE at index ${fail_detail#* })"
            else
                echo "run-spec K=1..8 FAIL (exit $runspec_rc, $runspec_passes/8 per-K passes)"
            fi
            tail -12 "$RUNSPEC_LOG"
            exit 1
        fi
        echo "run-spec K=1..8 self-consistency: PASS (Qwen 35B, 8/8)"
    elif [ ! -f "$Q35" ]; then
        echo "run-spec K=1..8: SKIP (no q35 model at $Q35)"
    else
        echo "run-spec K=1..8: SKIP (no q35 draft at $Q35_DRAFT)"
    fi
else
    echo "run-spec K=1..8: SKIP (MEMRA_CI_RUNSPEC=0)"
fi

G31="$MODELS/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf"
DEPTH=research/gemma4-bringup/depth-prompt-1736-ids.txt
if [ -f "$G31" ]; then
    # Calibrated argmax gate replaces the raw single-position run-gen assert here
    # (gemma-line merge, 2026-08-17): the ship campaign's exact-algebra folds moved
    # decode arithmetic; prompt-55's final position sits on a near-tie (decode top-2
    # margin 0.053 vs config spread 0.321) and the raw assert flips by position luck —
    # the exact coverage bug tools/argmax-margin-gate.sh was built for. The calibrated
    # gate keeps teeth: every flip must be margin-explained, one wide-margin flip fails
    # at any count (gemma-31B calibration row, banked invocations in the gate log).
    tools/argmax-margin-gate.sh "$G31" || { echo "argmax-margin-gate FAIL (31B)"; exit 1; }
    echo "argmax-margin-gate: PASS (31B, calibrated)"
    # shellcheck disable=SC2046
    out=$(MEMRA_VERIFY_GATE=7 target/release/gemma-gate "$G31" $(cat "$DEPTH") 2>&1)
    echo "$out" | grep -q "VERIFY-GATE K=7: PASS" || { echo "VERIFY-GATE FAIL (31B depth)"; exit 1; }
    echo "VERIFY-GATE K=7 depth: PASS (31B)"
    D31="$MODELS/gemma4-31b-tooluse-gguf/gemma-4-31B-it-Q4_0-MTP.gguf"
    if [ -f "$D31" ]; then
        # shellcheck disable=SC2046
        out=$(MEMRA_SPEC=6 MEMRA_DRAFT="$D31" MEMRA_NGEN=64 target/release/gemma-gate "$G31" \
            $(cat research/gemma4-bringup/e4b-chat-watercycle-ids.txt) 2>&1)
        echo "$out" | grep -qE "stream agreement 64/64" || { echo "spec self-consistency FAIL (31B)"; exit 1; }
        echo "spec self-consistency 64/64: PASS (31B)"
    fi
else
    echo "run-gen/VERIFY-GATE/spec: SKIP (no 31B model at $G31)"
fi

# gemma-4-12B (dense, MQA globals nkv=1 — the gqa=16 hd512 lane 31B never exercises).
G12="${MEMRA_G12_MODEL:-/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf}"
if [ -f "$G12" ]; then
    # shellcheck disable=SC2046
    out=$(MEMRA_NGEN=8 target/release/run-gen "$G12" $(cat "$DEPTH") 2>&1)
    echo "$out" | grep -q "MATCH" || { echo "run-gen argmax FAIL (12B depth)"; exit 1; }
    echo "run-gen argmax depth: MATCH (12B)"
    # shellcheck disable=SC2046
    out=$(MEMRA_VERIFY_GATE=7 target/release/gemma-gate "$G12" $(cat "$DEPTH") 2>&1)
    echo "$out" | grep -q "VERIFY-GATE K=7: PASS" || { echo "VERIFY-GATE FAIL (12B depth)"; exit 1; }
    echo "VERIFY-GATE K=7 depth: PASS (12B)"
else
    echo "12B run-gen/VERIFY-GATE: SKIP (no 12B model at $G12)"
fi
# BATCHED/SOLO DECODE EXACTNESS (serve-path phase 2, 2026-08-05). This battery guards the
# serving tick's numeric contract and it was rotting OUTSIDE the 5090 gate list — only
# validate-h100.sh ran it, so every sm_120 merge took it on trust. The law from the H100 lane:
# anything guarding a live lane belongs INSIDE the battery.
#
# What it pins here: (1) under live defaults, B=1 and B=N use one generic batched numeric class
# and are per-row bit-identical; (2) the explicit MEMRA_SERVE_B1FAST=1 diagnostic remains
# bit-identical to decode_step_h under the equalized strict environment; (3) device sampling
# greedy==host-argmax + sampled isolation + lean-logits.
#
# Strict runs on BOTH dtypes since lane/nvfp4-strict (2026-08-05). It used to be Q8_0-only:
# the equalizing env (MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1) was Q8/dp4a-shaped — the NVFP4
# gate+up/beta+alpha pair door (`matmul_pre_dual_noscale`) ignored MEMRA_MMVQ=0, so the oracle
# rode the fused MMVQ-family dual while the batched side fell to dp4a and strict FAILED on any
# NVFP4 model at pristine trees (train HEAD 70ce5a0f: gate1 maxdiff 1.639e-1 @ step 2 —
# research/servepath-p2-20260805/; q27 at 93420980: gate2 step-6 divergence —
# research/nvfp4-strict-20260805/repro.log). The engine fix pins that door under MMVQ=0 (the
# same FP-order law `q8_fused_params` always enforced for Q8_0), so a strict FAIL on NVFP4 is
# a REAL failure now. Q8_0 strict remains (it caught the pre-H3 B=1 deviation, maxdiff 1.591e-1).
DBG_NVFP4="${MEMRA_CI_DBG_NVFP4:-$MODELS/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
DBG_Q8="${MEMRA_CI_DBG_Q8:-$MODELS/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf}"
[ -x target/release/decode-batch-gate ] \
    || cargo build --release -p memra-engine --bin decode-batch-gate >/dev/null 2>&1
if [ -f "$DBG_NVFP4" ]; then
    out=$(target/release/decode-batch-gate "$DBG_NVFP4" --steps 32 --batch 8 --mode config 2>&1)
    echo "$out" | grep -q "ALL GREEN" \
        || { echo "$out" | tail -20; echo "decode-batch-gate FAIL (NVFP4 config B=8)"; exit 1; }
    echo "$out" | grep -Eq "global setting = OFF; effective .* = OFF" \
        || { echo "$out" | tail -20; echo "decode-batch-gate default B1 policy FAIL (NVFP4)"; exit 1; }
    echo "decode-batch-gate config B=8: ALL GREEN (9B NVFP4)"
    out=$(MEMRA_SERVE_B1FAST=1 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 target/release/decode-batch-gate \
        "$DBG_NVFP4" --steps 32 --batch 4 --mode strict 2>&1)
    echo "$out" | grep -q "ALL GREEN" \
        || { echo "$out" | tail -20; echo "decode-batch-gate FAIL (NVFP4 strict B=4)"; exit 1; }
    echo "decode-batch-gate strict B=4 equalized: ALL GREEN (9B NVFP4)"
elif [ -n "${MEMRA_CI_DBG_NVFP4:-}" ]; then
    # an EXPLICIT override that does not resolve is an operator error, not a skip
    echo "decode-batch-gate: MEMRA_CI_DBG_NVFP4 set but not a file: $DBG_NVFP4"; exit 1
else
    echo "decode-batch-gate NVFP4: SKIP (no model at $DBG_NVFP4)"
fi
if [ -f "$DBG_Q8" ]; then
    out=$(MEMRA_Q8RP=1 target/release/decode-batch-gate "$DBG_Q8" \
        --steps 32 --batch 8 --mode config 2>&1)
    echo "$out" | grep -q "ALL GREEN" \
        || { echo "$out" | tail -20; echo "decode-batch-gate FAIL (Q8_0 config B=8)"; exit 1; }
    echo "$out" | grep -Eq "global setting = OFF; effective .* = OFF" \
        || { echo "$out" | tail -20; echo "decode-batch-gate default B1 policy FAIL (Q8_0)"; exit 1; }
    echo "decode-batch-gate config B=8: ALL GREEN (9B Q8_0)"
    out=$(MEMRA_Q8RP=1 MEMRA_SERVE_B1FAST=1 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 target/release/decode-batch-gate \
        "$DBG_Q8" --steps 32 --batch 4 --mode strict 2>&1)
    echo "$out" | grep -q "ALL GREEN" \
        || { echo "$out" | tail -20; echo "decode-batch-gate FAIL (Q8_0 strict B=4)"; exit 1; }
    echo "decode-batch-gate strict B=4 equalized: ALL GREEN (9B Q8_0)"
elif [ -n "${MEMRA_CI_DBG_Q8:-}" ]; then
    echo "decode-batch-gate: MEMRA_CI_DBG_Q8 set but not a file: $DBG_Q8"; exit 1
else
    echo "decode-batch-gate Q8_0: SKIP (no model at $DBG_Q8)"
fi
# GRAPH-WARMUP STRESS (lane/graph-warmups, 2026-08-05): the pool-growth adversarial gate
# behind the MEMRA_GRAPH_WARMUPS=1 default. Large<->small session cycles + overlap arm force
# captures over freed async-pool blocks; every stream must be bit-identical to eager (the #68
# stale-baked-address class corrupts WITHOUT faulting) and the canary arm proves the
# comparator can catch injected graph-memory corruption. In-battery per the H100 lane law:
# gates outside the battery rot silently. MEMRA_CI_GWSTRESS=0 skips.
if [ "${MEMRA_CI_GWSTRESS:-1}" = "1" ] && [ -x tools/graph-warmup-stress-gate.sh ]; then
    tools/graph-warmup-stress-gate.sh || { echo "graph-warmup-stress FAIL"; exit 1; }
fi
# GRAPH-LANE EXACTNESS (rehomed 2026-09-02 from tools/validate-h100.sh, deleted with the
# Hopper CI lane): decode-dc (device counters, bit-identity), graph-decode (capture/replay
# bit-identity), graph-session (serving GraphSession vs generate_graph — the sm_120a serving
# decode path, NOT a Hopper artifact). Their old battery's round-35 incident is the origin of
# law 3: graph-decode-gate rotted OUTSIDE a battery for weeks, an emission off-by-one in the
# gate masquerading as 171/256 stream corruption. Deleting the battery without rehoming them
# recreated exactly that condition (caught in PR #73 review). Bins are built EXPLICITLY — the
# stale-binary rot the old battery refused to gate on. Verdict lines are grepped, never
# assumed from exit codes. MEMRA_CI_GRAPH=0 skips.
if [ "${MEMRA_CI_GRAPH:-1}" = "1" ]; then
    GRAPH_MODEL="${MEMRA_CI_GRAPH_MODEL:-$DBG_NVFP4}"
    if [ -f "$GRAPH_MODEL" ]; then
        cargo build --release -p memra-engine \
            --bin decode-dc-gate --bin graph-decode-gate --bin graph-session-gate \
            || { echo "graph-lane bins BUILD FAIL — refusing to gate on stale binaries"; exit 1; }
        out=$(target/release/decode-dc-gate "$GRAPH_MODEL" 2>&1)
        echo "$out" | tail -1 | grep -q "PASS" \
            || { echo "$out" | tail -5; echo "decode-dc-gate FAIL"; exit 1; }
        echo "decode-dc-gate: PASS"
        out=$(target/release/graph-decode-gate "$GRAPH_MODEL" 2>&1)
        echo "$out" | tail -1 | grep -q "PASS" \
            || { echo "$out" | tail -5; echo "graph-decode-gate FAIL"; exit 1; }
        echo "graph-decode-gate: PASS"
        out=$(target/release/graph-session-gate "$GRAPH_MODEL" 2>&1)
        echo "$out" | tail -1 | grep -q "ALL GREEN" \
            || { echo "$out" | tail -5; echo "graph-session-gate FAIL"; exit 1; }
        echo "graph-session-gate: ALL GREEN"
    elif [ -n "${MEMRA_CI_GRAPH_MODEL:-}" ]; then
        echo "graph-lane: MEMRA_CI_GRAPH_MODEL set but not a file: $GRAPH_MODEL"; exit 1
    else
        echo "graph-lane exactness: SKIP (no model at $GRAPH_MODEL)"
    fi
fi
echo "correctness stage: GREEN"

# normal-usage serving battery (2026-07-30): OpenAI surface, streaming, determinism,
# concurrency, lanes, spec==plain serving exactness. MEMRA_CI_SERVE=0 skips.
if [ "${MEMRA_CI_SERVE:-1}" = "1" ] && [ -x tools/serve-smoke.sh ]; then
    tools/serve-smoke.sh || { echo "serve-smoke FAIL"; exit 1; }
fi

# c=64 CONCURRENCY STRESS (lane/admit-oom, 2026-08-06): 64 staggered streaming clients on a
# 24GB card — the cell that was RED until the admission cost model charged the spec transient
# reserve and step-OOM learned to park instead of kill. In-battery per the H100 lane law
# (gates outside the battery rot silently); serving-density deliberately left it unwired while
# it was red, because wiring a known-red gate either blocks every merge or normalizes a red.
# Its own teeth: `tools/serve-stress-gate.sh --teeth` forces the reserve to 16MB and asserts
# the RED returns — run that whenever the admission math moves. MEMRA_CI_STRESS=0 skips.
if [ "${MEMRA_CI_STRESS:-1}" = "1" ] && [ -x tools/serve-stress-gate.sh ]; then
    tools/serve-stress-gate.sh || { echo "serve-stress FAIL"; exit 1; }
fi

# SERVED-SPEC ACCEPTANCE + LONG-TEXT ASSERTION (lane/accept-gate, 2026-08-06): the arm that
# closes a receipted blind spot in THIS battery. research/f8f4-flip-20260806 (merged c506317e)
# showed a kernel arm move served greedy text in 4 of 6 regime cells at temperature 0 and move
# spec acceptance up to -9.5pp while EVERY gate above stayed green — because (1) the token
# goldens stop at 20 tokens and both divergences landed at generated index 22 and 38, (2)
# `fast-gate --refresh-goldens` after such a change would silently re-pin the new arm, and (3)
# nothing here compared accepted-draft COUNTS, which are spec throughput, i.e. the product.
# Each arm was internally self-consistent and reproduced its own goldens, so self-consistency
# could never see it.
#
# This arm asserts, at the production serve config (real regime drafter attached, real serve K):
# exact (rounds, drafted, accepted) integers — temp 0 makes drafting deterministic — plus the
# full generated text sha256 to ngen=128, 6.4x past the golden window. In-battery per the H100
# lane law: gates outside the battery rot silently.
#
# Default arm = the smoke tier (ONE model, ONE cell: q27-p1, ~1 min incl. the 16G load) to keep
# the correctness stage near its ~3 min budget. The full 6-cell matrix (both NVFP4-reachable
# models x 3 prompt lengths) is `tools/accept-gate.sh --full`, and `--control` adds the
# second-boot determinism control. Its own teeth: `tools/accept-gate.sh --teeth` sets
# MEMRA_MMQ_F8F4=1 and REQUIRES the gate to fail — run that whenever the spec/draft or NVFP4
# prefill path moves. MEMRA_CI_ACCEPT=0 skips.
if [ "${MEMRA_CI_ACCEPT:-1}" = "1" ] && [ -x tools/accept-gate.sh ]; then
    tools/accept-gate.sh || { echo "accept-gate FAIL"; exit 1; }
fi

# SPEC ENGAGES ON PREFIX-CACHE HITS (lane/spec-on-cache-hit + lane/sampled-hit-spec): the arm
# that closes THIS battery's second receipted blind spot — a shipped headline that was inert in
# production while every gate stayed green.
#
# v0.93.0's headline was "spec engages on prefix-cache hits". It shipped GREEDY-ONLY, and the
# deploy verification on the DE box measured 3 cache hits, 3 plain-path downgrades and ZERO
# restores: the paying tenant's traffic is sampled, which is what the OpenAI surface defaults
# to (temperature 1.0). Nothing in this battery asked the question — the hit gate existed and
# was green, but it lived OUTSIDE the battery and only ever posed greedy rows. Both halves of
# that are fixed here: the gate is in, and it now carries sampled cells whose engagement
# assertion fails on a v0.93.0-shaped binary.
#
# What it asserts (own server boots, own GPU lock, ~4 min for the qwen arm): cold spec
# publishes a draft-plane entry; the identical repeat and the extended repeat both RE-ARM spec
# (usage.spec.accepted > 0, cached_tokens > 0); spec-on bytes == a spec-off twin boot's bytes
# on every greedy row; and for sampled traffic — engagement, per-seed byte identity against
# the COLD sampled leader over 3 seeds (the sampled-spec contract makes plain-identity
# unavailable, so cold-identity is the standard), exact acceptance parity, suffix-shape
# reproducibility, a plane-less refusal that names itself, penalized sampled hits (the refusal
# lane/sampled-spec-quality lifted), the three boundary-draw SITES with a deviation observed,
# and the growth/round-trip cells for extended-entry publication.
#
# The TEETH arm now runs IN the battery too (MEMRA_CI_HITGATE_TEETH=0 skips it and halves the
# hit-gate time). It boots with every door of this arc shut — MEMRA_SPEC_RESTORE_SAMPLED=0
# MEMRA_SPEC_SAMPLED_BOUNDARY=0 MEMRA_SPEC_PEN_SESSION=0 MEMRA_SPEC_RESTORE_REPUBLISH=0 — and
# REQUIRES the pre-lane behaviour back. That is not belt-and-braces: four default-ON doors whose
# closed posture nobody exercises are four rollbacks that might not work on the night they are
# needed, and a cell that cannot fail on a v0.93.0-shaped binary is how the inert headline
# shipped in the first place.
# MEMRA_CI_HITGATE=0 skips; the gemma arm additionally needs its 12B pair and skips silently
# when either artifact is absent. In-battery per the H100 lane law: gates outside the battery
# rot silently — this lane is the receipt for that law.
if [ "${MEMRA_CI_HITGATE:-1}" = "1" ] && [ -x tools/spec-on-cache-hit-gate.sh ]; then
    HITQ="$MODELS/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
    HITEV=${MEMRA_CI_HITGATE_EV:-/tmp/memra-ci-hitgate}
    # EXTERNAL QWEN DRAFTER FOR THE BATTERY'S QWEN ARM (2026-08-19). Without this the qwen arm's
    # drafter-attach assertion is a NO-OP: assert_mtp_drafter() returns 0 immediately when
    # MTP_DRAFT is empty, and this battery never set MEMRA_GATE_MTP_DRAFT — so in-battery coverage
    # of the assertion was the gemma arm ALONE, on a rig where the gemma arm can silently SKIP for
    # a missing artifact. The 9B trunk ships an external drafter beside it; attaching it makes the
    # arm exercise the real production shape (a served artifact whose drafter ships separately)
    # instead of drafting off the trunk's own embedded head.
    # SEAM: MEMRA_GATE_MTP_DRAFT feeds MEMRA_MTP_DRAFT (the qwen seam), NOT MEMRA_DRAFT (gemma's,
    # which attaches nothing on a qwen model while silently flipping wkv_on()).
    HITQD=${MEMRA_CI_HITGATE_QDRAFT:-"$MODELS/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"}
    if [ -f "$HITQD" ]; then
        echo "spec-on-cache-hit: qwen arm attaches the external drafter and ASSERTS it: $HITQD"
    else
        # LOUD, not a silent skip: a missing drafter downgrades this arm's coverage, and the
        # whole point of this lane is that unexercised assertions read as coverage.
        echo "spec-on-cache-hit: WARNING — qwen external drafter absent ($HITQD)." >&2
        echo "          The qwen drafter-attach assertion will NO-OP this run; in-battery" >&2
        echo "          coverage falls back to the gemma arm alone." >&2
        HITQD=""
    fi
    if [ -f "$HITQ" ]; then
        rm -rf "$HITEV/qwen" && mkdir -p "$HITEV/qwen"
        MEMRA_GATE_MTP_DRAFT="$HITQD" \
        tools/spec-on-cache-hit-gate.sh qwen "$HITQ" target/release/memra-server "$HITEV/qwen" \
            || { echo "spec-on-cache-hit (qwen) FAIL — evidence in $HITEV/qwen"; exit 1; }
        rm -rf "$HITEV/qwen" # green: keep /tmp clean (the tmp-hygiene law); a FAIL keeps it
        if [ "${MEMRA_CI_HITGATE_TEETH:-1}" = "1" ]; then
            rm -rf "$HITEV/qwen-teeth" && mkdir -p "$HITEV/qwen-teeth"
            MEMRA_HITGATE_TEETH=1 MEMRA_GATE_MTP_DRAFT="$HITQD" \
            tools/spec-on-cache-hit-gate.sh qwen "$HITQ" \
                target/release/memra-server "$HITEV/qwen-teeth" \
                || {
                    echo "spec-on-cache-hit TEETH (qwen) FAIL — evidence in $HITEV/qwen-teeth"
                    exit 1
                }
            rm -rf "$HITEV/qwen-teeth"
        fi
    else
        echo "spec-on-cache-hit: SKIP qwen arm (missing $HITQ)"
    fi
    HITG=${MEMRA_G12_MODEL:-/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf}
    HITGD="$MODELS/gemma4-12b-mtp-gguf/gemma-4-12B-it-qat-assistant-MTP-Q4_0.gguf"
    if [ "${MEMRA_CI_HITGATE_GEMMA:-1}" = "1" ] && [ -f "$HITG" ] && [ -f "$HITGD" ]; then
        rm -rf "$HITEV/gemma" && mkdir -p "$HITEV/gemma"
        # No MEMRA_GEMMA4_SPEC here: unset + MEMRA_DRAFT is the default-on K=5 posture the
        # banked gemma CERT ran at (worker.rs gemma4_spec_k_env), and pinning a different K
        # would silently stop reproducing it.
        tools/spec-on-cache-hit-gate.sh gemma "$HITG" "$HITGD" \
            target/release/memra-server "$HITEV/gemma" \
            || { echo "spec-on-cache-hit (gemma) FAIL — evidence in $HITEV/gemma"; exit 1; }
        rm -rf "$HITEV/gemma"
    else
        echo "spec-on-cache-hit: SKIP gemma arm (artifact or MEMRA_CI_HITGATE_GEMMA=0)"
    fi
fi

# The CPU chain must land before the stage is called green, and the engine's three
# `#[ignore]` GPU tests run here, under the lock this run already holds (the vision test
# names that condition in its own ignore reason). Serial runs have CPU_PID empty; join is a no-op.
join_cpu_chain
echo "== local-ci: memra-engine lib suite (GPU-only #[ignore] tests) =="
if ! cargo test --release -p memra-engine --lib -j8 -- --ignored; then
    echo "local-ci: memra-engine GPU-only lib tests FAILED"; exit 1
fi
[ "$MODE" = "--correctness" ] && exit 0

echo "== local-ci: perf stage ($MODE) =="
GIT_SHA=$(git rev-parse --short HEAD)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FAILS=0; WARNS=0

# Since lane/localci-lock-20260821 the whole run holds the lock (acquire_gpu_lock exports
# MEMRA_CI_LOCK_HELD=1), so this wrapper always self-skips here; it stays as defense in
# depth for any future caller that reaches this stage without the whole-run acquisition.
run_gpu_locked() {
    if [ "${MEMRA_CI_LOCK_HELD:-0}" = "1" ]; then
        "$@"
    else
        flock -w "${MEMRA_CI_LOCK_WAIT:-7200}" \
            "${MEMRA_CI_LOCK:-/tmp/memra-5090.lock}" "$@"
    fi
}

run_cell() {
    local id="$1" model="$2" mode="$3" prompt="$4" ngen="$5" k="$6" draft="$7" ranks="$8"
    local mp="$MODELS/$model"
    [ -f "$mp" ] || { echo "  $id: SKIP (no model)"; return 0; }
    local pfile; pfile=$(jq -r ".prompts[\"$prompt\"]" $MANIFEST)
    local best_toks="0" accept="" tokround="" cell_try
    for cell_try in 1 2; do
    best_toks="0"; accept=""; tokround=""
    for _rep in 1 2; do
        local out toks
        # The reps run UNDER THE SHARED GPU LOCK (2026-08-06, v0.71.0 release battery). The
        # window_free_now() recheck below samples only BETWEEN reps, so a neighbor lane that
        # starts and finishes inside a rep is invisible to it — that hole reported 10/10 cells
        # FAIL (-8.31%..-24.75%) at the v0.71.0 tag candidate while a concurrent Q8RP census
        # held run-gen on the same card, and every poisoned row still recorded
        # window_clean:true. Every other GPU consumer in this repo (fast-gate, the gate
        # scripts) already takes /tmp/memra-5090.lock; the perf stage — the one stage whose whole
        # output is a timing number — did not.
        if [ "$mode" = "plain" ]; then
            # shellcheck disable=SC2046
            out=$(run_gpu_locked \
                  env MEMRA_NGEN="$ngen" timeout 420 target/release/run-gen "$mp" $(cat "$pfile") 2>&1 || true)
            toks=$(echo "$out" | grep -oE "= [0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+" || echo 0)
        else
            local envs=(MEMRA_SPEC_ONLY=1 "MEMRA_SPEC=$k" "MEMRA_DRAFT=$MODELS/$draft" "MEMRA_NGEN=$ngen")
            [ -n "$ranks" ] && [ "$ranks" != "null" ] && envs+=("MEMRA_GEMMA_DRAFT_RANKS=$ranks")
            # shellcheck disable=SC2046
            out=$(run_gpu_locked \
                  env "${envs[@]}" timeout 420 target/release/gemma-gate "$mp" $(cat "$pfile") 2>&1 || true)
            toks=$(echo "$out" | grep -oE "spec: [0-9.]+" | grep -oE "[0-9.]+" || echo 0)
            accept=$(echo "$out" | grep -oE "accept-rate=[0-9.]+" | grep -oE "[0-9.]+" | tail -1 || true)
            tokround=$(echo "$out" | grep -oE "tok/round=[0-9.]+" | grep -oE "[0-9.]+" | tail -1 || true)
        fi
        awk -v a="$toks" -v b="$best_toks" 'BEGIN{exit !(a>b)}' && best_toks="$toks"
    done
    if window_free_now; then break; fi
    if [ "$cell_try" = 1 ]; then
        # BOUNDED wait, LATCHED once (2026-08-07, lane/spec-gate). This loop used to be
        # `while ! window_free_now; do sleep 40; done` — unbounded, so a PERSISTENT
        # co-resident deadlocked the whole perf stage and made the honest fallback two
        # lines below (record with window_clean=false) unreachable. Hit for real: the
        # owner's hermes-gateway.service holds a 394 MiB idle CUDA context 24/7 on this
        # box, 0% GPU util, and is not a lane's job to kill — the battery sat in that
        # loop through 31b-plain-short and produced no rows at all. A gate that hangs
        # forever is worse than one that records an honestly-labeled row.
        #
        # Latched, because the wait is only worth paying for a TRANSIENT joiner: once one
        # cell has proven the co-resident outlasts the wait, every later cell skips
        # straight to the labeled retry instead of re-paying it (10 cells x 600 s of
        # pure sleeping is not a gate, it is a hang with progress output).
        if [ "${PERSISTENT_CORESIDENT:-0}" = 1 ]; then
            echo "  $id: window DIRTY, co-resident already known persistent — retrying, row will be window_clean=false"
        else
            local wait_left="${MEMRA_CI_DIRTY_WAIT:-600}"
            echo "  $id: window went DIRTY mid-cell — waiting up to ${wait_left}s + retrying once"
            while ! window_free_now && [ "$wait_left" -gt 0 ]; do
                sleep 20; wait_left=$((wait_left - 20))
            done
            if ! window_free_now; then
                PERSISTENT_CORESIDENT=1
                echo "  $id: co-resident did not leave in ${MEMRA_CI_DIRTY_WAIT:-600}s — treating it as persistent; rows from here are window_clean=false"
            fi
        fi
    else
        echo "  $id: DIRTY twice — recording with window_clean=false"
        WINDOW_CLEAN=false
    fi
    done
    [ "$best_toks" = "0" ] && { echo "  $id: FAIL (no reading)"; FAILS=$((FAILS+1)); return 0; }

    # Rolling-median verdict from prior rows of this cell.
    #
    # WHAT THIS VERDICT IS, EXACTLY (2026-08-06): a DRIFT TRIPWIRE, not evidence. The
    # denominator is a median of rows measured on earlier days, so a tok/s FAIL here is a
    # CROSS-DAY comparison — precisely the form the measurement law (research/benchmarks.md,
    # the H100 lane's law 1) forbids as proof, because clock/thermal/power state drifts under
    # both numerator and denominator. It answers "did something move?", never "did this commit
    # regress?".
    #
    # THE PROTOCOL WHEN IT GOES RED (do not skip to a conclusion either way):
    #   build the last-green commit's binary, then run the SAME cell interleaved A/B/A/B, N>=5
    #   each, in ONE thermal window under one exclusive lock hold, and compare only within
    #   that window. See research/v071-prep-20260806/battery-logs/perf-ab.sh for the harness.
    # v0.71.0 is the worked example: 10/10 cells "FAIL" at -8.31%..-24.75%, and the
    # interleaved A/B put the last-green baseline binary at 37.87 tok/s against the candidate's
    # 37.87 (+0.00%) — the drop was the machine's state, and zero code had regressed. A uniform
    # multi-cell drop with correctness green is that signature, not ten simultaneous
    # regressions.
    #
    # ACCEPTANCE drops are the exception: acceptance is a RATIO, clock-independent, and
    # invisible to every exactness gate by construction. Treat an acceptance FAIL as real.
    local base verdict="OK" note="" rows
    rows=$(grep "\"cell\":\"$id\"" "$OUT" 2>/dev/null || true)
    base=$(printf '%s\n' "$rows" | tail -"$(jq -r .gates.baseline_window $MANIFEST)" \
        | jq -s 'map(.toks) | sort | .[length/2|floor] // 0' 2>/dev/null)
    base=${base:-0}
    if awk -v b="$base" 'BEGIN{exit !(b>0)}'; then
        local drop
        drop=$(awk -v n="$best_toks" -v b="$base" 'BEGIN{printf "%.2f", (b-n)/b*100}')
        if awk -v d="$drop" -v t="$(jq -r .gates.cell_drop_fail_pct $MANIFEST)" 'BEGIN{exit !(d>t)}'; then
            verdict="FAIL"; FAILS=$((FAILS+1)); note="tok/s -$drop% vs median $base"
        elif awk -v d="$drop" -v t="$(jq -r .gates.cell_drop_warn_pct $MANIFEST)" 'BEGIN{exit !(d>t)}'; then
            verdict="WARN"; WARNS=$((WARNS+1)); note="tok/s -$drop% vs median $base"
        fi
        if [ -n "$accept" ]; then
            local abase
            abase=$(python3 tools/perf_acceptance_baseline.py \
                --manifest "$MANIFEST" --history "$OUT" --cell "$id")
            abase=${abase:-0}
            if awk -v a="$accept" -v b="$abase" -v t="$(jq -r .gates.accept_drop_fail $MANIFEST)" \
                 'BEGIN{exit !(b>0 && b-a>t)}'; then
                verdict="FAIL"; FAILS=$((FAILS+1)); note="$note; ACCEPTANCE $abase -> $accept"
            fi
        fi
    else
        note="first row (baseline seed)"
    fi
    printf '{"ts":"%s","git":"%s","cell":"%s","toks":%s%s%s,"profile":"%s","load":%s,"window_clean":%s}\n' \
        "$TS" "$GIT_SHA" "$id" "$best_toks" \
        "${accept:+,\"accept\":$accept}" "${tokround:+,\"tok_round\":$tokround}" \
        "$PROFILE" "$LOAD" "$WINDOW_CLEAN" >> "$OUT"
    echo "  $id: $best_toks tok/s${accept:+ accept=$accept} [$verdict]${note:+ — $note}"
}

while read -r cell; do
    id=$(echo "$cell" | jq -r .id)
    if [ "$MODE" = "--perf-quick" ] && [[ "$id" != 31b-* ]]; then continue; fi
    # MEMRA_CI_CELLS: extended-regex cell-id filter (e.g. "26b-|e4b-") — run a subset
    # without touching the manifest; verdicts/rows behave exactly like a full run.
    if [ -n "${MEMRA_CI_CELLS:-}" ] && ! echo "$id" | grep -qE "$MEMRA_CI_CELLS"; then continue; fi
    run_cell "$id" "$(echo "$cell" | jq -r .model)" "$(echo "$cell" | jq -r .mode)" \
             "$(echo "$cell" | jq -r .prompt)" "$(echo "$cell" | jq -r .ngen)" \
             "$(echo "$cell" | jq -r '.k // 0')" "$(echo "$cell" | jq -r '.draft // ""')" \
             "$(echo "$cell" | jq -r '.ranks // ""')"
done < <(jq -c '.cells[]' $MANIFEST)

echo "perf stage: $FAILS fail, $WARNS warn"
if [ "$FAILS" -gt 0 ]; then
    cat <<'PERFRED'

  ^ A tok/s FAIL above is a DRIFT TRIPWIRE against a cross-day median, NOT a proven
    regression, and it is not by itself a merge/tag blocker. Settle it before concluding:
      1. build the last-green commit's binary for this cell,
      2. run that cell interleaved A/B/A/B, N>=5 each, ONE thermal window, one exclusive
         lock hold (harness: research/v071-prep-20260806/battery-logs/perf-ab.sh),
      3. compare medians WITHIN that window only.
    A uniform drop across many cells with correctness green points at machine state
    (power/thermal/profile) or a contended window, not at the diff. An ACCEPTANCE FAIL is
    different — acceptance is clock-independent, so treat it as real.
PERFRED
fi
[ "$FAILS" -eq 0 ] || exit 1
