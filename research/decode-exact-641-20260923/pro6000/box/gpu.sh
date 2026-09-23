#!/usr/bin/env bash
# lane E part 2, target-card cells under the collector's hold of /tmp/memra-gpu.lock:
#   python3 tools/tier-battery.py --rig pro-single --external-lock --execute bash gpu.sh @COLLECTOR_LOCK_FD@
set -uo pipefail
fd=$1
E=/root/e641
R=$E/out
mkdir -p $R
cd /root/wt-e || exit 1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
unset MEMRA_REWRITE_BUNDLE
M=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
LOG=$R/run.log
say() { echo "$*" | tee -a "$LOG"; }
card() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';'; }
state() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader; }

python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/LOCK.json"
say "== lane E part 2 on the target card $(date -u +%FT%TZ)"
say "tree $(git rev-parse HEAD) dirty=[$(git status --porcelain | tr '\n' ' ')]"
say "card $(nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader)"
say "bins:"; sed 's/^/  /' $E/bins.sha256 | tee -a "$LOG"
say "model $M $(stat -c %s "$M") bytes sha256 $(sha256sum "$M" | cut -c1-64)"
say "card at start: apps=[$(card)] state=[$(state)]"
nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv -lms 250 > "$R/telemetry-250ms.csv" &
TEL=$!

cell() { local name=$1; shift; say "-- $name start $(date -u +%FT%TZ) apps=[$(card)] state=[$(state)]"; "$@" > "$R/$name.out" 2>&1; local rc=$?; say "   $name rc=$rc"; return $rc; }

# 1. prime-batch-gate --exact rows on the fix, plus the canary
cell pbg-fix tools/prime-batch-exact-gate.sh "$M" --logdir "$R/pbg-fix"
grep -E '^    |^prime-batch-exact-gate: ' "$R/pbg-fix.out" | tee -a "$LOG"
cell pbg-fix-canary tools/prime-batch-exact-gate.sh "$M" --canary --logdir "$R/pbg-fix-canary"
grep -E '^    |^prime-batch-exact-gate: ' "$R/pbg-fix-canary.out" | tee -a "$LOG"

# 2. the same rows on the base binary (expected red, as on the 5090)
for row in "exact-b3-p24:--batch 3 --exact" "exact-b4-p1100:--batch 4 --plen 1100 --exact" "carried-b3-exact:--batch 3 --plen 600 --carried --exact"; do
    name=${row%%:*}; args=${row#*:}
    # shellcheck disable=SC2086
    cell "pbg-base-$name" $E/bins/base/prime-batch-gate "$M" $args
    grep -E '^(seq [0-9]+: exact|ALL GREEN|Error)' "$R/pbg-base-$name.out" | head -6 | sed 's/^/    /' | tee -a "$LOG"
done

# 3. the #641 tick-shape replay, if the family loads through HybridModel
cell ptick tools/prime-tick-exact-gate.sh "$M" --log "$R/ptick-naked.probe.log"
grep -E '^prime-tick-exact-gate: |^    ' "$R/ptick.out" | tee -a "$LOG"
if grep -q '^prime-tick-exact-gate: PASS' "$R/ptick.out"; then
    cell ptick-canary tools/prime-tick-exact-gate.sh "$M" --canary --log "$R/ptick-canary.probe.log"
    grep -E '^prime-tick-exact-gate: ' "$R/ptick-canary.out" | tee -a "$LOG"
else
    say "   ptick did not PASS; raw probe log tail:"; tail -5 "$R/ptick-naked.probe.log" | sed 's/^/     /' | tee -a "$LOG"
fi

# 4. memra#668's spec context-edge gate on the fix server
cell spec-ctx-edge env SCE_CTX=384 tools/spec-ctx-edge-gate.sh "$M" $E/bins/fix/memra-server "$R/spec-ctx-edge"
grep -E '^SPEC-CTX-EDGE GATE|-> FAIL|prompt_tokens' "$R/spec-ctx-edge.out" | cut -c1-200 | tee -a "$LOG"

# 5. batched prime cost A/B, B=3 T=1024, 6 pairs AB BA AB BA AB BA (A = base), one process per run
mkdir -p "$R/perf"
for p in 1 2 3 4 5 6; do
    if [ $((p % 2)) = 1 ]; then order="base fix"; else order="fix base"; fi
    for arm in $order; do
        say "-- pair $p arm $arm start $(date -u +%FT%TZ) state=[$(state)]"
        $E/bins/$arm/prime-batch-gate "$M" --batch 3 --bench 1024 > "$R/perf/pbg-$arm-pair$p.log" 2>&1
        say "   rc=$? $(grep -E '^bench B=' "$R/perf/pbg-$arm-pair$p.log")"
    done
done

kill $TEL 2>/dev/null; wait $TEL 2>/dev/null
say "card at end: apps=[$(card)] state=[$(state)]"
say "== done $(date -u +%FT%TZ)"
