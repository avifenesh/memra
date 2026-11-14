#!/usr/bin/env bash
# Dev desk: fold of the #1308 review on main bbc6f83aea. tree = the branch's tracked crates, tools
# and docs synced from the rig by checksum. Checks: the memra-server library suite, the engine's
# dflash tests (dflash.rs changed), clippy (engine and server, all targets, -D warnings), rustfmt
# on the touched files. Every cargo pinned to CPUs 14-31.
# usage: dd-checks.sh LABEL   -> logs/fold/LABEL/, logs/fold/LABEL.done
set -u
D=$HOME/.cache/memra-1304
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
LABEL=$1
L=$D/logs/fold/$LABEL
mkdir -p "$L"
rm -f "$D/logs/fold/$LABEL.done"
S=$L/summary.txt
cd "$D/tree" || exit 1
{ echo "start $(date -Is)"; sha256sum crates/memra-server/src/worker.rs crates/memra-server/src/worker/host_glm.rs crates/memra-engine/src/dflash.rs crates/memra-engine/src/glm_spec.rs | sed "s|$HOME|~|"; } > "$S"
taskset -c 14-31 nice -n 15 cargo test --release -p memra-server --lib -j 14 > "$L/suite.log" 2>&1
echo "server suite rc=$? $(grep -E '^test result' "$L/suite.log" | tail -n 1)" >> "$S"
grep -E "^test .*FAILED$" "$L/suite.log" | head -n 20 >> "$S"
taskset -c 14-31 nice -n 15 cargo test --release -p memra-engine --lib -j 14 -- dflash > "$L/engine-dflash.log" 2>&1
echo "engine dflash tests rc=$? $(grep -E '^test result' "$L/engine-dflash.log" | tail -n 1)" >> "$S"
grep -E "^test .*FAILED$" "$L/engine-dflash.log" | head -n 20 >> "$S"
taskset -c 14-31 nice -n 15 cargo clippy --release -p memra-engine -p memra-server --all-targets -j 14 -- -D warnings > "$L/clippy.log" 2>&1
echo "clippy rc=$? $(grep -cE '^(warning|error)' "$L/clippy.log") warning/error lines" >> "$S"
grep -E '^(error|warning)' "$L/clippy.log" | sort | uniq -c | head -n 12 >> "$S"
rustfmt --edition 2024 --check crates/memra-server/src/worker.rs crates/memra-server/src/worker/host_glm.rs crates/memra-engine/src/dflash.rs crates/memra-engine/src/glm_spec.rs > "$L/fmt.log" 2>&1
echo "rustfmt (touched files) rc=$?" >> "$S"
echo "done $(date -Is)" >> "$S"
touch "$D/logs/fold/$LABEL.done"
