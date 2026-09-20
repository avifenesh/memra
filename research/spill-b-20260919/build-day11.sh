#!/usr/bin/env bash
# Day 11 native build receipt: release gate binary, scoped clippy, kv/tier tests. No GPU cell.
# usage: build-day11.sh <rig pro-single|rtx5090> [receipt-dir]
set -uo pipefail
rig=${1:?rig}
case "$rig" in
  pro-single)
    wt=${WT:-/root/wt-b}
    out=${2:-/root/spill-receipts/b-day11/build}
    export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
    run() { "$@"; }
    ;;
  rtx5090)
    wt=${WT:-$HOME/projects/wt-spill-b}
    out=${2:-$wt/research/spill-b-20260919/rtx5090-day11/build}
    # Owner rule: no local CPU saturation.
    run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
    ;;
  *) echo "REFUSED: unknown rig $rig" >&2; exit 2 ;;
esac
cd "$wt"
mkdir -p "$out"
git rev-parse HEAD > "$out/source.txt"
git status --porcelain > "$out/dirty.txt"
run cargo build --release -p memra-engine --bin kv-tier-gate > "$out/build.log" 2>&1
echo $? > "$out/exit"
sha256sum target/release/kv-tier-gate > "$out/binary.sha256" 2>/dev/null || echo "missing" > "$out/binary.sha256"
run cargo clippy --release -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings > "$out/clippy.log" 2>&1
echo $? > "$out/clippy.exit"
run cargo test --release -p memra-kv -p memra-tier --offline --no-fail-fast > "$out/tests.log" 2>&1
echo $? > "$out/tests.exit"
cat "$out/exit" "$out/clippy.exit" "$out/tests.exit" | tr '\n' ' '; echo
