#!/usr/bin/env bash
# DAY88 section 3's CPU gates for phase 1 of the door's promotion, run under nice 19 inside the caller's scope (600% CPU,
# MemoryMax 12G): the tier suites, the engine library, clippy (-D warnings, all targets, the two crates), fmt, the
# flags census, git diff --check and rc-scan --live; each step's tail and rc.
cd /home/avifenesh/projects/wt-spill-c || exit 1
export PATH=/usr/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
echo "# DAY88 CPU gates at $(git rev-parse --short HEAD) (+ the working tree), $(date -u +%FT%TZ)"
echo "== tier suites"; nice -n 19 cargo test -p memra-tier 2>&1 | grep -E '^test result|FAILED|panicked|^error' | sort | uniq -c; echo "rc=${PIPESTATUS[0]}"
echo "== engine library"; nice -n 19 cargo test -p memra-engine --lib 2>&1 | grep -E '^test result|FAILED|panicked|^error'; echo "rc=${PIPESTATUS[0]}"
echo "== engine day88 cells"; nice -n 19 cargo test -p memra-engine --lib day88 2>&1 | grep -E '^test |test result'; echo "rc=${PIPESTATUS[0]}"
echo "== clippy"; nice -n 19 cargo clippy -p memra-tier -p memra-engine --all-targets -- -D warnings 2>&1 | grep -E '^(error|warning: unused)' -A8 | head -40; echo "rc=${PIPESTATUS[0]}"
echo "== fmt"; cargo fmt --all -- --check 2>&1 | head -20; echo "rc=${PIPESTATUS[0]}"
echo "== flags census"; bash tools/check-flags.sh 2>&1 | tail -3; echo "rc=${PIPESTATUS[0]}"
echo "== diff check"; git diff --check; echo "rc=$?"
echo "== rc-scan --live"; /usr/bin/python3 research/spill-c-20260919/rc-scan.py --live; echo "rc=$?"
