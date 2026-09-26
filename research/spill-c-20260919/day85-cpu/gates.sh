#!/usr/bin/env bash
# DAY85 section 3's CPU gates for I22, run inside the caller's CPU cap: the tier suites, the engine library, clippy
# (-D warnings, all targets, the two crates), fmt, and git diff --check; each step's tail and rc.
cd /home/avifenesh/projects/wt-spill-c || exit 1
export PATH=/usr/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
echo "# DAY85 CPU gates at $(git rev-parse --short HEAD) (+ the working tree), $(date -u +%FT%TZ)"
echo "== tier suites"; cargo test -p memra-tier 2>&1 | grep -E '^test result|FAILED|panicked|error' | sort | uniq -c; echo "rc=${PIPESTATUS[0]}"
echo "== engine library"; cargo test -p memra-engine --lib 2>&1 | grep -E '^test result|FAILED|panicked|^error' ; echo "rc=${PIPESTATUS[0]}"
echo "== clippy"; cargo clippy -p memra-tier -p memra-engine --all-targets -- -D warnings 2>&1 | tail -3; echo "rc=${PIPESTATUS[0]}"
echo "== fmt"; cargo fmt --all -- --check 2>&1 | head -20; echo "rc=${PIPESTATUS[0]}"
echo "== diff check"; git diff --check; echo "rc=$?"
echo "== rc-scan --live"; /usr/bin/python3 research/spill-c-20260919/rc-scan.py --live; echo "rc=$?"
