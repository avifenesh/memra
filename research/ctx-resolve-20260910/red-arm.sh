#!/bin/sh
# Red-arm demonstration for the MEMRA_CTX context-fallback fix.
# Reintroduces the pre-fix behaviour (an unset MEMRA_CTX resolving to a literal
# 8192) and re-runs the tests that exist to catch exactly that. FAILURES are the
# expected result; a green run here would mean the checks are vacuous.
set -eu
cd /data/ctxlane
cp crates/memra-server/src/worker.rs /tmp/worker.rs.green
python3 - <<'PY'
p = 'crates/memra-server/src/worker.rs'
s = open(p).read()
old = """        None => {
            if model_ctx > 0 {
                Ok(model_ctx)"""
new = """        None => {
            if model_ctx > 0 {
                Ok(8192) // RED ARM: the pre-fix literal"""
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, new, 1))
PY
export PATH=/data/cargo/bin:$PATH RUSTUP_HOME=/data/rustup CARGO_HOME=/data/cargo
cargo test -j 24 -p memra-server --lib -- --test-threads 8 ctx 2>&1 | tail -45
cp /tmp/worker.rs.green crates/memra-server/src/worker.rs
