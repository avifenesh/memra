#!/usr/bin/env bash
# Laptop: what main's commits 63c351a2f4..0db086f87a (the landing of memra#1310 and memra#1317, merged into
# memra#1315 after the support record's pin 6371a9975b) can reach in the deepseek-v4.1 and deepseek-v4 routes.
# Reads git objects only. usage: reach.sh REPO_DIR > reach.txt
set -u
cd "$1" || exit 1
A=63c351a2f4; B=0db086f87a
ROUTE="crates/memra-server/src/dsv41_serve.rs crates/memra-engine/src/dsv41_gpu.rs crates/memra-engine/src/dsv41_gpu crates/memra-server/src/dsv4_serve.rs crates/memra-server/src/dsv4_admit.rs"
export PATH=/usr/bin:$PATH
echo "## range"; git log --format='%h %s' $A..$B | cut -c1-150
echo; echo "## files changed under crates/, docs/, tools/"; git diff --name-only $A $B -- crates docs tools
echo; echo "## tools/cell_evidence.py need --base $A --head $B"; python3 tools/cell_evidence.py need --base $A --head $B 2>&1
echo; echo "## route files changed in the range"; git diff --name-only $A $B -- $ROUTE crates/memra-engine/src/dsv4_gpu.rs
strip() {  # the file at REV without comment lines and blank lines
  git show "$1:$2" | grep -v -E '^\s*//' | grep -v -E '^\s*$'
}
for f in crates/memra-engine/src/dsv41_gpu/experts.rs crates/memra-engine/src/dsv4_gpu.rs; do
  echo; echo "## $f: added and removed lines"
  git diff $A $B -- $f | grep -E '^[+-]' | grep -v -E '^(\+\+\+|---)'
  echo "## $f: lines that are not comments, sha256 at each end"
  echo "$A $(strip $A $f | sha256sum | cut -d' ' -f1)"
  echo "$B $(strip $B $f | sha256sum | cut -d' ' -f1)"
done
echo; echo "## what the route files name outside their own modules, among the files the range changed"
echo "# unified_memory, spec:: (non-comment lines of the route files at $B)"
for f in $(git ls-tree -r --name-only $B -- $ROUTE); do
  git show $B:$f | grep -n -E 'unified_memory|spec::' | grep -v -E '^[0-9]+:\s*//' | sed "s|^|$f:|"
done
echo "# (end of list)"
echo "# crate::worker items the server's route files use"
for f in crates/memra-server/src/dsv41_serve.rs crates/memra-server/src/dsv4_serve.rs crates/memra-server/src/dsv4_admit.rs; do
  git show $B:$f | grep -n -E 'use crate::worker|worker::[A-Za-z_0-9]+' | grep -v -E '^[0-9]+:\s*//' | sed "s|^|$f:|" | cut -c1-200
done
ITEMS="EngineError Event EventSender ModelCaps Request SpecUsage ENGINE_MAX_CTX resolve_ctx event_channel utf8_delta stop_token_reason resolve_env_ctx release_request_reservation prompt_source_limit_error plain_chat_render_path meminfo_available_bytes"
echo; echo "## those items in the range's diff of worker.rs (added or removed lines naming them)"
for it in $ITEMS; do
  n=$(git diff $A $B -- crates/memra-server/src/worker.rs | grep -E '^[+-]' | grep -v -E '^(\+\+\+|---)' | grep -c -E "\b$it\b")
  echo "$it: $n"
done
echo; echo "## the engine's lib.rs in the range (added and removed lines)"
git diff $A $B -- crates/memra-engine/src/lib.rs | grep -E '^[+-]' | grep -v -E '^(\+\+\+|---)'
echo; echo "## Engine::unified_memory or is_unified named by the route files at $B"
for f in $(git ls-tree -r --name-only $B -- $ROUTE); do
  git show $B:$f | grep -n -E '\.unified_memory\(\)|is_unified' | sed "s|^|$f:|"
done
echo "# (end of list)"
