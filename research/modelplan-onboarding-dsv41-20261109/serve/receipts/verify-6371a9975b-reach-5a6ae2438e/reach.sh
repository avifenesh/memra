#!/usr/bin/env bash
# Laptop: what main's commits 0db086f87a..5a6ae2438e (the landing of memra#1297 and memra#1320, merged into
# memra#1315 after the support record's pin 6371a9975b) can reach in the deepseek-v4.1 and deepseek-v4 routes.
# Reads git objects only. usage: reach2.sh REPO_DIR > reach.txt
set -u
cd "$1" || exit 1
A=0db086f87a; B=5a6ae2438e
ROUTE="crates/memra-server/src/dsv41_serve.rs crates/memra-engine/src/dsv41_gpu.rs crates/memra-engine/src/dsv41_gpu crates/memra-server/src/dsv4_serve.rs crates/memra-server/src/dsv4_admit.rs crates/memra-engine/src/dsv4_gpu.rs"
HF=crates/memra-engine/src/hybrid_forward.rs
GATE=crates/memra-engine/src/bin/dsv41_gate.rs
export PATH=/usr/bin:$PATH
echo "## range"; git log --format='%h %s' $A..$B | cut -c1-150
echo; echo "## files changed under crates/, docs/, tools/, cu/"; git diff --name-only $A $B -- crates docs tools cu
echo; echo "## files changed outside them, by top directories"; git diff --name-only $A $B -- . ':!crates' ':!docs' ':!tools' ':!cu' | awk -F/ '{print $1"/"$2}' | sort | uniq -c
echo; echo "## tools/change_impact.py --base $A --head $B"; python3 tools/change_impact.py --base $A --head $B 2>&1
echo; echo "## tools/cell_evidence.py need --base $A --head $B"; python3 tools/cell_evidence.py need --base $A --head $B 2>&1
echo; echo "## route files changed in the range"; git diff --name-only $A $B -- $ROUTE; echo "# (end of list)"
echo; echo "## the route files at $B"; git ls-tree -r --name-only $B -- $ROUTE

echo; echo "## $HF: the diff's hunks (the enclosing item of each)"
git diff $A $B -- $HF | grep -E '^@@' | cut -c1-160
echo; echo "## $HF: added and removed lines outside its test module (old lines below 32000)"
git diff -U0 $A $B -- $HF | awk '/^@@/ { split($2, a, ","); line = substr(a[1], 2) + 0; keep = (line < 32000); next } keep && /^[+-]/ && !/^(\+\+\+|---)/' | cut -c1-170
echo; echo "## the items those lines change, and every file under crates/ and cu/ that names each at $B (file:count)"
for it in step_gemm_prime_ranges step_gemm_call_rows_max CallRowsSchedule StepGemm fixed_prime_chunk_ranges prime_cache_overlaid_inner PrimeWorkspaceShape step_gemm; do
  echo "# $it"; git grep -c -w -e "$it" $B -- crates cu | sed "s|^$B:||"
done
echo; echo "## where step_gemm is set to anything but the literal false, and where a PrimeWorkspaceShape is made from a model"
git grep -n -E 'step_gemm:|prime_workspace_shape\(\)' $B -- crates | sed "s|^$B:||" | grep -v 'step_gemm: false' | cut -c1-170
echo; echo "## the receiver of prime_workspace_shape and of the GEMM prime entry: the impl blocks that hold them"
for f in $HF crates/memra-engine/src/hybrid.rs; do
  git show $B:$f | awk -v f=$f '/^impl / { blk = $0; at = NR } /pub fn prime_workspace_shape\(|fn prime_cache_overlaid_inner\(|fn uses_sliding_gated_moe_program\(/ { line = $0; sub(/^[ ]+/, "", line); print f":"NR": "line"   <- in `"blk"` (line "at")" }' | cut -c1-200
done
echo; echo "## non-comment lines of the route files at $B that name the hybrid model or any of the changed items"
for f in $(git ls-tree -r --name-only $B -- $ROUTE); do
  git show $B:$f | grep -n -E 'hybrid_forward|HybridModel|PrimeWorkspaceShape|step_gemm|StepGemm|CallRowsSchedule|prime_call_rows|prime_chunk_ranges|fixed_prime_chunk|prime_cache_overlaid|prime_workspace_shape|PRIME_CHUNK_MAX_TOKENS|PRIME_MIN_T|uses_sliding_gated_moe_program' | grep -v -E '^[0-9]+:\s*//' | sed "s|^|$f:|" | cut -c1-200
done
echo "# (end of list)"
echo "# the same names in comment lines of the route files (for the reader)"
for f in $(git ls-tree -r --name-only $B -- $ROUTE); do
  git show $B:$f | grep -n -E 'hybrid_forward|HybridModel|PrimeWorkspaceShape|step_gemm' | grep -E '^[0-9]+:\s*//' | sed "s|^|$f:|" | cut -c1-200
done
echo "# (end of list)"
echo; echo "## crate::worker items the server's route files use, and worker.rs in the range"
for f in crates/memra-server/src/dsv41_serve.rs crates/memra-server/src/dsv4_serve.rs crates/memra-server/src/dsv4_admit.rs; do
  git show $B:$f | grep -o -E 'worker::[A-Za-z_0-9]+' | sort | uniq -c | sed "s|^|$f: |"
done
echo "worker.rs changed in the range: $(git diff --name-only $A $B -- crates/memra-server/src/worker.rs | wc -l) file(s)"
echo "# every call of PrimeWorkspaceShape::call_rows in worker.rs (line: the call   <- the fn that holds it; tests included)"
git show $B:crates/memra-server/src/worker.rs | awk '/^[ ]*(pub(\([a-z]+\))? )?(async )?fn [a-z_0-9]+/ { fn = $0; sub(/^[ ]+/, "", fn) } /\.call_rows\(/ { line = $0; sub(/^[ ]+/, "", line); print NR": "line"   <- "fn }' | cut -c1-230
echo; echo "## $GATE: the gate binary. Where its name appears outside its own file at $B (non-comment lines)"
git grep -n -E 'dsv41_gate|dsv41-gate' $B -- crates cu ':!'$GATE | sed "s|^$B:||" | grep -v -E ':[0-9]+:\s*(//|\*)' | cut -c1-170
echo "# (end of list)"
echo "# its diff's hunks"
git diff $A $B -- $GATE | grep -E '^@@' | cut -c1-160
echo "# does the library or the server include the file (mod, #[path], include!)"
git grep -n -E 'mod dsv41_gate|path = ".*dsv41_gate|include!\(.*dsv41_gate' $B -- crates | sed "s|^$B:||" | grep -v 'Cargo.toml'
echo "# (end of list)"
