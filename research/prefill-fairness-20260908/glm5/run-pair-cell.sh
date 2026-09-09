#!/usr/bin/env bash
# Queue on the assigned non-production pair. No endpoint or provider is hardcoded.
set -euo pipefail
: "${PAIR_PROFILE:?JSON env map for this exact PP2 or TP2 placement}"
: "${PAIR_METADATA:?models.toml for the pinned artifact}"
: "${PAIR_RECEIPTS:?absolute new receipt directory}"
: "${PAIR_BINARY:?absolute release binary path}"
: "${PAIR_BINARY_SHA256:?sha256 of the reviewed binary}"
: "${PAIR_PROMPT_256K:?pinned vendor-default request JSON with generation room}"
: "${PAIR_PROMPT_1M:?pinned vendor-default request JSON with generation room}"
: "${PAIR_GATE_BINARY:?absolute glm5-tp2-box-probe binary from the same source}"
: "${PAIR_GATE_BINARY_SHA256:?sha256 of that probe binary}"
: "${PAIR_GATE_PROMPTS:?directory of pinned multichunk text prompts}"
export MEMRA_CUDA_ARCH=100a MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
root=$(git rev-parse --show-toplevel)
cd "$root"
count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
test "$count" -eq 2
# Record both immutable source inputs and placement before any scored launch.
mkdir -p "$PAIR_RECEIPTS"
git rev-parse HEAD > "$PAIR_RECEIPTS/engine.sha"
sha256sum "$PAIR_PROFILE" "$PAIR_METADATA" "$PAIR_PROMPT_256K" "$PAIR_PROMPT_1M" \
  > "$PAIR_RECEIPTS/inputs.sha256"
# The probe runs the real saved-prime path and an independent original-loop
# oracle, interleaving a peer prime plus four decode steps at each boundary.
flock "$MEMRA_GPU_LOCK" nice -n 19 python3 - "$PAIR_PROFILE" "$PAIR_GATE_BINARY" \
  "$PAIR_GATE_BINARY_SHA256" "$PAIR_GATE_PROMPTS" "$PAIR_RECEIPTS/rendezvous" <<'PY_GATE'
import hashlib, json, os, pathlib, subprocess, sys
env = json.loads(pathlib.Path(sys.argv[1]).read_text())
binary = pathlib.Path(sys.argv[2])
with binary.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == sys.argv[3]
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
model = env['MEMRA_MODELS'].split('=', 1)[1]
out = pathlib.Path(sys.argv[5]); out.mkdir(parents=True, exist_ok=False)
env.update(BOXP_MODE='prime-walker', MEMRA_TICK_TRACE='1', MEMRA_PRIME_CHUNK='4096')
with (out/'gate.log').open('w') as log:
    subprocess.run([str(binary), model, sys.argv[4], str(out)],
        env={'PATH': os.environ['PATH'], **env}, stdout=log, stderr=subprocess.STDOUT, check=True)
assert '[glm5-prime-walker-check] PASS' in (out/'gate.log').read_text()
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
PY_GATE
cells=research/prefill-fairness-20260908/glm5/http_cells.py
common=(--binary "$PAIR_BINARY" --binary-sha256 "$PAIR_BINARY_SHA256"
        --profile "$PAIR_PROFILE" --metadata "$PAIR_METADATA" --chunk 4096)
nice -n 19 python3 "$cells" "${common[@]}" --cell exactness \
  --boot-order 0,1 --out "$PAIR_RECEIPTS/exactness"
# Each context gets three fresh boots per arm, with the same fixed arrivals.
nice -n 19 python3 "$cells" "${common[@]}" --cell latency --boot-order 0,1,1,0,0,1 \
  --long-request "$PAIR_PROMPT_256K" --out "$PAIR_RECEIPTS/256k"
nice -n 19 python3 "$cells" "${common[@]}" --cell latency --boot-order 0,1,1,0,0,1 \
  --long-request "$PAIR_PROMPT_1M" --out "$PAIR_RECEIPTS/1m"
