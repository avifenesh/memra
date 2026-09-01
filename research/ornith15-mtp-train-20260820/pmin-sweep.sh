#!/usr/bin/env bash
# Confidence-gated drafting sweep (owner direction 2026-08-20): memra ships the
# gate (MEMRA_SPEC_PMIN early chain stop + MEMRA_SPEC_PMIN0 zero-draft rounds)
# but defaults it OFF — every round drafts full K regardless of confidence.
# llama.cpp's 35B spec win rides exactly this gate (spec.rs comment, mean draft
# len 2.5). Sweep p-min on the v2-head GGUF, greedy 256-tok probes.
# DIRECTIONAL numbers (other card may be busy): acceptance + drafted/rounds are
# timing-independent; tok/s here picks the candidate, the winner gets a clean
# interleaved pair on a quiet box before any default moves.
set -uo pipefail
cd "$HOME/models/ornith15"
mkdir -p pmin-sweep
BINS=$HOME/memra-src/target/release
M=$HOME/models/ornith15/Ornith-1.5-35B-A3B-NVFP4-MTP-v2.gguf
ADDR=127.0.0.1:8098
export CUDA_VISIBLE_DEVICES=0

probe() { # $1 label
  for p in "Write a Python function that parses an ISO-8601 timestamp string and returns seconds since the Unix epoch, handling timezone offsets. Include tests." "You are a coding agent in a Rust repository. The test test_kv_append_wrap fails with an off-by-one at the ring boundary. Describe, step by step, how you would locate and fix the bug, then give the patch."; do
    curl -sf -m 600 "http://$ADDR/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":256,\"temperature\":0}" \
      | python3 -c "
import json, sys, time
r = json.load(sys.stdin)
u = r['usage']
s = u.get('spec') or {}
print(json.dumps({'label': '$1', 'ct': u['completion_tokens'],
                  'elapsed_s': u.get('elapsed_s'), 'spec': s}))"
  done
}

run_arm() { # $1 label, $2 env
  env $2 MEMRA_MODELS="m=$M" MEMRA_ADDR=$ADDR "$BINS/memra-server" > "pmin-sweep/server-$1.log" 2>&1 &
  local pid=$!
  for _ in $(seq 240); do curl -sf "http://$ADDR/health" >/dev/null 2>&1 && break; sleep 2; done
  echo "--- arm $1"
  { time probe "$1"; } 2>&1
  kill $pid 2>/dev/null; wait $pid 2>/dev/null || true
}

run_arm "plain" "MEMRA_SERVE_SPEC=0"
run_arm "pmin0.0" ""
for pm in 0.3 0.5 0.7 0.85; do
  run_arm "pmin$pm" "MEMRA_SPEC_PMIN=$pm MEMRA_SPEC_PMIN0=1"
done
echo "PMIN-SWEEP DONE"
