#!/usr/bin/env bash
# Gemma-4 E4B qualification cells on the local RTX 5090 (run under gpulock).
# usage: e4b_gpu_cells.sh <repo-root> <target-dir> <inputs-dir> <evidence-dir> [cells...]
# cells: rungen argmax spec verify graph serve identity (default: the engine cells)
set -uo pipefail
ROOT="$1"; TGT="$2"; IN="$3"; EV="$4"; shift 4
CELLS=("$@"); [ ${#CELLS[@]} -eq 0 ] && CELLS=(rungen argmax spec verify graph)
MODEL=/data/ai-ml/hf-models/gemma4-e4b-qat-gguf/gemma-4-E4B_q4_0-it.gguf
DRAFT=/data/ai-ml/hf-models/gemma4-e4b-qat-gguf/drafter/MTP/gemma-4-E4B-it-assistant.Q8_0.gguf
BIN="$TGT/release"
IDS="$IN/e4b-chat-watercycle-ids.txt"
DEPTH="$IN/depth-prompt-1736-ids.txt"
mkdir -p "$EV"
cd "$ROOT" || exit 2
SUMMARY="$EV/summary.txt"
note() { echo "$*" | tee -a "$SUMMARY"; }
note "== e4b cells $(date -u +%FT%TZ) cells=${CELLS[*]} head=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.used,temperature.gpu,clocks.sm --format=csv,noheader >> "$SUMMARY"
for cell in "${CELLS[@]}"; do
  case "$cell" in
    rungen)
      # fast-gate e4b probe shape: MEMRA_NGEN=20 MEMRA_NMEASURE=0 run-gen on the pinned gemma ids.
      # shellcheck disable=SC2046
      env MEMRA_NGEN=20 MEMRA_NMEASURE=0 timeout 900 "$BIN/run-gen" "$MODEL" $(cat "$IDS") > "$EV/rungen.log" 2>&1
      rc=$?
      match=$(grep -cE "argmax=[0-9]+ +decode argmax=[0-9]+ .* MATCH$" "$EV/rungen.log")
      structured=$(grep -c "MISMATCH-STRUCTURED" "$EV/rungen.log")
      toks=$(grep -oE "^ *tokens: \[[0-9, ]*\]" "$EV/rungen.log" | head -1 | sed 's/^ *//')
      golden=$(sed -n '2p' tools/fast-gate/goldens/e4b.tokens)
      same=$([ "$toks" = "$golden" ] && echo yes || echo no)
      note "rungen rc=$rc match_lines=$match mismatch_structured=$structured golden_identical=$same"
      note "  got:    $toks"
      note "  golden: $golden"
      grep -E 'tok/s|maxdiff|argmax=|gemma4-e4b' "$EV/rungen.log" | head -8 | sed 's/^/  /' | tee -a "$SUMMARY"
      ;;
    argmax)
      env CARGO_TARGET_DIR="$TGT" tools/argmax-margin-gate.sh "$MODEL" --prompt "$IN/board-2048.txt" \
        --logdir "$EV/argmax" > "$EV/argmax.log" 2>&1
      rc=$?
      note "argmax-margin-gate rc=$rc"
      grep -E 'SUMMARY|PRIME-SUMMARY|PASS|FAIL|SKIP' "$EV/argmax.log" | head -8 | sed 's/^/  /' | tee -a "$SUMMARY"
      ;;
    spec)
      for k in 1 2 3 4 5 6 7 8; do
        # shellcheck disable=SC2046
        env MEMRA_SPEC=$k MEMRA_DRAFT="$DRAFT" MEMRA_NGEN=64 timeout 900 \
          "$BIN/gemma-gate" "$MODEL" $(cat "$IDS") > "$EV/spec-k$k.log" 2>&1
        rc=$?
        line=$(grep -E "stream agreement" "$EV/spec-k$k.log" | tail -1)
        acc=$(grep -E "^\[gemma-spec\] rounds=" "$EV/spec-k$k.log" | tail -1 | tr -s ' ')
        note "spec K=$k rc=$rc $line | $acc"
      done
      ;;
    verify)
      # shellcheck disable=SC2046
      env MEMRA_VERIFY_GATE=7 timeout 900 "$BIN/gemma-gate" "$MODEL" $(cat "$DEPTH") > "$EV/verify-k7.log" 2>&1
      rc=$?
      note "verify-gate K=7 depth rc=$rc $(grep -E 'VERIFY-GATE' "$EV/verify-k7.log" | tail -1)"
      ;;
    graph)
      # shellcheck disable=SC2046
      env MEMRA_E4B_GRAPH_GATE=256 timeout 900 "$BIN/gemma-gate" "$MODEL" $(cat "$IDS") > "$EV/graph-256.log" 2>&1
      rc=$?
      note "e4b-graph-gate 256 rc=$rc $(grep -E 'E4B-GRAPH-GATE' "$EV/graph-256.log" | tail -1)"
      ;;
    pp)
      # MEMRA_PP_STAGES=2 must refuse the E4B load by operation name before any upload.
      env MEMRA_PP_STAGES=2 MEMRA_NGEN=4 MEMRA_NMEASURE=0 timeout 600 "$BIN/run-gen" "$MODEL" 2 3 4 > "$EV/pp2-refusal.log" 2>&1
      rc=$?
      note "pp2-refusal rc=$rc $(grep -oE 'pipeline placement is unsupported[^;]*;[^]]*\]' "$EV/pp2-refusal.log" | head -1 | cut -c1-400)"
      ;;
    serve)
      # Same env as the build, so serve-smoke's own cargo build is a no-op.
      env MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR="$TGT" CARGO_BUILD_JOBS=2 \
        timeout 1800 tools/serve-smoke.sh "$MODEL" /nonexistent-no-nextn-draft > "$EV/serve-smoke.log" 2>&1
      rc=$?
      cp /tmp/serve-smoke.log "$EV/serve-smoke-server-last.log" 2>/dev/null
      note "serve-smoke rc=$rc"
      grep -E '^  (ok|FAIL):|^==|SKIP|FAILED|PASS' "$EV/serve-smoke.log" | head -40 | sed 's/^/  /' | tee -a "$SUMMARY"
      ;;
    identity)
      bash "$(dirname "$0")/e4b_serve_identity.sh" "$ROOT" "$MODEL" "$EV" "$BIN/memra-server" > "$EV/identity.log" 2>&1
      rc=$?
      note "served identity rc=$rc $(grep -E '^IDENTITY' "$EV/identity.log" | tail -1)"
      ;;
    serve-gemma)
      # serve-smoke with its gemma4 arm pointed at E4B (GEMMA_MODEL) and the Qwen3.6-35B cold
      # arm off (a different model; its replay script lives under research/).
      env MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR="$TGT" CARGO_BUILD_JOBS=2 \
        GEMMA_MODEL="$MODEL" MEMRA_Q35_COLD_MODEL=/nonexistent-q35 \
        timeout 1800 tools/serve-smoke.sh "$MODEL" /nonexistent-no-nextn-draft > "$EV/serve-smoke-gemma.log" 2>&1
      rc=$?
      cp /tmp/serve-smoke.log "$EV/serve-smoke-gemma-server-last.log" 2>/dev/null
      note "serve-smoke (gemma4 arm on E4B) rc=$rc"
      grep -E '^  (ok|FAIL):|^==|SKIP|FAILED|PASS|failed' "$EV/serve-smoke-gemma.log" | head -60 | sed 's/^/  /' | tee -a "$SUMMARY"
      ;;
    *) note "unknown cell $cell" ;;
  esac
done
note "== done $(date -u +%FT%TZ)"
