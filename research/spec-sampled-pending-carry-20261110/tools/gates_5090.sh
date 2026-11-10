#!/usr/bin/env bash
# Runs on the RTX 5090 machine: the pass/fail gates of this lane, one gate run per card lease,
# taken only when the card is free and nothing queued on it is a landing. No timed cell here.
# usage: B=<dir with base/ cand/ sg/ binaries> O=<output dir> W=<this tree> \
#        M27=<Qwen3.8-27B MTP gguf> D27=<its trimmed drafter> M9=<Qwen3.5-9B MTP gguf> D9=<its own-trim drafter> \
#        gates_5090.sh STEP...
# STEP: boundary-base boundary-cand inv-base inv-cand trim27-base trim27-cand trim9-base
#       trim9-cand seams-base seams-cand hit hit-teeth sample-check kernel-check
#       close-base close-cand demote-base demote-cand sg-<variant>
set -u
B=${B:?}; O=${O:?}; W=${W:?}; M27=${M27:?}; D27=${D27:-}; M9=${M9:-}; D9=${D9:-}
T=$W/research/spec-sampled-pending-carry-20261110/tools
mkdir -p "$O"
cd "$W" || exit 1
g() { local name=$1 ttl=$2 what=$3; shift 3
  waitfor --until "gpulock status | grep -q '^5090: free' && ! gpulock status | grep -qi landing" --every 20s --timeout 240m --quiet || return 1
  gpulock run --ttl "$ttl" --wait 3m --for "sampled pending carry: $what" -- "$@" > "$O/$name.log" 2>&1
  echo "$name exit $?" >> "$O/steps.rc"
}
for step in "$@"; do
  case $step in
    boundary-base)
      g "$step" 40m "burst-boundary gate, base" python3 tools/spec-burst-boundary-gate.py "$B/base/memra-server" "$M27" --out "$O/boundary-base" --carry no ;;
    boundary-cand)
      g "$step" 40m "burst-boundary gate, candidate" python3 tools/spec-burst-boundary-gate.py "$B/cand/memra-server" "$M27" --out "$O/boundary-cand" --compare "$O/boundary-base/runs.jsonl" ;;
    inv-base|inv-cand) arm=${step#inv-}
      g "$step" 40m "invariance gate, $arm" python3 tools/seeded-invariance-gate.py "$B/$arm/memra-server" "$M27" --out "$O/$step" ;;
    trim27-base|trim27-cand) arm=${step#trim27-}
      g "$step" 30m "invariance gate, $arm, trimmed drafter" python3 tools/seeded-invariance-gate.py "$B/$arm/memra-server" "$M27" --out "$O/$step" --cases ref,1,2,3 --port 18198 --env MEMRA_MTP_DRAFT="$D27" ;;
    trim9-base|trim9-cand) arm=${step#trim9-}
      g "$step" 30m "invariance gate, $arm, 9B with its own-trim drafter" python3 tools/seeded-invariance-gate.py "$B/$arm/memra-server" "$M9" --out "$O/$step" --cases ref,1,2,3 --port 18198 --env MEMRA_MTP_DRAFT="$D9" ;;
    seams-base|seams-cand) arm=${step#seams-}
      g "$step" 30m "rollback seam rows, $arm" python3 "$T/seam_rows.py" "$B/$arm/memra-server" "$M27" "$O/$step" ;;
    hit) mkdir -p "$O/hit"
      g hit 30m "spec-on-cache-hit gate, qwen arm" env MEMRA_GPU_LOCK="$O/hit/private.lock" MEMRA_GATE_MTP_DRAFT="$D9" tools/spec-on-cache-hit-gate.sh qwen "$M9" "$B/cand/memra-server" "$O/hit/qwen" ;;
    hit-teeth) mkdir -p "$O/hit"
      g hit-teeth 30m "spec-on-cache-hit gate, teeth arm" env MEMRA_GPU_LOCK="$O/hit/private.lock" MEMRA_HITGATE_TEETH=1 MEMRA_GATE_MTP_DRAFT="$D9" tools/spec-on-cache-hit-gate.sh qwen "$M9" "$B/cand/memra-server" "$O/hit/qwen-teeth" ;;
    sample-check)
      g sample-check 10m "sample_check" "$B/cand/sample_check" ;;
    kernel-check)
      g kernel-check 20m "kernel-check" "$B/cand/kernel-check" --require-manifest tools/kernel-check-27b.cells --require-manifest tools/kernel-check-step35.cells ;;
    close-base|close-cand|demote-base|demote-cand) mode=${step%-*}; arm=${step#*-}
      g "$step" 25m "end-of-request cell, $mode, $arm" env CELL_MODEL="$M27" MODE="$mode" python3 "$T/close_cell.py" "$O/close" "$B/$arm/memra-server" "$arm" ;;
    sg-*) v=${step#sg-}
      g "$step" 15m "session-gate, $v" env MEMRA_SPEC_K=3 "$B/sg/session-gate-$v" "$M27" ;;
  esac
done
