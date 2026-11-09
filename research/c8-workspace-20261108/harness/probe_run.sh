#!/usr/bin/env bash
# The workspace probe (measurement build, not scored): agentic c8 full, agentic c1 prime pass
# (solo primes, 1.4k-17k prompts), the mixed shape. Run under one gpulock lease.
set -u
H=<harness>
BIN=${1:?probe binary}
O=<runs>/probe
mkdir -p "$O"
bash "$H/arm2.sh" c8 "$BIN" "$O" agentic:8:1.0:768
bash "$H/arm2.sh" c1-prime "$BIN" "$O" agentic:1:1.0:768:prime
bash "$H/arm2.sh" mixed "$BIN" "$O" mixed MEMRA_MAX_SESSIONS=12
echo PROBE_DONE
