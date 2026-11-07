#!/usr/bin/env bash
# Preview (not scored): the mixed shape once per arm on the first candidate (7b385110e4, before the
# review's reclaim changes; MEMRA_ENV_AUDIT=warn because its door registry predates the FLAGS row).
set -u
H=<harness>
B=<bins>
O=<runs>/preview1a
mkdir -p "$O"
bash $H/arm.sh gds $B/memra-server-new-7b385110e4 $O mixed MEMRA_MAX_SESSIONS=12 MEMRA_ENV_AUDIT=warn
bash $H/arm.sh lru $B/memra-server-new-7b385110e4 $O mixed MEMRA_MAX_SESSIONS=12 MEMRA_ENV_AUDIT=warn MEMRA_PREFIX_EVICT=lru
echo PREVIEW_DONE
