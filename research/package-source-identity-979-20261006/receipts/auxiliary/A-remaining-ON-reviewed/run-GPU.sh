#!/usr/bin/env bash
set -euo pipefail
exec /home/evidence-user/.local/bin/gpulock run --ttl 11m --for "Root remainingONLY package335e cacheON prefixspec engagement" -- /usr/bin/env CUDA_VISIBLE_DEVICES=GPU-1a3cbffc-29df-926c-df5c-29b4c210ef5d MEMRA_RIG_LOCK_FD=9 /usr/bin/python3 -I -B /home/evidence-user/.cache/evidence-manager-20261006/A-remaining-ON-reviewed/run_cell.py --inputs /home/evidence-user/.cache/evidence-manager-20261006/A-serving-v3-reviewed/ROOT-SERVING-INPUTS.json
