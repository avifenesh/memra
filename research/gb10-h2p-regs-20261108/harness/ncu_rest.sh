#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: the remaining ncu arms at depth 131,072, h2p first as the
# same-unit baseline.
set -u
D=$HOME/.cache/memra-attn3
for spec in "h2p fa_prefill_qw_h2p" "qb14 cand_qb14" "pb0q15 cand_pb0q15" "pb0 cand_pb0"; do
  set -- $spec
  bash "$D/box/ncu_cand.sh" "$1" "$2" 131072
done
PROBE_CUBIN=$D/probe/x134.cubin bash "$D/box/ncu_cand.sh" xh2p fa_prefill_qw_h2p 131072
echo NCU-REST-DONE
