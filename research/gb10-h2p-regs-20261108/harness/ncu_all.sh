#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: ncu --set full at depth 131,072 on h2p and each lever's arms.
set -u
D=$HOME/.cache/memra-attn3
for spec in "h2p fa_prefill_qw_h2p" "rb cand_rb" "qb12 cand_qb12" "qb14 cand_qb14" "pb0q15 cand_pb0q15" "pb0 cand_pb0"; do
  set -- $spec
  bash "$D/box/ncu_cand.sh" "$1" "$2" 131072
done
# the ptxas 13.4 assembly of the same PTX (driver-API launch; the kernel name filter matches it too)
PROBE_CUBIN=$D/probe/x134.cubin bash "$D/box/ncu_cand.sh" xh2p fa_prefill_qw_h2p 131072
echo NCU-ALL-DONE
