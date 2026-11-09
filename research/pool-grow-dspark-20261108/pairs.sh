#!/usr/bin/env bash
# memra#1093 follow-up: warm == cold pair cells for the DSpark-pool grow on the RTX 5090.
# #1103's nine cases (its cells now expect turn 2 to resume where turn 1's park is too small for
# its cap: a grow) plus the agentic shape (the same max_tokens every turn, so every resume grows)
# greedy, T=1, penalized and raw, and the cache-preferred shape (a decode budget under prompt/8
# with the session's own boundary entry in the prefix cache), where the veto now yields to the
# rewound session. usage: pairs.sh BINARY OUT   (a YIELD file in OUT stops between cases)
set -u
B=$1 O=$2
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"
for c in \
  "dspark-1024-rw=$B|dspark|96,1024,512|_EXPECT=rewind:3:2:1" \
  "dspark-1024-t1-rw=$B|dspark|96,1024,512|_TEMP=1.0 _SEED=1234 _EXPECT=rewind:3:2:1" \
  "dspark-chain-rw=$B|dspark|1536,448,448|_EXPECT=rewind:3:2" \
  "dspark-chain-t1-rw=$B|dspark|1536,448,448|_TEMP=1.0 _SEED=1234 _EXPECT=rewind:3:2" \
  "dspark-1024-t1-pen-rw=$B|dspark|96,1024,512|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=rewind:3:2:1" \
  "dspark-chain-t1-pen-rw=$B|dspark|1536,448,448|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=rewind:3:2" \
  "dspark-raw-1024-rw=$B|dspark-raw|96,1024,512|_EXPECT=rewind:3:2:1" \
  "dspark-raw-chain-t1-rw=$B|dspark-raw|1536,448,448|_TEMP=1.0 _SEED=1234 _EXPECT=rewind:3:2" \
  "dspark-256-nopc-aff0-rw=$B|dspark|96,256,96|MEMRA_PREFIX_CACHE_MB=0 MEMRA_AFFINITY=0 _EXPECT=skip" \
  "dspark-agentic-rw=$B|dspark|512,512,512|_EXPECT=rewind:3:2:2" \
  "dspark-agentic-t1-rw=$B|dspark|512,512,512|_TEMP=1.0 _SEED=1234 _EXPECT=rewind:3:2:2" \
  "dspark-agentic-t1-pen-rw=$B|dspark|512,512,512|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=rewind:3:2:2" \
  "dspark-raw-agentic-t1-rw=$B|dspark-raw|512,512,512|_TEMP=1.0 _SEED=1234 _EXPECT=rewind:3:2:2" \
  "dspark-veto-rw=$B|dspark|256,256,256|_EXPECT=rewind:3:2:2" \
  "dspark-veto-t1-rw=$B|dspark|256,256,256|_TEMP=1.0 _SEED=1234 _EXPECT=rewind:3:2:2"; do
  if [ -e "$O/YIELD" ]; then echo "yield before ${c%%=*} $(date -Is)"; break; fi
  echo "case ${c%%=*} start $(date -Is)"
  CELL_PORT=18175 python3 "$H/pool_pair_cells.py" "$O/pairs" "$c" >> "$O/pairs.log" 2>&1
  echo "case ${c%%=*} exit $? $(date -Is)"
done
echo PAIRS_DONE
