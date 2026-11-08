set -u
for c in   "dspark-1024-t1-pen-rw=/home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server|dspark|96,1024,512|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=rewind:3:1"   "dspark-chain-t1-pen-rw=/home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server|dspark|1536,448,448|_TEMP=1.0 _SEED=1234 _FREQ=0.3 _PRES=0.2 _EXPECT=rewind:3:2"; do
  if [ -e /home/avifenesh/.cache/memra-pools/out-a2/pen/YIELD ]; then echo "yield before ${c%%=*} $(date -Is)"; break; fi
  echo "case ${c%%=*} start $(date -Is)"
  CELL_PORT=18175 python3 /home/avifenesh/.cache/memra-pools/wt/research/pool-rewind-dspark-20261108/pool_pair_cells.py /home/avifenesh/.cache/memra-pools/out-a2/pen/pairs "$c" >> /home/avifenesh/.cache/memra-pools/out-a2/pen/pairs.log 2>&1
  echo "case ${c%%=*} exit $? $(date -Is)"
done
echo PEN_DONE
