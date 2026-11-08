set -u
for c in   "dspark-1024-rw=/home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server|dspark|96,1024,512|_EXPECT=rewind:3:1"   "dspark-1024-t1-rw=/home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server|dspark|96,1024,512|_TEMP=1.0 _SEED=1234 _EXPECT=rewind:3:1"   "dspark-chain-rw=/home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server|dspark|1536,448,448|_EXPECT=rewind:3:2"   "dspark-chain-t1-rw=/home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server|dspark|1536,448,448|_TEMP=1.0 _SEED=1234 _EXPECT=rewind:3:2"   "dspark-raw-1024-rw=/home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server|dspark-raw|96,1024,512|_EXPECT=rewind:3:1"   "dspark-raw-chain-t1-rw=/home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server|dspark-raw|1536,448,448|_TEMP=1.0 _SEED=1234 _EXPECT=rewind:3:2"   "dspark-256-nopc-aff0-rw=/home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server|dspark|96,256,96|MEMRA_PREFIX_CACHE_MB=0 MEMRA_AFFINITY=0 _EXPECT=skip"; do
  if [ -e /home/avifenesh/.cache/memra-pools/out-a2/YIELD ]; then echo "yield before ${c%%=*} $(date -Is)"; break; fi
  echo "case ${c%%=*} start $(date -Is)"
  CELL_PORT=18175 python3 /home/avifenesh/.cache/memra-pools/wt/research/pool-rewind-dspark-20261108/pool_pair_cells.py /home/avifenesh/.cache/memra-pools/out-a2/pairs "$c" >> /home/avifenesh/.cache/memra-pools/out-a2/pairs.log 2>&1
  echo "case ${c%%=*} exit $? $(date -Is)"
done
echo PAIRS_DONE
