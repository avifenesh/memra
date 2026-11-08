set -u
i=0
for a in base fix fix base; do
  i=$((i+1))
  if [ -e /home/avifenesh/.cache/memra-pools/out-a2/ttft/YIELD ]; then echo "yield before boot $i $(date -Is)"; break; fi
  echo "boot $i $a start $(date -Is)"
  CELL_PORT=18177 python3 /home/avifenesh/.cache/memra-pools/wt/research/pool-resume-cost-20261107/cost_cell.py /home/avifenesh/.cache/memra-pools/out-a2/ttft/b$i-$a /home/avifenesh/.cache/memra-pools/cells-work/bins/b30d0b92ad7d0a26ff8f3c4aa29a0d4bcd63fba9-120a/memra-server /home/avifenesh/.cache/memra-pools/cells-work/bins/dev-608404e003/memra-server     chat-dspark-long:$a >> /home/avifenesh/.cache/memra-pools/out-a2/ttft/ttft.log 2>&1
  echo "boot $i $a exit $? $(date -Is)"
done
echo TTFT_DONE
