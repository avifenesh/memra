#!/bin/bash
# 5090 (under gpulock): the vendor F.linear's fp32 summation order at the router's and compressor's shapes (~1 min).
set -u
D=/home/avifenesh/.cache/dsv41-memra/r1-dig/bench
cd $D || exit 1
OMP_NUM_THREADS=8 /home/avifenesh/projects/ml100/.venv/bin/python -I $D/cublas_order.py $D/cublas-order-5090.txt \
  > $D/cublas-order-5090.log 2>&1
echo "rc $?" >> $D/cublas-order-5090.log
