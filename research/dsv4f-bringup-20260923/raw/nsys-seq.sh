#!/usr/bin/env bash
cd /root/box
EPENV="MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_EP=pair MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_SAMPLE_SORT=radix MEMRA_DSV4_INDEXER_SCORE=tiled MEMRA_DSV4_SINK_SCORE=tiled"
NSYS=1 ./prof.sh nsys-naked current MEMRA_DSV4_DRAFTER=dspark
NSYS=1 ./prof.sh nsys-ep-composed composed $EPENV
echo NSYS_DONE >> /root/rcpt/prof/summary.txt
