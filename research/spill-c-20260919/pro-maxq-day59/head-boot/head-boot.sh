#!/bin/bash
# DAY59 section 5's owed check: one boot of run-gen from the Max-Q arm commit (d9942ea8d7) on a Max-Q card, at the
# pressure shape, reading the slot cache's boot line. Untimed; checks the arm and its source, not the timing.
# Waits for the scheduler's manual gap to close, then runs on GPU 2 under gpu2.lock and the memra lock, only if
# it can start before 14:55 (GPU 2 belongs to the training session's eval batches from about 15:30).
set -u
D=/mnt/Data/work/memra-maxq/head-boot
WF=/home/<user>/.local/bin/waitfor
S=$D/status.txt
M=/mnt/Data/work/memra-maxq/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
echo "$(date +%F_%T) armed" >> $S
$WF --until 'grep -q "gap closed" /mnt/Data/work/memra-maxq/gap/status.txt' --timeout 8h --every 60s > $D/wait.log 2>&1 \
    || { echo "$(date +%F_%T) gap did not close in time" >> $S; exit 3; }
[ "$(date +%H%M)" -lt 1455 ] || { echo "$(date +%F_%T) too late (after 14:55); owed to the next window" >> $S; exit 3; }
exec 9> /mnt/Data/work/evals/gpu2.lock
flock -w 120 9 || { echo "$(date +%F_%T) gpu2.lock busy" >> $S; exit 3; }
exec 8> /tmp/memra-gpu.lock
flock -w 120 8 || { echo "$(date +%F_%T) memra-gpu.lock busy" >> $S; exit 3; }
used=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -lt 256 ] || { echo "$(date +%F_%T) GPU 2 not empty ($used MiB)" >> $S; exit 3; }
nvidia-smi -i 2 --query-gpu=index,name,power.limit,memory.used,temperature.gpu --format=csv,noheader >> $S
sha256sum $D/run-gen-head >> $S
env -u MEMRA_MOE_PREFETCH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 \
    MEMRA_MOE_SLOTS=9986 timeout 300 $D/run-gen-head $M 55 88 13 > $D/boot.log 2>&1
rc=$?
echo "$(date +%F_%T) run-gen rc=$rc boot line: $(grep -a -m1 'in-token prefetch' $D/boot.log)" >> $S
echo "$(date +%F_%T) done; GPU 2 $(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader)" >> $S
