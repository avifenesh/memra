#!/bin/bash
# The four SIB-7 sibling gates again, after pair-cells6.sh, on binaries with the census fix
# (b1470ea8d1, built at the stack top cd63409246; bin7/, sha256 in bin7/bin7.sha256). pair-cells6.sh
# deletes dsv4-nvfp4/ when it ends, so dsv4-keep/ holds hard links to the same files (no extra disk).
# Same env, cards and lock as pair-cells6's SIB-7 cells. No cell starts after 10:40; every cell is
# over by 10:52 (the 11:00 guard); dsv4-keep/ and bin7/ are deleted at the end on every path.
set -u
W=<pair-kit>
B=$W/bin7; M=$W/dsv4-keep; T=$W/tape.txt; O=$W/out6x
WF=<box-home>/.local/bin/waitfor
export PATH=<box-home>/.local/bin:$PATH
mkdir -p $O; S=$O/status.txt
cleanup() { rm -rf <pair-kit>/dsv4-keep <pair-kit>/bin7
  echo "extra6 cleaned $(date +%H:%M:%S); $(df -BG /mnt/Data | tail -1 | awk '{print $4}') free" >> $S; }
trap cleanup EXIT
echo "extra6 armed $(date '+%m-%d %H:%M:%S')" >> $S
LAST=$(date -d '2026-11-07 10:40' +%s); END=$(date -d '2026-11-07 10:52' +%s)
$WF --until 'test -s <pair-kit>/bin7/bin7.sha256' --timeout 3h --every 30s > $O/wait.log 2>&1 \
  || { echo "extra6: bin7 never arrived" >> $S; exit 3; }
$WF --no-proc 'bash <pair-kit>/pair-cells6.sh' --timeout 3h --every 30s >> $O/wait.log 2>&1
[ "$(date +%s)" -lt "$LAST" ] || { echo "extra6: pair-cells6 ended too late ($(date +%H:%M))" >> $S; exit 3; }
exec 9>/tmp/memra-gpu.lock; flock -w 600 9 || { echo "extra6: memra-gpu.lock busy" >> $S; exit 3; }
( cd $B && sha256sum -c --quiet bin7.sha256 ) > $O/bin-check.log 2>&1 || { echo "extra6: bin7 fails its manifest" >> $S; exit 3; }
[ -d $M ] || { echo "extra6: no dsv4-keep" >> $S; exit 3; }
export CUDA_VISIBLE_DEVICES=0,1 MEMRA_DSV4_DECODE_PATH=device
TPEP="MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
SIB="MEMRA_DSV4_DOTS_ARM=f32x MEMRA_DSV4_DRAFTER=off MEMRA_MOE_F16G=2 MEMRA_F16G_SK=32 MEMRA_DSV4_SAMPLER=device"
echo "extra6 start $(date +%H:%M:%S) bins verified" >> $S
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,fan.speed,power.draw,clocks.sm,memory.used --format=csv -i 0,1 -lms 1000 > $O/thermal.csv 2>&1 &
TPID=$!
cell() {  # cell NAME ENV... -- CMD...
    local name=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local now=$(date +%s)
    if [ "$now" -ge "$LAST" ]; then echo "$name SKIP deadline" >> $S; return; fi
    local to=$((END - now)); [ $to -gt 1200 ] && to=1200
    env "${envs[@]}" timeout "$to" "$@" > $O/$name.log 2>&1
    local rc=$?
    echo "$name rc=$rc $(( $(date +%s) - now ))s :: $(grep -a -E -m3 'PASS|FAIL|panicked|assertion|left:|right:' $O/$name.log | cut -c1-160 | tr '\n' '|') :: census $(grep -a -c -E '^census|^HC24 census' $O/$name.log)" >> $S
}
# shellcheck disable=SC2086
cell SIB-7x-dense-fast $TPEP $SIB -- $B/dsv4_dense_fast_gate $M $T $O/dense-fast
cell SIB-7x-dense-exact-tail $TPEP $SIB -- $B/dsv4_dense_exact_tail_gate $M $T $O/dense-exact-tail
cell SIB-7x-compose-cadence $TPEP $SIB -- $B/dsv4_compose_cadence_dense_gate $M $T $O/compose-cadence --defaults
cell SIB-7x-hcdot-defaults $TPEP $SIB -- $B/dsv4_hc_dot_split_gate $M $T $O/hcdot --defaults
kill $TPID 2>/dev/null
echo "extra6 done $(date +%H:%M:%S)" >> $S
