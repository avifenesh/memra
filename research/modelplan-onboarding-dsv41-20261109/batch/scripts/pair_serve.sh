#!/bin/bash
# The served cell on a pair of cards: memra-server on the 9-layer resident trunk, batching on, through the lane's
# shared client (decode_c1, conc8, dconc8; 3 samples a run), three arms interleaved one / peer / device / device / peer / one:
#   one     one rank on the pair's first card
#   peer    two ranks, a rank a card, MEMRA_TP_TRANSPORT=peer-pull
#   device  two ranks, a rank a card, MEMRA_TP_TRANSPORT=device-push
# Every run is its own two-card lease through the 4-card lane's wrapper (both box locks held), so the arms see the
# same pair and the same cores (TP4_CPUS).
# usage: pair_serve.sh TAG PAIR        e.g. TP4_CPUS=28-31,60-63 pair_serve.sh k16 01
# env: ARMS (default "one peer device device peer one"), RUN0 (the runs are numbered from RUN0 + 1), LINKLOG=1 (see
#      serve_bench.sh: a warm-up request and the cards' PCIe link state around it)
set -u
drops() { journalctl -k --no-pager 2>/dev/null | grep drop_caches | tail -n 2 | awk '{print $3" "$(NF-1)" "$NF}' | tr '\n' ';'; }
echo "$(date +%H:%M:%S) page-cache drops before: $(drops)"
TAG=$1; P=$2
S=$LANE/spec
TL=$LANE/bin/tp4_lease.sh
export TBIN=$S/bin/$TAG
[ -x $TBIN/memra-server ] || { echo "missing $TBIN/memra-server"; exit 2; }
i=${RUN0:-0}
for arm in ${ARMS:-one peer device device peer one}; do
  i=$((i + 1))
  case $arm in
    one) dev=0; tr= ;;
    peer) dev=0,1; tr=MEMRA_TP_TRANSPORT=peer-pull ;;
    device) dev=0,1; tr=MEMRA_TP_TRANSPORT=device-push ;;
  esac
  DEVICES=$dev EXTRA_BENCH=--no-smoke-gate bash $TL spec-ps-r$i 600 $P bash $S/scripts/serve_bench.sh \
    ps-$TAG/r$i-$arm decode_c1,conc8,dconc8 3 MEMRA_DSV41_TRUNK_LAYERS=9 MEMRA_CTX=8192 $tr > $S/logs/ps-$TAG-r$i-$arm.log 2>&1
  echo "$(date +%H:%M:%S) run $i arm $arm rc $? $(grep -h "rank(s) on devices" $S/out/ps-$TAG/r$i-$arm/server.log | cut -c1-60)"
done
python3 $S/scripts/ab_summary.py $S/out/ps-$TAG
echo "PAIR-SERVE-DONE $TAG"
echo "$(date +%H:%M:%S) page-cache drops after: $(drops)"
