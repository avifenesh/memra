#!/bin/bash
# Inside one lease of four cards: the served route on the tensor split, the full resident trunk, a rank a card,
# batching on (calls of 8 rows), MEMRA_TP_TRANSPORT from N4_TRANSPORT (default device-push).
#  1. identity: the lane's chat gate against the full-trunk native references (render, solo == native, five at once
#     == solo, stream, sampled, stops, penalties, budget). A red gate stops the step before any timing.
#  2. the shared client: decode_c1, conc8, dconc8, conc32 (N4_SUITE), N4_SAMPLES samples (default 3), 32 sessions.
# usage: n4_serve.sh TAG
set -u
TAG=$1
S=$LANE/spec
export TBIN=$S/bin/$TAG
TR=${N4_TRANSPORT:-device-push}
t0=$(date +%s)
GATE_LAYERS=full bash $S/scripts/serve_gate9.sh gate-$TAG-n4 MEMRA_DSV41_DEVICES=0,1,2,3 MEMRA_TP_TRANSPORT=$TR \
  > $S/logs/gate-$TAG-n4.log 2>&1
rc=$?
echo "$(date +%H:%M:%S) four-rank served gate rc $rc, $(grep -c PASS $S/logs/gate-$TAG-n4.log) pass lines, $(( $(date +%s) - t0 )) s"
grep -h "PASS\|FAIL" $S/logs/gate-$TAG-n4.log | cut -c1-160
if [ $rc -ne 0 ]; then echo "N4-SERVE RED $TAG: the gate failed, no timing"; tail -n 6 $S/logs/gate-$TAG-n4.log | cut -c1-300; exit 1; fi
DEVICES=0,1,2,3 SESSIONS=32 LINKLOG=1 EXTRA_BENCH=--no-smoke-gate bash $S/scripts/serve_bench.sh n4s-$TAG \
  ${N4_SUITE:-decode_c1,conc8,dconc8,conc32} ${N4_SAMPLES:-3} MEMRA_CTX=8192 MEMRA_TP_TRANSPORT=$TR > $S/logs/n4s-$TAG.log 2>&1
rc=$?
echo "$(date +%H:%M:%S) four-rank served bench rc $rc, $(( $(date +%s) - t0 )) s"
python3 - $S/out/n4s-$TAG/bench/summary.json <<'PY'
import json, sys
w = json.load(open(sys.argv[1]))["workloads"]
g = lambda v: v.get("median") if isinstance(v, dict) else v
for k, v in w.items():
    if "agg_out_tok_s" in v:
        print(f"{k}: aggregate {g(v['agg_out_tok_s'])} tok/s, per request {g(v.get('per_req_decode_tok_s_median'))}, ttft max {g(v.get('ttft_max_s'))} s, errors {v.get('n_err', 0)}")
    else:
        print(f"{k}: decode {g(v.get('decode_tok_s'))} tok/s, ttft {g(v.get('ttft_s'))} s")
PY
echo "N4-SERVE-DONE $TAG rc $rc"
