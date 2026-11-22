#!/bin/bash
# The four-card box, 2026-11-22: the deepseek-v4.1 served batch above 8 rows on four ranks (DSV41-ROWS-ABOVE-8-FOUR-RANKS),
# perf-bins/r16 (632ea9564b on main 63f806baf9: MEMRA_DSV41_BATCH_ROWS admits 1..16, unset 8). One window:
#   1. bytes: dsv41-spec batch-rows wide16 on four ranks, device-push (calls of 16 rows; every row equal to its
#      sequence's own one-token call). A red gate stops the window before any timing.
#   2. dsv41-spec batch-time distinct8 + distinct16 (n4t: one call of 16 against two of 8), device-push.
#   3. the four-rank served chat gate (serve_gate9r.sh GATE_LAYERS=full, fresh native references), unset batch rows: memra#1337's serve smoke.
#   3b. the served 16-wide byte gate (wide_serve.sh / wide_gate.py) at MEMRA_DSV41_BATCH_ROWS=16, four ranks.
#   4. TIMED: served A/B, A = MEMRA_DSV41_BATCH_ROWS=8, B = 16, the same binary, order A B B A A B (three a side when
#      the window allows; a run that does not fit is skipped and the A/B is then a first look only), one boot a run,
#      suite conc32,dconc16 one sample a boot, 32 sessions, MEMRA_CTX=8192, device-push. 250 ms telemetry of every card.
# Starts on /mnt/Data/work/evals/dssparks.window-0123 newer than this runner's armed marker, all four cards without a
# compute process on two reads 30 s apart and gpu0..3.lock free (flock -n; any held: abort, nothing run). Hard stop
# HARD_S (1200 s) after the start: each step gets what is left and is skipped when too little is. At the end releases
# gpu0..3.lock and touches dsv41.r16-4card-done. CPUs 24-31,56-63. GIVE_UP default tomorrow 06:00.
# Start: (cd /home/access/dsv41 && nohup setsid bash spec/scripts/w15_r16.sh) > spec/logs/w15.log 2>&1 < /dev/null &
set -u
D=/home/access/dsv41
EV=/mnt/Data/work/evals
P=$D/perf-bins/r16
S=$D/spec
R=$D/receipts-perf/w15-r16
HARD_S=${HARD_S:-1200}
CPUS=24-31,56-63
export LANE=$D TBIN=$P
mkdir -p $R
cd $D || exit 1
say() { echo "$(date +%T) $*" | tee -a $R/summary.txt; }
finish() { touch $EV/dsv41.r16-4card-done; say "touched dsv41.r16-4card-done"; }
for b in memra-server dsv41-spec dsv41-run; do [ -x $P/$b ] || { say "missing $P/$b"; exit 2; }; done
say "r16 $(cat $P/SOURCE)"; (cd $P && sha256sum memra-server dsv41-spec dsv41-run) >> $R/summary.txt
grep '^batch-rows wide16 ' $S/plans/n4b.plan > $R/w16.plan
[ -s $R/w16.plan ] || { say "no wide16 job in n4b.plan"; exit 2; }
touch $S/logs/w15.armed
gu=$(date -d "${GIVE_UP:-tomorrow 06:00}" +%s)
open() { [ -e $EV/dssparks.window-0123 ] && [ $EV/dssparks.window-0123 -nt $S/logs/w15.armed ]; }
free4() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
say "armed; waiting for dssparks.window-0123; give up $(date -d @$gu '+%F %H:%M')"
while :; do
  [ "$(date +%s)" -lt $gu ] || { say "no window by the give-up time: not run"; exit 3; }
  [ -e $QUIET_FILE ] && { say "QUIET set: not run"; exit 3; }
  if open && free4; then sleep 30; free4 && break; fi
  sleep 10
done
for i in 0 1 2 3; do
  exec {fd}>>"$EV/gpu$i.lock"
  if ! flock -n "$fd"; then say "ABORT: gpu$i.lock is held by someone else; nothing run"; finish; exit 2; fi
  eval "FD$i=$fd"
done
# ds-sparks may write the granted window length in seconds into the marker (40 min = 2400 fits A B B A A B); else HARD_S.
g=$(tr -dc 0-9 < $EV/dssparks.window-0123 | head -c 5)
[ -n "$g" ] && [ "$g" -ge 600 ] && [ "$g" -le 3600 ] && HARD_S=$g
t0=$(date +%s); dl=$((t0 + HARD_S))
left() { echo $((dl - $(date +%s))); }
release() { for i in 0 1 2 3; do eval "fd=\$FD$i"; flock -u "$fd"; eval "exec $fd>&-"; done; say "all four box locks released ($(( $(date +%s) - t0 )) s in the window)"; }
say "W15-START: window from ds-sparks; all four box locks held; host free $(free -g | awk '/Mem:/{print $7}') GiB; load $(cut -d' ' -f1-3 /proc/loadavg); hard stop $(date -d @$dl +%T)"
nvidia-smi --query-gpu=index,name,power.limit,temperature.gpu,clocks.sm --format=csv,noheader >> $R/summary.txt
nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,clocks_event_reasons.active \
  --format=csv -lms 250 > $R/telemetry.csv 2>&1 &
TEL=$!
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3

spec() { # NAME PLAN ARGS...: dsv41-spec on four ranks, 40 layers resident, device-push
  local name=$1 plan=$2 o=$R/$1; shift 2
  mkdir -p $o
  MEMRA_TP_TRANSPORT=device-push MEMRA_FULL_PREC=1 MEMRA_DSV41_TOKEN_MAP=$D/oracle/engram_constants.token_map.i32 \
    taskset -c $CPUS timeout -k 10 $(( $(left) - 5 )) $P/dsv41-spec $D/DeepSeek-V4.1-Flash-1017 --plan $plan --out $o \
    --devices 0,1,2,3 --layers 40 --resident --no-draft --stop-on-fail "$@" > $o/stdout.jsonl 2> $o/stderr.log
  local rc=$?
  python3 - $o/stdout.jsonl > $o/verdict.txt <<'EOF'
import json, sys
bad = n = 0
for line in open(sys.argv[1]):
    if not line.startswith("{"):
        continue
    d = json.loads(line)
    if d.get("stopped"):
        bad += 1
        print("STOPPED, skipped:", d["skipped"])
        continue
    n += 1
    ok = d.get("pass")
    bad += 0 if ok else 1
    if d.get("job") == "batch-rows":
        print(f"{'PASS' if ok else 'RED '} batch-rows {d['id']}: {d['sequences']} sequences, rows {d['rows']}, "
              f"logits equal {d['logit_rows_equal']}, taps equal {d['tap_rows_equal']}, device argmax equal "
              f"{d['device_argmax_equal']}, calls by width {d['calls_by_width']}, first_bad {d['first_bad']}")
    elif d.get("job") == "batch-time":
        ws = [w for w in (1, 2, 4, 8, 16, 32) if f"argmax_{w}_ms" in d]
        for m in ("argmax", "fwd", "batched", "alone", "chunk8"):
            if any(f"{m}_{w}_ms" in d for w in ws):
                v = " / ".join(f"{d[f'{m}_{w}_ms']:.2f}" if f"{m}_{w}_ms" in d else "-" for w in ws)
                print(f"{'PASS' if ok else 'RED '} batch-time {d['id']} {m:8s} ms at {ws}: {v}")
    else:
        print(("PASS " if ok else "RED  ") + json.dumps(d)[:400])
print(f"jobs {n}, red {bad}")
EOF
  say "$name rc $rc: $(tail -n 1 $o/verdict.txt)"; grep -v '^jobs ' $o/verdict.txt | cut -c1-300 | tee -a $R/summary.txt
  return $rc
}

# 1. bytes of a 16-row call
spec wide16 $R/w16.plan --batch-max 16
w16=$?
if [ $w16 -ne 0 ] || ! grep -q 'red 0$' $R/wide16/verdict.txt; then
  say "wide16 RED or incomplete: no timing in this window"; kill $TEL; release; finish; say W15-DONE; exit 1
fi
# 2. a 16-row call against two of 8
[ $(left) -gt 600 ] && spec batchtime $S/plans/n4t.plan || say "batchtime: skipped or failed"
# 3. the four-rank served chat gate
# The lane's full-trunk native references (dec/greedy, 2026-11-09) predate the CUTLASS lane's numerics: a 9-layer
# rehearsal on card 3 failed solo/stops against the old 9-layer references and passed 9 of 9 against fresh ones. So the
# gate makes fresh references with this binary's dsv41-run on the four ranks (GATE_REF an empty dir: nothing linked).
if [ $(left) -gt 540 ]; then
  mkdir -p $R/noref $R/greedyfull
  GATE_LAYERS=full GATE_REF=$R/noref GATE_REF_DIR=$R/greedyfull GATE_NATIVE_DEVICES=0,1,2,3 PORT=18394 taskset -c $CPUS \
    timeout -k 10 420 bash $S/scripts/serve_gate9r.sh w15-gate-n4 \
    MEMRA_DSV41_DEVICES=0,1,2,3 MEMRA_TP_TRANSPORT=device-push > $R/gate-n4.log 2>&1
  say "served gate on 0,1,2,3 rc $?: $(grep -c ': PASS' $R/gate-n4.log) pass lines; $(grep -h 'CHAT-GATE' $R/gate-n4.log | tail -n 1)"
  cp -a $S/out/w15-gate-n4 $R/gate-n4 2>/dev/null
fi
# 3b. the served 16-wide byte gate at 16 rows: 16 prompts greedy alone, then all at once (one 16-row call), equal bytes.
# A red gate stops the window before the A/B.
if [ $(left) -gt 420 ]; then
  TBIN=$P PORT=18393 taskset -c $CPUS timeout -k 10 240 bash $S/scripts/wide_serve.sh $R/wide-n4 0,1,2,3 \
    MEMRA_DSV41_BATCH_ROWS=16 MEMRA_TP_TRANSPORT=device-push > $R/wide-n4.log 2>&1
  wr=$?
  say "served wide gate on 0,1,2,3 rc $wr: $(grep -h 'wide-gate:' $R/wide-n4.log | tail -n 1); $(grep -o 'decode_rows=16' $R/wide-n4/server.log | wc -l) steps of 16 rows"
  if [ $wr -ne 0 ]; then say "wide gate RED: no A/B in this window"; kill $TEL; release; finish; say W15-DONE; exit 1; fi
fi
# 4. the served A/B
i=0
for arm in A B B A A B; do
  i=$((i + 1))
  [ $(left) -gt 210 ] || { say "ab run$i-$arm: skipped, $(left) s left"; continue; }
  rows=8; [ $arm = B ] && rows=16
  DEVICES=0,1,2,3 SESSIONS=32 LINKLOG=1 EXTRA_BENCH=--no-smoke-gate PORT=18395 taskset -c $CPUS \
    timeout -k 15 $(( $(left) - 10 )) bash $S/scripts/serve_bench.sh w15-ab/run$i-$arm conc32,dconc16 1 \
    MEMRA_DSV41_BATCH_ROWS=$rows MEMRA_CTX=8192 MEMRA_TP_TRANSPORT=device-push > $R/ab-run$i-$arm.log 2>&1
  say "ab run$i-$arm (rows $rows) rc $?"
  python3 - $S/out/w15-ab/run$i-$arm/bench/summary.json <<'PY' 2>&1 | tee -a $R/summary.txt
import json, sys
w = json.load(open(sys.argv[1]))["workloads"]
g = lambda v: v.get("median") if isinstance(v, dict) else v
for k, v in w.items():
    print(f"  {k}: aggregate {g(v.get('agg_out_tok_s'))} tok/s, per request {g(v.get('per_req_decode_tok_s_median'))}, ttft max {g(v.get('ttft_max_s'))} s, errors {v.get('n_err', 0)}")
PY
done
cp -a $S/out/w15-ab $R/ab 2>/dev/null
kill $TEL
release
finish
say W15-DONE
