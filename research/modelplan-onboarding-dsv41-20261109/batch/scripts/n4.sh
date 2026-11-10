#!/bin/bash
# Inside one lease of several cards (a rank a card): the batched decode call on the full resident trunk.
#  A. device-push, plan n4a, calls of 8 rows at most: the byte gate first (batch-rows chat8, mixed, nine, window:
#     every batched row must equal its sequence's own one-token call on the same ranks), then batch-time
#     distinct8 (1/2/4/8 sequences) and distinct16 (adds one call of 16 against two of 8), each with the launch
#     tally and every kernel's own time. A red gate job stops the run there (--stop-on-fail) and this script
#     prints the job's line (its first_bad names the sequence, step, row and position) and exits 1.
#  B. peer-pull, plan n4b with --batch-max 16: batch-rows chat8 again (the transport is not the variable) and
#     batch-rows wide16 (calls of 16 rows).
# usage: n4.sh TAG        env: DEVICES (default 0,1,2,3), LAYERS (default 40), SPEC_BIN (default spec/bin/TAG/dsv41-spec),
#        STEPS (default "A B")
# Outputs: spec/out/TAG-n4a, spec/out/TAG-n4b; the tables: python3 spec/scripts/cmp_arms.py spec/out TAG-n4a
set -u
TAG=$1
D=$LANE
S=$D/spec
BIN=${SPEC_BIN:-$S/bin/$TAG/dsv41-spec}
DEV=${DEVICES:-0,1,2,3}
LAYERS=${LAYERS:-40}
STEPS=${STEPS:-A B}
[ -x "$BIN" ] || { echo "missing $BIN"; exit 2; }
t0=$(date +%s)
run() { # NAME TRANSPORT PLAN ARGS...
  local name=$1 tr=$2 plan=$3 o; shift 3
  o=$S/out/$TAG-$name
  mkdir -p $o
  echo "$(date +%H:%M:%S) $name: $tr, plan $plan, devices $DEV, $LAYERS layers $*"
  MEMRA_TP_TRANSPORT=$tr MEMRA_FULL_PREC=1 MEMRA_DSV41_TOKEN_MAP=$D/oracle/engram_constants.token_map.i32 \
    $BIN $D/DeepSeek-V4.1-Flash-1017 --plan $S/plans/$plan.plan --out $o --devices $DEV --layers $LAYERS --resident \
    --no-draft --stop-on-fail "$@" > $o/stdout.jsonl 2> $o/stderr.log
  local rc=$?
  grep -h "dsv41-tp\]\|dsv41-resident\]\|up in" $o/stderr.log | cut -c1-260
  python3 - $o/stdout.jsonl <<'EOF'
import json, sys
bad = 0
for line in open(sys.argv[1]):
    if not line.startswith("{"):
        print(line.strip()[:200])
        continue
    d = json.loads(line)
    if d.get("stopped"):
        print("STOPPED, skipped:", d["skipped"])
        continue
    ok = d.get("pass")
    bad += 0 if ok else 1
    if d.get("job") == "batch-rows":
        print(f"{'PASS' if ok else 'RED '} batch-rows {d['id']}: {d['sequences']} sequences, rows {d['rows']}, "
              f"logits equal {d['logit_rows_equal']}, taps equal {d['tap_rows_equal']}, device argmax equal "
              f"{d['device_argmax_equal']}, calls by width {d['calls_by_width']}, first_bad {d['first_bad']}")
    elif d.get("job") == "batch-time":
        ws = [w for w in (1, 2, 4, 8, 16, 32) if f"argmax_{w}_ms" in d]
        for m in ("argmax", "fwd", "batched", "alone", "chunk8"):
            v = " / ".join(f"{d[f'{m}_{w}_ms']:.2f}" if f"{m}_{w}_ms" in d else "-" for w in ws)
            if not any(f"{m}_{w}_ms" in d for w in ws):
                continue
            print(f"{'PASS' if ok else 'RED '} batch-time {d['id']} {m:8s} ms at {ws}: {v}")
        print(f"     distinct experts a layer at the widest: {d.get('distinct_experts_per_layer_mean', 0):.1f}")
    else:
        print(("PASS " if ok else "RED  ") + json.dumps(d)[:400])
sys.exit(1 if bad else 0)
EOF
  local prc=$?
  echo "$(date +%H:%M:%S) $name: binary rc $rc, jobs rc $prc, $(( $(date +%s) - t0 )) s since the start"
  [ $rc -eq 0 ] && [ $prc -eq 0 ]
}
case " $STEPS " in *" A "*)
  if ! run n4a device-push n4a; then
    echo "N4 RED in step A (device-push): the run stopped at the job above; stderr tail:"
    tail -n 5 $S/out/$TAG-n4a/stderr.log | cut -c1-300
    exit 1
  fi ;;
esac
case " $STEPS " in *" B "*)
  if ! run n4b peer-pull n4b --batch-max 16; then
    echo "N4 RED in step B (peer-pull): stderr tail:"
    tail -n 5 $S/out/$TAG-n4b/stderr.log | cut -c1-300
    exit 1
  fi ;;
esac
python3 $S/scripts/cmp_arms.py $S/out $TAG-n4a > $S/logs/$TAG-n4a-tables.txt 2>&1
echo "N4-DONE $TAG green, $(( $(date +%s) - t0 )) s; tables: $S/logs/$TAG-n4a-tables.txt"
