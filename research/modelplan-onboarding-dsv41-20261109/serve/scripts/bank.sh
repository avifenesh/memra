#!/bin/bash
# Gather the serve lane's receipts on the run box into one directory for the repo: the residency gate, the checkpoint
# and serve gates' bundle, the chat gates, the suites and one line per GPU lease. Paths are made relative to the lane's
# run directory, card ids and trailing blanks are dropped; large files are gzipped; the full logits stay on the box (their sha256 is
# banked).
# usage: bank.sh OUT_DIR
. "${LANE_ENV:?LANE_ENV names the machine file that sets ROOT GPU_UUID BOX_LOCK QUIET_FILE CORES GPU_RUN BUILD}"
set -u
S=$ROOT/serve
O=$1
mkdir -p $O/residency $O/verify-res $O/chat $O/tests
clean() { sed -e "s#$ROOT/##g" -e 's#GPU-[0-9a-f]\{8\}-[0-9a-f-]\{27\}#<card>#g' -e 's/[ \t]*$//' "$@"; }
put() { local src=$1 dst=$2; [ -f "$src" ] && clean "$src" > "$dst"; }
put $S/pin_probe.out $O/pin_probe.out
# residency
put $S/res_gate.out $O/residency/res_gate.out
put $S/logs/res-forced.log $O/residency/res-forced.log
put $S/logs/res-greedy-floor.log $O/residency/res-greedy-floor.log
(cd $S/res-forced && find . -type f | sort | xargs sha256sum) > $O/residency/res-forced.sha256
(cd $ROOT/dec/native-final && find . -type f | sort | xargs sha256sum) > $O/residency/dec-native-final.sha256
(cd $S/res-greedy-floor && find . -type f \( -name '*.i32' -o -name '*.f32' -o -name '*.f64' \) | sort | xargs sha256sum) \
  > $O/residency/res-greedy-floor.sha256
(cd $ROOT/dec/greedy && for c in capital gsm hebrew sys_json multiturn; do sha256sum $c/*.i32 $c/*.f32 $c/*.f64; done \
  | sed 's#  #  ./#') > $O/residency/dec-greedy-5.sha256
# checkpoint and serve gates
V=$S/verify-res
for f in artifact.lock checkpoint-parity.tsv gates.txt serve-gate.tsv serve-response.json oracle-request.tsv \
  tokenizer-contract.tsv; do
  [ -f $V/$f ] && cp $V/$f $O/verify-res/
done
[ -f $V/native-oracle.tsv ] && gzip -nc $V/native-oracle.tsv > $O/verify-res/native-oracle.tsv.gz
[ -f $V/serve.log ] && grep -v '^\[meter\]' $V/serve.log | clean > $O/verify-res/serve.log
put $S/logs/verify-checkpoint-res.log $O/verify-res/verify-checkpoint-res.log
put $S/logs/verify-serve.log $O/verify-res/verify-serve.log
put $S/serve_gate_all.out $O/serve_gate_all.out
# chat
put $S/chat-gate.json $O/chat/chat-gate.json
put $S/logs/chat-gate.log $O/chat/chat-gate.log
grep -v '^\[meter\]' $S/logs/serve-chat.log | clean > $O/chat/serve-chat.log
put $S/rates.json $O/chat/rates.json
# the shared benchmark client against the route
B=$S/bench-client
mkdir -p $O/bench-client
for f in summary.json quality.json meta.json samples.jsonl; do put $B/$f $O/bench-client/$f; done
put $S/logs/bench-client.log $O/bench-client/bench-client.log
put $S/serve_gate_bench.out $O/bench-client/serve_gate_bench.out
[ -f $S/logs/serve-bench.log ] && grep -v '^\[meter\]' $S/logs/serve-bench.log | clean > $O/bench-client/serve-bench.log
# suites
put $S/test_build.out $O/tests/test_build.out
put $S/test_run.out $O/tests/test_run.out
for f in test-server test-cli test-engine-dsv41 clippy fmt; do
  [ -f $S/logs/$f.log ] && grep -E 'test result:|^error|^warning: unused|Finished|running [0-9]+ tests' $S/logs/$f.log \
    | clean > $O/tests/$f.summary
done
# one line per lease of this lane: name, start (UTC), seconds, exit code, card name
python3 - "$ROOT/leases" > $O/leases.tsv <<'PY'
import json, pathlib, sys, time
print("lease\tstarted_utc\tseconds\texit_code\ttimed_out\tcard")
for d in sorted(pathlib.Path(sys.argv[1]).glob("serve-*")):
    f = d / "lease.json"
    if not f.is_file():
        continue
    j = json.loads(f.read_text())
    t0, t1 = j.get("started_unix"), j.get("finished_unix")
    print("\t".join([d.name, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)) if t0 else "-",
                     f"{t1 - t0:.0f}" if t0 and t1 else "-", str(j.get("exit_code")), str(j.get("timed_out")),
                     ",".join(x["name"] for x in j.get("devices", []))]))
PY
if grep -rlE "$ROOT|GPU-[0-9a-f]{8}-" $O; then echo "machine paths or card ids left in the bank"; exit 1; fi
du -sh $O
echo BANKED
