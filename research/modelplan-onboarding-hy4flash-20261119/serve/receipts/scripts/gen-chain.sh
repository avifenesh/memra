#!/usr/bin/env bash
# GB10 box: the route's own greedy ids per request (hy4f-gpu --gen), then tapes.json and route-gen.json
set -u
D=$HOME/hy4flash/pr9
R=$HOME/hy4flash/pr7/requests
L=$HOME/hy4flash/logs/pr9-gen.log
: > $L
for spec in "chat 32" "long 64" "long-split 72"; do
  set -- $spec
  waitfor --quiet --timeout 60m --every 10s --no-proc 'target/release/(hy4f-gpu|hy4f-run|memra-server)' > /dev/null
  waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
  echo "GEN $1 $2 $(date '+%F %T')" >> $L
  systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -p CPUWeight=50 -- \
    $D/target/release/hy4f-gpu $HOME/hy4flash/hf-stage --request $R/$1.json --gen $2 \
    > $D/gen-$1.txt 2> $HOME/hy4flash/logs/pr9-gen-$1.log
  echo "EXIT $1 rc=$? $(date '+%F %T')" >> $L
done
python3 - <<'PY' >> $L 2>&1
import json, os
D = os.path.expanduser("~/hy4flash/pr9")
P8 = os.path.expanduser("~/hy4flash/pr8/native")
gen = {}
for n in ("chat", "long", "long-split"):
    line = [l for l in open(f"{D}/gen-{n}.txt") if l.startswith("gen ")][0]
    gen[n] = [int(x) for x in line[4:].strip().split(",")]
json.dump(gen, open(f"{D}/route-gen.json", "w"))
tapes = {}
for n, t, f in (("chat", "chat1f", None), ("long", "long1", "long1f"), ("long-split", "long-split1", "long-split1f")):
    tapes[n] = json.load(open(f"{P8}/{t}/tape.json"))["argmax"]
    if f:
        forced = json.load(open(f"{P8}/{f}/tape.json"))["argmax"]
        diff = [i for i, (a, b) in enumerate(zip(tapes[n], forced)) if a != b]
        print("ARGMAX free-vs-forced", n, "differ at", len(diff), "positions", diff[:10])
json.dump(tapes, open(f"{D}/tapes.json", "w"))
print("GEN", {k: v[:12] for k, v in gen.items()})
PY
echo "GEN_DONE $(date '+%F %T')" >> $L
