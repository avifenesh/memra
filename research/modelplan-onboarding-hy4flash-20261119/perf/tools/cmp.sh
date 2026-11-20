#!/usr/bin/env bash
# b23d: compare a native tape's arrays bitwise with a PR 8 reference tape. usage: cmp.sh TAG REF
D=$HOME/hy4flash/perf/native/$1; R=$HOME/hy4flash/pr8ref/$2
ok=1
for f in logits.f32 router_scores.f32 router_sel.i32 indexer_logits.f32 indexer_sel.i32; do
  if cmp -s $D/$f $R/$f; then echo "EQUAL $f"; else echo "DIFF $f"; ok=0; fi
done
python3 - "$D/tape.json" "$R/tape.json" <<'PY'
import json,sys
a,b=(json.load(open(p)) for p in sys.argv[1:3])
print("engram equal" if a["engram"]==b["engram"] else "engram DIFF", "argmax equal" if a["argmax"]==b["argmax"] else "argmax DIFF")
print("seconds", a["seconds"], "ref", b["seconds"])
PY
[ $ok = 1 ] && echo BITWISE_EQUAL || echo NOT_EQUAL
