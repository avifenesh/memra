#!/bin/bash
# lane H's batching gate for a P head before the landing builds on it, on one ink2-gpu (BIN), to OUT:
# 1. batched == solo, both classes, short and long schedules; red arms batch-rows-rotated and
#    batch-attn-one-call (the dispatch before every batched row ran its solo attention) must differ;
# 2. past the load's context (the serve route's sessions): sessions of 4096 positions on a load of 64
#    and on a load of 8192 give the same solo digest, batched == solo there too; red arm
#    t4-scratch-absent (the dispatch before t4s) must change the digest;
# 3. batch_bisect.sh, the precision groups.
L=/home/avifenesh/.cache/inkling2mini
T=$L/oracle/tapes
M=/home/avifenesh/models/inkling2-mini
mkdir -p $OUT; cd $OUT
cat $BIN/COMMIT $BIN/SHA256SUMS
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; tail -n 1 $tag.log; }
for cls in shipped nvfp4-rig; do
  run batch-$cls $BIN/ink2-gpu $M --class $cls --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out batch-$cls.json
  run batch-long-$cls $BIN/ink2-gpu $M --class $cls --tape $T/p2_window.exact_f32 --batch-check 4,16,256,1000,60 --out batch-long-$cls.json
  run past64-$cls $BIN/ink2-gpu $M --class $cls --context 64 --batch-seq-cap 4096 --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out past64-$cls.json
  run past8k-$cls $BIN/ink2-gpu $M --class $cls --context 8192 --batch-seq-cap 4096 --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out past8k-$cls.json
done
run red-rotated $BIN/ink2-gpu $M --class shipped --red batch-rows-rotated --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out red-rotated.json
run red-one-call $BIN/ink2-gpu $M --class shipped --red batch-attn-one-call --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out red-one-call.json
run red-t4-absent $BIN/ink2-gpu $M --class shipped --red t4-scratch-absent --context 64 --batch-seq-cap 4096 --tape $T/p1_chat.exact_f32 --batch-check 4,12,7 --out red-t4-absent.json
python3 - <<'EOF'
import json
d = lambda f: json.load(open(f))["batch_check"]
ok = True
for c in ["shipped", "nvfp4-rig"]:
    for f in [f"batch-{c}.json", f"batch-long-{c}.json", f"past64-{c}.json", f"past8k-{c}.json"]:
        b = d(f); print(f, "differ", b["decode_rows_differ"]); ok &= b["decode_rows_differ"] == 0
    a, b = d(f"past64-{c}.json")["solo_digest"], d(f"past8k-{c}.json")["solo_digest"]
    print(c, "past-load-context digest", a, b, "EQUAL" if a == b else "DIFFER"); ok &= a == b
for f in ["red-rotated.json", "red-one-call.json"]:
    b = d(f); print(f, "differ", b["decode_rows_differ"], "of", b["decode_rows_compared"]); ok &= b["decode_rows_differ"] > 0
r = d("red-t4-absent.json")["solo_digest"]; g = d("past8k-shipped.json")["solo_digest"]
print("red-t4-absent digest", r, "vs", g, "MOVES" if r != g else "DOES NOT MOVE"); ok &= r != g
print("BATCH_GATE", "PASS" if ok else "FAIL")
EOF
BIN=$BIN OUT=$OUT/bisect bash $(dirname "$0")/batch_bisect.sh
echo BATCH_GATE_DONE
