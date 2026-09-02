#!/bin/bash
# Byte-identity of the STREAMED loader vs the PRE-STREAMING loader on the REAL artifact.
#
# Reference arm = /root/realgate/downsel/*-spec-ab-rep0-off.* , produced by the
# pre-streaming binary (sha256 69c44eb85b82d4ee..., commit 7ef03558) on the SAME
# artifact with the SAME --goldens dump and the SAME seams env, by an unrelated lane.
# An independently-banked reference is a stronger control than a fresh twin run: nothing
# about it was arranged by this lane.
#
# args: <new_label>
set -uo pipefail
new="$1"
out=/root/realgate/loaderout
ref=/root/realgate/downsel
fail=0
say() { printf "%s\n" "$*"; }

# 1. probe logits: raw f32 prefill logits over the goldens probe rows. Same weights +
#    same ids + same program => same bytes. This is the load path's byte oracle.
a=$(sha256sum "$ref/probe-logits-spec-ab-rep0-off.bin" | cut -d" " -f1)
b=$(sha256sum "$out/probe-logits-$new.bin" | cut -d" " -f1)
say "probe-logits prestream sha256 = $a"
say "probe-logits streamed   sha256 = $b"
if [ "$a" = "$b" ]; then say "probe-logits: BYTE-IDENTICAL"; else say "probe-logits: DIVERGED"; fail=1; fi

# 2. per-layer wide-stream envelope table: deterministic numbers, no timings.
if diff <(grep -v "^#" "$ref/hidden-gate-spec-ab-rep0-off.tsv") \
        <(grep -v "^#" "$out/hidden-gate-$new.tsv") > "$out/hidden-gate-$new.diff"; then
  say "hidden-gate envelope table: IDENTICAL ($(grep -vc "^#" "$out/hidden-gate-$new.tsv") rows)"
else
  say "hidden-gate envelope table: DIVERGED (see hidden-gate-$new.diff)"; fail=1
fi

# 3. greedy chains + first divergence per prompt. Timing columns are wall clocks and are
#    excluded by name; the CHAIN and first_div are what the loader can move.
gref="$ref/greedy-gate-spec-ab-rep0-off.tsv"
gnew="$out/greedy-gate-$new.tsv"
if [ -f "$gref" ] && [ -f "$gnew" ]; then
  if diff <(grep -v "^#" "$gref" | cut -f1-5,9,10) <(grep -v "^#" "$gnew" | cut -f1-5,9,10) \
        > "$out/greedy-gate-$new.diff"; then
    say "greedy chains: IDENTICAL ($(grep -vc "^#" "$gnew") rows)"
  else
    say "greedy chains: DIVERGED (see greedy-gate-$new.diff)"; fail=1
  fi
else
  say "greedy chains: MISSING ONE SIDE — not compared (ref=$gref new=$gnew)"; fail=1
fi
say "identity_failures=$fail"
exit "$fail"
