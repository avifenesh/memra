#!/bin/bash
# lane S's drafter gates on the landing's ink2-spec (LAND_BIN), lane S's method: the tiny gate, run-spec
# K=1..7 over the four gate prompts with --check-plain, both classes (28 cells each, spec equal to plain)
B=${LAND_BIN}
C=/home/avifenesh/models/inkling2-mini
G=/home/avifenesh/.cache/inkling2mini/wt-land/research/modelplan-onboarding-inkling2mini-20261121/drafter/gate-prompts.jsonl
O=${LAND_SPEC_OUT}
mkdir -p $O; cd $O
sha256sum $G
$B/ink2-spec --tiny-gate > tiny.log 2>&1; echo "RC tiny=$?"; grep -c PASS tiny.log
$B/ink2-spec $C --prompts $G --gen 64 --ks 1,2,3,4,5,6,7 --check-plain --context 2048 --out gate.jsonl > gate.log 2>&1; echo "RC gate=$? $(grep -c 'Some(true)' gate.log)/28"
$B/ink2-spec $C --prompts $G --gen 64 --ks 1,2,3,4,5,6,7 --check-plain --class nvfp4-rig --context 2048 --out gate-rig.jsonl > gate-rig.log 2>&1; echo "RC gate-rig=$? $(grep -c 'Some(true)' gate-rig.log)/28"
echo LAND_SPEC_DONE
