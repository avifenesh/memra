#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
bash $P/cls2.sh lk6-rig nvfp4-rig $P/a34-bin "" $P/a33-bin "" > $P/lk6-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh lk6-sh shipped $P/a34-bin "" $P/a33-bin "" > $P/lk6-sh.log 2>&1; echo RC_sh=$?
bash $P/cls2.sh lk6-rig4k nvfp4-rig $P/a34-bin "" $P/a33-bin "" 4096 > $P/lk6-rig4k.log 2>&1; echo RC_rig4k=$?
