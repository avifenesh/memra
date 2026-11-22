#!/bin/bash
P=/home/avifenesh/.cache/inkling2mini/perf
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh; bash $C; rc=$?; echo CLAMP_RC=$rc; if [ $rc -ne 0 ]; then sudo -n systemctl reset-failed nvidia-powerd; sudo -n systemctl restart nvidia-powerd; waitfor --until "bash $C" --timeout 60s; echo CLAMP_AFTER=$?; fi
bash $P/cls2.sh lk9-rig nvfp4-rig $P/a38-bin "" $P/a37-bin "" > $P/lk9-rig.log 2>&1; echo RC_rig=$?
bash $P/cls2.sh lk9-sh shipped $P/a38-bin "" $P/a37-bin "" > $P/lk9-sh.log 2>&1; echo RC_sh=$?
bash $P/cls2.sh lk9-rig4k nvfp4-rig $P/a38-bin "" $P/a37-bin "" 4096 > $P/lk9-rig4k.log 2>&1; echo RC_rig4k=$?
