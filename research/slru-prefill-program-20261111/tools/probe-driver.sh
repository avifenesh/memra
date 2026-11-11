#!/usr/bin/env bash
cd ~/.cache/memra-1217 || exit 1
bash scripts/cells.sh bin/argmax-margin-probe-main step3/main 'res-e:' 'res-b:DECODE_BATCH' 'slru-e:MEMRA_MOE_RESIDENT=0' 'slru-b:MEMRA_MOE_RESIDENT=0,DECODE_BATCH' 'slru500-e:MEMRA_MOE_RESIDENT=0,MEMRA_MOE_SLOTS=500'
bash scripts/cells.sh bin/argmax-margin-probe-fix step3/fix 'res-e:' 'res-b:DECODE_BATCH' 'slru-e:MEMRA_MOE_RESIDENT=0' 'slru-b:MEMRA_MOE_RESIDENT=0,DECODE_BATCH' 'slru500-e:MEMRA_MOE_RESIDENT=0,MEMRA_MOE_SLOTS=500,MEMRA_MOE_SLRU_PAIRS_TRACE=1' 'slru500-b:MEMRA_MOE_RESIDENT=0,MEMRA_MOE_SLOTS=500,DECODE_BATCH' 'slru2000-e:MEMRA_MOE_RESIDENT=0,MEMRA_MOE_SLOTS=2000,MEMRA_MOE_SLRU_PAIRS_TRACE=1'
echo PROBES-DONE
