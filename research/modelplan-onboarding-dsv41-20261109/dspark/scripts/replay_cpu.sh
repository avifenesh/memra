#!/bin/bash
# The vendor draft layers on the CPU backend over the GPU tapes' own inputs (the floor for the draft gates).
D=${LANE:?LANE names the working directory on the run box (checkpoint, oracle, dec ids, spec/, target-spec/)}
exec taskset -c 28-31,60-63 nice -n 10 ${TORCH_PY:?a python with torch} -I $D/spec/oracle/spec_oracle.py \
  $D/DeepSeek-V4.1-Flash-1017 $D/ref $D/oracle/v2_cases.json $D/spec/tapes-cpu --threads 8 --device cpu \
  --replay $D/spec/tapes --only gsm,code,copy,story,hebrew_para,code_pal,sys_json,gsm_think,needle,capital,hebrew,multiturn --full-logits 4
