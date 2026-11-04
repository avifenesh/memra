#!/bin/bash
# Owner decision S04:F0: measure MEMRA_GEMMA_FA_ONE_PROGRAM=1 against the default on the 5090,
# on gemma4 dense (31B) and MoE (26B-A4B): calibrated run-gen argmax, gemma-gate spec K=1..8
# stream agreement + acceptance on a short prompt (the path the switch changes), and the
# prefix-KV exactness probe (rows [0..900) of a <1024-row prime vs a >1024-row prime).
cd /home/avifenesh/.cache/memra-fix/engine-dsv4-glm || exit 2
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/home/avifenesh/.cache/memra-fix/target-cuda}
R=$CARGO_TARGET_DIR/release
L=/home/avifenesh/.cache/memra-fix/logs-gemma-c; mkdir -p $L
S=$L/status${STATUS_TAG:+-$STATUS_TAG}; : > $S
H=/data/ai-ml/hf-models
declare -A MODEL=( [g31]=$H/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf [g26]=$H/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf )
declare -A DRAFT=( [g31]=$H/gemma4-31b-qat-gguf/drafter/MTP/gemma-4-31B-it-assistant.Q8_0.gguf [g26]=$H/gemma4-26b-a4b-qat-gguf/drafter/MTP/gemma-4-26B-A4B-it-assistant.Q8_0.gguf )
IDS=$(cat research/gemma4-bringup/e4b-chat-watercycle-ids.txt)
A_TXT=$(python3 -c "print(' '.join('Sentence %d: the water cycle moves water between the sea, the air and the land.' % i for i in range(1, 49)))")
B_TXT=$(python3 -c "print(' '.join('Sentence %d: the water cycle moves water between the sea, the air and the land.' % i for i in range(1, 80)))")
# OFF is the default: the switch unset (env-audit refuses unknown MEMRA_* names).
arm_env() { [ "$1" = on ] && echo MEMRA_GEMMA_FA_ONE_PROGRAM=1 || echo "-u MEMRA_GEMMA_FA_ONE_PROGRAM"; }
echo "head $(git rev-parse --short=12 HEAD) server-stamp=$(grep -a -c "$(git rev-parse --short=12 HEAD)" $R/memra-server)" >> $S
for m in ${MODELS:-g31 g26}; do
  M=${MODEL[$m]}; D=${DRAFT[$m]}
  for arm in off on; do
    E=$(arm_env $arm)
    gpulock run --ttl 1h --for "memra gemma one-program $m $arm argmax" --wait 2h -- env $E bash tools/argmax-margin-gate.sh "$M" --logdir $L/argmax-$m-$arm > $L/argmax-$m-$arm.log 2>&1
    echo "$m $arm argmax rc=$? $(grep -m1 -o 'SUMMARY flips=[0-9]* bad=[0-9]*' $L/argmax-$m-$arm.log)" >> $S
    for k in 1 2 3 4 5 6 7 8; do
      gpulock run --ttl 30m --for "memra gemma one-program $m $arm spec K=$k" --wait 2h -- env $E MEMRA_SPEC=$k MEMRA_DRAFT="$D" MEMRA_NGEN=128 $R/gemma-gate "$M" $IDS > $L/spec-$m-$arm-k$k.log 2>&1
      echo "$m $arm spec K=$k rc=$? $(grep -m1 -o 'stream agreement [0-9]*/[0-9]*' $L/spec-$m-$arm-k$k.log) $(grep -m1 -i -o 'accept[a-z]* [^|]*' $L/spec-$m-$arm-k$k.log | head -c 80)" >> $S
    done
    for p in A B; do
      [ $p = A ] && TXT=$A_TXT || TXT=$B_TXT
      LOG=$L/kvprobe-$m-$arm-$p.log
      gpulock run --ttl 30m --for "memra gemma one-program $m $arm kvprobe $p" --wait 2h -- bash -c '
        env '"$E"' MEMRA_COMPAT=openai MEMRA_MODELS="g4='"$M"'" MEMRA_ADDR=127.0.0.1:8191 MEMRA_KVPROBE_AT=900 '"$R"'/memra-server > '"$LOG"' 2>&1 &
        SP=$!
        for i in $(seq 240); do curl -sf http://127.0.0.1:8191/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
        python3 -c "import json,sys; print(json.dumps({\"model\":\"g4\",\"messages\":[{\"role\":\"user\",\"content\":sys.argv[1]}],\"max_tokens\":4,\"temperature\":0,\"stream\":False}))" "$0" > '"$L"'/req.json
        curl -sf -m 600 http://127.0.0.1:8191/v1/chat/completions -H "Content-Type: application/json" -d @'"$L"'/req.json > /dev/null
        kill $SP 2>/dev/null; wait $SP' "$TXT"
      echo "$m $arm kvprobe $p $(grep -m1 -o 'pos=[0-9]* at=900 kv_sha256=[0-9a-f]*' $LOG)$(grep -m1 -o 'FATAL.*' $LOG | head -c 160)" >> $S
    done
  done
done
echo done >> $S
