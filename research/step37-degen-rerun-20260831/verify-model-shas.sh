#!/usr/bin/env bash
# Verify the on-box step37 artifact against darklanes ops/serving/artifact-registry.tsv.
set -uo pipefail
D=/home/ubuntu/degen-rerun
M=/data/models/step37-flash-nvfp4
OUT=$D/receipts/model-sha256.txt
: > "$OUT"
for f in model-0000{1,2,3,4,5,6,7,8,9}-of-00013.safetensors model-000{10,11,12,13}-of-00013.safetensors \
         model-mtp-bf16.safetensors model.safetensors.index.json chat_template.jinja config.json \
         generation_config.json hf_quant_config.json special_tokens_map.json tokenizer.json tokenizer_config.json; do
  nice -n 19 ionice -c3 sha256sum "$M/$f" >> "$OUT"
done
echo "MODEL_SHA_DONE $(wc -l < "$OUT") files"
