#!/usr/bin/env bash
# Generate and extract one resumable DSpark corpus chunk without competing with serving lanes.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 OFFSET LIMIT [LABEL]" >&2
  exit 64
fi

offset=$1
limit=$2
label=${3:-smoke}

if [[ ! $offset =~ ^[0-9]+$ || ! $limit =~ ^[1-9][0-9]*$ ]]; then
  echo "OFFSET must be non-negative and LIMIT must be positive" >&2
  exit 64
fi
if [[ ! $label =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "LABEL must match [a-z0-9][a-z0-9-]*" >&2
  exit 64
fi

workspace=/home/avifenesh/projects/wt-cx-dspark2
model=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
prompt_root=/data/projects/dspark/qwen35-9b-corpus-pilot/prompts-v2
corpus_root=/data/projects/dspark/qwen35-9b-corpus-pilot/chunks
remote_root=/home/ubuntu/dspark2/corpus/chunks
chunk_tag=$(printf '%s-%05d-%05d' "$label" "$offset" "$((offset + limit))")
chunk_root=$corpus_root/$chunk_tag
generated=$chunk_root/generated
extracted=$chunk_root/extracted
raw=$workspace/research/dspark2-20260811/raw/corpus-$chunk_tag.log
manifest=$chunk_root/sha256.txt
generate=$workspace/target/release/dspark-generate
extract=$workspace/target/release/dspark-extract
validate=$workspace/research/dspark2-20260811/validate-corpus.py

mkdir -p "$(dirname "$raw")" "$chunk_root"
exec > >(tee -a "$raw") 2>&1

echo "DSPARK-CHUNK-BEGIN $(date -Is) offset=$offset limit=$limit label=$label"
echo "workspace=$workspace"
echo "chunk_root=$chunk_root"

for required in \
  "$model" \
  "$prompt_root/prompts.promptpack" \
  "$prompt_root/prompts.tsv" \
  "$generate" \
  "$extract" \
  "$validate"; do
  if [[ ! -f $required ]]; then
    echo "missing required file: $required" >&2
    exit 66
  fi
done

expected_model=52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de
actual_model=$(sha256sum "$model" | awk '{print $1}')
if [[ $actual_model != "$expected_model" ]]; then
  echo "model hash mismatch: expected=$expected_model actual=$actual_model" >&2
  exit 65
fi

expected_prompt_pack=20a061ddee54bb3113a25cd2abbb150e7e51c65a05670c3be127c242221fffd9
expected_prompt_tsv=c2f9504c5761de8bfd88657433e259496240e795ec60cf709e4202d76de58e7f
expected_generate=dc757120c3765d10bc670ac68fd1cec22789317057ecaf2d833121ddf6f53129
expected_extract=e8abb3bc7674b69ad46766e729ef178c8ed326b44769302d2e6c7f9cf44f044e
for frozen in \
  "$expected_prompt_pack:$prompt_root/prompts.promptpack" \
  "$expected_prompt_tsv:$prompt_root/prompts.tsv" \
  "$expected_generate:$generate" \
  "$expected_extract:$extract"; do
  expected=${frozen%%:*}
  path=${frozen#*:}
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ $actual != "$expected" ]]; then
    echo "frozen input hash mismatch: path=$path expected=$expected actual=$actual" >&2
    exit 65
  fi
done

sha256sum "$model" "$prompt_root/prompts.promptpack" "$prompt_root/prompts.tsv" "$generate" "$extract"
nvidia-smi --query-gpu=timestamp,name,uuid,compute_cap,memory.total,memory.used,temperature.gpu,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader || true

exec 9>/tmp/memra-gpu.lock
if ! flock -n 9; then
  echo "DSPARK-CHUNK-LOCK-BUSY $(date -Is)"
  exit 75
fi
echo "DSPARK-CHUNK-LOCK-ACQUIRED $(date -Is)"

nice -n 15 ionice -c 2 -n 7 "$generate" \
  "$model" \
  "$prompt_root/prompts.promptpack" \
  "$prompt_root/prompts.tsv" \
  "$generated" \
  --offset "$offset" \
  --limit "$limit" \
  --max-new 512 \
  --temperature 0.7 \
  --seed 20260811

nice -n 15 ionice -c 2 -n 7 "$extract" \
  "$model" \
  "$generated/pairs.tsv" \
  "$extracted" \
  --anchors 4 \
  --gamma 5 \
  --top-k 64 \
  --chunk 512 \
  --temperature 0.7 \
  --seed 20260811

nvidia-smi --query-gpu=timestamp,name,uuid,compute_cap,memory.total,memory.used,temperature.gpu,power.draw --format=csv,noheader
echo "DSPARK-CHUNK-LOCK-RELEASE $(date -Is)"
flock -u 9

python3 -B "$validate" "$extracted" --receipt "$chunk_root/validation.json"
(cd "$chunk_root" && find generated extracted validation.json -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > sha256.txt)
sha256sum "$manifest"
du -sh "$chunk_root"

remote_chunk=$remote_root/$chunk_tag
ssh fpv-recognition-teacher mkdir -p "$remote_chunk"
rsync -a --partial "$chunk_root/" "fpv-recognition-teacher:$remote_chunk/"
ssh fpv-recognition-teacher bash -s -- "$remote_chunk" <<'REMOTE_VERIFY'
set -euo pipefail
cd -- "$1"
sha256sum -c sha256.txt
REMOTE_VERIFY
touch "$chunk_root/.remote-verified"

echo "DSPARK-CHUNK-DONE $(date -Is) tag=$chunk_tag"
