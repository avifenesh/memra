#!/usr/bin/env bash
# Freeze the train-only 32,768-row d2t map after the corrected 2K pilot is complete.

set -euo pipefail

workspace=/home/avifenesh/projects/wt-cx-dspark2
corpus_root=/data/projects/dspark/qwen35-9b-corpus-pilot/chunks
artifact_root=/data/projects/dspark/qwen35-9b-corpus-pilot/ranks/pilot-02000
remote_root=/home/ubuntu/dspark2/artifacts/pilot-02000
backfill=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/owngen-ranks-32768.gguf.txt
builder=$workspace/research/dspark2-20260811/build-own-ranks.py
raw=$workspace/research/dspark2-20260811/raw/ranks-pilot-02000.log
expected_backfill=4ac5f4866a531966c3bc014d996b119781eb766a399acaf6d78c1d85f66cf876

mkdir -p "$artifact_root" "$(dirname "$raw")"
exec > >(tee "$raw") 2>&1

echo "DSPARK-RANK-FREEZE-BEGIN $(date -Is)"
actual_backfill=$(sha256sum "$backfill" | awk '{print $1}')
if [[ $actual_backfill != "$expected_backfill" ]]; then
  echo "backfill hash mismatch: expected=$expected_backfill actual=$actual_backfill" >&2
  exit 65
fi

pairs=()
offset=0
while (( offset < 2000 )); do
  finish=$((offset + 64))
  if (( finish > 2000 )); then
    finish=2000
  fi
  tag=$(printf 'pilot-%05d-%05d' "$offset" "$finish")
  chunk=$corpus_root/$tag
  if [[ ! -f $chunk/.remote-verified ]]; then
    echo "missing remote-verification marker: $tag" >&2
    exit 66
  fi
  (cd "$chunk" && sha256sum -c sha256.txt >/dev/null)
  pairs+=("$chunk/generated/pairs.tsv")
  offset=$finish
done

python3 -B "$builder" \
  --pairs "${pairs[@]}" \
  --out "$artifact_root/ranks-32768.txt" \
  --summary "$artifact_root/summary.json" \
  --size 32768 \
  --backfill "$backfill"

jq -e \
  --arg backfill "$expected_backfill" \
  '.format == "memra-dspark-d2t-v1"
   and .pairs == 2000
   and .pair_id_min == 0
   and .pair_id_max == 1999
   and .draft_vocab_size == 32768
   and .ranking_split == "train"
   and .backfill_sha256 == $backfill
   and .coverage.train > 0.0
   and .coverage.heldout > 0.0' \
  "$artifact_root/summary.json" >/dev/null

(cd "$artifact_root" && sha256sum ranks-32768.txt summary.json > sha256.txt)
sha256sum "$artifact_root/sha256.txt"
ssh fpv-recognition-teacher mkdir -p "$remote_root"
rsync -a --partial "$artifact_root/" "fpv-recognition-teacher:$remote_root/"
ssh fpv-recognition-teacher bash -s -- "$remote_root" <<'REMOTE_VERIFY'
set -euo pipefail
cd -- "$1"
sha256sum -c sha256.txt
REMOTE_VERIFY
echo "DSPARK-RANK-FREEZE-DONE $(date -Is) remote=$remote_root"
