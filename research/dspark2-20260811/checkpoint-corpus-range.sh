#!/usr/bin/env bash
# Commit one fail-closed evidence checkpoint after each remotely verified corpus chunk.

set -euo pipefail

if [[ $# != 4 ]]; then
  echo "usage: $0 LABEL COMMITTED_END END CHUNK_SIZE" >&2
  exit 64
fi

label=$1
offset=$2
end=$3
chunk_size=$4

if [[ ! $label =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "LABEL must match [a-z0-9][a-z0-9-]*" >&2
  exit 64
fi
for value in "$offset" "$end" "$chunk_size"; do
  if [[ ! $value =~ ^[0-9]+$ ]]; then
    echo "range values must be non-negative integers" >&2
    exit 64
  fi
done
if (( end <= offset || chunk_size == 0 )); then
  echo "require END > COMMITTED_END and CHUNK_SIZE > 0" >&2
  exit 64
fi

workspace=/home/avifenesh/projects/wt-cx-dspark2
corpus_root=/data/projects/dspark/qwen35-9b-corpus-pilot/chunks
raw_root=$workspace/research/dspark2-20260811/raw
summarizer=$workspace/research/dspark2-20260811/summarize-corpus.py

git_retry() {
  local attempt
  for ((attempt = 1; attempt <= 30; attempt++)); do
    if git "$@"; then
      return 0
    fi
    echo "DSPARK-CHECKPOINT-GIT-RETRY $(date -Is) attempt=$attempt command=git $*" >&2
    sleep 2
  done
  return 1
}

cd "$workspace"
echo "DSPARK-CHECKPOINT-BEGIN $(date -Is) label=$label committed_end=$offset end=$end"
while (( offset < end )); do
  finish=$((offset + chunk_size))
  if (( finish > end )); then
    finish=$end
  fi
  chunk_tag=$(printf '%s-%05d-%05d' "$label" "$offset" "$finish")
  chunk_root=$corpus_root/$chunk_tag
  raw_log=$(printf '%s/corpus-%s-%05d-%05d.log' "$raw_root" "$label" "$offset" "$finish")
  summary=$(printf '%s/corpus-%s-summary-%05d.json' "$raw_root" "$label" "$finish")

  while [[ ! -f $chunk_root/.remote-verified ]] ||
    ! grep -Eq "^DSPARK-CHUNK-DONE [^ ]+ tag=${chunk_tag}$" "$raw_log" 2>/dev/null; do
    sleep 10
  done

  if ! GIT_OPTIONAL_LOCKS=0 git diff --cached --quiet; then
    echo "refusing to checkpoint with pre-existing staged changes" >&2
    exit 65
  fi
  python3 "$summarizer" \
    --root "$corpus_root" \
    --label "$label" \
    --start 0 \
    --end "$finish" \
    --out "$summary"
  git_retry add -- "$raw_log" "$summary"
  mapfile -t staged < <(GIT_OPTIONAL_LOCKS=0 git diff --cached --name-only)
  if [[ ${#staged[@]} != 2 ]] ||
    [[ ${staged[0]} != "research/dspark2-20260811/raw/$(basename "$raw_log")" ]] ||
    [[ ${staged[1]} != "research/dspark2-20260811/raw/$(basename "$summary")" ]]; then
    echo "refusing unexpected staged path set" >&2
    printf 'staged: %s\n' "${staged[@]}" >&2
    exit 65
  fi
  git_retry commit -m "data(dspark2): checkpoint $label corpus through $finish"
  echo "DSPARK-CHECKPOINT-DONE $(date -Is) tag=$chunk_tag commit=$(git rev-parse --short HEAD)"
  offset=$finish
done
echo "DSPARK-CHECKPOINT-RANGE-DONE $(date -Is) label=$label end=$end"
