#!/usr/bin/env bash
# Cooperatively fill a corpus range one bounded lock window at a time.

set -euo pipefail

if [[ $# != 4 ]]; then
  echo "usage: $0 LABEL START END CHUNK_SIZE" >&2
  exit 64
fi

label=$1
start=$2
end=$3
chunk_size=$4

if [[ ! $label =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "LABEL must match [a-z0-9][a-z0-9-]*" >&2
  exit 64
fi
for value in "$start" "$end" "$chunk_size"; do
  if [[ ! $value =~ ^[0-9]+$ ]]; then
    echo "range values must be non-negative integers" >&2
    exit 64
  fi
done
if (( end <= start || chunk_size == 0 )); then
  echo "require END > START and CHUNK_SIZE > 0" >&2
  exit 64
fi

workspace=/home/avifenesh/projects/wt-cx-dspark2
corpus_root=/data/projects/dspark/qwen35-9b-corpus-pilot/chunks
stop_file=/data/projects/dspark/qwen35-9b-corpus-pilot/STOP-AFTER-CHUNK
driver=$workspace/research/dspark2-20260811/run-corpus-chunk.sh
offset=$start

echo "DSPARK-RANGE-BEGIN $(date -Is) label=$label start=$start end=$end chunk_size=$chunk_size"
while (( offset < end )); do
  limit=$chunk_size
  if (( offset + limit > end )); then
    limit=$((end - offset))
  fi
  chunk_tag=$(printf '%s-%05d-%05d' "$label" "$offset" "$((offset + limit))")
  chunk_root=$corpus_root/$chunk_tag

  if [[ -f $chunk_root/.remote-verified ]]; then
    if ! (cd "$chunk_root" && sha256sum -c sha256.txt >/dev/null); then
      echo "completed chunk failed local manifest validation: $chunk_tag" >&2
      exit 65
    fi
    echo "DSPARK-RANGE-SKIP $(date -Is) tag=$chunk_tag"
    offset=$((offset + limit))
    continue
  fi

  if [[ -f $stop_file ]]; then
    echo "DSPARK-RANGE-STOP $(date -Is) sentinel=$stop_file next=$chunk_tag"
    exit 0
  fi
  if ! flock -n /tmp/memra-gpu.lock -c true; then
    echo "DSPARK-RANGE-WAIT $(date -Is) next=$chunk_tag"
    sleep 30
    continue
  fi

  set +e
  "$driver" "$offset" "$limit" "$label"
  rc=$?
  set -e
  if (( rc == 75 )); then
    echo "DSPARK-RANGE-RACED $(date -Is) next=$chunk_tag"
    sleep 30
    continue
  fi
  if (( rc != 0 )); then
    echo "DSPARK-RANGE-FAIL $(date -Is) tag=$chunk_tag rc=$rc" >&2
    exit "$rc"
  fi
  echo "DSPARK-RANGE-COMPLETE $(date -Is) tag=$chunk_tag"
  offset=$((offset + limit))
done
echo "DSPARK-RANGE-DONE $(date -Is) label=$label start=$start end=$end"
