#!/usr/bin/env bash
# batch.sh <name> : run a list of arms sequentially (one GPU job at a time), detached by caller.
# Lines of $base/<name>.arms: <tag> <model-key> <rows> [VAR=VALUE ...]
set -uo pipefail
base=$HOME/.cache/memra-1203
orn=/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
q38=/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
while read -r tag key rows rest; do
  [ -z "${tag:-}" ] && continue
  case $tag in \#*) continue;; esac
  case $key in orn) m=$orn;; q38) m=$q38;; *) echo "bad key $key"; continue;; esac
  if [ -f "$base/runs/$tag/result.txt" ] && grep -q RUN-DONE "$base/runs/$tag/result.txt"; then
    echo "skip $tag (done)"; continue
  fi
  echo "start $tag $(date +%T)"
  # shellcheck disable=SC2086
  "$base/run2.sh" "$tag" "$m" "$rows" $rest
  echo "end $tag $(date +%T) $(grep '^rc=' "$base/runs/$tag/result.txt")"
done < "$base/$1.arms"
echo BATCH-DONE
