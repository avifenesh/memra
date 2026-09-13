#!/bin/bash
set -euo pipefail
cd /workspace/adaptive-draft
exec > >(tee -a raw/campaign.log) 2>&1
test -f CUDA_OK
test -f BOOTSTRAP_OK
source /root/.cargo/env
date -u --iso-8601=seconds
sha256sum engine-source.tar.gz pilot-tools.tar > raw/source-archives.sha256
git init -q control
git -C control -c user.name=TIYUVTA -c user.email=avifenesh@tiyuvta.ai commit --allow-empty -qm 'Research snapshot base'
git -C control worktree add -b lane/adaptive-draft-joint-20260913 ../memra
tar -xf engine-source.tar.gz -C memra
tar -xf pilot-tools.tar -C memra
git -C memra add .
git -C memra -c user.name=TIYUVTA -c user.email=avifenesh@tiyuvta.ai commit -qm 'Pinned source and research pilot'
LANE=memra/research/adaptive-draft-joint-20260913
python3 "$LANE/stage.py" models > raw/stage.log 2>&1 &
stage_pid=$!
(
  cd memra
  MEMRA_CUDA_ARCH=120a cargo build --release -p memra-engine --bin gemma-gate --bin run-spec --bin frspec-owngen
  cargo build --release -p memra-tokenizer --bin draft_prompt
) > raw/build.log 2>&1 &
build_pid=$!
wait "$stage_pid"
wait "$build_pid"
date -u --iso-8601=seconds
python3 "$LANE/pilot.py"
find raw -type f -print0 | sort -z | xargs -0 sha256sum > raw-manifest.sha256
date -u --iso-8601=seconds
