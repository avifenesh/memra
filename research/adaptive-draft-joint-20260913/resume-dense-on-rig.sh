#!/usr/bin/env bash
# Run only on the designated, non-serving RTX 5090. Preserve every failed attempt.
set -euo pipefail
study_root=/workspace/adaptive-draft
study_repo=$(git rev-parse --show-toplevel)
study_lane="$study_repo/research/adaptive-draft-joint-20260913"
test "$(readlink -f "$study_root/memra")" = "$(readlink -f "$study_repo")" || {
    echo 'Expected /workspace/adaptive-draft/memra to resolve to this isolated checkout.' >&2
    exit 1
}
test -z "$(git -C "$study_repo" status --porcelain)" || {
    echo 'Commit or preserve checkout changes before binding a research source revision.' >&2
    exit 1
}
for study_output in raw/dense-admission raw/fresh-admission2 checkpoints/admission-dense-model.json checkpoints/dense-admission-prompts checkpoints/fresh-admission2-prompts; do
    test ! -e "$study_root/$study_output" || {
        echo "Refusing to overwrite $study_output; inspect the prior run and resume its failed stage." >&2
        exit 1
    }
done
study_commit=$(git -C "$study_repo" rev-parse HEAD)
if test -e "$study_root/SOURCE_COMMIT"; then
    test "$(cat "$study_root/SOURCE_COMMIT")" = "$study_commit" || {
        echo 'Existing SOURCE_COMMIT belongs to another revision; preserve its evidence first.' >&2
        exit 1
    }
fi
mkdir -p "$study_root/raw" "$study_root/checkpoints"
study_setup="$study_root/raw/rig-setup-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir "$study_setup"
exec > >(tee "$study_setup/run.log") 2>&1
trap 'echo "Rig study failed; preserve logs and resume the failed stage. No outputs were removed." >&2' ERR
date -u --iso-8601=seconds
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = 'NVIDIA GeForce RTX 5090' || {
    echo 'This registered run requires a dedicated single RTX 5090; register other rig shapes separately.' >&2
    exit 1
}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || {
    echo 'GPU already has a compute process; this study cannot share a serving GPU.' >&2
    exit 1
}
nvcc --version
rustc --version
cargo --version
python3 --version
cat > "$study_setup/cuda-probe.cu" <<'CUDA'
#include <cuda_runtime.h>
#include <cstdio>
int main() {
    void *p = nullptr;
    auto err = cudaMalloc(&p, 1048576);
    std::printf("cudaMalloc: %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) return 1;
    err = cudaMemset(p, 1, 1048576);
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    cudaFree(p);
    return err == cudaSuccess ? 0 : 2;
}
CUDA
nvcc "$study_setup/cuda-probe.cu" -o "$study_setup/cuda-probe"
"$study_setup/cuda-probe"
cd "$study_repo"
cargo fmt --all -- --check
python3 "$study_lane/verify_export.py"
MEMRA_CUDA_ARCH=120a cargo build --release -p memra-engine --bin gemma-gate
cargo build --release -p memra-tokenizer --bin draft_prompt
python3 "$study_lane/stage.py" "$study_root/models"
test -z "$(git status --porcelain)"
printf '%s\n' "$study_commit" > "$study_root/SOURCE_COMMIT"
sha256sum target/release/gemma-gate target/release/draft_prompt > "$study_setup/binaries.sha256"
git ls-files -z crates .cargo Cargo.toml Cargo.lock rustfmt.toml | xargs -0 sha256sum > "$study_setup/build-inputs.sha256"
python3 - "$study_lane" "$study_root" <<'PY'
from pathlib import Path
import hashlib
import json
import shutil
import sys
lane, root = map(Path, sys.argv[1:])
receipts = lane / 'candidate-gates-receipts'
manifest = json.loads((receipts / 'MANIFEST.json').read_text())
for name in ['checkpoints/admission-model.json', 'raw/admission-model-frozen.sha256']:
    source = receipts / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest[name]
    destination = root / name
    if destination.exists():
        assert destination.read_bytes() == source.read_bytes(), name
    else:
        shutil.copyfile(source, destination)
PY
cd "$study_root"
python3 "$study_lane/dense_admission.py"
python3 "$study_lane/evaluate_admission.py" . --original "$study_lane/row-oracle-receipts" --study dense-admission --out raw/dense-audit.json
python3 "$study_lane/admission_model.py" . --dense --out checkpoints/admission-dense-model.json
sha256sum checkpoints/admission-dense-model.json > raw/admission-dense-model-frozen.sha256
python3 "$study_lane/fresh_admission2_prompts.py" checkpoints/fresh-admission2-prompts
cp checkpoints/fresh-admission2-prompts/manifest.json raw/fresh2-prompts-before-generation.json
python3 "$study_lane/fresh_admission2.py"
python3 "$study_lane/evaluate_admission.py" . --original "$study_lane/row-oracle-receipts" --study fresh-admission2 --out raw/fresh2-audit.json
echo 'Dense training and second fresh audit complete. Bank raw files and checkpoints before any next policy experiment.'
