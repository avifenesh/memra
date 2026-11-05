#!/usr/bin/env bash
# Clean rebuild of the c60 bins. The first build reused memra-sampling, memra-validate and memra-lanes
# from the shared target dir (every kit tree is version 0.138.0, so cargo gives the same artifact name);
# their source tree is unknown. Touch every tracked file so all 11 workspace crates compile from wt-c60.
# Run only as:
#   flock /home/avifenesh/.cache/memra-rig-build.lock nice -n 15 env MEMRA_CUDA_ARCH=120a \
#     CARGO_TARGET_DIR=/home/avifenesh/.cache/memra-maxq/target bash /home/avifenesh/.cache/memra-maxq/moe-prefetch/rebuild-clean.sh
set -euo pipefail
K=/home/avifenesh/.cache/memra-maxq/moe-prefetch
WT=$K/wt-c60
SHA=da649107c7e85f7681ddfd179650543b8400270d
LOG=$K/kit/build-c60.log
[ -f $K/kit/build-c60-attempt1.log ] || cp $LOG $K/kit/build-c60-attempt1.log
[ -d $WT ] || git -C /home/avifenesh/projects/memra worktree add --detach $WT $SHA
cd $WT
git ls-files -z | xargs -0 touch -c
bash $K/kit/build.sh
missing=""
for c in memra-engine memra-gguf memra-kv memra-lanes memra-reference memra-runtime memra-sampling memra-server memra-tier memra-tokenizer memra-validate; do
    grep -qF "Compiling $c v0.138.0 ($WT/crates/$c)" $LOG || missing="$missing $c"
done
if [ -n "$missing" ]; then echo "NOT CLEAN: not compiled from $WT:$missing"; exit 1; fi
echo "closure: all 11 workspace crates of memra-server, run-gen and run-spec compiled from $WT" | tee -a $LOG
cd $K/kit/bins/c60
sha256sum run-gen-c60 run-spec-c60 memra-server-c60 > bins.sha256
cat bins.sha256
