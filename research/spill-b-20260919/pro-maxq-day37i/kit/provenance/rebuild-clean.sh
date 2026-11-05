#!/usr/bin/env bash
# Clean rebuild of the kv-vmm bins. The first build compiled only memra-engine and memra-server in each tree;
# the other nine workspace crates came from the shared target dir (every kit tree is version 0.138.0), source
# tree unknown. This touches every tracked file of a tree right before its build, so each label compiles
# all 11 workspace crates from its own worktree, then rewrites PROVENANCE.txt and KIT-MANIFEST.sha256.
# Run only as:
#   flock /home/avifenesh/.cache/memra-rig-build.lock nice -n 15 env MEMRA_CUDA_ARCH=120a \
#     CARGO_TARGET_DIR=/home/avifenesh/.cache/memra-maxq/target bash /home/avifenesh/.cache/memra-maxq/kv-vmm/rebuild-clean.sh
set -euo pipefail
B=/home/avifenesh/.cache/memra-maxq/kv-vmm
R=/home/avifenesh/projects/memra
P=$B/kit/provenance
[ -d $B/wt-lane ] || git -C $R worktree add --detach $B/wt-lane c6f9282c26e71349c0263b88dc7beb59c4a57797
[ -d $B/wt-main ] || git -C $R worktree add --detach $B/wt-main 17dceb9817d050d0960a8e2d0be372921bfbee80
[ -f $P/build-attempt1.log ] || cp $P/build.log $P/build-attempt1.log
sed 's|^  cd "\$wt"$|  cd "$wt"; git ls-files -z \| xargs -0 touch -c|' $P/build.sh > $B/build-clean.sh
grep -q 'xargs -0 touch -c' $B/build-clean.sh
bash $B/build-clean.sh > $P/build.log 2>&1
missing=""
for label in lane main; do
    sec=$(sed -n "/^$label build start/,/^$label build end/p" $P/build.log)
    for c in memra-engine memra-gguf memra-kv memra-lanes memra-reference memra-runtime memra-sampling memra-server memra-tier memra-tokenizer memra-validate; do
        grep -qF "Compiling $c v0.138.0 ($B/wt-$label/crates/$c)" <<<"$sec" || missing="$missing $label:$c"
    done
done
if [ -n "$missing" ]; then echo "NOT CLEAN:$missing"; exit 1; fi
echo "closure: lane and main each compiled all 11 workspace crates from their own worktree (rebuild-clean.sh)" | tee -a $P/build.log
cp $B/rebuild-clean.sh $P/rebuild-clean.sh
python3 $B/finalize.py
cat $B/kit/bins/lane/SHA256SUMS $B/kit/bins/main/SHA256SUMS
