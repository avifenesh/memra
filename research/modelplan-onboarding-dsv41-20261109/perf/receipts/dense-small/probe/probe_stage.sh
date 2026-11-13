#!/bin/bash
# Laptop: stage the probe's three sources on the dev desk (k3 = 565f0ff3e1, old = 498473abd8, new = the fix worktree)
# with the probe and its build script.
set -u
F=/home/avifenesh/.cache/dsv41-memra/cutlass-decode/fix
WT=/home/avifenesh/.cache/dsv41-memra/wt-cutlass
S=$F/stage
mkdir -p $S/src/k3 $S/src/old $S/src/new
git -C $WT show 565f0ff3e1:crates/memra-engine/cu/dsv41_gpu.cu > $S/src/k3/dsv41_gpu.cu
git -C $WT show 498473abd8:crates/memra-engine/cu/dsv41_gpu.cu > $S/src/old/dsv41_gpu.cu
git -C $WT show 498473abd8:crates/memra-engine/cu/dsv41_cutlass.cu > $S/src/old/dsv41_cutlass.cu
cp $WT/crates/memra-engine/cu/dsv41_gpu.cu $WT/crates/memra-engine/cu/dsv41_cutlass.cu $S/src/new/
cp $F/probe.cu $F/probe_build.sh $S/
ssh devdesk 'mkdir -p ~/build/cdec2' && rsync -a $S/ devdesk:build/cdec2/ && echo staged
sha256sum $S/src/*/*.cu $S/probe.cu | cut -c1-16
