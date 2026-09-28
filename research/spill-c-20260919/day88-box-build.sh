#!/usr/bin/env bash
# Day 88 builds on the target card's box (one RTX PRO 6000 Blackwell Workstation Edition), each label from its exact
# commit, one after the other in this lane's box build worktree (the sitting tree stays at the lane tip): run-gen for
# every label, plus run-spec for `p88` (the promotion). Per label: detached checkout, a clean-tree check, `cargo build
# --release -p memra-engine` of those bins, then a COPY (not the cargo hardlink) into <out>/bins with its SHA-256 and
# the tree it came from. No lock is taken: a build is not a scored run and the card is not used. Stops at the first
# failure; the build's own rc is read before anything else prints. No host, id or price here.
# usage: day88-box-build.sh <box_build_worktree> <out> <label>=<sha> [<label>=<sha> ...]
set -uo pipefail
WT=$1; OUT=$2; shift 2
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
mkdir -p "$OUT/bins"
cd "$WT" || exit 1
for spec in "$@"; do
    label=${spec%%=*}; sha=${spec#*=}
    log=$OUT/build-$label.log
    {
        echo "label=$label sha=$sha start=$(date -u +%FT%TZ)"
        if ! git checkout -q --detach "$sha"; then echo "checkout failed"; echo "rc=1"; exit 1; fi
        head=$(git rev-parse HEAD)
        echo "head=$head"
        if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "tree not clean"; git status --short | head; echo "rc=1"; exit 1; fi
        nvcc --version | tail -2
        cargo --version
        bins=(run-gen); [ "$label" = p88 ] && bins+=(run-spec)
        args=(); for b in "${bins[@]}"; do args+=(--bin "$b"); done
        cargo build --release -p memra-engine "${args[@]}" 2>&1 | tail -5
        brc=${PIPESTATUS[0]}
        if [ "$brc" -ne 0 ]; then echo "rc=$brc"; exit 1; fi
        for b in "${bins[@]}"; do
            cp --no-preserve=links "target/release/$b" "$OUT/bins/$b-$label"
            echo "binary=$b-$label tree=$head sha256=$(sha256sum "$OUT/bins/$b-$label" | cut -d' ' -f1)"
        done
        echo "end=$(date -u +%FT%TZ)"
        echo "rc=0"
    } 2>&1 | tee "$log"
    grep -q '^rc=0$' "$log" || { echo "BUILD $label FAILED" | tee -a "$OUT/builds.log"; exit 1; }
    grep '^binary=' "$log" | tee -a "$OUT/builds.log"
done
echo "BUILDS-DONE $(date -u +%FT%TZ)" | tee -a "$OUT/builds.log"
