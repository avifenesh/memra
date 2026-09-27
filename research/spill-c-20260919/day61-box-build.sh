#!/usr/bin/env bash
# Days 59 to 61 builds on the target card's box (one RTX PRO 6000 Blackwell Workstation Edition), each label from the
# exact commit named in DAY59 section 1a, DAY60 section 1a and DAY61 section 2b, one after the other in this lane's
# own box build worktree (the sitting tree stays at the lane tip). No lock is taken: a build is not a scored run and
# the card is not used. Per label: detached checkout, a clean-tree check, the cargo build, then a COPY (not the cargo
# hardlink) into <out>/bins with its SHA-256 and the tree it came from. Labels: c60 builds run-gen, run-spec and
# memra-server (DAY59's three binaries and DAY60's run-gen); i11 and i12 build run-gen. Stops at the first failure.
# No host, id or price here.
# usage: day61-box-build.sh <box_build_worktree> <out> <label>=<sha> [<label>=<sha> ...]
set -uo pipefail
BWT=$1; OUT=$2; shift 2
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
mkdir -p "$OUT/bins"
cd "$BWT" || exit 1
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
        bins=(--bin run-gen); [ "$label" = c60 ] && bins+=(--bin run-spec)
        cargo build --release -p memra-engine "${bins[@]}" 2>&1 | tail -5
        brc=${PIPESTATUS[0]}
        if [ "$brc" -ne 0 ]; then echo "rc=$brc"; exit 1; fi
        cp --no-preserve=links target/release/run-gen "$OUT/bins/run-gen-$label"
        echo "binary=run-gen-$label tree=$head sha256=$(sha256sum "$OUT/bins/run-gen-$label" | cut -d' ' -f1)"
        if [ "$label" = c60 ]; then
            cp --no-preserve=links target/release/run-spec "$OUT/bins/run-spec-c60"
            echo "binary=run-spec-c60 tree=$head sha256=$(sha256sum "$OUT/bins/run-spec-c60" | cut -d' ' -f1)"
            cargo build --release -p memra-server 2>&1 | tail -5
            brc=${PIPESTATUS[0]}
            if [ "$brc" -ne 0 ]; then echo "rc=$brc"; exit 1; fi
            cp --no-preserve=links target/release/memra-server "$OUT/bins/memra-server-c60"
            echo "binary=memra-server-c60 tree=$head sha256=$(sha256sum "$OUT/bins/memra-server-c60" | cut -d' ' -f1)"
        fi
        echo "end=$(date -u +%FT%TZ)"
        echo "rc=0"
    } > "$log" 2>&1
    grep -q '^rc=0$' "$log" || { echo "build $label failed (see $log)"; exit 1; }
    echo "built $label"
done
