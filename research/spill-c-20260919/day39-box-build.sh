#!/usr/bin/env bash
# Day 39 builds on the target card's box (one RTX PRO 6000 Blackwell): the three memra-server binaries of the
# day-39 cell (DAY39.md section 1), each from an exact SHA recorded in DAY39.md before this ran, built one after
# the other in this lane's own box worktree (never the read-only reference build, never another lane's worktree).
# No lock is taken: a build is not a scored run and the card is not used. Per binary: detached checkout of the SHA,
# a clean-tree check, `cargo build --release -p memra-server`, then a COPY (not the cargo hardlink) of the binary
# into <out>/bin/<label>-memra-server with its SHA-256 and the tree SHA it came from. Stops at the first failure.
# No host, id or price here.
# usage: day39-box-build.sh <box_worktree> <out> <label>=<sha> [<label>=<sha> ...]
set -uo pipefail
WT=$1; OUT=$2; shift 2
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
mkdir -p "$OUT/bin"
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
        cargo build --release -p memra-server 2>&1 | tail -5
        brc=${PIPESTATUS[0]}
        if [ "$brc" -ne 0 ]; then echo "rc=$brc"; exit 1; fi
        cp --no-preserve=links target/release/memra-server "$OUT/bin/$label-memra-server"
        echo "binary=$label-memra-server tree=$head sha256=$(sha256sum "$OUT/bin/$label-memra-server" | cut -d' ' -f1)"
        echo "end=$(date -u +%FT%TZ)"
        echo "rc=0"
    } 2>&1 | tee "$log"
    grep -q '^rc=0$' "$log" || { echo "BUILD $label FAILED" | tee -a "$OUT/builds.log"; exit 1; }
    grep '^binary=' "$log" | tee -a "$OUT/builds.log"
done
echo "BUILDS-DONE $(date -u +%FT%TZ)" | tee -a "$OUT/builds.log"
