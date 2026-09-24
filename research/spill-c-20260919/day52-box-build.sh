#!/usr/bin/env bash
# Day 52 builds on the target card's box (one RTX PRO 6000 Blackwell Workstation Edition): the run-gen binary of
# every MoE slot cache door rung and run-gen plus run-spec of the final tree (DAY52.md section 2), and the label
# `server` builds memra-server for the verify digest v3 gates (section 4), each from the
# exact commit recorded there, one after the other in this lane's own box worktree. No lock is taken: a build is
# not a scored run and the card is not used. Per label: detached checkout, a clean-tree check, `cargo build
# --release -p memra-engine --bin run-gen` (plus `--bin run-spec` for `final`), then a COPY (not the cargo
# hardlink) into <out>/bins with its SHA-256 and the tree it came from. The day-40 binary keeps its day-40 name
# (`base` -> bins/run-gen, which day40-cell.sh runs). Stops at the first failure. No host, id or price here.
# usage: day52-box-build.sh <box_worktree> <out> <label>=<sha> [<label>=<sha> ...]
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
        if [ "$label" = server ]; then
            # DAY52 section 4: the verify digest v3 server (OWED C6) for the host-tier gates.
            cargo build --release -p memra-server 2>&1 | tail -5
            brc=${PIPESTATUS[0]}
            if [ "$brc" -ne 0 ]; then echo "rc=$brc"; exit 1; fi
            cp --no-preserve=links target/release/memra-server "$OUT/bins/memra-server-v3"
            echo "binary=memra-server-v3 tree=$head sha256=$(sha256sum "$OUT/bins/memra-server-v3" | cut -d' ' -f1)"
            echo "end=$(date -u +%FT%TZ)"
            echo "rc=0"
            exit 0
        fi
        bins=(--bin run-gen); [ "$label" = final ] && bins+=(--bin run-spec)
        cargo build --release -p memra-engine "${bins[@]}" 2>&1 | tail -5
        brc=${PIPESTATUS[0]}
        if [ "$brc" -ne 0 ]; then echo "rc=$brc"; exit 1; fi
        name=run-gen-$label; [ "$label" = base ] && name=run-gen
        cp --no-preserve=links target/release/run-gen "$OUT/bins/$name"
        echo "binary=$name tree=$head sha256=$(sha256sum "$OUT/bins/$name" | cut -d' ' -f1)"
        if [ "$label" = final ]; then
            cp --no-preserve=links target/release/run-spec "$OUT/bins/run-spec-final"
            echo "binary=run-spec-final tree=$head sha256=$(sha256sum "$OUT/bins/run-spec-final" | cut -d' ' -f1)"
        fi
        echo "end=$(date -u +%FT%TZ)"
        echo "rc=0"
    } 2>&1 | tee "$log"
    grep -q '^rc=0$' "$log" || { echo "BUILD $label FAILED" | tee -a "$OUT/builds.log"; exit 1; }
    grep '^binary=' "$log" | tee -a "$OUT/builds.log"
done
echo "BUILDS-DONE $(date -u +%FT%TZ)" | tee -a "$OUT/builds.log"
