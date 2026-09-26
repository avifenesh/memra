#!/usr/bin/env bash
# Lane C local builds for the RTX 5090 queue (rtx5090-queue-v9-20260926.sh), after the 2026-09-26 reboot wiped the
# /tmp binary dirs. Each label from its exact commit, one after the other, in a detached build worktree under the
# lane's target/ (never /tmp): detached checkout, a clean-tree check, `cargo build --release` of each named bin, then
# a COPY (not the cargo hardlink) into <out> as <bin>-<label> with its SHA-256 and the tree it came from. Run inside
# the caller's CPU cap. Skips a binary already in <out>; stops at the first failure. A log per label in <out>/logs.
# usage: c-local-build.sh <build_worktree> <out> <label>=<sha>:<pkg>/<bin>[,<pkg>/<bin>...] [...]
set -uo pipefail
BWT=$1; OUT=$2; shift 2
mkdir -p "$OUT/logs"
cd "$BWT" || exit 1
for spec in "$@"; do
    label=${spec%%=*}; rest=${spec#*=}; sha=${rest%%:*}; bins=${rest#*:}
    log=$OUT/logs/build-$label.log
    need=0
    for pb in ${bins//,/ }; do [ -x "$OUT/${pb#*/}-$label" ] || need=1; done
    if [ "$need" = 0 ]; then echo "have $label"; continue; fi
    {
        echo "label=$label sha=$sha bins=$bins start=$(date -u +%FT%TZ)"
        if ! git checkout -q --detach "$sha"; then echo "checkout failed"; echo "rc=1"; exit 1; fi
        head=$(git rev-parse HEAD)
        echo "head=$head"
        if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "tree not clean"; git status --short | head; echo "rc=1"; exit 1; fi
        nvcc --version | tail -2
        cargo --version
        for pb in ${bins//,/ }; do
            pkg=${pb%%/*}; bin=${pb#*/}
            cargo build --release -p "$pkg" --bin "$bin" 2>&1 | tail -5
            brc=${PIPESTATUS[0]}
            if [ "$brc" -ne 0 ]; then echo "rc=$brc"; exit 1; fi
            cp --no-preserve=links "target/release/$bin" "$OUT/$bin-$label"
            echo "binary=$bin-$label tree=$head sha256=$(sha256sum "$OUT/$bin-$label" | cut -d' ' -f1)"
        done
        echo "end=$(date -u +%FT%TZ)"
        echo "rc=0"
    } > "$log" 2>&1
    grep -q '^rc=0$' "$log" || { echo "build $label failed (see $log)"; exit 1; }
    echo "built $label"
done
echo "all builds done"
