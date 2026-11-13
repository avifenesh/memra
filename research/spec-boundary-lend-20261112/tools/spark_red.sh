#!/usr/bin/env bash
# Runs ON a GB10 unit (CPU only, under the caller's lease): memra#1221 part 2's red-arm test
# binaries. For each NAME, the head tree (src-head) with that guard disarmed (red_patch.py) builds
# the lib test binary of the crate that holds its GPU cell, and the touched files are put back from
# the tree's tarball. Stops at CAP_MIN.
# usage: spark_red.sh CAP_MIN HEADTAG NAME...   -> bins/red-NAME-HEADTAG/{kv,engine,server}-lib-test
set -u
R=$HOME/.cache/memra-1221b; CAP=${1:-25}; TAG=$2; shift 2
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target-head
L=$R/logs/build-red-$TAG.log
end=$(( $(date +%s) + CAP * 60 ))
cd "$R/src-head" || exit 1
for name in "$@"; do
  OUT=$R/bins/red-$name-$TAG; mkdir -p "$OUT"
  case $name in scratch) crates="engine:memra-engine" ;; *) crates="kv:memra-kv" ;; esac
  python3 "$R/box/red_patch.py" "$R/src-head" "$name" >> "$L" 2>&1 || exit 1
  for c in $crates; do
    n=${c%%:*}; p=${c#*:}
    [ -x "$OUT/$n-lib-test" ] && continue
    left=$(( end - $(date +%s) ))
    [ "$left" -gt 60 ] || { echo "$(date -Is) cap reached before red $name $n" >> "$L"; break; }
    timeout -s INT "$left" nice -n 10 cargo test --release --locked -j 16 -p "$p" --lib --no-run --message-format=json > "$R/logs/red-$name-$n.json" 2>> "$L"
    exe=$(grep -o '"executable":"[^"]*"' "$R/logs/red-$name-$n.json" | grep -v null | tail -1 | cut -d'"' -f4)
    [ -x "$exe" ] && cp -a "$exe" "$OUT/$n-lib-test" || echo "$(date -Is) red $name $n: no test executable" >> "$L"
  done
  tar -xzf "$R/$TAG.tar.gz" -C "$R/src-head" crates/memra-kv/src/lib.rs crates/memra-kv/src/plane.rs crates/memra-engine/src/spec.rs
  touch crates/memra-kv/src/lib.rs crates/memra-kv/src/plane.rs crates/memra-engine/src/spec.rs
  echo "$(date -Is) red $name built: $(ls "$OUT")" >> "$L"
done
echo "$(date -Is) RED-DONE $*" >> "$L"
