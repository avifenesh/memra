#!/usr/bin/env bash
# Runs ON a GB10 unit (CPU only, under the caller's lease): memra#1221 part 2's head build from
# its own tree (src-head, unpacked from head-lite.tar.gz) and target directory (target-head): the
# memra-server binary and the lib test binaries of memra-kv, memra-engine and memra-server.
# Stops at CAP_MIN (default 25); the next call resumes. TAG names the output directory.
# usage: spark_build_head.sh CAP_MIN TAG   -> bins/TAG/{memra-server,kv-lib-test,engine-lib-test,server-lib-test}
set -u
R=$HOME/.cache/memra-1221b; CAP=${1:-25}; TAG=$2
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target-head
L=$R/logs/build-$TAG.log; mkdir -p "$R/logs" "$R/bins/$TAG"
end=$(( $(date +%s) + CAP * 60 ))
left() { echo $(( end - $(date +%s) )); }
if [ ! -f "$R/src-head/.tag-$TAG" ]; then
  mkdir -p "$R/src-head" && tar -xzf "$R/$TAG.tar.gz" -C "$R/src-head" && touch "$R/src-head/.tag-$TAG" || exit 1
fi
cd "$R/src-head" || exit 1
echo "== $TAG $(date -Is)" >> "$L"
if [ "${SKIP_SERVER:-0}" != 1 ] && [ ! -x "$R/bins/$TAG/memra-server" ]; then
  timeout -s INT "$(left)" nice -n 10 cargo build --release --locked -j 16 -p memra-server --bin memra-server >> "$L" 2>&1 \
    || { echo "$(date -Is) BUILD-STOP server (rc=$?)" >> "$L"; exit 1; }
  cp -a "$CARGO_TARGET_DIR/release/memra-server" "$R/bins/$TAG/"
fi
for c in kv:memra-kv engine:memra-engine server:memra-server; do
  n=${c%%:*}; p=${c#*:}
  [ -x "$R/bins/$TAG/$n-lib-test" ] && continue
  [ "$(left)" -gt 60 ] || { echo "$(date -Is) cap reached before $n tests" >> "$L"; exit 1; }
  timeout -s INT "$(left)" nice -n 10 cargo test --release --locked -j 16 -p "$p" --lib --no-run --message-format=json > "$R/logs/tests-$TAG-$n.json" 2>> "$L" \
    || { echo "$(date -Is) BUILD-STOP $n tests (rc=$?)" >> "$L"; exit 1; }
  exe=$(grep -o '"executable":"[^"]*"' "$R/logs/tests-$TAG-$n.json" | grep -v null | tail -1 | cut -d'"' -f4)
  [ -x "$exe" ] || { echo "$(date -Is) no test executable for $n" >> "$L"; exit 1; }
  cp -a "$exe" "$R/bins/$TAG/$n-lib-test"
done
sha256sum "$R/bins/$TAG/"* | sed "s|$HOME|~|" >> "$L"
echo "$(date -Is) BUILD-DONE $TAG" >> "$L"
