#!/usr/bin/env bash
# Runs ON a GB10 host, unleased, only while no timed cell runs there: round 5 lever 1's two
# servers (B = main a8a980a017, T = perf/spec-eos-burst-end) and the session gate, banked as
# bins/eosb-<sha> and bins/eost-<sha>. usage: eos_build.sh BASE_SHA HEAD_SHA
set -u
D=$HOME/.cache/memra-gb10-r4
B=$1; T=$2
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
L=$D/logs/eos-build.log
git -C "$D/tree" fetch -q origin main perf/spec-eos-burst-end || { echo EOS-BUILD-FAIL fetch; exit 1; }
for arm in B T; do
  sha=$B; [ "$arm" = T ] && sha=$T
  git -C "$D/tree" checkout -q --detach "$sha" || { echo EOS-BUILD-FAIL checkout; exit 1; }
  (cd "$D/tree" && CARGO_TARGET_DIR=$D/target nice -n 10 cargo build --release --locked -j 16 \
    -p memra-server --bin memra-server -p memra-engine --bin session-gate) >> "$L" 2>&1 || { echo "EOS-BUILD-FAIL build $arm"; exit 1; }
  out=$D/bins/eos${arm,,}-${sha:0:10}; mkdir -p "$out"
  cp -a "$D/target/release/memra-server" "$D/target/release/session-gate" "$out/"
  sha256sum "$out/memra-server" >> "$L"
done
echo EOS-BUILD-DONE
