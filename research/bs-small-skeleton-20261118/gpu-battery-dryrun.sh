#!/usr/bin/env bash
# Dry runs of gpu-battery.sh with fake binaries (no GPU, no cargo). One case a call: harness.sh CASE
# Prints the case, the script's exit code and its last line.
set -u
D=$HOME/.cache/bs-small/dry
S=$(dirname "$0")/gpu-battery.sh
CASE=$1
W=$D/w-$CASE
rm -rf "$W"; mkdir -p "$W/fakebin" "$W/data/fixtures-gpu/rand" "$W/ckpt"
FB=$W/fakebin
# fake nvidia-smi and cargo
printf '#!/bin/sh\necho "0, FAKE, 300 W, 30, 0 MiB"\n' > "$FB/nvidia-smi"
printf '#!/bin/sh\nexit 0\n' > "$FB/cargo"
chmod +x "$FB"/*
for arm in BASE HEAD; do
  src=$W/src-$arm; mkdir -p "$src/tools" "$src/research/tune-data"
  printf '{"supported_models": [{"model": "M1"}, {"model": "M 2"}]}\n' > "$src/research/tune-data/current-board.json"
  printf '#!/bin/sh\n[ "${FAKE_RB_FAIL:-0}" = 1 ] && exit 1\necho "release-battery card $CUDA_VISIBLE_DEVICES PASS"\n' > "$src/tools/release-battery.sh"
  chmod +x "$src/tools/release-battery.sh"
  git -C "$src" init -q && git -C "$src" -c user.name=x -c user.email=x@x commit -q --allow-empty -m x
  t=$W/out/target-$arm/release; mkdir -p "$t"
  cat > "$t/dsv41-gate" <<'EOF'
#!/bin/sh
if [ "$1" = kernels ]; then
  echo "PASS a: 10 bytes, 0 differ"
  if [ "${FAKE_KG_DIFF:-}" = "$ARM" ]; then echo "PASS b: 10 bytes, 1 differ"; else echo "PASS b: 10 bytes, 0 differ"; fi
  n=${FAKE_KG_N:-1091}; [ "${FAKE_KG_HEAD_N:+x}" = x ] && [ "$ARM" = HEAD ] && n=$FAKE_KG_HEAD_N
  echo "dsv41-gate: $n checks, 0 failed"; exit 0
fi
# tp: the default transport must arrive with MEMRA_TP_TRANSPORT unset
want=916; [ -n "${MEMRA_TP_TRANSPORT+x}" ] && want=914
i=0; while [ $i -lt 180 ]; do echo "HASH h$i x"; i=$((i+1)); done
echo "dsv41-gate: $want checks, 0 failed"
EOF
  cat > "$t/kernel-check" <<'EOF'
#!/bin/sh
echo "ALL GREEN (120 cells, 3 skipped)"
EOF
  cat > "$t/run-gen" <<'EOF'
#!/bin/sh
case "${FAKE_RG:-match}" in
  match) echo "prefill argmax=1  decode argmax=1  logit maxdiff=0.0  MATCH"
         echo "batched-prime argmax=1  tokenwise argmax=1  logit maxdiff=0.0  MATCH" ;;
  neartie) echo "prefill argmax=1  decode argmax=1  logit maxdiff=0.0  MATCH"
           echo "batched-prime argmax=1  tokenwise argmax=2  logit maxdiff=0.1  FLIP-NEARTIE (tokenwise margin 0.01)" ;;
  mismatch) echo "prefill argmax=1  decode argmax=2  logit maxdiff=0.3  MISMATCH"; exit 1 ;;
esac
EOF
  cat > "$t/run-spec" <<'EOF'
#!/bin/sh
k=1; while [ $k -le 8 ]; do echo "K=$k self-consistency: PASS"; k=$((k+1)); done
echo "=== SELF-CONSISTENCY PASS ==="
EOF
  cp "$t/run-spec" "$t/argmax-margin-probe"
  for f in "$t"/*; do sed -i "2i ARM=$arm" "$f"; done
  chmod +x "$t"/*
done
printf '1 2 3\n' > "$W/ids"
mkdir -p "$W/models"; : > "$W/models/m1.gguf"; : > "$W/models/m2.gguf"
printf 'M1\t%s\t%s\nM 2\t%s\t%s\n' "$W/models/m1.gguf" "$W/ids" "$W/models/m2.gguf" "$W/ids" > "$W/board.tsv"
export PATH=$FB:$PATH OUT=$W/out SRC_BASE=$W/src-BASE SRC_HEAD=$W/src-HEAD DATA=$W/data CKPT=$W/ckpt
export CARD1=0 CARDS4=0,1,2,3 PAIR=0,1 BOARD=$W/board.tsv MEMRA_MODELS_DIR=$W/models MEMRA_GPU_LOCK=$W/gpu.lock
export MEMRA_TP_TRANSPORT=device-push   # set in the caller's env: the default arm must unset it
PH=pair
case $CASE in
  pass) ;;
  pass-all) PH=kgate; bash "$S" kgate > "$W/kg.out" 2>&1; echo "  kgate rc=$? $(tail -n 1 "$W/kg.out")"; PH=tp4 ;;
  board-missing) BOARD=$W/nope.tsv ;;
  board-partial) printf 'M1\t%s\t%s\n' "$W/models/m1.gguf" "$W/ids" > "$W/board.tsv" ;;
  board-nonl) printf 'M1\t%s\t%s\nM 2\t%s\t%s' "$W/models/m1.gguf" "$W/ids" "$W/models/m2.gguf" "$W/ids" > "$W/board.tsv" ;;
  model-absent) rm "$W/models/m2.gguf" ;;
  lock-unopenable) MEMRA_GPU_LOCK=$W/no-such-dir/gpu.lock ;;
  binary-missing) rm "$W/out/target-HEAD/release/run-spec"; PH=build ;;
  rg-mismatch) export FAKE_RG=mismatch ;;
  rg-neartie) export FAKE_RG=neartie ;;
  rb-fail) export FAKE_RB_FAIL=1 ;;
  kgate-differ) export FAKE_KG_DIFF=HEAD; PH=kgate ;;
  kgate-count-differ) export FAKE_KG_HEAD_N=1090; PH=kgate ;;
  kgate-below-floor) export FAKE_KG_N=1088; PH=kgate ;;
  kgate-pinned) export EXPECT_KGATE=1092; PH=kgate ;;
  relative-out) OUT=relative/out ;;
esac
export BOARD MEMRA_GPU_LOCK OUT
bash "$S" $PH > "$W/run.out" 2>&1
rc=$?
echo "$CASE ($PH): rc=$rc  last: $(tail -n 1 "$W/run.out")"
grep -a 'FAIL:' "$W/run.out" | sed 's/^/    /' | head -6
grep -a -E 'board:|FLIP|release-battery' "$W/run.out" | sed 's/^/    /' | head -6
