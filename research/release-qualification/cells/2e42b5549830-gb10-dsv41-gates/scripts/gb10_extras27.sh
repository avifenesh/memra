#!/usr/bin/env bash
# Runs ON a GB10 under the caller's sparklock lease, AFTER the landing's sm_121a record is sealed: the GB10 items of
# #1315's and #1327's landing lists on the landing head, as scoped extras (receipts beside the record, not records).
#   build   dsv41-gate in the record's own tree-121a, with cell_evidence's build command and environment plus that
#           one binary, so nothing else is rebuilt; the six record binaries must still equal the record's cache.
#   #1315   kernel gate (expect 1,027 checks, 0 failed); rowfault a-d on the first 2 trunk layers (resident and
#           --experts-host, one rank and --devices 0,0; 9 checks, 0 failed each); the PR's smoke5.sh with the
#           record's memra-server (SMOKE PASS, 18 PASS lines). The commands are the PR's gset.sh and smoke5.sh.
#   #1327   the tp gate with two ranks on the card (--devices 0,0) under the three transports, 0 failed each
#           (the PR's gb.sh); then, if PART4=1, the same with four ranks on the card (--devices 0,0,0,0), the form
#           the PR's own GB10 receipts ran. Its kernel gate is the one above.
# Every gate with MEMRA_FULL_PREC=1. Model inputs: ~/models/DeepSeek-V4.1-Flash-1017 and ~/.cache/dsv41-speed/data
# (the DSv4.1 lane's kit on the unit); nothing is written there.
# usage: gb10_extras27.sh HEAD_SHA WORK_DIR [PART4=1]   -> ~/.cache/memra-land27/extras/gb10-<sha10>/summary.txt
#        markers: BUILD-DONE, P1315-DONE, P1327-DONE, EXTRAS-GB10 PASS|FAIL, EXTRAS-GB10-DONE
set -u
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
SHA=$1; W=$2; PART4=${3:-1}
L=$HOME/.cache/memra-land27
T=$W/tree-121a
BINS=$W/bins/$SHA-121a
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
FX=$D/fixtures-gpu/rand
SM=$L/cells-src/research/modelplan-onboarding-dsv41-20261109/row-faults/receipts/gb10-50413a3a4d/scripts/smoke5.sh
O=$L/extras/gb10-${SHA:0:10}
mkdir -p "$O"
S=$O/summary.txt
echo $$ > "$L/extras.pgid"
bad=0
fin() { [ "$1" = 0 ] && echo "EXTRAS-GB10 PASS" >> "$S" || echo "EXTRAS-GB10 FAIL" >> "$S"; echo "EXTRAS-GB10-DONE $(date -Is)" >> "$S"; exit "$1"; }
echo "head $SHA start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" > "$S"
[ "$(git -C "$T" rev-parse HEAD)" = "$SHA" ] || { echo "tree-121a is not at $SHA" >> "$S"; fin 2; }
for f in "$TM" "$FX" "$SM" "$BINS/memra-server"; do [ -e "$f" ] || { echo "missing $f" >> "$S"; fin 2; }; done
[ -d "$C" ] || { echo "missing $C" >> "$S"; fin 2; }

# ---- build: one more binary in the record's tree
t0=$SECONDS
( cd "$T" && env CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_LTO=fat \
    MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR="$T/target" \
    flock "$HOME/.cache/memra-rig-build.lock" nice -n 15 cargo build --release --locked --quiet \
    -p memra-engine --bin kernel-check --bin run-gen --bin run-spec --bin argmax-margin-probe --bin dsv41-gate \
    -p memra-server --bin memra-server -p memra-tokenizer --bin tok-parity ) > "$O/build.log" 2>&1
rc=$?
echo "build dsv41-gate rc=$rc $((SECONDS - t0)) s" >> "$S"
[ $rc = 0 ] || { tail -n 20 "$O/build.log" >> "$S"; fin 1; }
same=0
for b in kernel-check run-gen run-spec argmax-margin-probe memra-server tok-parity; do cmp -s "$T/target/release/$b" "$BINS/$b" && same=$((same + 1)); done
echo "record binaries unchanged by the build: $same of 6" >> "$S"
[ $same = 6 ] || bad=1
mkdir -p "$O/bin" && cp "$T/target/release/dsv41-gate" "$O/bin/dsv41-gate"
G=$O/bin/dsv41-gate
sha256sum "$G" "$BINS/memra-server" | sed "s|$HOME|~|" >> "$S"
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader >> "$S" 2>&1
echo "BUILD-DONE $(date -Is)" >> "$S"

# GB10's memory is the host's: drop the checkpoint's cached pages before each load (the PR's gset.sh)
drop_cache() {
  python3 - "$C" <<'PY'
import os, sys, glob
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
PY
}
# gate NAME WANT("0 failed" must hold; EXPECT_CHECKS or "-") TIMEOUT CMD...
gate() {
  local name=$1 want=$2 to=$3; shift 3
  local t0=$SECONDS
  MEMRA_FULL_PREC=1 timeout "$to" "$@" --out "$O/$name.json" > "$O/$name.log" 2>&1
  local rc=$? line fails ok=no
  line=$(grep -ao '[0-9,]* checks, [0-9]* failed' "$O/$name.log" | tail -n 1)
  fails=$(grep -ac '^FAIL' "$O/$name.log")
  if [ $rc = 0 ] && [ -n "$line" ] && [ "${line##*, }" = "0 failed" ] && [ "$fails" = 0 ]; then
    if [ "$want" = - ] || [ "${line%% *}" = "$want" ]; then ok=yes; fi
  fi
  [ $ok = yes ] || bad=1
  echo "$name: rc=$rc; ${line:-no checks line}; FAIL lines $fails; $((SECONDS - t0)) s; as expected: $ok" >> "$S"
}

# ---- #1315
gate kernels 1027 20m "$G" kernels "$FX"
for dev in 0 0,0; do
  for arm in resident host; do
    drop_cache
    XF=(); [ $arm = host ] && XF=(--experts-host)
    gate "rowfault-$arm-d${dev/,/_}" 9 12m "$G" rowfault "$C" --token-map "$TM" --layers 2 "${XF[@]}" --devices "$dev"
  done
done
sed -e 's|^B=\$R/bins/\$TAG$|B=${SMOKE_BINS:?}|' -e 's|^O=\$R/out/smoke5-\$TAG; |O=${SMOKE_OUT:?}; |' "$SM" > "$O/smoke27.sh"
diff "$SM" "$O/smoke27.sh" > "$O/smoke27.diff"
t0=$SECONDS
if [ "$(grep -c '^[<>]' "$O/smoke27.diff")" = 4 ]; then
  SMOKE_BINS=$BINS SMOKE_OUT=$O/smoke bash "$O/smoke27.sh" l27 > "$O/smoke.out" 2>&1
  src=$?
else
  src=99; echo "the smoke script's two path lines were not found: not run" > "$O/smoke.out"
fi
sp=$(grep -c '^PASS' "$O/smoke/s.txt" 2> /dev/null); sf=$(grep -c '^FAIL' "$O/smoke/s.txt" 2> /dev/null)
sok=no; grep -q '^SMOKE PASS' "$O/smoke/s.txt" 2> /dev/null && [ "${sf:-1}" = 0 ] && sok=yes
[ $sok = yes ] || bad=1
echo "smoke5: rc=$src; ${sp:-0} PASS lines, ${sf:-?} FAIL; $((SECONDS - t0)) s; as expected: $sok" >> "$S"
echo "P1315-DONE $(date -Is)" >> "$S"

# ---- #1327: the tp gate, two ranks on the card, three transports; then four ranks
tp() {  # DEVLIST
  local dev=$1 tr
  for tr in default peer-pull host-canonical; do
    drop_cache
    if [ $tr = default ]; then
      gate "tp-$tr-${dev//,/_}" - 13m env -u MEMRA_TP_TRANSPORT "$G" tp "$C" "$D/captures-v2/needle" --layers 1,2,14,20,21,24 --devices "$dev" --decode 2 --token-map "$TM"
    else
      gate "tp-$tr-${dev//,/_}" - 13m env MEMRA_TP_TRANSPORT=$tr "$G" tp "$C" "$D/captures-v2/needle" --layers 1,2,14,20,21,24 --devices "$dev" --decode 2 --token-map "$TM"
    fi
    grep -a "stress/" "$O/tp-$tr-${dev//,/_}.log" | cut -c1-200 >> "$S"
  done
}
tp 0,0
echo "P1327-DONE $(date -Is)" >> "$S"
if [ "$PART4" = 1 ]; then
  tp 0,0,0,0
  echo "P1327-FOUR-RANKS-DONE $(date -Is)" >> "$S"
fi
echo "end $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$S"
fin $bad
