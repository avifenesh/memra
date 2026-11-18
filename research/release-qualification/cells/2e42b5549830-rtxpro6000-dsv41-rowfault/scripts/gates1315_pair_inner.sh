#!/usr/bin/env bash
# the PRO 6000 box, landing 27, runs INSIDE a two-card lease (pair27.sh): #1315's pair items on the landing head (its landing test
# list): rowfault 6f (2 layers, resident), 6g (2 layers, --experts-host), 6h (40 layers, --experts-host), each on
# --devices 0,1 (the two leased cards) with MEMRA_FULL_PREC=1; each must read "9 checks, 0 failed".
# Binary: $L/bins1315/green/dsv41-gate (build1315.sh). Model inputs are read from the DSv4.1 lane's kit, nothing is
# written there.
# usage: gates1315_pair_inner.sh HEAD_SHA OUT RCPT     -> OUT/summary.txt, last line "G1315-PAIR PASS|FAIL"
set -u
. <land27>/kit/env27.sh
SHA=$1; O=$2
K=~/dsv41
C=$K/DeepSeek-V4.1-Flash-1017
TM=$K/oracle/engram_constants.token_map.i32
GR=$L/bins1315/green/dsv41-gate
mkdir -p "$O"
S=$O/summary.txt
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
for f in "$GR" "$TM"; do [ -e "$f" ] || { echo "G1315-PAIR FAIL missing $f" | tee -a "$S"; exit 2; }; done
[ "$(cat "$L/bins1315/SOURCE")" = "$SHA" ] || { echo "G1315-PAIR FAIL bins1315 is not built from $SHA" | tee -a "$S"; exit 2; }
{ echo "head $SHA start $(date -Is) CUDA_VISIBLE_DEVICES count $(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)"; sha256sum "$GR"; free -g | sed -n 2p; } > "$S"
cd "$K" || exit 1
bad=0
gate() {
  local name=$1; shift
  local t0=$SECONDS
  MEMRA_FULL_PREC=1 "$GR" rowfault "$C" --token-map "$TM" "$@" --devices 0,1 --out "$O/$name.json" > "$O/$name.log" 2>&1
  local rc=$? line fails
  line=$(grep -ao '[0-9,]* checks, [0-9]* failed' "$O/$name.log" | tail -n 1)
  fails=$(grep -ac '^FAIL' "$O/$name.log")
  local ok=no
  [ $rc = 0 ] && [ "$line" = "9 checks, 0 failed" ] && [ "$fails" = 0 ] && ok=yes
  [ $ok = yes ] || bad=1
  echo "$name: rc=$rc; ${line:-no checks line}; FAIL lines $fails; $((SECONDS - t0)) s; as expected: $ok" >> "$S"
}
gate rowfault-f --layers 2
gate rowfault-g --layers 2 --experts-host
gate rowfault-h --layers 40 --experts-host
echo "end $(date -Is)" >> "$S"
if [ $bad = 0 ]; then echo "G1315-PAIR PASS" >> "$S"; else echo "G1315-PAIR FAIL" >> "$S"; fi
exit $bad
