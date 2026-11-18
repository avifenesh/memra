#!/usr/bin/env bash
# the PRO 6000 box, landing 27, runs INSIDE a one-card lease (lease27.sh): #1315's one-card RTX PRO 6000 items on the landing head
# (its landing test list): 5 the kernel gate, 6a/6b rowfault at 2 layers (resident, --experts-host), 6e rowfault over
# the 40 layers on the one-card residency, 7 the 40-layer serve smoke (the PR's smoke_wrx2.sh, with only its binary
# directory pointed at this landing's memra-server), then the red arm: dsv41-gate built with the PR's red.patch must
# pass 6a and fail 6b with "route: 0 of 174 pairs placed". Every gate with MEMRA_FULL_PREC=1; the smoke as the PR ran
# it (its own environment). Model inputs are read from the DSv4.1 lane's kit (~/dsv41), nothing is written there.
# Binaries: $L/bins1315/{green,red}/dsv41-gate (build1315.sh) and the cell record's memra-server ($L/cells-work/bins).
# usage: gates1315_inner.sh HEAD_SHA OUT RCPT     -> OUT/summary.txt, last line "G1315 PASS|FAIL ..."
set -u
. <land27>/kit/env27.sh
SHA=$1; O=$2
K=~/dsv41
C=$K/DeepSeek-V4.1-Flash-1017
TM=$K/oracle/engram_constants.token_map.i32
FX=$K/fixtures-gpu/rand
GR=$L/bins1315/green/dsv41-gate
RD=$L/bins1315/red/dsv41-gate
SB=$L/cells-work/bins/$SHA-120a
SM=$L/src1315/research/modelplan-onboarding-dsv41-20261109/row-faults/receipts/pro6000-6371a9975b/scripts/smoke_wrx2.sh
mkdir -p "$O"
S=$O/summary.txt
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
for f in "$GR" "$RD" "$SB/memra-server" "$TM" "$SM"; do [ -e "$f" ] || { echo "G1315 FAIL missing $f" | tee -a "$S"; exit 2; }; done
[ "$(cat "$L/bins1315/SOURCE")" = "$SHA" ] || { echo "G1315 FAIL bins1315 is not built from $SHA" | tee -a "$S"; exit 2; }
{ echo "head $SHA start $(date -Is)"; sha256sum "$GR" "$RD" "$SB/memra-server"; nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.limit --format=csv,noheader; free -g | sed -n 2p; } > "$S"
cd "$K" || exit 1
bad=0
# gate NAME WANT_FAILED_COUNT_IS_ZERO(1|0) BINARY ARGS...
gate() {
  local name=$1 green=$2; shift 2
  local t0=$SECONDS
  MEMRA_FULL_PREC=1 "$@" --out "$O/$name.json" > "$O/$name.log" 2>&1
  local rc=$? line
  line=$(grep -ao '[0-9,]* checks, [0-9]* failed' "$O/$name.log" | tail -n 1)
  local fails; fails=$(grep -ac '^FAIL' "$O/$name.log")
  local ok=no
  if [ "$green" = 1 ]; then
    [ $rc = 0 ] && [ -n "$line" ] && [ "${line##*, }" = "0 failed" ] && [ "$fails" = 0 ] && ok=yes
  else
    [ $rc != 0 ] && grep -aq 'route: 0 of 174 pairs placed' "$O/$name.log" && ok=yes
  fi
  [ $ok = yes ] || bad=1
  echo "$name: rc=$rc; ${line:-no checks line}; FAIL lines $fails; $((SECONDS - t0)) s; as expected: $ok" >> "$S"
}
gate kernels 1 "$GR" kernels "$FX"
gate rowfault-a 1 "$GR" rowfault "$C" --token-map "$TM" --layers 2 --devices 0
gate rowfault-b 1 "$GR" rowfault "$C" --token-map "$TM" --layers 2 --experts-host --devices 0
gate rowfault-e 1 "$GR" rowfault "$C" --token-map "$TM" --layers 40 --experts-host
# 7: the PR's script; its line "B=$D/perf-bins/$TAG" is the only one changed
sed 's|^B=\$D/perf-bins/\$TAG$|B=${SMOKE_BINS:?}|' "$SM" > "$O/smoke27.sh"
diff "$SM" "$O/smoke27.sh" > "$O/smoke27.diff"
t0=$SECONDS
if [ "$(grep -c '^[<>]' "$O/smoke27.diff")" = 2 ]; then
  SMOKE_BINS=$SB bash "$O/smoke27.sh" l27 "$O/smoke" > "$O/smoke.out" 2>&1
  src=$?
else
  src=99; echo "the smoke script's binary line was not found: not run" > "$O/smoke.out"
fi
sp=$(grep -c '^PASS' "$O/smoke/s.txt" 2> /dev/null); sf=$(grep -c '^FAIL' "$O/smoke/s.txt" 2> /dev/null)
sok=no; grep -q '^SMOKE PASS' "$O/smoke/s.txt" 2> /dev/null && [ "${sf:-1}" = 0 ] && sok=yes
[ $sok = yes ] || bad=1
echo "smoke_wrx2: rc=$src; ${sp:-0} PASS lines, ${sf:-?} FAIL; $((SECONDS - t0)) s; as expected: $sok" >> "$S"
gate red-rowfault-a 1 "$RD" rowfault "$C" --token-map "$TM" --layers 2 --devices 0
gate red-rowfault-b 0 "$RD" rowfault "$C" --token-map "$TM" --layers 2 --experts-host --devices 0
echo "end $(date -Is)" >> "$S"
if [ $bad = 0 ]; then echo "G1315 PASS" >> "$S"; else echo "G1315 FAIL" >> "$S"; fi
exit $bad
