#!/usr/bin/env bash
# RED ARM for the card-isolation refusal (memra#264).
#
# The refusal exists because the battery once passed on an empty card and failed on a shared
# one with the same binary. A check that has never said no is not a check, so this holds a
# real allocation on the card and asserts the battery REFUSES rather than producing either
# verdict. It also asserts the receipt names the tenants, which is the line that would have
# saved a bisect.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
fail() { echo "FAIL: $*" >&2; exit 1; }

command -v nvidia-smi >/dev/null || { echo "SKIP: no nvidia-smi (the refusal is a no-op without a card)"; exit 0; }
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | head -1)
[ -n "$TOTAL" ] || fail "could not read card total"

# 1. PURE ARM: the arithmetic, with no card involved.
need=$(CARD_HEADROOM_MIB=1024; sz=$( { stat -c %s "$0"; } ); echo $(( sz / 1048576 + 1024 )))
[ "$need" -ge 1024 ] || fail "model_need_mib arithmetic lost the headroom"

# 2. RED ARM: hold most of the card, then require the refusal.
#
# The holder is a 12-line cudaMalloc compiled here with nvcc, NOT a python/torch import.
# The first version of this test used torch, torch is not on the rig, and it answered SKIP —
# which is precisely the shape this repo refuses ("a gate you satisfy by not having the file
# is not a gate"). nvcc is already required to build the engine, so if it is missing there is
# no binary to gate in the first place.
command -v nvcc >/dev/null || { echo "SKIP: no nvcc — there is no engine build to gate here"; exit 0; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"; kill $HOLDER 2>/dev/null' EXIT
cat > "$TMP/hold.cu" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
int main(int argc, char** argv) {
  size_t mib = (size_t)atoll(argv[1]);
  void* p = nullptr;
  if (cudaMalloc(&p, mib * 1024ull * 1024ull) != cudaSuccess) { printf("HOLD-FAILED\n"); fflush(stdout); return 1; }
  printf("HELD\n"); fflush(stdout);
  // Hold until killed; the caller kills us as soon as the battery has answered.
  for (;;) { struct timespec t{1, 0}; nanosleep(&t, nullptr); }
}
CU
nvcc -o "$TMP/hold" "$TMP/hold.cu" 2>/dev/null || { echo "SKIP: nvcc could not build the holder"; exit 0; }
# Size the hold from what is FREE, not from the card total: the rig's card usually has a few
# GB already spoken for, and asking for total-512 simply fails to allocate (which the first
# run of this arm proved by refusing to arm itself rather than reporting a hollow pass).
# Leave ~2 GB, which is far under any roster model's need.
FREE0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1)
HOLD=$(( FREE0 - 2048 ))
[ "$HOLD" -gt 1024 ] || { echo "SKIP: card already too full to stage the arm (free ${FREE0}MiB)"; exit 0; }
"$TMP/hold" "$HOLD" & HOLDER=$!
sleep 10
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1)
[ "${FREE:-999999}" -gt 4096 ] && fail "the holder did not take the card (free ${FREE}MiB) — the red arm cannot arm itself"

# The battery refuses on missing binaries BEFORE it ever looks at the card, so the arm needs
# a build to reach the check under test. Borrow one rather than spend 25 minutes compiling:
# this arm is about the card gate, not about the linker.
mkdir -p "$ROOT/target/release"
for t in kernel-check run-spec argmax-margin-probe; do
  [ -x "$ROOT/target/release/$t" ] && continue
  src=$(ls -1 "$ROOT"/../*/target/release/"$t" 2>/dev/null | head -1)
  [ -n "$src" ] || { echo "SKIP: no built $t anywhere to borrow — build once, then this arm runs"; exit 0; }
  ln -sf "$src" "$ROOT/target/release/$t"
done

OUT=$(cd "$ROOT" && CARD_WAIT_S=10 timeout 600 tools/release-battery.sh 2>&1 || true)
kill $HOLDER 2>/dev/null; wait $HOLDER 2>/dev/null

printf '%s' "$OUT" | grep -q "REFUSED   card cannot seat the model" \
  || fail "a starved card must produce REFUSED, got:"$'\n'"$(printf '%s' "$OUT" | tail -5)"
printf '%s' "$OUT" | grep -q "on the card:" \
  || fail "the refusal must name the other tenants on the card"
printf '%s' "$OUT" | grep -q "RELEASE BATTERY PASS" \
  && fail "a starved card must never produce a PASS receipt"
echo "PASS: the battery refuses a starved card and names what is holding it"
