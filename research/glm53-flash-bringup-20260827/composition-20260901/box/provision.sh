#!/usr/bin/env bash
# Composition-lane box provisioning (lane/glm5-composition). Runs ON the box as root.
# Provider/fleet specifics live in the PRIVATE ops repo; this script is provider-agnostic.
#
#   provision.sh <memra-git-sha> [n-cards=2] [artifact-dir=/root/models/glm53-nvfp4]
#
# Exit contract (stated exactly, #80 + #82 reviews):
#   0  = hardware gates green AND staging certified
#   65 = DISQUALIFIED (hardware/build/artifact-integrity failure — abandon the box)
#   70 = STAGING-INCOMPLETE (hardware GREEN, artifact not pulled yet — keep the box)
# HARD gates: card count, 600 W power class, driver >= 580, real P2P matrix pair, disk
# floor, build + binary markers, shard count, drafter sha. ADVISORY: the host ST probe
# (a slow host caveats absolute tok/s rows; within-box A/B ratios stay valid) — a BROKEN
# probe still disqualifies, which is the distinction #82 found collapsed.
# Raw outputs are teed to /root/provision-raw.log FIRST and parsed from files second, never
# by positional tail of a shared log (the pipe-swallow law).
set -uo pipefail
SHA=${1:?pinned memra sha}
NCARDS=${2:-2}
ART=${3:-/root/models/glm53-nvfp4}
DRAFTER=${DRAFTER:-/root/models/glm53-dflash2}
DRAFTER_SHA=b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b
RAW=/root/provision-raw.log
: > "$RAW"
fail() { echo "DISQUALIFIED: $*"; exit 65; }

echo "== 1/5 cards + power + driver =="
nvidia-smi --query-gpu=index,name,power.limit,power.max_limit,driver_version \
  --format=csv,noheader | tee -a "$RAW"
n=$(grep -c . <<<"$(nvidia-smi -L)")
[ "$n" -ge "$NCARDS" ] || fail "cards=$n < required $NCARDS"
maxp_raw=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | sort -n | head -1)
case "$maxp_raw" in
  *N/A*|'') fail "power.max_limit unreadable ($maxp_raw) — cannot prove the 600 W class" ;;
esac
maxp=${maxp_raw%%.*}
[ "$maxp" -ge 600 ] || fail "power class ${maxp}W < 600W (Max-Q or soft-capped; the stated bar is 600)"
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$drv" -ge 580 ] || fail "driver $drv < 580 (native CUDA 13 userspace required)"

echo "== 2/5 host single-thread gate (ADVISORY: slow host caveats absolute rows, never a DQ) =="
command -v /usr/bin/time >/dev/null || fail "/usr/bin/time missing — install 'time' before gating"
# -o writes ONLY the timing to its own file: no positional tail of a shared log (a stray
# line there made a broken probe indistinguishable from a slow host — #82 review).
/usr/bin/time -o /root/st.time -f %e python3 -c 'i=0
while i<20000000: i+=1' > /dev/null 2>> "$RAW" || fail "the ST probe itself failed (see $RAW)"
st=$(cat /root/st.time)
case "$st" in
  ''|*[!0-9.]*) fail "ST probe produced an unparsable measurement (${st@Q}) — broken probe, not a slow host" ;;
esac
echo "20M-iter loop: ${st}s (bar <= 0.6)"
if ! python3 -c "import sys; sys.exit(0 if float('$st') <= 0.6 else 1)"; then
  echo "HOST-CLASS CAVEAT: ST ${st}s > 0.6s — every ABSOLUTE tok/s receipt from this box"
  echo "  must name the host class; within-box A/B ratios stay valid. Not a DQ."
  echo "HOST_ST_SLOW=${st}" >> /root/provision-verdict.txt
fi

echo "== 3/5 p2p topology (peer-pull prerequisite) =="
# Capture ONCE, tee, and parse the MATRIX ROWS ONLY: nvidia-smi appends a legend containing
# "OK = Status Ok" to every topo output, so a bare `grep -q OK` passes on a box with no peer
# pair at all (reproduced live on a 1-GPU box — #82 review's confirmed vacuous gate).
nvidia-smi topo -p2p r > /root/p2p.txt 2>&1 || fail "nvidia-smi topo -p2p r failed"
cat /root/p2p.txt | tee -a "$RAW"
# Matrix rows start with GPU<N>; a pair cell reading OK on such a row is a real peer pair.
if [ "$NCARDS" -gt 1 ]; then
  grep -E '^[[:space:]]*GPU[0-9]+' /root/p2p.txt | grep -qw "OK" \
    || fail "no OK pair in the topo -p2p r MATRIX rows — peer-pull arms cannot run"
fi

echo "== 4/5 disk =="
df -B1G / | tee -a "$RAW"
freeg=$(df -B1G --output=avail / | tail -1 | tr -d ' ')
# 200G floor with MEMRA_RP=0 pinned (no repack cache); the 2x-artifacts law applies the
# moment any arm unpins RP.
[ "$freeg" -ge 200 ] || fail "free disk ${freeg}G < 200G (artifact is 191G; RP=0 pinned)"

echo "== 5/5 memra build @ $SHA =="
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
if [ ! -d /root/memra ]; then
  git clone https://github.com/avifenesh/memra.git /root/memra || fail "clone failed"
fi
cd /root/memra
git fetch origin && git checkout -q "$SHA" || fail "checkout $SHA failed"
git log -1 --format='%H %s'
t0=$(date +%s)
nice -n19 cargo build --release --bin memra-server --bin glm5-tp2-box-probe --bin glm5-tp-gate \
  >> "$RAW" 2>&1 || fail "build failed (see $RAW)"
t1=$(date +%s)
echo "BUILD_WALL_S=$((t1-t0))"
# rebuild-after-checkout law: a ~0s build after a fresh checkout is a FAILED checkout.
[ $((t1-t0)) -ge 5 ] || echo "SUSPECT: build finished in $((t1-t0))s after checkout — verify binaries"
grep -aq 'glm5-tp-preflight' target/release/glm5-tp2-box-probe \
  || fail "probe binary missing the TP preflight marker"
grep -aq 'spec x TP composition ARMED' target/release/memra-server \
  || fail "server binary missing the composition announce"

echo "== artifact + drafter (STAGING check: absent artifact is NOT a hardware DQ) =="
if [ ! -d "$ART" ]; then
  # The box qualified; the bytes are simply not here yet. A distinct exit code and NO
  # DISQUALIFIED string, so a qualification loop keyed on that word never discards good
  # hardware over a pending transfer (#82 review).
  echo "STAGING-INCOMPLETE: artifact dir $ART absent — hardware gates 1-5 GREEN; pull the"
  echo "  artifact per the private ops doc, then re-run provision.sh to certify staging."
  exit 70
fi
shards=$(ls "$ART"/*.safetensors 2>/dev/null | wc -l)
[ "$shards" -ge 20 ] || fail "artifact carries $shards shards, expected 20"
[ -f "$DRAFTER/model.safetensors" ] || fail "drafter absent at $DRAFTER"
got=$(sha256sum "$DRAFTER/model.safetensors" | awk '{print $1}')
[ "$got" = "$DRAFTER_SHA" ] || fail "drafter sha $got != pinned $DRAFTER_SHA"

echo "provision.sh DONE (all gates green) — run the fixture gate next:"
echo "  cd /root/memra && NVIDIA_TF32_OVERRIDE=0 ./target/release/glm5-tp-gate 16 12"
