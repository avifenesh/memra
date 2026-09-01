#!/usr/bin/env bash
# HEALTH.sh — the box-window boot receipt (lane/glm5-tp-transport, 2026-09-01).
#
# RUN THIS FIRST, BEFORE ANY MEASUREMENT, ON EVERY BOX WINDOW. Every check below is a
# documented case of a box reporting 100% utilisation with clean logs and no error while
# delivering a fraction of its capability. Sources are section numbers in darklanes
# research/pro6000-multicard-research-20260901/RESEARCH.md.
#
# Exit 0 = every hard check passed. Non-zero = do not open the window on this box; fix or
# re-place first. A window that measures on a degraded box banks the degradation as a result.
#
# Usage:  bash HEALTH.sh [OUTDIR]      (default OUTDIR=./health-$(hostname)-$(date +%s))
#
# Deliberately NOT here: `ncu`. §6.14 — "Profiling every rank deadlocks (the profiler
# serialises the observed kernel while its peers wait), and any metric set needing more than
# one pass deadlocks the same way. Reading the .cu alongside the DSL found all four of the
# above; the profiler found none of them."

set -u -o pipefail

OUT="${1:-./health-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
LOG="$OUT/health.log"
HARD_FAILS=0
SOFT_WARNS=0

log()  { printf '%s\n' "$*" | tee -a "$LOG" ; }
fail() { printf 'HARD-FAIL  %s\n' "$*" | tee -a "$LOG" ; HARD_FAILS=$((HARD_FAILS+1)) ; }
warn() { printf 'WARN       %s\n' "$*" | tee -a "$LOG" ; SOFT_WARNS=$((SOFT_WARNS+1)) ; }
ok()   { printf 'ok         %s\n' "$*" | tee -a "$LOG" ; }

log "=== HEALTH.sh $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) out=$OUT ==="

# ---------------------------------------------------------------------------------------
# 0. Driver, and the 5-key ForceP2P caveat pinned against THIS box's driver
# ---------------------------------------------------------------------------------------
# §2.3b: verify the LOADED params, never the file — "the sysfs node reads empty even when
# active — it's a charp the module copies without retaining". And the CAUTION: the 5-key
# recipe (adding RMForceP2PType=1;RMPcieP2PType=2) BROKE real peer copies with "invalid
# device ordinal" on driver 580.167.08 while cudaDeviceCanAccessPeer still read 1. Use the
# 3-key subset only: ForceP2P=0x11;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1
log ""
log "--- 0. driver + loaded module params (RESEARCH.md 2.3b) ---"
if [ -r /proc/driver/nvidia/version ]; then
  cat /proc/driver/nvidia/version | tee "$OUT/driver-version.txt" | tee -a "$LOG"
else
  fail "no /proc/driver/nvidia/version — nvidia module not loaded"
fi
if [ -r /proc/driver/nvidia/params ]; then
  cp /proc/driver/nvidia/params "$OUT/driver-params.txt"
  grep -E 'RegistryDwords|EnableResizableBar|DmaRemapPeerMmio|GrdmaPciTopoCheckOverride' \
    /proc/driver/nvidia/params | tee -a "$LOG"
  if grep -q 'RMForceP2PType\|RMPcieP2PType' /proc/driver/nvidia/params; then
    fail "the 5-KEY ForceP2P form is loaded (RMForceP2PType / RMPcieP2PType). §2.3b: this broke real peer copies with 'invalid device ordinal' on 580.167.08 while canAccessPeer still read 1. Use the 3-key subset."
  else
    ok "no 5-key ForceP2P keys loaded"
  fi
else
  warn "no /proc/driver/nvidia/params — cannot verify loaded params"
fi

# ---------------------------------------------------------------------------------------
# 1. Power limit ACTUAL vs MAX
# ---------------------------------------------------------------------------------------
# §6.14 / rtx6kpro #79: a card sat at 400 W of 600 W from two independent persistent sources
# (a lactd config and a systemd ExecStartPre), throttling dense-GEMM prefill 25.3% with HBM
# bandwidth unaffected. "Verify the machine before trusting a benchmark."
log ""
log "--- 1. power limit actual vs max (RESEARCH.md 6.14) ---"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.max_limit,power.min_limit \
  --format=csv | tee "$OUT/power-limits.csv" | tee -a "$LOG"
while IFS=, read -r idx _name lim _def maxlim _min; do
  case "$idx" in index|"") continue;; esac
  L=$(printf '%s' "$lim"  | tr -dc '0-9.'); M=$(printf '%s' "$maxlim" | tr -dc '0-9.')
  [ -z "$L" ] || [ -z "$M" ] && { warn "gpu$idx power limit unreadable"; continue; }
  # awk, not bash: these are floats.
  if awk -v l="$L" -v m="$M" 'BEGIN{exit !(l < m*0.95)}'; then
    fail "gpu$idx power limit ${L}W is below 95% of max ${M}W — find and remove the persistent cap before measuring (§6.14: 400/600 W cost 25.3% of prefill silently)"
  else
    ok "gpu$idx power limit ${L}W of ${M}W"
  fi
done < "$OUT/power-limits.csv"

# ---------------------------------------------------------------------------------------
# 2. The false-600W / ~600 MHz degradation — alarm on the COMBINATION
# ---------------------------------------------------------------------------------------
# §5.4: a 600 W WS card can drop to 577-675 MHz at 34-37 °C with sw_power_cap asserted,
# delivering ~1/10 to 1/20 of its FP32 throughput; reproduced across three machines, both
# OSes, four driver branches; VBIOS reflash does NOT fix it. NEVER flash VBIOS in-fleet.
# The signature is the COMBINATION: power at cap AND clocks.sm < 1 GHz AND temp < 50 °C.
log ""
log "--- 2. false-600W degradation signature (RESEARCH.md 5.4) — NEVER flash VBIOS in-fleet ---"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,temperature.gpu,power.draw,power.limit,pstate,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_thermal_slowdown \
  --format=csv | tee "$OUT/clocks-idle.csv" | tee -a "$LOG"
while IFS=, read -r idx sm _maxsm _mc _maxmc temp draw lim _ps _act swcap hwsd _thsd; do
  case "$idx" in index|"") continue;; esac
  S=$(printf '%s' "$sm" | tr -dc '0-9'); T=$(printf '%s' "$temp" | tr -dc '0-9')
  D=$(printf '%s' "$draw" | tr -dc '0-9.'); L=$(printf '%s' "$lim" | tr -dc '0-9.')
  CAP=$(printf '%s' "$swcap" | tr -d ' ')
  if [ -n "$S" ] && [ -n "$T" ] && [ -n "$D" ] && [ -n "$L" ] \
     && [ "$S" -lt 1000 ] && [ "$T" -lt 50 ] \
     && [ "$CAP" = "Active" ] \
     && awk -v d="$D" -v l="$L" 'BEGIN{exit !(d > l*0.9)}'; then
    fail "gpu$idx FALSE-CAP SIGNATURE: sm=${S}MHz temp=${T}C draw=${D}/${L}W sw_power_cap=Active — §5.4's ~600 MHz degradation. Do NOT flash VBIOS. Re-place the workload."
  else
    ok "gpu$idx sm=${S}MHz temp=${T}C draw=${D}W sw_power_cap=${CAP}"
  fi
  if [ "$(printf '%s' "$hwsd" | tr -d ' ')" = "Active" ]; then
    warn "gpu$idx hw_slowdown ACTIVE at idle"
  fi
done < "$OUT/clocks-idle.csv"

# ---------------------------------------------------------------------------------------
# 3. PCIe link gen + width, NEGOTIATED vs MAX
# ---------------------------------------------------------------------------------------
# §3.7: a card sat at Gen2 x16 for 3.5 HOURS of production — ~8 GB/s instead of 64 — while
# "nothing logged an error. The card reported 100% utilisation the whole time." Lock links
# only AFTER verifying maximum.
log ""
log "--- 3. PCIe link gen/width negotiated vs max (RESEARCH.md 3.7) ---"
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
  --format=csv | tee "$OUT/pcie-link.csv" | tee -a "$LOG"
while IFS=, read -r idx g gm w wm; do
  case "$idx" in index|"") continue;; esac
  G=$(printf '%s' "$g" | tr -dc '0-9'); GM=$(printf '%s' "$gm" | tr -dc '0-9')
  W=$(printf '%s' "$w" | tr -dc '0-9'); WM=$(printf '%s' "$wm" | tr -dc '0-9')
  if [ -n "$G" ] && [ -n "$GM" ] && [ "$G" -lt "$GM" ]; then
    fail "gpu$idx PCIe gen$G of max gen$GM — §3.7's silent downgrade (Gen2 x16 ran 3.5 h of production undetected)"
  elif [ -n "$W" ] && [ -n "$WM" ] && [ "$W" -lt "$WM" ]; then
    fail "gpu$idx PCIe width x$W of max x$WM"
  else
    ok "gpu$idx PCIe gen$G x$W (max gen$GM x$WM)"
  fi
done < "$OUT/pcie-link.csv"

# ---------------------------------------------------------------------------------------
# 4. BAR1
# ---------------------------------------------------------------------------------------
# §2.10: "Expected size: 96 GB (matching VRAM) per GPU"; "Common issue: some BIOS
# configurations default to 256 MB BAR1, which cripples P2P performance."
log ""
log "--- 4. BAR1 sizing (RESEARCH.md 2.10) ---"
nvidia-smi -q 2>/dev/null | grep -A3 'BAR1 Memory Usage' | tee "$OUT/bar1.txt" | tee -a "$LOG"
B1=$(nvidia-smi -q 2>/dev/null | awk '/BAR1 Memory Usage/{f=1} f&&/Total/{print $3; exit}')
if [ -n "${B1:-}" ]; then
  if [ "$B1" -lt 32768 ]; then
    fail "BAR1 total ${B1} MiB — far below the expected VRAM-matching size; a 256 MB BAR1 cripples P2P (§2.10). Fix in BIOS (Above 4G decoding / Resizable BAR)."
  else
    ok "BAR1 total ${B1} MiB"
  fi
else
  warn "BAR1 total unreadable"
fi

# ---------------------------------------------------------------------------------------
# 5. CPU / NUMA affinity per GPU
# ---------------------------------------------------------------------------------------
# §3.5 / NCCL#2361: a strtok() race cost 25% of all-reduce bandwidth (32.47 -> 24.23 GB/s)
# with the ONLY signature an out-of-range affinity mask. "The out-of-range 128-159 on a
# 128-CPU host is the clearest signature." Hidden by default because
# NCCL_IGNORE_CPU_AFFINITY=0 ANDs the bad mask with the process mask.
log ""
log "--- 5. CPU/NUMA affinity per GPU (RESEARCH.md 3.5) ---"
NCPU=$(nproc --all 2>/dev/null || echo 0)
log "host CPUs (nproc --all) = $NCPU"
nvidia-smi topo -m > "$OUT/topo-m.txt" 2>&1
cat "$OUT/topo-m.txt" | tee -a "$LOG"
# Any CPU id named in an affinity range must be < NCPU.
awk -v ncpu="$NCPU" '
  /^GPU[0-9]+/ {
    for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) {
      n = split($i, parts, ",");
      for (p = 1; p <= n; p++) {
        split(parts[p], r, "-");
        hi = (r[2] == "" ? r[1] : r[2]);
        if (ncpu > 0 && hi + 0 >= ncpu) { print "OUT-OF-RANGE " $1 " affinity " $i " on a " ncpu "-CPU host"; }
      }
      break;
    }
  }' "$OUT/topo-m.txt" > "$OUT/affinity-anomalies.txt" 2>/dev/null
if [ -s "$OUT/affinity-anomalies.txt" ]; then
  cat "$OUT/affinity-anomalies.txt" | tee -a "$LOG"
  fail "an affinity mask names a CPU outside this host's range — §3.5's 25% all-reduce loss signature"
else
  ok "no out-of-range CPU affinity mask"
fi
log "NUMA nodes: $(lscpu 2>/dev/null | awk -F: '/NUMA node\(s\)/{gsub(/ /,"",$2); print $2}')"

# ---------------------------------------------------------------------------------------
# 6. IOMMU mode and ACS — captured per box, which we have NEVER done
# ---------------------------------------------------------------------------------------
# §2.8: NVIDIA (NCCL 2.31.2 troubleshooting) — CUDA and the driver "do not support
# IOMMU-enabled PCIe peer-to-peer memory transfer"; the IOMMU "must be disabled on Linux
# bare-metal systems to prevent SILENT DEVICE MEMORY CORRUPTION". Concerning dmesg output is
# "iommu: Default domain type: Translated" / "DMAR: IOMMU enabled".
# ACS: when ReqRedir/CmpltRedir are set, ALL P2P traffic is forced through the upstream root
# port. Measured GPU0<->GPU1 ~50 GB/s ACS ON -> ~103 GB/s ACS OFF. Verify with
# `lspci -vv | grep ACSCtl: | grep -c ReqRedir+` -> should be 0. And: our fleet is VMs, where
# "virtual machines require ACS to function, hence disabling ACS is not an option" — so this
# is a RECORD-IT check, not always a fix-it one.
log ""
log "--- 6. IOMMU mode + ACS state (RESEARCH.md 2.8) — RECORD per box ---"
cat /proc/cmdline > "$OUT/cmdline.txt" 2>/dev/null && log "cmdline: $(cat /proc/cmdline)"
dmesg 2>/dev/null | grep -i -E 'iommu|dmar|default domain' | tail -30 > "$OUT/iommu-dmesg.txt"
if [ -s "$OUT/iommu-dmesg.txt" ]; then cat "$OUT/iommu-dmesg.txt" | tee -a "$LOG"; else log "(dmesg iommu lines unavailable — needs privileges)"; fi
if dmesg 2>/dev/null | grep -qi 'Default domain type: Translated'; then
  warn "IOMMU is in TRANSLATED mode. §2.8: NVIDIA's words are 'disabled', and the stake is SILENT DEVICE MEMORY CORRUPTION, not throughput. Record the decision and its reason (New flags: ON or OFF by design, and written)."
fi
ACS_REDIR=$(lspci -vv 2>/dev/null | grep "ACSCtl:" | grep -c "ReqRedir+" || true)
log "ACS ReqRedir+ ports: ${ACS_REDIR:-unknown} (expected 0; nonzero forces P2P through the root port, measured ~50 vs ~103 GB/s)"
lspci -vv 2>/dev/null | grep "ACSCtl:" > "$OUT/acs-ctl.txt" || true
if [ "${ACS_REDIR:-0}" != "0" ]; then
  warn "ACS ReqRedir is set on ${ACS_REDIR} port(s). On a VM this is expected and not removable; record it beside every bandwidth number from this box."
fi

# ---------------------------------------------------------------------------------------
# 7. topo -p2p — TIER-1 ONLY, and -p2p a returning NS is EXPECTED
# ---------------------------------------------------------------------------------------
# §2.4, NVIDIA's own words: "a positive `nvidia-smi topo -p2p p` result is useful, but it is
# NOT a complete system sanity check for this issue."
# §2.1: NativeAtomicSupported=0 on every SM120 pair — `-p2p a` = NS is CORRECT, not a fault.
log ""
log "--- 7. topo -p2p (TIER-1 signal only; RESEARCH.md 2.4, 2.1) ---"
for mode in r w a; do
  log "topo -p2p $mode:"
  nvidia-smi topo -p2p "$mode" 2>&1 | tee "$OUT/topo-p2p-$mode.txt" | tee -a "$LOG"
done
log "NOTE: '-p2p a' reporting NS is EXPECTED on SM120 (no native peer atomics, §2.1). Our"
log "      transport is atomics-free by design; NS is not a failure."
log "NOTE: a PASS here proves NOTHING about kernel peer reads. Section 8 is the real check."

# ---------------------------------------------------------------------------------------
# 8. The KERNEL peer-read probe — the only check that catches SysMem staging
# ---------------------------------------------------------------------------------------
log ""
log "--- 8. simpleP2P-class KERNEL peer read (RESEARCH.md 2.3b, 2.4, 2.10) ---"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_SRC="$HERE/peer-read-probe.cu"
PROBE_BIN="$OUT/peer-read-probe"
if [ ! -r "$PROBE_SRC" ]; then
  fail "peer-read-probe.cu missing beside HEALTH.sh — the kernel peer-read check cannot run"
elif ! command -v nvcc >/dev/null 2>&1; then
  fail "nvcc not on PATH — cannot build the kernel peer-read probe. This is the ONE check that detects the driver's SysMem-staging default (§2.3b); do not skip it."
else
  ARCH="${MEMRA_CUDA_ARCH:-sm_120}"
  case "$ARCH" in sm_*) ;; *) ARCH="sm_${ARCH%%a}";; esac
  if nvcc -O2 -arch="$ARCH" -o "$PROBE_BIN" "$PROBE_SRC" >"$OUT/peer-read-probe.build.log" 2>&1; then
    ok "peer-read-probe built ($ARCH)"
    set +e
    "$PROBE_BIN" 2>&1 | tee "$OUT/peer-read-probe.log" | tee -a "$LOG"
    PRC=${PIPESTATUS[0]}
    set -e 2>/dev/null || true
    case "$PRC" in
      0) ok "kernel peer read: Test passed (bytes validated both directions, 4 B .. 64 MiB)" ;;
      2) fail "kernel peer read returned WRONG BYTES. §2.4: 'peer access reports Yes, cudaMemcpyPeer runs at 26 GB/s, but kernel peer-reads return zeros.' A fused pull collective is BLOCKED on this box until the 3-key ForceP2P form is applied and this passes." ;;
      4) warn "fewer than two devices — kernel peer read not applicable (expected on the single-card rig)" ;;
      5) fail "no peer-capable pair on this host. §3.2: place every TP group INSIDE a peer island; a group spanning an island boundary has no peer path at all." ;;
      *) fail "peer-read-probe exited $PRC (CUDA error) — see $OUT/peer-read-probe.log" ;;
    esac
  else
    fail "peer-read-probe failed to build — see $OUT/peer-read-probe.build.log"
    cat "$OUT/peer-read-probe.build.log" | tail -20 | tee -a "$LOG"
  fi
fi

# ---------------------------------------------------------------------------------------
# 9. P-state, and the P8 wait before any timing
# ---------------------------------------------------------------------------------------
# §6.14 / b12x #141: "An idle transition changes the same executable graph from approximately
# 54.2 to 56.4 tok/s even though the GPUs return to P1 for measurement ... P8 waiting is
# therefore a required benchmark-normalization condition."
# §6.14 / b12x #131: one wedged clock governor (a single GPU pinned at 2610 MHz with zero
# throttle reasons) made 15 peers absorb it as spin-waits and hid a 3% difference: "every arm
# of that A/B was equally hostage to the straggler."
log ""
log "--- 9. P-state normalization (RESEARCH.md 6.14) ---"
nvidia-smi --query-gpu=index,pstate,clocks.sm,persistence_mode,compute_mode --format=csv \
  | tee "$OUT/pstate.csv" | tee -a "$LOG"
log "MEASUREMENT LAW: wait for every card to reach P8 before the first timed arm, and"
log "      interleave arms x5 (interleaved-ab-protocol-law; §2.4 independently measures a cold"
log "      card ~4% FASTER on decode). A single card wedged at max clock with no throttle"
log "      reason invalidates the whole A/B — check the clocks.sm spread above."
NSTUCK=$(awk -F, 'NR>1 && $2 !~ /P8/ {n++} END{print n+0}' "$OUT/pstate.csv")
if [ "${NSTUCK:-0}" != "0" ]; then
  log "note: $NSTUCK card(s) not at P8 right now — that is fine at boot, but re-check immediately before timing."
fi

# ---------------------------------------------------------------------------------------
# 10. Fleet identity, so the receipt is attributable
# ---------------------------------------------------------------------------------------
log ""
log "--- 10. identity ---"
{
  echo "host=$(hostname)"
  echo "uname=$(uname -srvmo)"
  echo "cpu=$(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
  echo "sockets=$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2); print $2}')"
  echo "numa_nodes=$(lscpu 2>/dev/null | awk -F: '/NUMA node\(s\)/{gsub(/ /,"",$2); print $2}')"
  echo "mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)"
  nvidia-smi --query-gpu=index,name,uuid,serial,vbios_version,memory.total --format=csv
} | tee "$OUT/identity.txt" | tee -a "$LOG"

# ---------------------------------------------------------------------------------------
log ""
log "=== HEALTH.sh summary: hard_fails=$HARD_FAILS warns=$SOFT_WARNS receipts=$OUT ==="
if [ "$HARD_FAILS" -ne 0 ]; then
  log "DO NOT OPEN THE WINDOW ON THIS BOX. Fix or re-place, then re-run."
  exit 1
fi
log "Box is clean for measurement. Bank $OUT beside the window's receipts."
exit 0
