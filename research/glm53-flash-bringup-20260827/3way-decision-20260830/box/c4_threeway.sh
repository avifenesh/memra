#!/usr/bin/env bash
# CELL 4 — THE THREE-WAY (timed). PLAIN vs NATIVE-MTP-spec vs DFLASH2-spec.
#
# Protocol laws honoured:
#  * interleaved x5 (box clock drift invalidates cross-run perf claims) — the loop runs
#    plain, native, dflash IN THAT ORDER five times, fresh boot every arm every round;
#  * boot-nonce arm identity on every boot (health-200 proves a listener, not which server);
#  * TIMED WINDOW: the caller raises /root/TIMING-IN-FLIGHT before this script and removes it
#    after — held for the WHOLE window so arm conditions are consistent;
#  * decode tok/s on BOTH pools (c=1), TTFT at ~0.4k (l3-WARM) and ~3.7k (l3-A4630) cold,
#    ONE vendor-default sampled row per boot (never-serve-greedy law: the real traffic shape,
#    no sampling params on the wire, spec receipt from the log);
#  * the 8-turn larger-prompt cache-on twin per the 2026-08-21 owner law, MEMRA_PREFIX_CACHE_MB
#    =2000 on the twin boots ONLY (named deviation). glm5 prefix entries REFUSE to snapshot
#    (latent MLA/DSA planes), so cached_tokens will be 0 — the refusal line IS the receipt that
#    the twin ran honestly (spec-battery-20260830 finding, re-banked here per arm).
set -uo pipefail
OUT=/root/out-3way/s4
mkdir -p "$OUT"
NAT=(MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1)
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)

boot_and_time() {  # name, extras...
  local name="$1"; shift
  echo "######## S4 BOOT $name ########"
  /root/out-3way/serve.sh start "$name" "$@" || { echo "S4_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py sample --out "$OUT/$name" || { echo "S4_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-3way/logs/boot-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  grep -m1 '\[glm5-spec\] serve route ARMED' "$log" || true
  echo "S4_${name}_EXIT=0"
}

boot_and_twin() {  # name, extras...
  local name="$1"; shift
  echo "######## S4 TWIN BOOT $name ########"
  /root/out-3way/serve.sh start "$name" "$@" || { echo "S4_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py sample --out "$OUT/$name" || true
  python3 /root/out-3way/run_pool.py twin --out "$OUT/$name"
  local log=/root/out-3way/logs/boot-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  echo "--- prefix-cache lines (the honest-twin receipt) ---"
  grep -iE "prefix.cache|snapshot failed|budget" "$log" | sort -u | head -6 || echo none
  echo "S4_${name}_EXIT=0"
}

# ---- interleaved x5, three arms per round, fresh boot each ----
for i in 1 2 3 4 5; do
  boot_and_time "s4-plain$i"
  boot_and_time "s4-nat$i" "${NAT[@]}"
  boot_and_time "s4-dfl$i" "${DFL[@]}"
done

# ---- 8-turn larger-prompt cache-on twin per arm (owner law) ----
boot_and_twin "s4twin-plain-c2000" MEMRA_PREFIX_CACHE_MB=2000
boot_and_twin "s4twin-nat-c2000" "${NAT[@]}" MEMRA_PREFIX_CACHE_MB=2000
boot_and_twin "s4twin-dfl-c2000" "${DFL[@]}" MEMRA_PREFIX_CACHE_MB=2000

/root/out-3way/serve.sh stop
echo "=== LOOP-LAW SCREEN (all s4 tapes) ==="
python3 /root/out-3way/looplaw_screen.py "$OUT"/*/
echo "S4_ALL_DONE"
