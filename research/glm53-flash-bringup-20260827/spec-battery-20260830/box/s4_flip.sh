#!/usr/bin/env bash
# stage-4 flip A/B: interleaved x5 fresh boots per arm, spec ON (nopin = K=3 policy
# default) vs OFF, then the 8-turn larger-prompt cache twins (cache 2000 vs 0, both arms).
# TIMED WINDOW: caller raises /root/TIMING-IN-FLIGHT before this script and removes after.
set -uo pipefail
OUT=/root/out-specbat/s4
mkdir -p "$OUT"
SPEC_ENV=(MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1)

boot_and_time() {
  local name="$1"; shift
  echo "######## S4 BOOT $name ########"
  /root/out-specbat/serve.sh start "$name" "$@" || { echo "S4_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-specbat/run_pool.py sample --out "$OUT/$name" || { echo "S4_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-specbat/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-specbat/logs/boot-$name.log
  echo "engagement: glm5spec_lines=$(grep -c "\[glm5-spec\]" "$log") acc_lines=$(grep -c "\[glm5-acc\]" "$log")"
  echo "S4_${name}_EXIT=0"
}

boot_and_twin() {
  local name="$1"; shift
  echo "######## S4 TWIN BOOT $name ########"
  /root/out-specbat/serve.sh start "$name" "$@" || { echo "S4_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-specbat/run_pool.py sample --out "$OUT/$name" || true
  python3 /root/out-specbat/run_pool.py twin --out "$OUT/$name"
  local log=/root/out-specbat/logs/boot-$name.log
  echo "engagement: glm5spec_lines=$(grep -c "\[glm5-spec\]" "$log") acc_lines=$(grep -c "\[glm5-acc\]" "$log") cache_lines=$(grep -ciE "prefix.cache|cache hit" "$log")"
  echo "S4_${name}_EXIT=0"
}

for i in 1 2 3 4 5; do
  boot_and_time "s4-off$i"
  boot_and_time "s4-on$i" "${SPEC_ENV[@]}"
done

# 8-turn larger-prompt cache twins (owner law). MEMRA_PREFIX_CACHE_MB=2000 is the NAMED
# deviation on the c2000 twins only.
boot_and_twin "s4twin-off-c2000" MEMRA_PREFIX_CACHE_MB=2000
boot_and_twin "s4twin-on-c2000" "${SPEC_ENV[@]}" MEMRA_PREFIX_CACHE_MB=2000
boot_and_twin "s4twin-off-c0"
boot_and_twin "s4twin-on-c0" "${SPEC_ENV[@]}"

/root/out-specbat/serve.sh stop
echo "S4_ALL_DONE"
