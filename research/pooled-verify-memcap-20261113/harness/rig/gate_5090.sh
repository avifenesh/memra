#!/usr/bin/env bash
# Rig, inside a gpulock lease on the RTX 5090 (memra#1266): one exactness gate on one binary, in
# its own process group, stopped (with every server it booted) when it outlives CAP_MIN.
#   inv     tools/seeded-invariance-gate.py --penalised device  (MTP route: draft lengths, the pair
#           sent at once, resumes, demotion), the masked 32,768-row MTP head
#   resend  tools/prefix-exact-resend-gate.py --external-lock 9 (plain path: exact re-send, tails)
# usage: gate_5090.sh KIND BIN_PATH OUT_NAME [CAP_MIN]  -> runs/gates/OUT_NAME{,.log,.guard}; exit 0 on PASS
set -u
D=<lane>
KIND=$1; BIN=$2; OUT=$D/runs/gates/$3; CAP=${4:-27}
M=<models>/q38-gguf
T=$D/wt/tools
mkdir -p "$D/runs/gates"
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
case $KIND in
  inv) set -- python3 "$T/seeded-invariance-gate.py" "$BIN" "$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" --out "$OUT" --penalised device \
         --env "MEMRA_MTP_DRAFT=$M/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"; PASS='SEEDED INVARIANCE: PASS' ;;
  resend) set -- python3 "$T/prefix-exact-resend-gate.py" --external-lock 9 --model "$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" --bin "$BIN" --out "$OUT"; PASS='-> PASS' ;;
  *) echo "unknown kind $KIND"; exit 2 ;;
esac
{ date -Is; sha256sum "$BIN" | sed "s|$HOME|~|"; nvidia-smi --query-gpu=name,driver_version,enforced.power.limit,temperature.gpu --format=csv,noheader
  echo "ac=$(cat /sys/class/power_supply/ADP0/online) nv_tgp=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/nv_tgp/current_value) ppt_pl1_spl=$(cat /sys/class/firmware-attributes/asus-armoury/attributes/ppt_pl1_spl/current_value) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable)"; uptime; } > "$OUT.guard"
setsid "$@" > "$OUT.log" 2>&1 < /dev/null &
p=$!
if ! waitfor --gone "$p" --timeout "${CAP}m" --quiet; then
  echo "$(date -Is) gate_5090: cap ${CAP} min reached; stopping group $p" >> "$OUT.log"
  kill -TERM -- -"$p"; waitfor --gone "$p" --timeout 1m --quiet || kill -KILL -- -"$p"
fi
wait "$p"
echo "profile_end=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unreadable) $(date -Is)" >> "$OUT.guard"
grep -q -- "$PASS" "$OUT.log"
