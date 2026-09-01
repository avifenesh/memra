#!/usr/bin/env bash
# CELL 6b — interleaved CONFIRMATION of the winning K.
#
# Why this exists: cell 6's sweep uses ONE boot per K, which is enough to LOCATE the knee but
# not enough to CLAIM a flip (interleaved-A/B protocol law: box clock drift invalidates
# cross-run perf claims). So the sweep locates, and this cell claims: plain vs dfl@K*
# interleaved x3 fresh boots each, same binary, same placement, marker held.
#
# usage: c6b_confirm.sh <K> [rounds]
set -uo pipefail
K="${1:?usage: c6b_confirm.sh <K> [rounds]}"
R="${2:-3}"
OUT=/root/out-3way/c6b
mkdir -p "$OUT"
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)

boot_and_time() {
  local name="$1"; shift
  echo "######## C6B BOOT $name ########"
  /root/out-3way/serve.sh start "$name" "$@" || { echo "C6B_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py sample --out "$OUT/$name" || { echo "C6B_${name}_EXIT=SAMPLEFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py timed --out "$OUT/$name" --max-tokens 256
  local log=/root/out-3way/logs/boot-$name.log
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  grep -m1 -E '\[glm5-spec\] route=spec K=' "$log" || true
  echo "C6B_${name}_EXIT=0"
}

for i in $(seq 1 "$R"); do
  boot_and_time "c6b-plain$i"
  boot_and_time "c6b-dfl$i" "${DFL[@]}" MEMRA_SPEC_K="$K"
done

/root/out-3way/serve.sh stop
echo "=== LOOP-LAW SCREEN ==="
python3 /root/out-3way/looplaw_screen.py "$OUT"/*/
echo "=== CONFIRMATION TABLE (plain vs dfl@K=$K, interleaved x$R) ==="
python3 /root/out-3way/summarize.py "$OUT" --prefix c6b- --arms plain,dfl --rounds "$(seq -s, 1 "$R")"
echo "=== PER-PROMPT BREAK-EVEN ==="
python3 /root/out-3way/breakeven.py "$OUT" --prefix c6b- --arm dfl --base plain --rounds "$(seq -s, 1 "$R")"
echo "C6B_ALL_DONE"
