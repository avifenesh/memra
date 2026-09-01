#!/usr/bin/env bash
# CELL 5 — concurrency row on the LEADING spec arm (arg $1 = "dfl" or "nat"), c=4 mixed pool.
#
# Two spec arms, because the K-shed policy makes them different questions on this placement:
#  * NOPIN = the DEPLOYED behavior. 3 PP stages is not the pp2 cross-device shape, so
#    spec_gate_defaults gives LOW=2/HIGH=4: at projected_active > 2 choose_spec_k returns K=0
#    with reason=Concurrency, i.e. c=4 SHEDS to plain. The receipt is the [spec-k] admission
#    line naming the shed, not an inference from the tok/s.
#  * PINNED K=3 = the counterfactual. An operator pin disables automatic demotion (the boot log
#    says so), so this arm shows what spec actually COSTS at c=4 if the shed were removed.
# Plain c=4 is the baseline both are read against.
set -uo pipefail
ARM="${1:-dfl}"
OUT=/root/out-3way/c5
mkdir -p "$OUT"
case "$ARM" in
  dfl) SPEC=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1) ;;
  nat) SPEC=(MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1) ;;
  *) echo "usage: c5_conc.sh dfl|nat"; exit 2 ;;
esac

run() {  # name, extras...
  local name="$1"; shift
  echo "######## C5 BOOT $name ########"
  /root/out-3way/serve.sh start "c5-$name" "$@" || { echo "C5_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-3way/run_pool.py sample --out "$OUT/$name" || { echo "C5_${name}_EXIT=SAMPLEFAIL"; return 1; }
  echo "--- c=1 reference on the same boot ---"
  python3 /root/out-3way/run_pool.py conc --out "$OUT/$name" --n 1 --mode greedy --max-tokens 256
  echo "--- c=4 mixed pool ---"
  python3 /root/out-3way/run_pool.py conc --out "$OUT/$name" --n 4 --mode greedy --max-tokens 256
  local log=/root/out-3way/logs/boot-c5-$name.log
  echo "--- K policy / shed receipts ---"
  grep -E '\[spec-gate\]|\[spec-k\]' "$log" | sort -u | head -6 || true
  echo "--- per-request admission lines (route + K + wave) ---"
  grep -oE '\[glm5-spec\] route=[a-z]+ K=[0-9]+[^m]*wave=[0-9]+' "$log" | sort | uniq -c | head -12 || true
  echo "engagement: glm5spec=$(grep -c '\[glm5-spec\]' "$log") acc=$(grep -c '\[glm5-acc\]' "$log")"
  echo "C5_${name}_EXIT=0"
}

run plain
run "$ARM-nopin" "${SPEC[@]}"
run "$ARM-k3pin" "${SPEC[@]}" MEMRA_SPEC_K=3

/root/out-3way/serve.sh stop
echo "=== LOOP-LAW SCREEN ==="
python3 /root/out-3way/looplaw_screen.py "$OUT"/*/
echo "=== AGGREGATE COMPARISON ==="
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob("/root/out-3way/c5/*/conc-*-greedy.json")):
    d = json.load(open(f))
    arm = os.path.basename(os.path.dirname(f))
    spec_rows = sum(1 for r in d["rows"] if r and r.get("spec"))
    print(f"{arm:<12} c={d['n']} wall={d['wall_s']:.1f}s tokens={d['total_completion_tokens']:>5} "
          f"agg_tok_s={d['aggregate_tok_s'] and round(d['aggregate_tok_s'],1):>6} "
          f"spec_rows={spec_rows}/{len(d['rows'])} "
          f"per_row_tok_s={[r and r['decode_tok_s'] and round(r['decode_tok_s'],1) for r in d['rows']]}")
PY
echo "C5_ALL_DONE"
