#!/usr/bin/env bash
# MV-DOORS CELL 5 — census re-run on the WINNER (LANE.md §7: the census IS the door-X/M
# efficiency receipt). Duration-bounded nsys (the diet-battery instrument trap: memra-server
# dies on TERM without CUPTI's atexit flush and ignores INT — `--duration=N --kill=none`
# detaches and flushes while the server keeps running; then a scoped TERM).
# Instrument config mirrors c8-ship EXACTLY (CTX=8192 MAX_SESSIONS=1 PREFIX_CACHE_MB=0
# MOE_FUSED_EPI=1, port 18412, same census prompt p0, 192 completion tokens) so the
# per-kernel table diffs against §1 of matvec LANE.md row by row.
# Usage: c5_census.sh "<winner door flags>" "<winner K pin env or ''>"
set -uo pipefail
WINNER_DOORS="${1:?pass the winner door flags (or '' for none)}"
WINNER_K="${2:-}"
OUT=/root/out-mv/c5
BIN=/root/memra-mv/target/release/memra-server
PORT=18412
DUR=180
mkdir -p "$OUT/prompts"
cd "$OUT"

python3 - <<'PY'
import json
d = json.load(open("/root/memra/research/glm53-flash-bringup-20260827/decode-attribution-receipts/prompts.json"))
open("prompts/p0.txt", "w").write(d["decode"][0]["text"])
PY
python3 - <<'PY'
import json
text = open("prompts/p0.txt").read()
json.dump({"model": "zai/glm-5.3-flash",
           "messages": [{"role": "user", "content": text}],
           "max_tokens": 192, "temperature": 0.0, "stream": False},
          open("census-prompt.json", "w"))
PY

{
  echo "== provenance =="
  git -C /root/memra-mv log -1 --format='%H %s'
  ls -la "$BIN"; sha256sum "$BIN" | cut -c1-16
  echo "WINNER_DOORS=$WINNER_DOORS WINNER_K=$WINNER_K"
  echo "instrument: nsys --duration=$DUR --kill=none (CUPTI-flush trap fix), port $PORT"
  LE=/root/memra-mv/target/release/launch-econ
  [ -x "$LE" ] && "$LE" 3200 || echo "launch-econ missing (banked box constant 2.049 us/launch applies)"
} 2>&1 | tee "$OUT/mv-census-provenance.txt"

PP_ENV="MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7 $WINNER_DOORS $WINNER_K"
echo "PP_ENV=$PP_ENV" >> "$OUT/mv-census-provenance.txt"

env CUDA_VISIBLE_DEVICES=0,1,2 NVIDIA_TF32_OVERRIDE=0 \
  MEMRA_ADDR="127.0.0.1:${PORT}" MEMRA_COMPAT=openai MEMRA_CTX=8192 \
  MEMRA_MAX_SESSIONS=1 MEMRA_PREFIX_CACHE_MB=0 MEMRA_MOE_FUSED_EPI=1 \
  MEMRA_MODELS="zai/glm-5.3-flash=/root/models/glm53-nvfp4" \
  $PP_ENV \
  nsys profile --trace=cuda,osrt --duration=$DUR --kill=none \
    -o "$OUT/mv-census" "$BIN" > "$OUT/census-server.log" 2>&1 &
NSYS=$!

ready=0
for _ in $(seq 1 120); do
  curl -sf -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$NSYS" 2>/dev/null || { echo "nsys/server exited early"; tail -20 "$OUT/census-server.log"; exit 1; }
  sleep 2
done
[ "$ready" = 1 ] || { echo "census server never ready"; exit 1; }
echo "census server READY; sending the census request"
T0=$(date +%s.%N)
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @"$OUT/census-prompt.json" > "$OUT/mv-census-response.json"
T1=$(date +%s.%N)
python3 - "$OUT/mv-census-response.json" "$T0" "$T1" <<'PY' | tee "$OUT/mv-census-wall.txt"
import json, sys
r = json.load(open(sys.argv[1]))
u = r.get("usage", {})
wall = float(sys.argv[3]) - float(sys.argv[2])
print(f"wall_s={wall:.3f} prompt_tokens={u.get('prompt_tokens')} "
      f"completion_tokens={u.get('completion_tokens')} spec={u.get('spec')}")
PY

echo "waiting for the duration window + report finalize (nsys pid $NSYS)"
wait "$NSYS" || true
ls -la "$OUT"/mv-census.nsys-rep

# scoped stop of the still-running server (--kill=none leaves it up)
SPID=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)
if [ -n "${SPID:-}" ]; then
  exe=$(readlink -f "/proc/$SPID/exe" 2>/dev/null || true)
  case "$exe" in
    "$BIN"|"$BIN (deleted)") echo "scoped TERM census server pid=$SPID"; kill -TERM "$SPID";
      for _ in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
      kill -0 "$SPID" 2>/dev/null && kill -KILL "$SPID" ;;
    *) echo "REFUSE stop pid=$SPID exe=$exe (not our binary)";;
  esac
fi

nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/mv-census.nsys-rep" > "$OUT/mv-census-kernsum.csv"
nsys stats --report cuda_api_sum      --format csv "$OUT/mv-census.nsys-rep" > "$OUT/mv-census-apisum.csv" || true
CT=$(grep -o 'completion_tokens=[0-9]*' "$OUT/mv-census-wall.txt" | cut -d= -f2)
python3 /root/out-mv/c8_buckets.py "$OUT/mv-census-kernsum.csv" "$OUT/mv-census-apisum.csv" "${CT:-192}" \
  | tee "$OUT/mv-census-phase-buckets.txt"

echo "=== §7 kernel-row extract (vs matvec LANE.md §1) ==="
python3 - "$OUT/mv-census-kernsum.csv" <<'PY' | tee "$OUT/mv-census-doors-extract.txt"
import csv, io, sys
lines = open(sys.argv[1]).read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
want = ("matvec_bf16", "topk_rows", "moe_gate_up_preclamp8", "moe_down8_fma")
print(f"{'kernel':56} {'inst':>6} {'total_ms':>9} {'avg_us':>8}")
for r in rows:
    n = r.get("Name", "")
    if any(w in n for w in want):
        ns = float(r.get("Total Time (ns)", 0) or 0); inst = int(r.get("Instances", 0) or 0)
        print(f"{n[:56]:56} {inst:>6} {ns/1e6:>9.1f} {ns/inst/1e3 if inst else 0:>8.1f}")
PY
echo "C5_CENSUS_DONE"
