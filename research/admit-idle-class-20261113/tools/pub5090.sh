#!/usr/bin/env bash
# memra#1304 second review fold, runs on the RTX 5090 rig under `gpulock run` (balanced profile;
# pass/fail, no timing): the discrete publisher cell. For each arm (a memra-server binary), a
# fresh server on Qwen3.8-27B GGUF with the DFlash2 drafter, the prefix cache on and the DFlash
# restore switch on, then the same g8k request twice: the first primes cold and its boundary is
# published (trunk planes and the drafter tail, the copy publisher), the second hits that entry.
# The arms must publish the same entry and write the same texts.
# usage: pub5090.sh BOX_DIR OUT_DIR ARM=BINARY...
set -u
BOX=$1; OUT=$2; shift 2
M=/data/ai-ml/models/q38-gguf
mkdir -p "$OUT"
[ "$(cat /sys/class/power_supply/ADP0/online)" = 1 ] || { echo "not on AC: not run" | tee "$OUT/cell.log"; exit 3; }
for spec in "$@"; do [ -x "${spec#*=}" ] || { echo "binary missing: ${spec#*=}" | tee "$OUT/cell.log"; exit 4; }; done
: > "$OUT/cell.log"
for spec in "$@"; do
  arm=${spec%%=*}; BIN=${spec#*=}; D=$OUT/$arm; mkdir -p "$D"
  { echo "$(date -Is) arm $arm profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null) ac=$(cat /sys/class/power_supply/ADP0/online)"; sha256sum "$BIN" | sed "s|$HOME|~|";
    nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu --format=csv,noheader; } > "$D/arm.txt"
  env MEMRA_COMPAT=openai MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_ADDR=127.0.0.1:8093 \
    MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=1 MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_CACHE_MB=4000 \
    MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$M/Qwen3.8-27B-DFlash2-memra \
    MEMRA_FRSPEC_TRIM=$M/q38-ranks-sxc32768.gguf.txt MEMRA_DSPARK_PREFIX_RESTORE=1 \
    MEMRA_PREFIX_STABLE_BOUNDARY=1 MEMRA_TICK_TRACE=1 \
    python3 "$BOX/tsrun.py" "$BIN" > "$D/server.log" 2>&1 &
  SP=$!
  if ! waitfor --http http://127.0.0.1:8093/readyz --timeout 10m --every 5s --fail-if "! kill -0 $SP" >> "$D/arm.txt" 2>&1; then
    echo "$arm boot FAILED" | tee -a "$OUT/cell.log"; kill -TERM $SP 2>/dev/null; wait $SP 2>/dev/null; continue
  fi
  for i in 1 2; do
    BENCH_BASE=http://127.0.0.1:8093 BENCH_MODEL=q38 python3 "$BOX/idle_req.py" "$D/ab.jsonl" "$arm-$i" 8000 >> "$D/arm.txt" 2>&1
  done
  kill -TERM $SP; waitfor --gone $SP --timeout 90s >> "$D/arm.txt" 2>&1 || kill -KILL $SP
  wait $SP 2>/dev/null
  grep -E '\[server\] build:|\[(specpub|prefix-cache|dspark|admission|admit-oom)\]|DSPARK|out of memory|OUT_OF_MEMORY' "$D/server.log" | grep -v '\[prefix-cache\] lookup' > "$D/trace.txt"
  grep '^{' "$D/ab.jsonl" | cut -c1-260 | sed "s/^/$arm /" >> "$OUT/cell.log"
  echo "$arm published: $(grep -c 'insert (dspark-boundary)' "$D/server.log") dspark-boundary inserts, $(grep -c 'DSPARK restore\|dspark.*restor' "$D/server.log") restore lines, $(grep -ci 'out of memory\|OUT_OF_MEMORY' "$D/server.log") OOM lines" >> "$OUT/cell.log"
done
python3 - "$OUT" "$@" <<'PY' | tee -a "$OUT/cell.log"
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); arms = [a.split("=")[0] for a in sys.argv[2:]]
rows = {a: [json.loads(x) for x in (out / a / "ab.jsonl").read_text().splitlines() if x.strip()] if (out / a / "ab.jsonl").exists() else [] for a in arms}
ok = all(len(rows[a]) == 2 and all(r["status"] == 200 and r.get("completion_tokens") for r in rows[a]) for a in arms)
texts = {a: [r["text_sha256"] for r in rows[a]] for a in arms}
same = ok and len({tuple(t) for t in texts.values()}) == 1 and all(t[0] == t[1] for t in texts.values())
for a in arms:
    print(a, [(r["status"], r.get("completion_tokens"), r.get("finish"), r["text_sha256"][:8]) for r in rows[a]])
print("PUB5090", "PASS" if same else "FAIL", "the arms write the same text, cold and on the hit" if same else "texts differ or a request failed")
PY
