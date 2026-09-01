#!/bin/bash
# Depth ladder for the 262k 2-card PINNED-RECIPE cell: cold primes at ~16k, ~45k, ~130k,
# ~250k tokens of REAL text (Gutenberg corpus, sha256-banked by corpus-build; prefix cache
# is OFF so every prime is cold). Greedy, max_tokens 32 per lane craft (greedy is the
# instrument; bounded tokens bound loop damage). chars->tokens ratio is calibrated from the
# first rung's server-reported usage.prompt_tokens; later rungs slice chars = target*ratio.
# Per rung it banks: the probe row (TTFD/prefill tok/s from usage+elapsed), per-card VRAM
# before/after, and the server-log lines appended during the rung (error census).
# Any engine-error/OOM: the ladder STOPS - the depth where it died is the wall, banked.
# usage: ladder.sh <outdir> <server-log>
set -u
OUT=$1; SLOG=$2
CORPUS=/root/out-262k-2c/corpus/corpus-1m.txt
CELL=/root/out-262k-2c
RATIO=4.2   # initial planning figure; replaced by rung-1 calibration
mkdir -p "$OUT"

for TARGET in 16000 45000 130000 250000; do
  CHARS=$(python3 -c "print(int($TARGET*$RATIO))")
  NAME="rung-${TARGET}"
  echo "=== $NAME: target=$TARGET tok ratio=$RATIO chars=$CHARS $(date -u +%FT%TZ) ==="
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/$NAME.vram-before.csv"
  OFF=$(stat -c%s "$SLOG")
  date -u +%FT%T.%3NZ > "$OUT/$NAME.t-start"
  python3 "$CELL/primeprobe262.py" "$NAME" "$CORPUS" "$CHARS" 32 greedy "$OUT/$NAME.json"
  RC=$?
  date -u +%FT%T.%3NZ > "$OUT/$NAME.t-end"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/$NAME.vram-after.csv"
  tail -c +$((OFF+1)) "$SLOG" > "$OUT/$NAME.server-log-slice.txt"
  grep -icE "error|oom|out of memory|alloc|fatal|panic" "$OUT/$NAME.server-log-slice.txt" \
    > "$OUT/$NAME.errcount" || true
  # calibrate ratio from the server's own token count (rung 1 and refine each rung)
  NEWRATIO=$(python3 - "$OUT/$NAME.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
pt=(d.get("usage") or {}).get("prompt_tokens")
print(round(d["chars"]/pt,4) if pt else "")
EOF
)
  [ -n "$NEWRATIO" ] && RATIO=$NEWRATIO
  if [ $RC -ne 0 ]; then
    # calibration-drift guard: a 400 naming the context ceiling is NOT the wall under
    # test (VRAM) - it means the char slice overshot 262144 tokens. Retry ONCE at a
    # 3% smaller slice with the freshly calibrated ratio, named in the receipt.
    ERRSHAPE=$(python3 - "$OUT/$NAME.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
e=(d.get("error") or "").lower()
print("ctx400" if d.get("status")==400 and any(k in e for k in ("context","ctx","length","max")) else "other")
EOF
)
    if [ "$ERRSHAPE" = "ctx400" ] && [ ! -f "$OUT/$NAME.retried" ]; then
      touch "$OUT/$NAME.retried"
      CHARS=$(python3 -c "print(int($TARGET*0.97*$RATIO))")
      echo "=== $NAME hit the admission ceiling (400), retrying once at chars=$CHARS ==="
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/$NAME.vram-before.csv"
      OFF=$(stat -c%s "$SLOG")
      date -u +%FT%T.%3NZ > "$OUT/$NAME.t-start"
      python3 "$CELL/primeprobe262.py" "$NAME" "$CORPUS" "$CHARS" 32 greedy "$OUT/$NAME.json"
      RC=$?
      date -u +%FT%T.%3NZ > "$OUT/$NAME.t-end"
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/$NAME.vram-after.csv"
      tail -c +$((OFF+1)) "$SLOG" > "$OUT/$NAME.server-log-slice.txt"
    fi
  fi
  if [ $RC -ne 0 ]; then
    echo "=== LADDER STOPPED at $NAME (probe rc=$RC) - this depth is the wall candidate ==="
    echo "$NAME" > "$OUT/WALL"
    exit $RC
  fi
done
echo "=== ladder complete: all four rungs served ==="
