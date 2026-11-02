#!/usr/bin/env bash
# Runs ON gb10-b under the caller's sparklock lease: nsys kernel census of one cold prefill.
#   BIN=~/memra-run/bin-<sha12> TOKENS=32768 TAG=<name> [MODEL=gguf|nv] [EXTRA="K=V ..."] profile-prefill.sh
# Writes ~/memra-run/prof/<TAG>.{nsys-rep,kern.csv,log}; prints the top kernels by total time.
set -u
BIN=${BIN:?bin}; TAG=${TAG:?tag}; TOKENS=${TOKENS:-32768}
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
case "${MODEL:-gguf}" in
  gguf) M=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf ;;
  nv) M=$HOME/models/Qwen3.8-27B-NVFP4 ;;
esac
O=$HOME/memra-run/prof; mkdir -p "$O"
P=$O/prompt-v2-$TOKENS.txt
if [ ! -s "$P" ]; then
  # Deterministic filler: numbered prose lines, ~14.4 tokens each (measured on the Qwen3.8 tokenizer).
  python3 - "$P" "$TOKENS" <<'EOF'
import sys
path, n = sys.argv[1], int(sys.argv[2])
words = "the quick brown fox jumps over the lazy dog while seven engineers measure prefill".split()
lines, k = [], 0
while len(lines) * 14.4 < n:
    lines.append(f"{k}: " + " ".join(words[(k + i) % len(words)] for i in range(9)) + ".")
    k += 1
open(path, "w").write("\n".join(lines))
EOF
fi
for kv in ${EXTRA:-}; do export "$kv"; done
export MEMRA_NGEN=${NGEN:-4}
# Pure prefill: one warmup and one timed monolithic forward, no decode gate.
export MEMRA_PP_ONLY=1 MEMRA_PP_REPS=${REPS:-1} MEMRA_PP_WARMUP=1
export MEMRA_PROMPT_FILE="$P"
export MEMRA_CTX=${CTX:-65536}
cd "$O"
/usr/bin/time -v nsys profile -t cuda,nvtx --stats=false --force-overwrite true -o "$O/$TAG" \
  "$BIN/run-gen" "$M" > "$O/$TAG.log" 2>&1
echo "RUN_EXIT=$?"
grep -E 'prefill|prompt|tok/s|ms' "$O/$TAG.log" | tail -8
nsys stats --force-export=true --force-overwrite=true --report cuda_gpu_kern_sum --format csv --output "$O/$TAG" "$O/$TAG.nsys-rep" > /dev/null 2>&1
python3 - "$O/${TAG}_cuda_gpu_kern_sum.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
print(f"kernels={len(rows)} total_gpu_s={tot/1e9:.2f}")
for r in rows[:25]:
    t = float(r["Total Time (ns)"])
    print(f"{100*t/tot:5.1f}% {t/1e9:7.3f}s n={r['Instances']:>6} avg_us={float(r['Avg (ns)'])/1e3:9.1f} {r['Name'][:110]}")
EOF
