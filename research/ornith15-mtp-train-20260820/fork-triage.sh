#!/usr/bin/env bash
# CLI-vs-server fork triage (diverge@8 on the qwen35moe ST NVFP4+FP8 arm).
# Isolates which program pair forks:
#   A = run-gen CLI tokenwise          (gate's cli arm)
#   B = server batched-prime + decode  (gate's srv arm, MEMRA_SERVE_SPEC=0)
#   C = server tokenwise twin          (MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0)
# A==C, B!=C  -> batched-vs-tokenwise inside the worker (one-numeric-program class)
# A!=C        -> CLI-vs-serve tokenwise (loader/render/decode-config class)
# Repeats the same three arms on the v2 GGUF to test ST-specificity.
set -uo pipefail
cd "$HOME/models/ornith15"
mkdir -p fork-triage
BINS=$HOME/memra-src/target/release
PROMPT="What is the capital of France? Answer in one short sentence."
NGEN=48
ADDR=127.0.0.1:8097
export CUDA_VISIBLE_DEVICES=1

run_cli() { # $1 model path, $2 out
  MEMRA_CHAT=1 MEMRA_NGEN=$NGEN "$BINS/run-gen" "$1" --prompt "$PROMPT" > "$2" 2>&1
  grep '^tokens: ' "$2" | tail -1 | sed 's/^tokens: //'
}
run_srv() { # $1 model path, $2 extra env, $3 log
  env $2 MEMRA_MODELS="st=$1" MEMRA_ADDR=$ADDR "$BINS/memra-server" > "$3" 2>&1 &
  local pid=$!
  for _ in $(seq 300); do curl -sf "http://$ADDR/health" >/dev/null 2>&1 && break; sleep 2; done
  local toks
  toks=$(curl -sf -m 300 "http://$ADDR/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"st\",\"prompt\":\"$PROMPT\",\"chat\":true,\"max_tokens\":$NGEN,\"temperature\":0}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["tokens"])')
  kill $pid 2>/dev/null; wait $pid 2>/dev/null || true
  echo "$toks"
}

for target in "st:nvfp4-official" "gguf:Ornith-1.5-35B-A3B-NVFP4-MTP-v2.gguf"; do
  name=${target%%:*}; path=$HOME/models/ornith15/${target#*:}
  echo "=== target $name ==="
  A=$(run_cli "$path" "fork-triage/cli-$name.log")
  B=$(run_srv "$path" "MEMRA_SERVE_SPEC=0" "fork-triage/srv-batched-$name.log")
  C=$(run_srv "$path" "MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0" "fork-triage/srv-tokenwise-$name.log")
  python3 - "$A" "$B" "$C" "$name" <<'PYEOF'
import ast, sys
A, B, C = (ast.literal_eval(x) for x in sys.argv[1:4])
name = sys.argv[4]
def cmp(x, y, lx, ly):
    n = min(len(x), len(y))
    div = next((i for i in range(n) if x[i] != y[i]), None)
    if div is None and abs(len(x) - len(y)) <= 1:
        return f"{lx}=={ly} (identical, lens {len(x)}/{len(y)})"
    return f"{lx}!={ly} DIVERGE@{div} {lx}={x[div:div+5] if div is not None else '-'} {ly}={y[div:div+5] if div is not None else '-'} (lens {len(x)}/{len(y)})"
print(f"[{name}] " + cmp(A, C, "CLI", "SRV-TOKENWISE"))
print(f"[{name}] " + cmp(C, B, "SRV-TOKENWISE", "SRV-BATCHED"))
print(f"[{name}] " + cmp(A, B, "CLI", "SRV-BATCHED"))
PYEOF
done
echo "FORK-TRIAGE DONE"
