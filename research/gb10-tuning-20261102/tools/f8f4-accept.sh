#!/usr/bin/env bash
# Runs ON gb10-b under the caller's lease: served acceptance A/B of the F8F4 prefill route on
# the GB10 serve config (GGUF trunk + masked MTP draft, automatic K table), OFF / ON / OFF2.
# Greedy (T=0) so drafting is deterministic and acceptance is a count, raw-prompt /v1/completions,
# one server per arm (MEMRA_MMQ_F8F4 is read once). Protocol of
# research/f8f4-flip-20260806/tools/regime_accept_ab.sh on this device.
#   BIN=~/memra-run/bin-<sha12> f8f4-accept.sh      Log: ~/memra-run/f8f4/accept.log
set -u
BIN=${BIN:?bin}
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
O=$HOME/memra-run/f8f4; mkdir -p "$O"; L=$O/accept.log
PORT=8117
PROMPTS=("$HOME/memra-run/prompts/p2-code-medium.txt" "$HOME/memra-run/prompts/p1-code-short.txt"
         "$HOME/memra-run/prompts/p3-agentic-long-v3.txt" "$HOME/memra-run/prompts/p4-16k.txt")
{ echo "[start] $(date -Is)"; sha256sum "$BIN/memra-server"; echo "[proto] T=0, max_tokens 128, OFF/ON/OFF2"; } > "$L"
for ARM in OFF ON OFF2; do
  aenv=(); [ "$ARM" = ON ] && aenv=(MEMRA_MMQ_F8F4=1)
  echo "=== ARM $ARM $(date -Is) $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" >> "$L"
  env "${aenv[@]}" MEMRA_COMPAT=openai MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
    MEMRA_MTP_DRAFT="$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf" MEMRA_PREFIX_STABLE_BOUNDARY=1 \
    MEMRA_ADDR=127.0.0.1:$PORT MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=1 MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_CACHE_MB=0 \
    "$BIN/memra-server" > "$O/accept-server-$ARM.log" 2>&1 &
  SPID=$!
  ok=0
  for i in $(seq 1 150); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 2
  done
  [ $ok = 1 ] || { echo "[FATAL] $ARM server not healthy" >> "$L"; kill $SPID 2>/dev/null; continue; }
  for p in "${PROMPTS[@]}"; do
    body=$(python3 -c 'import json,sys; print(json.dumps({"model":"q38","prompt":open(sys.argv[1]).read(),"max_tokens":128,"temperature":0,"stream":False}))' "$p")
    echo "--- prompt $(basename "$p")" >> "$L"
    curl -s -m 900 -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/v1/completions" \
      | python3 -c 'import json,sys,hashlib; d=json.load(sys.stdin); u=d.get("usage",{}); print("USAGE",json.dumps({k:u.get(k) for k in ("prompt_tokens","completion_tokens","elapsed_s")}),"SPEC",json.dumps(u.get("spec",{}))); t=d["choices"][0]["text"] if "choices" in d else "ERROR "+json.dumps(d)[:300]; print("TEXT_SHA",hashlib.sha256(t.encode()).hexdigest()[:16]); print("TEXT_HEAD",repr(t[:100]))' >> "$L" 2>&1
  done
  kill -TERM $SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 3
done
echo "[done] $(date -Is)" >> "$L"
grep -E '^=== ARM|^--- prompt|^USAGE|^TEXT_SHA|FATAL' "$L"
