#!/usr/bin/env bash
# Runs ON unit2 under the lease: build the K2-at-32 head (k232), the dp4a probe, kernel-check
# synthetic arms, then the prime A/B M = the residual-epilogue binary vs A = k232 (K2 at 32 and the
# vectorized conv) and X = k232 with MEMRA_GDN_K2_32=0 (the vectorized conv alone).
set -u
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
P=$(ls -d "$D"/bins/resid-* | tail -n 1)
R=$D/runs/k232
mkdir -p "$R"
bash "$D/box/lane_build2.sh" k232 "$D/lane9.bundle" > "$R/build.out" 2>&1
L=$(ls -d "$D"/bins/k232-* 2>/dev/null | tail -n 1)
echo "base=$P lane=$L $(tail -n 1 "$R/build.out")" > "$R/chain.txt"
[ -n "$L" ] || { echo CHAIN10-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
(cd "$D/probe" && nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 probe_dp4a.cu -o probe_dp4a) > "$R/probe-build.log" 2>&1
"$D/probe/probe_dp4a" > "$R/probe-dp4a.log" 2>&1
echo "probe rc=$?" >> "$R/chain.txt"
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" >> "$R/chain.txt"
grep -E "GDN-DB-GQA|GDN-CONV-L2|Q-SPLIT-NORM-ROPE|FAIL" "$R/kc-fast.log" >> "$R/chain.txt"
ARM_ENV_EXTRA="MEMRA_GDN_K2_32=0" LANE=$L MAIN=$P bash "$D/box/pp_ab4.sh" "$R/pp33k" 1240 "M A X X A M"
echo CHAIN10-DONE >> "$R/chain.txt"
