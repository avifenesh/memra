# Sourced (memra#1316): the serving both occurrences booted, memra#1307's serve1307.sh in its
# dflash mode: Qwen3.8-27B (GGUF NVFP4 trunk, Q5_K output head) with the DFlash2 drafter,
# 32,768 context, 4 sessions, prefix cache and reuse pool off, on 127.0.0.1:8090. Every MEMRA_*
# variable of the calling shell is cleared first.
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
Q=$HOME/models/memra-q38
G8=$Q/Qwen3.8-27B-NVFP4-MTP-GGUF
export MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:8090 MEMRA_KV_HOST_MB=0 MEMRA_TICK_TRACE=1
export MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0 MEMRA_REUSE_POOL=0
export MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
export MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$Q/Qwen3.8-27B-DFlash2-memra
