#!/usr/bin/env bash
# WP-B DAY42 target-card half (DAY42.md 1.4 to 1.5, addendum A): one RTX PRO 6000 Blackwell Workstation Edition, the 27B
# at the checkpoint's context, the host tier at 32,768 MB, burst 64. Builds S42's memra-server (build-arms.sh), runs the
# eight boots (day42-run.sh), the reader, a manifest. Refuses to start without ss or lsof. Never a signal to anything this
# lane did not start. Receipts under /root/spill-receipts/b-day42.
set -uo pipefail
R=/root/spill-receipts/b-day42; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S42=${S42:-1d11d5426fc137401e8166bafa094cb15ae04a0e}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
command -v ss >/dev/null || command -v lsof >/dev/null || { log "neither ss nor lsof on PATH; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
[ "$(free -g | awk '/^Mem:/{print $2}')" -ge 96 ] || { log "host RAM under 96 GB for a 32 GB pinned tier; not run"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
export WT
TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$S42" tip > "$R/bins/build.out" 2>&1 \
  || { log "build failed: $(tail -1 "$R/bins/build.out")"; exit 1; }
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS")"
export RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/tip/memra-server" MODEL MODEL_KEY=q38 BOOT_CTX='' NO_SCOPE=1 HOST_MB=32768 BURST=64
bash research/spill-b-20260919/day42-run.sh "$R" ontick-O1:ontick offtick-O1:offtick offtick-O2:offtick ontick-O2:ontick \
  ontick-nocontracts:ontick-nocontracts fault-d2h-delay:fault-d2h-delay fault-d2h-source-flip:fault-d2h-source-flip \
  fault-sources-helper-gone:fault-sources-helper-gone || log "boots stopped rc=$?"
python3 research/spill-b-20260919/day42-read.py pro6000 "$R" > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY42-BOX-DONE"
