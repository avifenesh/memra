#!/usr/bin/env bash
# WP-B DAY38 addendum D target-card half (DAY38.md 1.11): one RTX PRO 6000 Blackwell Workstation Edition, the 27B at the
# checkpoint's context. green = B2_SHA (the local 5090 half's source), red = green plus day38-red.patch, built with
# build-arms.sh; the boots of 1.3 with shape Xp in place of X (day38d-run.sh), the target card's lengths; day38d-read.py.
# Never a signal to anything this lane did not start. Receipts under /root/spill-receipts/b-day38d.
set -uo pipefail
R=/root/spill-receipts/b-day38d; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
B2_SHA=${B2_SHA:-a803d308080d047eefedf76fc18cf5c09484869e}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
export WT; TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$B2_SHA" green red:day38-red.patch \
  > "$R/bins/build.out" 2>&1 || { log "builds failed: $(tail -1 "$R/bins/build.out")"; exit 1; }
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS")"
export WT RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/green/memra-server" RED_BIN="$R/bins/red/memra-server" MODEL MODEL_KEY=q38 \
  BOOT_CTX='' LENGTHS=6144,30720,122880 NO_SCOPE=1
bash research/spill-b-20260919/day38d-run.sh "$R" \
  main-O1-off:off main-O1-on:on main-O2-on:on main-O2-off:off \
  fault-batch:on-fault-batch fault-nobatch:on-fault-nobatch fault-nobatch-red:on-fault-nobatch-red \
  vmm-off:vmm-off vmm-on:vmm-on
python3 research/spill-b-20260919/day38d-read.py pro6000 "$R" > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY38D-BOX-DONE"
