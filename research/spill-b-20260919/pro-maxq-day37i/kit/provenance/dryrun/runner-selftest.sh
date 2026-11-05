#!/usr/bin/env bash
# Laptop self-test of the kit's runner plumbing with a fake nvidia-smi: no GPU is touched. Exercises the window-lock
# check, the canonical lock hold inherited by the collector (tier-battery.py port) and by boots-f.sh (port), the
# card-pinned nvidia-smi calls, the boots-f resume rules (skip executed, cut and re-ask), the deadline watchdog kill,
# the reader and rule on a partial root, and the cleanup. Test copy only: the shipped kit is not modified.
set -uo pipefail
T=/home/avifenesh/.cache/memra-maxq/kv-vmm/scratch/selftest
KIT=/home/avifenesh/.cache/memra-maxq/kv-vmm/kit
C=$T/kv-vmm
rm -rf "$T/kv-vmm" "$T/models" "$T/evals"
mkdir -p "$C" "$T/models" "$T/evals"
cp -a "$KIT/." "$C/"
head -c 1048576 /dev/urandom > "$T/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
fake=$(sha256sum "$T/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" | cut -d' ' -f1)
sed -i "s/^WANT_MODEL=.*/WANT_MODEL=$fake/" "$C/lib.sh"
sed -i "s#/data/ai-ml/#/nonexistent-selftest/#g" "$C/lib.sh"
# Stub cell: writes what the reader and boots-f read, then the exit file.
cat > "$C/harness/research/spill-b-20260919/run-day26-cell.sh" <<'EOF'
#!/usr/bin/env bash
cell=$1; C=$RIGDIR/$cell; rm -rf "$C"; mkdir -p "$C"
echo "lock=$LOCK card=$CARD cvd=$CUDA_VISIBLE_DEVICES grow=${MEMRA_KV_VMM_GROW:-unset} alloc=${MEMRA_KV_ALLOCATOR:-unset}" > "$C/stub.txt"
echo "1 [kv-vmm] door=OFF" > "$C/server.log"
nvidia-smi -i "$CARD" --query-gpu=name --format=csv > "$C/gpu-before.csv"
echo 0 > "$C.exit"
EOF
(cd "$C/harness" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum | sed 's#  \./#  #' > MANIFEST.sha256)
# Test chain: collector under the inherited hold, boots-f resume rules, then a long sleep the deadline must kill.
cat > "$C/chain-untimed.sh" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
R=$OUT; cd "$KIT_HARNESS" || exit 1
python3 tools/tier-battery.py --rig pro-single --timeout 60 --out "$R/st-collector" --external-lock --execute \
  bash -c 'python3 tools/tier-lock-proof.py --fd "$0" --lock /tmp/memra-gpu.lock' @COLLECTOR_LOCK_FD@ > "$R/st-collector.log" 2>&1
echo "$(date -u +%FT%TZ) selftest collector rc=$?" >> "$R/chain.log"
python3 tools/tier-battery.py --rig pro-single --timeout 60 --out "$R/st-collector2" --execute true > "$R/st-collector2.log" 2>&1
echo "$(date -u +%FT%TZ) selftest collector2 rc=$?" >> "$R/chain.log"
mkdir -p "$R/boots"; echo 0 > "$R/boots/mix-spec-O1-pooled.exit"; echo "arm=pooled" > "$R/boots/mix-spec-O1-vmm.arm.txt"; mkdir -p "$R/boots/mix-spec-O1-vmm"
export WT=$KIT_HARNESS RIG_LOCK=/tmp/memra-gpu.lock LANE_BIN=$K/bins/lane/memra-server MAIN_BIN=$K/bins/main/memra-server MODEL_KEY=q38 BOOT_CTX='' STREAM_CONC=16 MEMRA_KV_VMM_GROW=inline
bash research/spill-b-20260919/rtx5090-day37/boots-f.sh "$R" mix-spec-O1-pooled:pooled:mixspec mix-spec-O1-vmm:vmm:mixspec off-main:main:mixspec
echo "$(date -u +%FT%TZ) selftest boots rc=$?" >> "$R/chain.log"
sleep 600 &
sleep 600
EOF
cat > "$C/chain-timed.sh" <<'EOF'
#!/usr/bin/env bash
echo "$(date -u +%FT%TZ) selftest timed stub card=$CARD scope=${KIT_IDLE_SCOPE:-} lockfd=$KIT_LOCK_FD" >> "$OUT/chain.log"
echo "$(date -u +%FT%TZ) MAXQ-KVVMM-TIMED-DONE" >> "$OUT/chain.log"
EOF
(cd "$C" && awk '{print $2}' KIT-MANIFEST.sha256 | xargs sha256sum > KIT-MANIFEST.new && mv KIT-MANIFEST.new KIT-MANIFEST.sha256)
export PATH=$T/fakebin:$PATH FAKE_SMI_LOG=$T/smi.log KIT_EVALS_DIR=$T/evals
rm -f "$T/smi.log"
: > "$T/evals/gpu2.lock"
echo "== dry run (no window held)"
( CARD=2 DRY_RUN=1 bash "$C/run.sh" ) 2>&1 | tail -6
echo "== refusal without the window"
( CARD=2 bash "$C/run.sh" ) 2>&1 | tail -2
echo "== run with the window held, WINDOW_MIN=1"
(
  exec 9>"$T/evals/gpu2.lock"; flock -n 9 || exit 9
  CARD=2 WINDOW_MIN=1 bash "$C/run.sh"
) > "$T/run.out" 2>&1
echo "run rc=$?"; tail -8 "$T/run.out"
OUTD=$(grep -m1 '^out: ' "$T/run.out" | cut -d' ' -f2)
echo "== chain.log"; cat "$OUTD/chain.log"
echo "== boots"; cat "$OUTD/run.log"; ls "$OUTD/boots" "$OUTD/boots/cut"; cat "$OUTD/boots/mix-spec-O1-vmm/stub.txt" "$OUTD/boots/off-main.arm.txt"
echo "== collector"; grep -h 'lock\|status' "$OUTD/st-collector/CELL.jsonl" | head -3 | cut -c1-300; tail -2 "$OUTD/st-collector.log"; ls "$OUTD/st-collector"
echo "== verdict"; cat "$OUTD/VERDICT.txt"
echo "== leftovers"; ls -a "$OUTD" | tr '\n' ' '; echo
if flock -n /tmp/memra-gpu.lock true; then echo "memra lock free after the run"; else echo "memra lock STILL HELD"; fi
echo "== nvidia-smi calls without -i"; grep -v -- '-i ' "$T/smi.log" | sort | uniq -c
echo "== nvidia-smi calls total $(wc -l < "$T/smi.log")"
echo "== run-timed on GPU 0 (refuse) and on GPU 2 into the same OUT"
( exec 9>"$T/evals/gpu0.lock"; flock -n 9 || exit 9; CARD=0 OUT=$OUTD bash "$C/run-timed.sh" ) 2>&1 | tail -1
( exec 9>"$T/evals/gpu2.lock"; flock -n 9 || exit 9; CARD=2 OUT=$OUTD bash "$C/run-timed.sh" ) > "$T/run-timed.out" 2>&1
echo "run-timed rc=$?"; tail -4 "$T/run-timed.out"; tail -6 "$OUTD/chain.log"; cat "$OUTD/stops-earlier.txt"
if flock -n /tmp/memra-gpu.lock true; then echo "memra lock free after run-timed"; else echo "memra lock STILL HELD"; fi
