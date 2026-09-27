#!/usr/bin/env bash
# OWED 20 sitting: storage-bound Step-3.7-Flash IQ4_XS on the qualified PP-2 path plus the disk tier
# (M1-PREREG.md section H; SITTING.md). Run as root from the repository root at the lane tip:
#   mkdir -p /root/spill-receipts/f-s20
#   setsid nohup timeout 10h bash research/spill-f-20260919/owed20/run-sitting.sh \
#     > /root/spill-receipts/f-s20/sitting.log 2>&1 < /dev/null & disown
# Every GPU step is its own collector cell on /tmp/memra-gpu.lock (`--rig pro-pair`); a failing step
# stops the sitting with its receipts kept. Nothing else may run on the box.
set -uo pipefail
S=${S:-/scratch/spill-f}
R=${R:-/root/spill-receipts/f-s20}
PRIV=${PRIV:-/root/f-private}
F=research/spill-f-20260919
L=$F/owed20/s20-arms.lock.json
REV=0b69336d2fd2adfdef9c66e425f7778196c31482
HF=https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF/resolve/$REV
SHARD1=$S/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
MTP=$S/Step3.7-flash-mtp-Q8_0.gguf
mkdir -p "$S/bin" "$R" "$PRIV"
chmod 700 "$PRIV"

step() {
  local name=$1; shift
  echo "== $name start $(date -u +%FT%TZ)"
  "$@" > "$R/$name.driver.log" 2>&1
  local rc=$?
  echo "== $name end $(date -u +%FT%TZ) rc=$rc"
  if [ "$rc" -ne 0 ]; then echo "STOPPED at $name (rc=$rc); receipts kept in $R"; exit "$rc"; fi
}
col() {  # col NAME TIMEOUT -- command...  (a collector cell on the proven scratch, both cards)
  local name=$1 timeout=$2; shift 3
  local ext=()
  case " $* " in *@COLLECTOR_LOCK_FD@*) ext=(--external-lock) ;; esac
  step "$name" python3 tools/tier-battery.py --rig pro-pair --timeout "$timeout" "${ext[@]}" \
    --storage-root "$S" --storage-proof "$R/m1-proof/PROOF.json" --out "$R/$name" --execute "$@"
}

# 0. The storage-bound precondition: host RAM below the expert bank (section H).
step ram python3 -c "
import json;b=json.load(open('$L'))['artifact']['expert_bank_bytes']
t=int([l for l in open('/proc/meminfo') if l.startswith('MemTotal:')][0].split()[1])*1024
print(json.dumps({'mem_total_bytes':t,'expert_bank_bytes':b}));raise SystemExit(0 if t<b else 3)"

# 1. Build and freeze the binaries (the OWED 26 fix build).
git rev-parse HEAD > "$R/commit.txt"
step build bash -c "MEMRA_CUDA_ARCH=120a cargo build --release -p memra-engine --bin run-gen --bin run-spec"
cp target/release/run-gen target/release/run-spec "$S/bin/"
(cd "$S/bin" && sha256sum run-gen run-spec) > "$R/binaries.sha256"

# 2. Stage the pinned artifact, every file byte-verified against the lock.
stage() {
  python3 - "$L" <<'PY' | while read -r path size sha; do
import json, sys
for f in json.load(open(sys.argv[1]))["artifact"]["files"]:
    print(f["path"], f["bytes"], f["sha256"])
PY
    name=$(basename "$path")
    curl -fL --retry 5 -o "$S/$name.part" "$HF/$path" || return 1
    [ "$(stat -c %s "$S/$name.part")" = "$size" ] || return 1
    echo "$sha  $S/$name.part" | sha256sum -c || return 1
    mv "$S/$name.part" "$S/$name"
  done
}
step stage stage

# 3. M1 proof of the scratch (a FAIL stops here).
step m1-proof python3 tools/tier-battery.py --rig pro-pair --timeout 600 --out "$R/m1-proof" --execute \
  python3 $F/m1-nvme-proof.py --path "$S" --bind-bytes 1073741824 \
  --private-out "$PRIV/m1-proof.json" --public-out "$R/m1-proof/PROOF.json"

SR="python3 $F/owed20/m1-step-runner.py"
# 4. The qualified program's own tokens (resident PP-2, no disk tier): the byte oracle.
col reference 3600 -- $SR reference --arms-lock $L --binary "$S/bin/run-gen" --shard-dir "$S" \
  --mtp-draft "$MTP" --out "$R/reference/visit" --lock-fd @COLLECTOR_LOCK_FD@
ORACLE=$R/reference/visit/oracle-tokens.json

RUN="$SR run --shard-dir $S --mtp-draft $MTP -- --arms-lock $L --binary $S/bin/run-gen --artifact $SHARD1 \
  --proof $PRIV/m1-proof.json --rig pro-pair --lock-fd @COLLECTOR_LOCK_FD@ --oracle-tokens $ORACLE"

# 5. The disk tier must reproduce the qualified program: one smoke round, then run-spec on worker16.
col s20-smoke 7200 -- $RUN --regime cold --rounds 1 --smoke --out "$R/s20-smoke/visits"
step s20-smoke-gate python3 $F/m1-b3-pool.py "$R/s20-smoke" --fallback-unclean --require-correct
if grep -h '\[spill-pread\] reads=' "$R"/s20-smoke/visits/r*/run.log | grep -Eq 'fallbacks=[1-9]|demand_wait_timeouts=[1-9]'; then
  echo "STOPPED: fallbacks or demand-wait timeouts in the smoke"; exit 5
fi
col s20-spec-worker16 7200 -- env MEMRA_MTP_DRAFT="$MTP" python3 $F/m1-spec-cell.py --arms-lock $L --arm worker16 \
  --binary "$S/bin/run-spec" --artifact "$SHARD1"
grep -q "=== SELF-CONSISTENCY PASS ===" "$R/s20-spec-worker16/command.log" || { echo "STOPPED: spec worker16"; exit 4; }

# 6. The storage-bound regime: ten rounds, six arms.
col s20-storage-bound 28800 -- $RUN --regime cold --out "$R/s20-storage-bound/visits"
python3 $F/m1-b3-pool.py "$R/s20-storage-bound" --fallback-unclean > "$R/s20-storage-bound.pool.log" 2>&1

(cd "$R" && find . -type f ! -name MANIFEST.sha256 -exec sha256sum {} + | sort -k2) > "$R/MANIFEST.sha256"
echo "SITTING DONE $(date -u +%FT%TZ)"
