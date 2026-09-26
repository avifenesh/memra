#!/usr/bin/env bash
# PRO 6000 sitting for OWED 17, 18 and 26 (SITTING.md; M1-PREREG.md sections E, F, G).
# Run from the repository root at the lane tip, as root, on a box of BOX27's class:
#   setsid nohup timeout 6h bash research/spill-f-20260919/pro-sitting/run-sitting.sh \
#     > /root/spill-receipts/f-pro/sitting.log 2>&1 < /dev/null & disown
# Every GPU step is its own collector cell holding /tmp/memra-gpu.lock (`--rig pro-single`); a
# failing step stops the sitting with its receipts kept. Nothing else may run on the box.
set -uo pipefail
S=${S:-/scratch/spill-f}
R=${R:-/root/spill-receipts/f-pro}
PRIV=${PRIV:-/root/f-private}
F=research/spill-f-20260919
ART=$S/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
URL=https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
SHA=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
F17=$F/m1-prereg/f17-arms.lock.json
ORACLE=$F/m1-prereg/f17-oracle-tokens.json
mkdir -p "$S/bin" "$S/b2" "$R" "$PRIV"
chmod 700 "$PRIV"

step() {
  local name=$1; shift
  echo "== $name start $(date -u +%FT%TZ)"
  "$@" > "$R/$name.driver.log" 2>&1
  local rc=$?
  echo "== $name end $(date -u +%FT%TZ) rc=$rc"
  if [ "$rc" -ne 0 ]; then echo "STOPPED at $name (rc=$rc); receipts kept in $R"; exit "$rc"; fi
}
col() {  # a collector cell on the proven scratch: col NAME TIMEOUT -- command...
  # The collector takes /tmp/memra-gpu.lock itself; --external-lock (as on BOX27) hands its FD to a
  # child that names @COLLECTOR_LOCK_FD@ (runner, handoff driver) and needs exactly one such token.
  local name=$1 timeout=$2; shift 3
  local ext=()
  case " $* " in *@COLLECTOR_LOCK_FD@*) ext=(--external-lock) ;; esac
  step "$name" python3 tools/tier-battery.py --rig pro-single --timeout "$timeout" "${ext[@]}" \
    --storage-root "$S" --storage-proof "$R/m1-proof/PROOF.json" --out "$R/$name" --execute "$@"
}

# 1. Build and freeze the binaries (the fix build: OWED 17 door, OWED 18 door, OWED 26 demand wait).
git rev-parse HEAD > "$R/commit.txt"
step build bash -c "MEMRA_CUDA_ARCH=120a cargo build --release -p memra-engine --bin run-gen --bin run-spec \
  -p memra-server --bin kv-handoff-gate --bin memra-server && \
  MEMRA_CUDA_ARCH=120a cargo test --release -p memra-engine --lib --no-run"
cp target/release/run-gen target/release/run-spec target/release/kv-handoff-gate target/release/memra-server "$S/bin/"
TESTBIN=$(grep -o 'Executable unittests src/lib.rs ([^)]*)' "$R/build.driver.log" | tail -1 | sed 's/.*(\(.*\))/\1/')
cp "$TESTBIN" "$S/bin/engine-lib-tests"
(cd "$S/bin" && sha256sum run-gen run-spec kv-handoff-gate memra-server engine-lib-tests) > "$R/binaries.sha256"

# 2. Stage the pinned artifact (byte-verified) and the registered B2 prompts.
step stage bash -c "cd '$S' && curl -fL --retry 5 -o Qwen3.6-35B-A3B-UD-IQ4_XS.gguf.part '$URL' && \
  echo '$SHA  Qwen3.6-35B-A3B-UD-IQ4_XS.gguf.part' | sha256sum -c && \
  mv Qwen3.6-35B-A3B-UD-IQ4_XS.gguf.part Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
step b2-prompts python3 $F/m1-b2-prompts.py --out "$S/b2-prompts.jsonl" --check $F/m1-prereg/b2-prompts.manifest.json

# 3. M1 proof of the scratch (a FAIL stops here: no cell without a PASS proof).
step m1-proof python3 tools/tier-battery.py --rig pro-single --timeout 600 --out "$R/m1-proof" --execute \
  python3 $F/m1-nvme-proof.py --path "$S" --bind-bytes 1073741824 \
  --private-out "$PRIV/m1-proof.json" --public-out "$R/m1-proof/PROOF.json"

# 4. OWED 26 on the target card: every pool GPU cell, serially.
col owed26-cells 1800 -- "$S/bin/engine-lib-tests" --ignored spill_pread::tests --test-threads 1 --nocapture

RUN="python3 $F/m1-spill-runner.py run --arms-lock $F17 --binary $S/bin/run-gen --artifact $ART \
  --proof $PRIV/m1-proof.json --rig pro-single --lock-fd @COLLECTOR_LOCK_FD@ --oracle-tokens $ORACLE"

# 5. OWED 17 correctness: smoke and its gate, then run-spec K=1..8 with each bypass arm.
col f17-smoke 3600 -- $RUN --regime cold --rounds 1 --smoke --out "$R/f17-smoke/visits"
step f17-smoke-gate python3 $F/m1-b3-pool.py "$R/f17-smoke" --bypass-check --fallback-unclean --require-correct
for arm in bypass-staged bypass-mapped; do
  col "f17-spec-$arm" 3600 -- python3 $F/m1-spec-cell.py --arms-lock $F17 --arm "$arm" \
    --binary "$S/bin/run-spec" --artifact "$ART"
  grep -q "=== SELF-CONSISTENCY PASS ===" "$R/f17-spec-$arm/command.log" || { echo "STOPPED: spec $arm"; exit 4; }
done

# 6. OWED 17 timing: cold and bounded, ten rounds each.
col f17-cold 10800 -- $RUN --regime cold --out "$R/f17-cold/visits"
col f17-bounded 14400 -- $RUN --regime bounded --balloon-touch --floor-bytes 2147483648 --out "$R/f17-bounded/visits"
python3 $F/m1-b3-pool.py "$R/f17-cold" --bypass-check --fallback-unclean > "$R/f17-cold.pool.log" 2>&1
python3 $F/m1-b3-pool.py "$R/f17-bounded" --bypass-check --fallback-unclean > "$R/f17-bounded.pool.log" 2>&1

# 7. OWED 18: ten alternating pairs per size (BOX27's B2 conditions; 8 GiB at tenant 100%).
SCHED=$(python3 -c "print(','.join('buffered,direct' if k % 2 == 0 else 'direct,buffered' for k in range(10)))")
HO="python3 $F/m1-handoff-driver.py run --gate $S/bin/kv-handoff-gate --server $S/bin/memra-server \
  --artifact $ART --prompts $S/b2-prompts.jsonl --proof $PRIV/m1-proof.json --scratch $S/b2 \
  --host-mb 16384 --rig pro-single --lock-fd @COLLECTOR_LOCK_FD@ --io-schedule $SCHED"
col handoff-1g 7200 -- $HO --size-bytes 1073741824 --out "$R/handoff-1g/visits"
col handoff-8g 10800 -- $HO --size-bytes 8589934592 --tenant-pct 100 --out "$R/handoff-8g/visits"
python3 $F/m1-handoff-pairs.py "$R/handoff-1g" > "$R/handoff-1g.pairs.log" 2>&1
python3 $F/m1-handoff-pairs.py "$R/handoff-8g" > "$R/handoff-8g.pairs.log" 2>&1

(cd "$R" && find . -type f ! -name MANIFEST.sha256 -exec sha256sum {} + | sort -k2) > "$R/MANIFEST.sha256"
echo "SITTING DONE $(date -u +%FT%TZ)"
