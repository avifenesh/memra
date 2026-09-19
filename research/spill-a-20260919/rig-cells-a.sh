#!/usr/bin/env bash
# Correctness/qualification launcher, not a performance receipt by itself.
# Every command is raw-logged before parsing. Unsupported A2/M1 commands FAIL.
set -euo pipefail
usage() {
    echo "usage: $0 <existing-local-NVMe-directory> [--approved-non-serving] [--dry-run [--stub executable]]" >&2
    exit 2
}
[[ $# -ge 1 ]] || usage
nvme=$1; shift
dry=0; approved=0; stub=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --approved-non-serving) approved=1; shift ;;
        --stub) [[ $# -ge 2 ]] || usage; stub=$2; shift 2 ;;
        *) usage ;;
    esac
done
[[ -z "$stub" || $dry -eq 1 ]] || usage
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
failures=0
scratch=
run() {
    local name=$1; shift
    if [[ $dry -eq 1 ]]; then
        printf 'DRY-RUN %s:' "$name"; printf ' %q' "$@"; printf '\n'
        [[ -z "$stub" ]] || "$stub" "$name" "$@"
        return
    fi
    local rc
    local pipeline
    set +e
    "$@" 2>&1 | tee "$out/$name.log"
    pipeline=("${PIPESTATUS[@]}")
    rc=${pipeline[0]}
    if [[ $rc -eq 0 && ${pipeline[1]} -ne 0 ]]; then rc=${pipeline[1]}; fi
    set -e
    # Parse ONLY the saved raw file. Each row records the exact argv and exit.
    python3 - "$out" "$name" "$rc" "$@" <<'PY' || { failures=$((failures + 1)); return 1; }
import json, pathlib, sys
out, name, rc, *argv = sys.argv[1:]
p = pathlib.Path(out)
row = dict(cell=name, argv=argv, exit=int(rc), status='pass' if rc == '0' else 'failed')
with (p/'commands.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
if name.startswith(('roundtrip-', 'restore-')) and rc == '0':
    sample = json.loads((p/(name+'.log')).read_text())
    assert sample['physical_bytes'] is None  # not an instrumented SSD-speed result
    with (p/'samples.jsonl').open('a') as f: f.write(json.dumps(sample)+'\n')
PY
    if [[ $rc -ne 0 ]]; then failures=$((failures + 1)); fi
    return "$rc"
}
if [[ $dry -eq 0 ]]; then
    [[ $(uname -s) == Linux && $approved -eq 1 && -d "$nvme" ]] || usage
    for command in flock findmnt lsblk nvidia-smi nvcc cargo python3 timeout sha256sum; do
        command -v "$command" >/dev/null || { echo "BLOCKED: missing $command" >&2; exit 2; }
    done
    nvme=$(cd "$nvme" && pwd -P)
    device=$(findmnt -n -o SOURCE -T "$nvme")
    # Refuse network volumes/tmpfs; dm-backed local NVMe is allowed through ancestry.
    lsblk -s -r -n -o KNAME "$device" | grep -q '^nvme' || { echo 'BLOCKED: local NVMe ancestry not proven' >&2; exit 2; }
    git diff --quiet && git diff --cached --quiet || { echo 'BLOCKED: tracked source is dirty' >&2; exit 2; }
    exec 9>/tmp/memra-5090.lock
    flock -x 9
    host=$(hostname -s | tr -cd 'A-Za-z0-9_.-')
    utc=$(date -u +%Y%m%dT%H%M%S.%NZ)
    out="research/spill-a-20260919/raw/$host-$utc"
    mkdir "$out"
    scratch=$(mktemp -d "$nvme/spill-a.XXXXXX")
    cleanup() { [[ -z "$scratch" ]] || rm -rf -- "$scratch"; }
    trap cleanup EXIT
    run git-revision git rev-parse HEAD
    run git-status git status --short
    run mount findmnt -T "$nvme"
    run storage lsblk -s -r -o KNAME,TYPE,ROTA,TRAN "$device"
    run gpu-topology nvidia-smi topo -m
    run gpu-before nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    run compute-before nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
else
    out='research/spill-a-20260919/raw/<host>-<utc>'
    scratch="$nvme/spill-a.DRY-RUN"
    printf 'DRY-RUN lock: flock -x /tmp/memra-5090.lock (entire window)\n'
fi
# CPU conformance rerun on the actual Linux filesystem; these do not qualify GPU I/O.
run a1-sharded-catalog cargo test -p memra-tier --offline --test storage day4::sharded_catalog -- --nocapture
run a1-gc cargo test -p memra-tier --offline --test storage gc_ -- --nocapture
run a1-telemetry-join cargo test -p memra-tier --offline --test storage telemetry:: -- --nocapture
run build-storage cargo build -p memra-engine --release --bin storage-bench
run build-worker cargo test -p memra-engine --lib --no-run
run binary-hash sha256sum target/release/storage-bench
# N=1 correctness cells. No thermal/cache control or measured physical bytes:
# these are filesystem characterization, NEVER scored spill-speed results.
for backend in buffered uncached direct; do
    for size in 264 4097 1048576 4194568; do
        path="$scratch/$backend-$size"
        run "roundtrip-$backend-$size" target/release/storage-bench roundtrip "$path" "$size" "$backend" || true
        run "restore-$backend-$size" target/release/storage-bench restore "$path" "$size" "$backend" || true
    done
done
run a2-existing-worker timeout 300 cargo test -p memra-engine --lib \
    spill_pread::tests::worker_positioned_reads_preserve_exact_bytes_and_reuse_after_short_read \
    -- --ignored --exact --nocapture || true
# These intended interfaces in CELLS.md are NOT implemented at day 3. Execute
# fail-closed probes and preserve their actual stderr; never turn unsupported into
# a skip/pass. The 5s cap is an admission probe, NOT the eventual scored window.
run a2-byte-roundtrip-unimplemented timeout 5 target/release/storage-bench roundtrip \
    --directions h2d,d2h --sizes 1,264,288,4095,4096,4097,1048576,933232640 \
    --slot-bytes 1048576 --slots 4 --reserved-demand-slots 1 --repeats 100 \
    --faults cancel,late-fence,lost-fence --telemetry-ms 250 --out "$out" || true
run m1-row-unimplemented timeout 5 target/release/storage-bench trace \
    --trace opaque-row264x48-v1 --logical-table-bytes 202758032400 \
    --requests-per-second 800 --seconds 1800 --backends worker,pread,mmap,direct \
    --slot-bytes 1048576 --slots 8 --reserved-demand-slots 2 --scratch "$scratch/row" \
    --order ab5,ba5 --telemetry-ms 250 --out "$out" || true
run m1-bulk-unimplemented timeout 5 target/release/storage-bench trace \
    --trace opaque-bulk-v1 --sizes 116654080,933232640 --restores-per-second 1 \
    --seconds 1800 --backends worker,direct --slot-bytes 1048576 --slots 8 \
    --reserved-demand-slots 2 --scratch "$scratch/bulk" --order ab5,ba5 --telemetry-ms 250 --out "$out" || true
run m1-mixed-unimplemented timeout 5 target/release/storage-bench trace \
    --trace opaque-mixed-v1 --row-batches-per-second 800 --rows-per-batch 48 \
    --row-bytes 264 --restore-bytes 116654080 --restores-per-second 1 \
    --backup-bytes-per-second 712000 --seconds 1800 --backends worker,direct \
    --slot-bytes 1048576 --slots 8 --reserved-demand-slots 2 --scratch "$scratch/mixed" \
    --order ab5,ba5 --telemetry-ms 250 --out "$out" || true
if [[ $dry -eq 0 ]]; then
    run compute-after nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
    echo "Receipts: $out; failed commands: $failures"
    # Even if a future binary accepts these spellings, this launcher lacks the
    # full 250ms/physical-I/O/model-consumer collector. No false green promotion.
    echo 'BLOCKED: full A2/M1 qualification and telemetry collector not implemented' >&2
    exit 1
fi
echo 'DRY-RUN ONLY: no hardware, filesystem mutation, lock, build or qualification ran'
