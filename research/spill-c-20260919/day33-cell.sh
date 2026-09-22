#!/usr/bin/env bash
# Day 33: the hash micro-cell on the local RTX 5090 Laptop GPU at the pair cell's entry size (54,800,000 B,
# the server's `54.8MB` at 1e6 per MB) and at 160 MiB (day 18's size), plus the two-step arm (a memcpy of the
# write-combined buffer into a cached pinned buffer, then the hash of the copy) against the single-pass hash
# over write-combined memory. Four hash-micro invocations in ONE collector lock hold on /tmp/memra-5090.lock:
# micro-54m, twostep-54m, micro-160m, twostep-160m (this order, fixed before the run). N=5 per kind per order,
# two orders per invocation, N=10 pooled per kind. Before them, `tier-transfer-gate roundtrip` prints the
# pinned arm this device resolves to (`PINNED-DEFAULT device=... kind=... flags=...`), the premise receipt.
# The collector's 250 ms sampler is the regime; this script adds a 1 s nvidia-smi sampler and snapshots.
# Every cell is executed-not-qualified development evidence. No host, id or price here.
# usage: day33-cell.sh <lockfd> <tree> <receipts_root> <hash_micro_bin> <gate_bin>
set -uo pipefail
fd=$1; TREE=$2; R=$3; HM=$4; GATE=$5
cd "$TREE" || exit 1
EV=$R/hashwc/hash-wc/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-5090.lock --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/tree.sha"
sha256sum "$HM" | tee "$EV/binary.sha256"
{
    echo "tree=$(git rev-parse HEAD)"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "bytes_small=54800000 bytes_large=167772160 n_per_order=5 orders=2 pooled=10"
    echo "invocations=micro-54m,twostep-54m,micro-160m,twostep-160m"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
: > "$EV/marks.tsv"
# stamp: prefix every line of stdin with a UTC ms timestamp (line arrival time); nothing is dropped.
stamp() { python3 -c '
import sys, datetime
for line in sys.stdin.buffer:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    sys.stdout.buffer.write(ts.encode() + b"\t" + line); sys.stdout.buffer.flush()
' > "$1"; }
run_hm() { # $1 label  $2.. argv
    local label=$1; shift
    mark "$label start"
    "$@" 2>&1 | stamp "$EV/$label.log"
    local rc=${PIPESTATUS[0]}
    echo "$rc" > "$EV/$label.exit"
    mark "$label end rc=$rc"
    return 0
}
snap() { # label
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.$1.csv" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw,pstate --format=csv > "$EV/card.$1.csv" 2>&1
}
snap before
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,memory.used,clocks.sm --format=csv -l 1 > "$EV/card.during.csv" 2> "$EV/card.during.err" &
SAMPLER=$!
uptime | tee "$EV/loadavg.before.txt"
# The pinned destination arm this device resolves to, printed by the transfer gate's setup (PinnedKind::for_device).
if [ -x "$GATE" ]; then
    CUDA_VISIBLE_DEVICES=0 "$GATE" roundtrip > "$EV/pinned-default.log" 2>&1
    echo "rc=$?" >> "$EV/pinned-default.log"
else
    echo "tier_transfer_gate binary absent at $GATE" > "$EV/pinned-default.log"
fi
grep -h 'PINNED-DEFAULT device=' "$EV/pinned-default.log" || true
export CUDA_VISIBLE_DEVICES=0
run_hm micro-54m    "$HM" --bytes 54800000 --n 5
run_hm twostep-54m  "$HM" --two-step --bytes 54800000 --n 5
run_hm micro-160m   "$HM" --bytes 167772160 --n 5
run_hm twostep-160m "$HM" --two-step --bytes 167772160 --n 5
uptime | tee "$EV/loadavg.after.txt"
kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
snap after
rules=$(grep -h 'HASH-MICRO' "$EV"/micro-54m.log "$EV"/twostep-54m.log "$EV"/micro-160m.log "$EV"/twostep-160m.log | grep -c 'rule')
echo "day33 cell done: rule lines=$rules"
[ "$rules" -eq 4 ] || exit 1
