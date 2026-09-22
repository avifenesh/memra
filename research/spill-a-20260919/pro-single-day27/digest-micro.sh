#!/usr/bin/env bash
# WP-A day 27: the digest micro-cell (DAY27.md task 2 option (b); pre-registered there before this ran). One
# collector lock hold per card's host (tools/tier-battery.py --external-lock passes the lock fd as $1); the
# cell is CPU-only and reads no GPU, but it runs inside the rig hold so no GPU lane shares the host's CPU and
# thermal regime while it measures, the shape C's day-18 hash-micro used. Every cell is executed-not-qualified
# development evidence. No host, id or price here.
# usage: digest-micro.sh <lockfd> <tree> <receipts_root> <bin> <lockpath>
set -uo pipefail
fd=$1; TREE=$2; R=$3; BIN=$4; LOCK=$5
cd "$TREE" || exit 1
EV=$R/digest-micro/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock "$LOCK" --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/tree.sha"
sha256sum "$BIN" | tee "$EV/binary.sha256"
{
    echo "tree=$(git rev-parse HEAD)"
    echo "source_sha256=$(sha256sum research/spill-a-20260919/day27-digest-micro/src/main.rs | cut -d' ' -f1)"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "shape=--bytes 167772160 --n 5 (two orders, interleaved call by call, pooled N=10 per program)"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.before.csv" 2>&1
cat /proc/loadavg > "$EV/loadavg.before"
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
stamp() { python3 -c '
import sys, datetime
for line in sys.stdin.buffer:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    sys.stdout.buffer.write(ts.encode() + b"\t" + line); sys.stdout.buffer.flush()
' > "$1"; }
: > "$EV/marks.tsv"
mark "digest-micro start"
"$BIN" --bytes 167772160 --n 5 2>&1 | stamp "$EV/digest-micro.log"
rc=${PIPESTATUS[0]}
echo "$rc" > "$EV/digest-micro.exit"
mark "digest-micro end rc=$rc"
cat /proc/loadavg > "$EV/loadavg.after"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
grep 'DIGEST-MICRO rule' "$EV/digest-micro.log" || { echo "no rule line"; echo 1 > "$EV/exit.txt"; exit 1; }
echo "$rc" > "$EV/exit.txt"
exit "$rc"
