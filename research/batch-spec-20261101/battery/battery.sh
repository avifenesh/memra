#!/usr/bin/env bash
# Full battery for one binary pair. usage: battery.sh BINDIR LABEL
set -u
BINDIR=$1
LABEL=$2
OUT=$HOME/.cache/memra-batchspec-battery/$LABEL
H=$HOME/.cache/memra-batchspec-harness
mkdir -p "$OUT"
sha256sum "$BINDIR/run-spec" "$BINDIR/memra-server" > "$OUT/bin.sha256"
bash "$H/runspec_battery.sh" "$BINDIR/run-spec" "$OUT/runspec"
python3 "$H/serve_battery.py" "$BINDIR/memra-server" "$OUT/serve" 8343 > "$OUT/serve.stdout" 2>&1
echo "serve rc=$?" >> "$OUT/runspec/rc.txt"
echo "battery $LABEL done"
