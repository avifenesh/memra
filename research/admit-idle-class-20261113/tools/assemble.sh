#!/usr/bin/env bash
# memra#1304, runs on the rig: copies each cell's receipts from the unit into gb10/<cell>/ with
# home paths written as `~` and the unit's hostname as `gb10` (run.log, guard.log, holder.log,
# server.log, trace.txt, mem.jsonl, ab.jsonl, kernel-oom-count.txt and, from campaign3 on,
# kernel.log, the kernel journal since the cell's start; not the GPU telemetry), and the campaign
# log. The first two campaigns' kernel journal is gb10/kernel.log, read back from the unit for
# their window on 2026-11-14.
# usage: assemble.sh UNIT CAMPAIGN DEST_DIR
set -u
U=$1; C=$2; DEST=$3
S=$HOME/.cache/memra-1304/stage/$C
mkdir -p "$S" "$DEST"
rsync -a --include='*/' --include='run.log' --include='guard.log' --include='holder.log' \
  --include='server.log' --include='trace.txt' --include='mem.jsonl' --include='ab.jsonl' \
  --include='kernel-oom-count.txt' --include='kernel.log' --include='campaign.log' --exclude='*' \
  "$U:.cache/memra-1304/runs/$C/" "$S/"
HOST=$(ssh "$U" hostname)
python3 - "$S" "$DEST" "$HOST" <<'EOF'
import sys
from pathlib import Path
src, dst, host = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
home = str(Path.home())
for f in sorted(src.rglob("*")):
    if f.is_file():
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f.read_text(errors="replace").replace(home, "~").replace(host, "gb10"))
EOF
