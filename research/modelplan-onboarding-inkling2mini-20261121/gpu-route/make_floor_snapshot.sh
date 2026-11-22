#!/bin/bash
# A config-only variant of the checkpoint: every file a symlink into the real directory except
# config.json, whose text_config.log_scaling_n_floor is set to $2. Laid out as a hub snapshot
# (snapshots/s plus blobs) because the loader refuses an index path outside a snapshot.
set -euo pipefail
SRC=$(readlink -f "$1"); FLOOR=$2; OUT=$3
mkdir -p "$OUT/snapshots/s"
ln -sfn "$SRC" "$OUT/blobs"
for f in "$SRC"/*; do
  b=$(basename "$f")
  [ "$b" = config.json ] && continue
  ln -sfn "$f" "$OUT/snapshots/s/$b"
done
python3 - "$SRC/config.json" "$OUT/snapshots/s/config.json" "$FLOOR" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
tc = c.get("text_config", c)
assert "log_scaling_n_floor" in tc, "no log_scaling_n_floor"
tc["log_scaling_n_floor"] = int(sys.argv[3])
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
echo "$OUT/snapshots/s"
