#!/usr/bin/env bash
# Dry-run of the kit's readers on the Workstation receipt (reference/ws-receipt, a copy of
# research/spill-b-20260919/pro-single-day49d/ at 2ccb038d1c). No GPU, no server.
#   bash dryrun-reader.sh <dest dir>      (the dest dir must not exist)
# 1. The Workstation receipt as it is: the registered serve reader must reproduce the committed
#    read-serve.log byte for byte, and maxq-verdict.py must print VERDICT ... -> ON.
# 2. Three mutated copies, so the reader is shown to say the other words (3. adds two run-dir cases):
#    off-red   O1-on's first burst row turned into a 503: a registered serving clause fails -> OFF.
#    skip      j1's control boot prints the HEAD tree's OFF boot line: the gate would SKIP and the
#              on arm is not the on arm -> NOT-READ.
#    maxq-off  the boots carry the HEAD tree's boot lines, but O2-on says OFF (a wrong binary on an
#              on arm) -> NOT-READ.
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd)
D=${1:?dest dir}
[ -e "$D" ] && { echo "dest $D exists"; exit 2; }
mkdir -p "$D"
W=$K/reference/ws-receipt
V=$K/maxq-verdict.py
python3 -I "$V" --card pro6000 --gate "$W/box/j1" --gate "$W/box/j2" --vmm "$W/box/vmm-green" "$W/box/vmm-red" \
  --serve "$W/box-boots" --log "$D/ws-read-serve.log" > "$D/ws.out" 2>&1
if cmp -s "$D/ws-read-serve.log" "$W/box-boots/read-serve.log"; then echo "ws: read-serve.log identical to the committed receipt"
else echo "ws: read-serve.log DIFFERS from the committed receipt"; fi
echo "ws: $(tail -1 "$D/ws.out")"

mut() { # <name>: a private copy of the receipt
  mkdir -p "$D/$1"; cp -r "$W/box" "$W/box-boots" "$D/$1/"
}
mut off-red
python3 -I - "$D/off-red/box-boots/boots/O1-on/client.jsonl" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p)]
for r in rows:
    if r["phase"] == "burst":
        r.update(status=503, content_sha256=None, error='{"error":{"message":"mutated for the dry run"}}')
        break
open(p, "w").write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
PY
python3 -I "$V" --card pro6000 --gate "$D/off-red/box/j1" --gate "$D/off-red/box/j2" --serve "$D/off-red/box-boots" \
  --log "$D/off-red.read-serve.log" > "$D/off-red.out" 2>&1
echo "off-red: $(tail -1 "$D/off-red.out")"

mut skip
sed -i '1i [admit-oom] batch OOM recovery: OFF (no receipt for this card: a batched chunk'"'"'s OOM ends the chunk'"'"'s sessions)' \
  "$D/skip/box/j1/j-ctrl/server.log"
python3 -I "$V" --card pro6000 --gate "$D/skip/box/j1" --gate "$D/skip/box/j2" --serve "$D/skip/box-boots" \
  --log "$D/skip.read-serve.log" > "$D/skip.out" 2>&1
echo "skip: $(tail -1 "$D/skip.out")"

mut maxq-off
for b in O1-off O2-off P1-off P2-off O2-on; do
  sed -i '1i 0 [admit-oom] batch OOM recovery: OFF (no receipt for this card: a batched chunk'"'"'s OOM ends the chunk'"'"'s sessions)' \
    "$D/maxq-off/box-boots/boots/$b/server.log"
done
for b in O1-on P1-on P2-on; do
  sed -i '1i 0 [admit-oom] batch OOM recovery: ON (RTX PRO 6000 Blackwell Workstation or Max-Q Workstation Edition: an untouched batched chunk takes one reclaim rung and retries once)' \
    "$D/maxq-off/box-boots/boots/$b/server.log"
done
python3 -I "$V" --card pro6000 --gate "$D/maxq-off/box/j1" --gate "$D/maxq-off/box/j2" --serve "$D/maxq-off/box-boots" \
  --log "$D/maxq-off.read-serve.log" > "$D/maxq-off.out" 2>&1
echo "maxq-off: $(tail -1 "$D/maxq-off.out")"

# 3. The kit's validity clauses: a run dir with a thermal stop, and a card that is not the Max-Q.
mkdir -p "$D/thermal"
printf 'index, name\n2, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition\n' > "$D/thermal/card.csv"
echo "card 2 temp=95.0 C fan=100.0 % streak=1: the Max-Q thermal rule (dry run)" > "$D/thermal/THERMAL-STOP.txt"
python3 -I "$V" --card pro6000 --gate "$W/box/j1" --gate "$W/box/j2" --serve "$W/box-boots" --run-dir "$D/thermal" \
  --expect-name "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" --log "$D/thermal.read-serve.log" > "$D/thermal.out" 2>&1
echo "thermal: $(tail -1 "$D/thermal.out")"
mkdir -p "$D/wrong-card"
cp "$W/box-boots/card.csv" "$D/wrong-card/card.csv"
python3 -I "$V" --card pro6000 --gate "$W/box/j1" --gate "$W/box/j2" --serve "$W/box-boots" --run-dir "$D/wrong-card" \
  --expect-name "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" --log "$D/wrong-card.read-serve.log" > "$D/wrong-card.out" 2>&1
echo "wrong-card: $(tail -1 "$D/wrong-card.out")"
