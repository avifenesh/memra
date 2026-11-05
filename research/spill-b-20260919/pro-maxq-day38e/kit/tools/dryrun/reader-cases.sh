#!/usr/bin/env bash
# Reader dry runs on the Workstation box-d receipts (no GPU): the part split, the combined reading and the kit checks.
set -uo pipefail
D=/home/avifenesh/.cache/memra-maxq/park-compact/dryrun
KIT=/home/avifenesh/.cache/memra-maxq/park-compact/kit
V=$KIT/maxq-verdict.py
SRC=$D/box-d
last() { tail -n 1 "$1"; }

# (b) part 1 alone, part 2 alone, both combined
for c in part1 part2 combined; do rm -rf "$D/case-$c"; mkdir -p "$D/case-$c/boots"; done
for b in main-O1-off main-O1-on fault-batch fault-nobatch fault-nobatch-red; do
  cp -a "$SRC/boots/$b" "$SRC/boots/$b.arm.txt" "$D/case-part1/boots/"; done
for b in main-O2-on main-O2-off; do cp -a "$SRC/boots/$b" "$SRC/boots/$b.arm.txt" "$D/case-part2/boots/"; done
cp -a "$D/case-part1/boots/." "$D/case-combined/boots/"; cp -a "$D/case-part2/boots/." "$D/case-combined/boots/"
for c in part1 part2 combined; do
  python3 -I "$V" "$D/case-$c" --card pro6000 --regime workstation > "$D/case-$c.txt" 2>&1
  echo "case $c: $(last "$D/case-$c.txt")"
done
cmp <(grep -v '^VERDICT' "$D/case-combined.txt") "$SRC/SUMMARY.txt" && echo "case combined: reader lines byte-identical to box-d SUMMARY.txt (vmm pair absent, so its line drops)" \
  || echo "case combined: reader lines differ from box-d SUMMARY.txt only by the absent vmm pair: $(diff <(grep -v '^VERDICT' "$D/case-combined.txt") "$SRC/SUMMARY.txt" | grep -c '^[<>]') line(s)"

# (a) kit checks on the unmodified Workstation receipts: they must refuse (other card, env arms, no pin record)
mkdir -p "$D/fakekit/bins" "$D/fakekit/reference/box-d"
printf '%s  off/memra-server\n%s  on/memra-server\n%s  red/memra-server\n' \
  c65f30761a15858954c3bb925b02b853eb7af68d8ebc30578fdbfcd34665313c \
  1111111111111111111111111111111111111111111111111111111111111111 \
  0673a71d04269abdebe9a9bd0a5f383392d63b26f336da65543e7a76c2bafd1f > "$D/fakekit/bins/SHA256SUMS"
cp "$KIT/reference/box-d/stream.json" "$D/fakekit/reference/box-d/stream.json"
python3 -I "$V" "$D/case-combined" --card pro6000 --kit "$D/fakekit" --card-index 0 --regime workstation > "$D/case-kitcheck-neg.txt" 2>&1
echo "case kit checks on Workstation receipts: $(last "$D/case-kitcheck-neg.txt" | cut -c1-600)"

# (c) kit checks satisfied: the combined receipts relabelled as a Max-Q run of the kit's three arm binaries
rm -rf "$D/case-kitcheck-pos"; cp -a "$D/case-combined" "$D/case-kitcheck-pos"
printf '%s  off/memra-server\n%s  on/memra-server\n%s  red/memra-server\n' \
  aaaa000000000000000000000000000000000000000000000000000000000000 \
  bbbb000000000000000000000000000000000000000000000000000000000000 \
  cccc000000000000000000000000000000000000000000000000000000000000 > "$D/fakekit/bins/SHA256SUMS"
python3 - "$D/case-kitcheck-pos" <<'EOF'
import os, sys
root = sys.argv[1]
arm = {"main-O1-off": "aaaa", "main-O2-off": "aaaa", "main-O1-on": "bbbb", "main-O2-on": "bbbb",
       "fault-batch": "bbbb", "fault-nobatch": "bbbb", "fault-nobatch-red": "cccc"}
for n, h in arm.items():
    p = os.path.join(root, "boots", n + ".arm.txt")
    lines = [l if not l.startswith("bin_sha256=") else "bin_sha256=" + h + "0" * 60 for l in open(p).read().splitlines()]
    open(p, "w").write("\n".join(lines) + "\n")
    b = os.path.join(root, "boots", n)
    open(os.path.join(b, "server.pid"), "w").write("4242\n")
    open(os.path.join(b, "compute-apps-ready.csv"), "w").write("pid, process_name, used_memory [MiB]\n4242, memra-server, 93000 MiB\n")
    open(os.path.join(b, "gpu-before.csv"), "w").write("name, memory.used [MiB]\nNVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 3 MiB\n")
    open(os.path.join(b, "dirty.txt"), "w").write("")
EOF
python3 -I "$V" "$D/case-kitcheck-pos" --card pro6000-maxq --kit "$D/fakekit" --card-index 2 --regime gap > "$D/case-kitcheck-pos.txt" 2>&1
grep -E '^DAY38D (STREAM|REGIME)' "$D/case-kitcheck-pos.txt"
echo "case kit checks satisfied: $(last "$D/case-kitcheck-pos.txt")"
