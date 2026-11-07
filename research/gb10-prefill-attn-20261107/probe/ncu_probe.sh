#!/usr/bin/env bash
# Runs ON a GB10 host: ncu --set full on one attention probe kernel launch (the first timed rep).
# usage: ncu_probe.sh NAME KERNEL T TKV
set -u
D=$HOME/.cache/memra-attn-gb10
NAME=$1; KN=$2; T=$3; TKV=$4
mkdir -p "$D/ncu"
fn=$(awk -v k="$KN" '$1 == k {print $2}' "$D/probe/ncu_names.txt"); [ -n "$fn" ] || { echo "unknown $KN"; exit 2; }
sudo -n -E env PATH="$PATH" /usr/local/cuda/bin/ncu -k "regex:^${fn}\$" -s 1 -c 1 --set full \
  --import-source no -f -o "$D/ncu/$NAME" "$D/probe/probe_attn" "$T" "$TKV" 1 "$KN" > "$D/ncu/$NAME.log" 2>&1
echo "ncu rc=$?"
sudo -n chown "$USER" "$D/ncu/$NAME.ncu-rep" 2>/dev/null
/usr/local/cuda/bin/ncu -i "$D/ncu/$NAME.ncu-rep" --page details --print-units base > "$D/ncu/$NAME.txt" 2>&1
/usr/local/cuda/bin/ncu -i "$D/ncu/$NAME.ncu-rep" --page raw --csv > "$D/ncu/$NAME.raw.csv" 2>&1
/usr/local/cuda/bin/ncu -i "$D/ncu/$NAME.ncu-rep" --page source --csv --print-source sass > "$D/ncu/$NAME-sass.csv" 2>&1
python3 "$D/probe/sass_top.py" "$D/ncu/$NAME-sass.csv" > "$D/ncu/$NAME-sass-top.txt" 2>&1
python3 "$D/probe/raw_pick.py" "$D/ncu/$NAME.raw.csv" > "$D/ncu/$NAME-raw-pick.txt" 2>&1
grep -E "Duration|Registers Per|Achieved Active Warps|Theoretical Active|Issue Slots Busy|No Eligible|Eligible Warps|L2 Cache Throughput|DRAM Throughput|L1/TEX Cache Throughput|L2 Hit Rate|Warp Cycles Per Issued|Dynamic Shared|Compute \(SM\) Throughput|Executed Ipc A" "$D/ncu/$NAME.txt" | head -n 30
grep -iE "tensor|is the highest|bank conflict|Est. Speedup|stall" "$D/ncu/$NAME.txt" | head -n 24
cat "$D/ncu/$NAME-raw-pick.txt"; head -n 40 "$D/ncu/$NAME-sass-top.txt"
