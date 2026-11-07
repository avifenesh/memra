#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: ncu --set full on one launch of KERNEL (probe3 name -> entry
# symbol) at T 4,096 and depth DEPTH, then the details, raw and SASS-source pages and the stall
# parse. usage: ncu_cand.sh NAME SYMBOL [DEPTH]
set -u
NAME=$1; SYM=$2; DEPTH=${3:-131072}
D=$HOME/.cache/memra-attn3
P=$D/probe; O=$D/ncu; mkdir -p "$O"
export PATH="/usr/local/cuda/bin:$PATH"
cd "$P" || exit 1
sudo -n -E env PATH="$PATH" ncu -k "regex:^$SYM\$" -s 1 -c 1 --set full --import-source no -f \
  -o "$O/$NAME-$DEPTH" "$P/probe3" 4096 "$DEPTH" 1 "$NAME" > "$O/$NAME-$DEPTH.log" 2>&1
echo "ncu rc=$?" >> "$O/$NAME-$DEPTH.log"
sudo -n chown "$USER" "$O/$NAME-$DEPTH.ncu-rep" 2>/dev/null
ncu -i "$O/$NAME-$DEPTH.ncu-rep" --page details --print-units base > "$O/$NAME-$DEPTH.txt" 2>&1
ncu -i "$O/$NAME-$DEPTH.ncu-rep" --page raw --csv > "$O/$NAME-$DEPTH.raw.csv" 2>&1
ncu -i "$O/$NAME-$DEPTH.ncu-rep" --page source --csv --print-source sass > "$O/$NAME-$DEPTH-sass.csv" 2>&1
python3 "$P/sass_wait.py" "$O/$NAME-$DEPTH-sass.csv" > "$O/$NAME-$DEPTH-wait.txt" 2>&1
python3 "$P/ncu_sum.py" "$O/$NAME-$DEPTH.raw.csv" > "$O/$NAME-$DEPTH-sum.txt" 2>&1
echo "NCU-DONE $NAME" >> "$O/$NAME-$DEPTH.log"
tail -n 2 "$O/$NAME-$DEPTH.log"
