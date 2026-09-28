#!/usr/bin/env bash
# DAY91 (day88-cpu/dry-check-cell.sh for day89-cell.sh): a control-flow dry run of day89-cell.sh's three cells in a sandbox: a sandbox tree with a stub lock proof,
# stub run-gen-i22, run-gen-i24 and run-spec-i24 (each logs its argv and environment words, prints nothing, exits
# 0), a stub nvidia-smi, a tiny stand-in artifact. No GPU and no lock: the fd is a pipe. Prints, per cell, the calls
# per binary with their flags and the environment each arm set, the run order and the exits.
set -euo pipefail
L=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d /tmp/c89cell.XXXXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/tree/tools" "$D/bins" "$D/bin" "$D/R"
printf 'print("{\\"stub\\": true}")\n' > "$D/tree/tools/tier-lock-proof.py"
git -C "$D/tree" init -q; git -C "$D/tree" -c user.email=x -c user.name=x commit -q --allow-empty -m x
echo a > "$D/art.gguf"
for b in run-gen-i22 run-gen-i24 run-spec-i24; do
    printf '#!/usr/bin/env bash\necho "%s env=[$(env | grep -E "^MEMRA_(EXPERTS_VIA_TIER|MOE_PREFETCH|MOE_RESIDENT|MOE_SLOTS)=" | sort | tr "\\n" " ")] $*" >> %s/calls.log\nexit 0\n' "$b" "$D" > "$D/bins/$b"
    chmod +x "$D/bins/$b"
done
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/bin/nvidia-smi"; chmod +x "$D/bin/nvidia-smi"
for cell in promo promo-res promo-spec; do
    : > "$D/calls.log"
    (cd "$D" && PATH=$D/bin:$PATH D40_R=$D/R D40_BINS=$D/bins D40_TREE=$D/tree D40_LOCK=$D/none D40_ART=$D/art.gguf \
        bash "$L/day89-cell.sh" "$cell" 0 > "cell-$cell.log" 2>&1; echo "== $cell rc=$?")
    echo "== $cell calls: $(wc -l < "$D/calls.log")"
    sed -E "s#$D#D#g; s# D/art.gguf 55 88 13##" "$D/calls.log" | sort | uniq -c
    echo "== $cell order: $(awk -F'\t' '/start/ {print $2}' "$D/R/$cell/ev/marks.tsv" | sed 's/ start//' | paste -sd' ' | cut -c1-400)"
    echo "== $cell exits: $(cat "$D/R/$cell/ev"/*.exit | sort | uniq -c | tr '\n' ' ')"
done
