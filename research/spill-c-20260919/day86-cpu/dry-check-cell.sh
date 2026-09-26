#!/usr/bin/env bash
# DAY86: a control-flow dry run of day86-cell.sh `slow86` in a sandbox, twice: with run-gen-i22 (five arms) and without
# it (three arms): a sandbox tree with a stub lock proof, stub run-gen binaries (print nothing, exit 0), a stub
# nvidia-smi, a tiny stand-in artifact. No GPU and no lock: the fd is a pipe. Prints the calls per binary and flag,
# the run order, the exits, and whether every snapshot carries the vmstat and buddyinfo sections.
set -euo pipefail
L=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d /tmp/c86cell.XXXXXX)
trap 'rm -rf "$D"' EXIT
for variant in with-i22 without-i22; do
    rm -rf "${D:?}/tree" "${D:?}/bins" "${D:?}/bin" "${D:?}/R" "${D:?}/calls.log"
    mkdir -p "$D/tree/tools" "$D/bins" "$D/bin" "$D/R"
    printf 'print("{\\"stub\\": true}")\n' > "$D/tree/tools/tier-lock-proof.py"
    git -C "$D/tree" init -q; git -C "$D/tree" -c user.email=x -c user.name=x commit -q --allow-empty -m x
    echo a > "$D/art.gguf"
    bins=(run-gen-i20 run-gen-i21); [ $variant = with-i22 ] && bins+=(run-gen-i22)
    for b in "${bins[@]}"; do printf '#!/usr/bin/env bash\necho "%s $*" >> %s/calls.log\nexit 0\n' "$b" "$D" > "$D/bins/$b"; chmod +x "$D/bins/$b"; done
    printf '#!/usr/bin/env bash\nexit 0\n' > "$D/bin/nvidia-smi"; chmod +x "$D/bin/nvidia-smi"
    (cd "$D" && PATH=$D/bin:$PATH D40_R=$D/R D40_BINS=$D/bins D40_TREE=$D/tree D40_LOCK=$D/none D40_ART=$D/art.gguf \
        bash "$L/day86-cell.sh" slow86 0 > cell.log 2>&1; echo "== $variant cell rc=$?")
    EV=$D/R/slow86/ev
    echo "== $variant arms: $(cat "$EV/arms.txt")"
    echo "== $variant calls: $(wc -l < "$D/calls.log") lines"
    sed -E "s#$D#D#g; s# D/art.gguf 55 88 13##" "$D/calls.log" | sort | uniq -c
    echo "== $variant order: $(awk -F'\t' '/start/ {print $2}' "$EV/marks.tsv" | sed 's/ start//' | paste -sd' ')"
    echo "== $variant exits: $(cat "$EV"/*.exit | sort | uniq -c | tr '\n' ' ')"
    n=$(find "$EV" -name '*.snap' | wc -l); v=$(grep -l '^## vmstat' "$EV"/*.snap | wc -l); b=$(grep -l '^## buddyinfo' "$EV"/*.snap | wc -l)
    echo "== $variant snapshots: $n, with vmstat $v, with buddyinfo $b"
    echo "== $variant cell tail: $(tail -1 "$D/cell.log")"
done
