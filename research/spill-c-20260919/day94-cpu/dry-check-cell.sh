#!/usr/bin/env bash
# DAY94: a control-flow dry run of day94-cell.sh in a sandbox: a sandbox tree with a stub lock proof, stub run-gen-i25
# and run-gen-i24 (each logs its argv, its environment words and its CPU affinity, prints nothing, exits 0), a stub
# nvidia-smi, a tiny stand-in artifact. No GPU and no lock: the fd is 0. Prints the calls per binary with their flags,
# environment and affinity, the run order, the exits, and each run's recorded CPU list.
set -euo pipefail
L=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d /tmp/c94cell.XXXXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/tree/tools" "$D/bins" "$D/bin" "$D/R"
printf 'print("{\\"stub\\": true}")\n' > "$D/tree/tools/tier-lock-proof.py"
git -C "$D/tree" init -q; git -C "$D/tree" -c user.email=x -c user.name=x commit -q --allow-empty -m x
echo a > "$D/art.gguf"
for b in run-gen-i25 run-gen-i24; do
    printf '#!/usr/bin/env bash\necho "%s cpus=$(taskset -pc $$ | sed "s/.*: //") env=[$(env | grep -E "^MEMRA_(EXPERTS_VIA_TIER|MOE_PREFETCH|MOE_RESIDENT|MOE_SLOTS|NGEN)=" | sort | tr "\\n" " ")] $*" >> %s/calls.log\nexit 0\n' "$b" "$D" > "$D/bins/$b"
    chmod +x "$D/bins/$b"
done
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/bin/nvidia-smi"; chmod +x "$D/bin/nvidia-smi"
(cd "$D" && PATH=$D/bin:$PATH D40_R=$D/R D40_BINS=$D/bins D40_TREE=$D/tree D40_LOCK=$D/none D40_ART=$D/art.gguf \
    D94_WIDE=0-11 D94_PCORES=0-3 bash "$L/day94-cell.sh" where285 0 > cell.log 2>&1; echo "== rc=$?")
echo "== calls: $(wc -l < "$D/calls.log")"
sed -E "s#$D#D#g; s# D/art.gguf 55 88 13##" "$D/calls.log" | sort | uniq -c
echo "== order: $(awk -F'\t' '/start/ {print $2}' "$D/R/where285/ev/marks.tsv" | sed 's/ start//' | paste -sd' ' | cut -c1-500)"
echo "== exits: $(cat "$D/R/where285/ev"/*.exit | sort | uniq -c | tr '\n' ' ')"
echo "== cpu files: $(cat "$D/R/where285/ev"/*.cpus | sort | uniq -c | tr '\n' ' ')"
cat "$D/R/where285/ev/cpus.txt"
