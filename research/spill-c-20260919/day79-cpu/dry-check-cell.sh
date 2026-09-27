#!/usr/bin/env bash
# DAY79: a control-flow dry run of day79-cell.sh `i18` in a sandbox: a sandbox tree with a stub lock proof, a stub
# run-gen-i17 (prints nothing, exits 0), a stub nsys on PATH (`profile` runs the command after its options and writes
# the -o report; `export` writes the -o file; `--version` prints a line), a stub nvidia-smi, a tiny stand-in artifact.
# No GPU and no lock: the fd is a pipe. Prints the calls, the run labels and exits, and the profile files.
set -euo pipefail
L=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d /tmp/c79cell.XXXXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/tree/tools" "$D/bins" "$D/bin" "$D/R"
printf 'print("{\\"stub\\": true}")\n' > "$D/tree/tools/tier-lock-proof.py"
git -C "$D/tree" init -q; git -C "$D/tree" -c user.email=x -c user.name=x commit -q --allow-empty -m x
echo a > "$D/art.gguf"
for b in run-gen-i17 run-gen-i18; do printf '#!/usr/bin/env bash\necho "%s $*" >> %s/calls.log\nexit 0\n' "$b" "$D" > "$D/bins/$b"; chmod +x "$D/bins/$b"; done
cat > "$D/bin/nsys" <<STUB
#!/usr/bin/env bash
echo "nsys \$*" >> $D/calls.log
case \$1 in
--version) echo "NVIDIA Nsight Systems version stub" ;;
profile) shift; out=; while [ "\${1#-}" != "\$1" ]; do case \$1 in -o) out=\$2; shift 2;; *) shift;; esac; done
         "\$@"; rc=\$?; echo rep > "\$out.nsys-rep"; exit \$rc ;;
export) out=; while [ \$# -gt 0 ]; do case \$1 in -o) out=\$2; shift 2;; *) shift;; esac; done; echo db > "\$out" ;;
esac
STUB
chmod +x "$D/bin/nsys"
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/bin/nvidia-smi"; chmod +x "$D/bin/nvidia-smi"
cd "$D"
PATH=$D/bin:$PATH D40_R=$D/R D40_BINS=$D/bins D40_TREE=$D/tree D40_LOCK=$D/none D40_ART=$D/art.gguf \
    bash "$L/day79-cell.sh" i18 0 > cell.log 2>&1
echo "cell rc=$?"
EV=$D/R/i18/ev
echo "== nsys.txt"; cat "$EV/nsys.txt"
echo "== calls: $(wc -l < calls.log) lines; per binary: $(grep -oE "^run-gen-i1[78]" calls.log | sort | uniq -c | tr "\n" " ")"; grep -E "^nsys (profile|export)" calls.log | sed "s#$D#D#g" | cut -c1-170
echo "== run exits"; for f in "$EV"/*.exit; do echo "$(basename "$f" .exit)=$(cat "$f")"; done | sort | paste -sd' '
echo "== profile files"; find "$EV" -name "*.nsys-rep" -o -name "*.sqlite" -o -name "*.export.log" | sed "s#$D#D#g" | sort
echo "== cell tail"; tail -1 cell.log
