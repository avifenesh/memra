#!/usr/bin/env bash
# DAY78: a control-flow dry run of day78-cell.sh `pages` in a sandbox: a sandbox tree holding the cell's sampler and a
# stub lock proof, a stub run-gen-p78 (the counters check prints a counted line; a run prints nothing and exits 0),
# a stub nvidia-smi (a dmon header and one row a second until killed; the snapshots' queries print nothing), a tiny
# stand-in artifact. No GPU and no lock: the fd is a pipe. Prints the cell's pins, counters decision, the calls,
# the sampler's row kinds, the dmon rows, and that no stub, sampler or dmon process outlives the cell.
set -euo pipefail
L=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d /tmp/c78cell.XXXXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/tree/research/spill-c-20260919" "$D/tree/tools" "$D/bins" "$D/bin" "$D/R"
cp "$L/day73-sampler.py" "$L/day74-compactor.py" "$L/day78-pages.py" "$D/tree/research/spill-c-20260919/"
printf 'print("{\\"stub\\": true}")\n' > "$D/tree/tools/tier-lock-proof.py"
git -C "$D/tree" init -q; git -C "$D/tree" -c user.email=x -c user.name=x commit -q --allow-empty -m x
echo a > "$D/art.gguf"
cat > "$D/bins/run-gen-p78" <<STUB
#!/usr/bin/env bash
echo "\$*" >> $D/calls.log
if [ "\${1:-}" = --cpu-probe-counters-check ]; then
    echo "[cpu-probe] counters-check cpu=1 rdpru=ok cpu_after=1 wall_ns=1528888 tsc=6571234 mperf=6571000 aperf=8745000" >&2
    exit 0
fi
echo "[cpu-probe] phase=gate cpu=1 compute_ns=1.050" >&2
sleep 1.5
exit 0
STUB
chmod +x "$D/bins/run-gen-p78"
printf '#!/usr/bin/env bash\necho "strace $*" >> %s/calls.log\n[ "$1" = -V ] && { echo "strace -- version stub"; exit 0; }\nwhile [ "${1#-}" != "$1" ]; do case $1 in -o) o=$2; echo "% time calls syscall" > "$o"; shift 2;; -S) shift 2;; *) shift;; esac; done\nexec "$@"\n' "$D" > "$D/bin/strace"; chmod +x "$D/bin/strace"
cat > "$D/bin/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
if [ "${1:-}" = dmon ]; then
    echo "#Time        gpu  rxpci  txpci"; echo "#HH:MM:SS    Idx   MB/s   MB/s"
    while true; do echo "$(date -u +%T)      0    100     20"; sleep 1; done
fi
exit 0
STUB
chmod +x "$D/bin/nvidia-smi"
cd "$D"
PATH=$D/bin:$PATH D74_DRY_F=1 D40_R=$D/R D40_BINS=$D/bins D40_TREE=$D/tree D40_LOCK=$D/none D40_ART=$D/art.gguf \
    bash "$L/day78-cell.sh" pages 0 > cell.log 2>&1
echo "cell rc=$?"
EV=$D/R/pages/ev
echo "== pins and counters"; cat "$EV/pins.txt" "$EV/counters.txt"
echo "== counters check"; cat "$EV/counters-check.txt"
echo "== calls (first 3 and count)"; head -3 calls.log | sed "s#$D#D#g"; wc -l < calls.log
echo "== run exits"; cat "$EV"/*.exit | sort | uniq -c
echo "== sampler row kinds"; cut -f2 "$EV/sched.tsv" | sort | uniq -c | tr '\n' ' '; echo
echo "== census runs: $(find "$EV" -name "c[12]-*-r1.exit" | wc -l), census attempts: $(find "$EV" -name "*.pages.log" | wc -l) (the stub is a script, named bash, so the census finds no run-gen-p78 and writes nothing)"; echo "== inducer"; cat "$EV/inducer.txt"; echo "== rewarm rows: $(wc -l < "$EV/rewarm.tsv")"; head -2 "$EV/rewarm.tsv"; echo "== compactor rows (labels, ready, loop, stopped)"; grep -P "\t(o[12]-|ready|loop|stopped)" "$EV/compactor.tsv" | cut -f2- | cut -c1-60
echo "== host state files"; for f in "$EV"/host-*.txt; do echo "$(basename "$f") $(wc -l < "$f") lines, $(head -1 "$f")"; done
echo "== cell tail"; tail -2 cell.log
sleep 1
echo "== leftover processes (none expected)"; ps -eo pid,args | grep -E "[d]ay7[34]-(sampler|compactor).py $D" || echo none
