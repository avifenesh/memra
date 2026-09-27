#!/usr/bin/env bash
# DAY73: a control-flow dry run of day73-cell.sh `compact` in a sandbox: a sandbox tree holding the cell's sampler and a
# stub lock proof, a stub run-gen-p71 (the counters check prints a counted line; a run prints nothing and exits 0),
# a stub nvidia-smi (a dmon header and one row a second until killed; the snapshots' queries print nothing), a tiny
# stand-in artifact. No GPU and no lock: the fd is a pipe. Prints the cell's pins, counters decision, the calls,
# the sampler's row kinds, the dmon rows, and that no stub, sampler or dmon process outlives the cell.
set -euo pipefail
L=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d /tmp/c73cell.XXXXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/tree/research/spill-c-20260919" "$D/tree/tools" "$D/bins" "$D/bin" "$D/R"
cp "$L/day73-sampler.py" "$D/tree/research/spill-c-20260919/"
printf 'print("{\\"stub\\": true}")\n' > "$D/tree/tools/tier-lock-proof.py"
git -C "$D/tree" init -q; git -C "$D/tree" -c user.email=x -c user.name=x commit -q --allow-empty -m x
echo a > "$D/art.gguf"
cat > "$D/bins/run-gen-p71" <<STUB
#!/usr/bin/env bash
echo "\$*" >> $D/calls.log
if [ "\${1:-}" = --cpu-probe-counters-check ]; then
    echo "[cpu-probe] counters-check cpu=1 rdpru=ok cpu_after=1 wall_ns=1528888 tsc=6571234 mperf=6571000 aperf=8745000" >&2
fi
exit 0
STUB
chmod +x "$D/bins/run-gen-p71"
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
PATH=$D/bin:$PATH D40_R=$D/R D40_BINS=$D/bins D40_TREE=$D/tree D40_LOCK=$D/none D40_ART=$D/art.gguf \
    bash "$L/day73-cell.sh" compact 0 > cell.log 2>&1
echo "cell rc=$?"
EV=$D/R/compact/ev
echo "== pins and counters"; cat "$EV/pins.txt" "$EV/counters.txt"
echo "== counters check"; cat "$EV/counters-check.txt"
echo "== calls (first 3 and count)"; head -3 calls.log | sed "s#$D#D#g"; wc -l < calls.log
echo "== run exits"; cat "$EV"/*.exit | sort | uniq -c
echo "== sampler row kinds"; cut -f2 "$EV/sched.tsv" | sort | uniq -c | tr '\n' ' '; echo
echo "== strace"; cat "$EV/strace.txt"; find "$EV" -name "*.strace" | sed "s#$D#D#g" | sort
echo "== host state files"; for f in "$EV"/host-*.txt; do echo "$(basename "$f") $(wc -l < "$f") lines, $(head -1 "$f")"; done
echo "== cell tail"; tail -2 cell.log
sleep 1
echo "== leftover processes (none expected)"; ps -eo pid,args | grep -E "[d]ay71-sampler.py $D" || echo none
