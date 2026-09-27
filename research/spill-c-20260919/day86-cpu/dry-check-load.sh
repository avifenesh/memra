#!/usr/bin/env bash
# DAY86 section 1c: a control-flow dry run of day86-load.sh in a sandbox: the /root paths and the lock redirected under
# a temp dir, the build replaced by a stub that leaves a stub run-gen-i22 (prints a MATCH line, a gen line and 200
# trace lines, exits 0; every seventh call exits 3), about 20 seconds of load. Prints the driver log, the runs table's
# shape, and which runs kept their full log.
set -euo pipefail
L=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d /tmp/c86load.XXXXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/root/artifacts" "$D/root/wt-c/research/spill-c-20260919"
echo a > "$D/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
sed -e "s#/root#$D/root#g" -e "s#/tmp/memra-gpu.lock#$D/gpu.lock#" "$L/day86-load.sh" > "$D/day86-load.sh"
cat > "$D/root/wt-c/research/spill-c-20260919/day63-box-build.sh" <<STUB
#!/usr/bin/env bash
echo "STUB day63-box-build.sh \$*"
mkdir -p \$2/bins
cat > \$2/bins/run-gen-i22 <<'BIN'
#!/usr/bin/env bash
c=\$(cat $D/count 2>/dev/null || echo 0); c=\$((c + 1)); echo \$c > $D/count
echo "prefill argmax=198  decode argmax=198  MATCH"; for i in \$(seq 200); do echo "[expert-host-slru] key=0:0:\$i bytes=1 slot=\$i hit=true victim=-"; done
echo "generated 32 tokens in 0.32\$((c % 10))s = 100 tok/s"; sleep 0.3
[ \$((c % 7)) -eq 0 ] && exit 3; exit 0
BIN
chmod +x \$2/bins/run-gen-i22
STUB
git -C "$D/root/wt-c" init -q; git -C "$D/root/wt-c" -c user.email=x -c user.name=x commit -q --allow-empty -m x
D86_LOAD_HOURS=0.005 bash "$D/day86-load.sh" > "$D/out.log" 2>&1 || echo "load rc=$?"
echo "== driver log"; sed "s#$D#D#g" "$D/root/spill-receipts/c-day86-load/load-driver.log"
echo "== runs.tsv: $(($(wc -l < "$D/root/spill-receipts/c-day86-load/runs.tsv") - 1)) runs"; head -3 "$D/root/spill-receipts/c-day86-load/runs.tsv" | cut -c1-160
echo "== full logs kept: $(cd "$D/root/spill-receipts/c-day86-load/ev" && ls ./*.log 2>/dev/null | paste -sd' ')"
echo "== a summary: $(cd "$D/root/spill-receipts/c-day86-load/ev" && cat load-r2.sum | paste -sd'|')"
