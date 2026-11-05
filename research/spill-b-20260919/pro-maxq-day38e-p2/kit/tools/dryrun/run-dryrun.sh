#!/usr/bin/env bash
# CPU dry run of the kit's runner (no GPU, no model): a copy of kit/ under the test root with the <box> paths
# rewritten, a fake nvidia-smi on PATH and a stub memra-server per arm. Scenarios: refusal on a busy card, part 1,
# part 2 with the combined reading, auto refusal once the cell is complete, a budget cut, a thermal stop.
set -uo pipefail
ST=/home/avifenesh/.cache/memra-maxq/park-compact/dryrun/stub
KIT0=/home/avifenesh/.cache/memra-maxq/park-compact/kit
T=$ST/root/kit
rm -rf "$ST/root"; mkdir -p "$T" "$ST/root/models" "$ST/logs"
cp -a "$KIT0/." "$T/"
rm -rf "$T/bins"
for arm in off on red; do
  mkdir -p "$T/bins/$arm"
  printf '#!/usr/bin/env bash\nexec python3 %s "$0"\n' "$ST/stub_server.py" > "$T/bins/$arm/memra-server"
  chmod +x "$T/bins/$arm/memra-server"
  echo "stub $arm" >> "$T/bins/$arm/memra-server"   # distinct bytes per arm
done
(cd "$T/bins" && sha256sum off/memra-server on/memra-server red/memra-server) > "$T/bins/SHA256SUMS"
head -c 1048576 /dev/urandom > "$ST/root/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
MSHA=$(sha256sum "$ST/root/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" | cut -d' ' -f1)
python3 - "$T" "$ST/root" "$MSHA" <<'EOF'
import sys
t, root, msha = sys.argv[1:]
for f in ("run.sh", "maxq-verdict.py"):
    p = f"{t}/{f}"
    s = open(p).read()
    s = s.replace("/mnt/Data/work/memra-maxq/park-compact", t).replace("/mnt/Data/work/memra-maxq/models", f"{root}/models")
    s = s.replace("1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a", msha)
    s = s.replace("MODEL_BYTES=15705922304", "MODEL_BYTES=1048576").replace("LOCK=/tmp/memra-gpu.lock", f"LOCK={root}/memra-gpu.lock")
    s = s.replace("PORT=18526", "PORT=18626")  # a sibling kit's laptop rehearsal holds 18526
    open(p, "w").write(s)
EOF
(cd "$T" && find . -type f ! -name KIT-MANIFEST.sha256 ! -path './out-*' ! -path './scratch/*' -print0 | sort -z | xargs -0 sha256sum) > "$ST/manifest.tmp" && mv "$ST/manifest.tmp" "$T/KIT-MANIFEST.sha256"
echo 2ccb038d1c16358edde65582597e7b840e3f63fb > "$T/SOURCE_SHA"
(cd "$T" && find . -type f ! -name KIT-MANIFEST.sha256 ! -path './out-*' ! -path './scratch/*' -print0 | sort -z | xargs -0 sha256sum) > "$ST/manifest.tmp" && mv "$ST/manifest.tmp" "$T/KIT-MANIFEST.sha256"
export PATH=$ST/bin:$PATH FAKE_SMI_KIT=$T FAKE_SMI_STATE=$ST/root/smi.json CARD=2
run() { local n=$1; shift; env "$@" bash "$T/run.sh" > "$ST/logs/$n.log" 2>&1; echo "== $n rc=$? :: $(grep -E '^(REFUSED|VERDICT|OUT)' "$ST/logs/$n.log" | tr '\n' ' ' | cut -c1-700)"; }

echo '{"memory.used": 5000}' > "$ST/root/smi.json"
run busy-card PART=1
rm -f "$ST/root/smi.json"
run part1 PART=auto LENGTHS_UNUSED=1
run part2 PART=auto
run auto-after-complete PART=auto
run budget-cut PART=1 BUDGET_S=200 MAIN_EST_S=10
( sleep 12; echo '{"temperature.gpu": 96, "fan.speed": 80}' > "$ST/root/smi.json" ) &
run thermal PART=1
rm -f "$ST/root/smi.json"
echo "== servers left: $(for p in $(pgrep -x memra-server); do [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$T" ] && echo $p; done | wc -l); scratch left: $([ -e "$T/scratch" ] && echo yes || echo no)"
ls -1d "$T"/out-* | sed "s|$T/||"
