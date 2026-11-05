#!/usr/bin/env bash
# Laptop rehearsal of kit/run.sh against the registered driver and the repo's CPU stub server.
# A copy of the kit with the <box> paths re-rooted here, stub wrappers in place of the binaries,
# a 1 MiB fake artifact, MemAvailable floor 4 GiB, and a fake nvidia-smi on PATH. No GPU work.
set -euo pipefail
REH=/home/avifenesh/.cache/memra-maxq/handoff-direct/rehearsal
KIT=/home/avifenesh/.cache/memra-maxq/handoff-direct/kit
WT=/home/avifenesh/.cache/memra-maxq/handoff-direct/wt-box36
R=$REH/hd
mkdir -p "$R" "$REH/models"
cp -r "$KIT/tools" "$KIT/research" "$KIT/SOURCE_SHA" "$KIT/kit-procs.py" "$KIT/maxq-verdict.py" "$R/"
mkdir -p "$R/bins/box36"
for pair in kv-handoff-gate:gate memra-server:server; do
  name=${pair%%:*} kind=${pair##*:}
  printf '#!/usr/bin/env bash\nM1_STUB_KIND=%s exec python3 %s "$@"\n' "$kind" "$WT/research/spill-f-20260919/m1-stub-kv-server.py" > "$R/bins/box36/$name"
  chmod +x "$R/bins/box36/$name"
done
(cd "$R/bins/box36" && sha256sum kv-handoff-gate memra-server > SHA256SUMS)
head -c 1048576 /dev/urandom > "$REH/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
SHA=$(sha256sum "$REH/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" | cut -d' ' -f1)
sed -e "s#/mnt/Data/work/memra-maxq/handoff-direct#$R#g" \
    -e "s#/mnt/Data/work/memra-maxq/models#$REH/models#g" \
    -e "s#^ART_SHA=.*#ART_SHA=$SHA#" -e "s#^ART_BYTES=.*#ART_BYTES=1048576#" \
    -e 's#64 \* 1024 \* 1024#4 * 1024 * 1024#' \
    "$KIT/run.sh" > "$R/run.sh"
diff "$KIT/run.sh" "$R/run.sh" > "$REH/run.sh.rehearsal.diff" || true
chmod +x "$REH/bin/nvidia-smi"
echo "rehearsal kit at $R"
