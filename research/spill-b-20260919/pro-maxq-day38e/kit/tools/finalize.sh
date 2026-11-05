#!/usr/bin/env bash
# After build.sh: SOURCE_SHA, PROVENANCE.txt and KIT-MANIFEST.sha256 for the park-compact Max-Q kit.
set -euo pipefail
P=/home/avifenesh/.cache/memra-maxq/park-compact
K=$P/kit
W=$P/wt-tip
SHA=2ccb038d1c16358edde65582597e7b840e3f63fb
cd "$K"
grep -q '^rc=0$' build.log
(cd bins && sha256sum -c --quiet SHA256SUMS)
echo "$SHA" > SOURCE_SHA
mkdir -p tools
cp "$P/build.sh" tools/build.sh
cp "$P/port-run-day26-cell.py" tools/port-run-day26-cell.py
cp "$P/finalize.sh" tools/finalize.sh
{
  echo "park-compact Max-Q kit provenance, written $(date -u +%FT%TZ) on the laptop"
  echo
  echo "source_sha=$SHA (origin/fix4/gates, the fix-stack tip; the three arms build from it)"
  echo "workstation_deciding_tree=a803d308080d047eefedf76fc18cf5c09484869e (box-d arms: green c65f30761a15858954c3bb925b02b853eb7af68d8ebc30578fdbfcd34665313c, red 0673a71d04269abdebe9a9bd0a5f383392d63b26f336da65543e7a76c2bafd1f)"
  echo "workstation_harness_tree=3476ea4dca27c4c3c738017b8429f507389f1223 (box-d source.txt; docs/SERVING.md taken from it)"
  echo
  echo "patches:"
  (cd patches && sha256sum maxq-arm.patch day38-red.patch | sed 's/^/  /')
  echo "  day38-red.patch is research/spill-b-20260919/day38-red.patch at the tip, the sha256 box-d bins/*/source.patches records"
  echo
  echo "binaries (cargo build --release -j2 -p memra-server --bin memra-server, default features):"
  for arm in off on red; do
    echo "  $arm: sha256=$(sha256sum "bins/$arm/memra-server" | cut -d' ' -f1) bytes=$(stat -c %s "bins/$arm/memra-server") source=$(cat "bins/$arm/source.commit") patches=[$(cut -d' ' -f3 "bins/$arm/source.patches" | grep -E '\.patch$' | tr '\n' ' ' | sed 's/ $//')]"
  done
  echo
  echo "toolchain (kit/build.log):"
  grep -E '^MEMRA_CUDA_ARCH|^Build cuda|^cargo |^rustc ' build.log | sed 's/^/  /'
  echo "  every one of the 12 workspace crates of memra-server compiled from the kit worktree in the off arm (build.log);"
  echo "  the on arm rebuilt memra-engine and memra-server, the red arm memra-server"
  echo
  echo "harness files (sha256; the first three byte-identical to the tip and to box-d inputs.sha256 where box-d records them):"
  sha256sum research/spill-b-20260919/day38-client.py research/spill-b-20260919/day38-parse.py \
    research/spill-b-20260919/day38d-read.py docs/SERVING.md research/spill-b-20260919/day38d-run.sh \
    research/spill-b-20260919/run-day26-cell.sh | sed 's/^/  /'
  echo "  ported scripts: PORTABILITY.diff against the tip's day38d-run.sh and run-day26-cell.sh"
  echo
  echo "model: Qwen3.8-27B-NVFP4-Q5K-mtp.gguf sha256=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a bytes=15705922304"
  echo "  (research/mtp-learned-depth-20260920/qwen-artifacts.lock.json and box-d model.sha256; run.sh verifies it before the first boot)"
} > PROVENANCE.txt
find . -type f ! -name KIT-MANIFEST.sha256 ! -path './out-*' ! -path './scratch/*' -print0 \
  | sort -z | xargs -0 sha256sum > "$P/manifest.tmp"
mv "$P/manifest.tmp" KIT-MANIFEST.sha256
sha256sum -c --quiet KIT-MANIFEST.sha256
echo "kit manifest: $(wc -l < KIT-MANIFEST.sha256) files, $(sha256sum KIT-MANIFEST.sha256 | cut -c1-16)"
