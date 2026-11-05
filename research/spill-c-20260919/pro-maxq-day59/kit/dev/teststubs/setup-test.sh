#!/usr/bin/env bash
# Stage a laptop plumbing test of kit/run.sh: the kit's scripts, stub binaries and a stub nvidia-smi (no GPU use),
# the real local artifact by symlink (so the SHA check is real). usage: setup-test.sh <test-dir-name>
set -euo pipefail
B=/home/avifenesh/.cache/memra-maxq/moe-prefetch
T=$B/$1
test ! -e "$T" || { echo "$T exists"; exit 1; }
mkdir -p "$T/bins/c60" "$T/models"
cp -r "$B/kit/tree" "$T/tree"
cp "$B/kit/run.sh" "$T/run.sh"
[ -f "$B/kit/PREREG.md" ] && cp "$B/kit/PREREG.md" "$T/PREREG.md"
cp "$B/teststubs/replay-bin.py" "$T/bins/c60/run-gen-c60"
cp "$B/teststubs/replay-bin.py" "$T/bins/c60/run-spec-c60"
cp "$B/teststubs/stub-server.py" "$T/bins/c60/memra-server-c60"
chmod +x "$T/bins/c60/"*
cp "$B/kit/bins/c60/BUILD_SHA" "$T/bins/c60/BUILD_SHA"
( cd "$T/bins/c60" && sha256sum run-gen-c60 run-spec-c60 memra-server-c60 > bins.sha256 )
ln -s /data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf "$T/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
chmod +x "$B/teststubs/stubbin/nvidia-smi"
echo "staged $T"
