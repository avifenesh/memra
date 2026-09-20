#!/bin/bash
set -euo pipefail
cd /root/artifacts
f=Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
curl --fail --location --retry 3 --retry-delay 10 --connect-timeout 30 --max-time 3600 --output "$f.part" "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/$f"
printf "df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf  %s.part\n" "$f" | sha256sum -c -
mv -n "$f.part" "$f"
printf "df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf  %s\n" "$f" > "$f.sha256"
sha256sum -c "$f.sha256"
echo ARTIFACT_SHA256_MATCH
