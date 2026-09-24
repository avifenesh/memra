#!/usr/bin/env bash
# DAY52: a control-flow dry run of day52-box.sh in a sandbox: the /root paths redirected under a temp dir, tiny
# stand-in artifacts with their SHA-256 substituted, every cell script, reader and the build replaced by a stub that
# logs its arguments, a stand-in nvidia-smi. Run 1 without `final` in the build list, run 2 with it (the finished
# cells skipped, DAY51's three cells run). No GPU, no build, nothing outside the temp dir. Prints both runs' calls.
set -euo pipefail
L=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d /tmp/c52dry.XXXXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/root/artifacts/q38-dflash2" "$D/bin" "$D/root/wt-c/research/spill-c-20260919" "$D/root/wt-c/tools"
echo a > "$D/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"; echo b > "$D/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
echo c > "$D/root/artifacts/q38-dflash2/config.json"; echo d > "$D/root/artifacts/q38-dflash2/model.safetensors"
SA=$(sha256sum "$D/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" | cut -d' ' -f1)
SB=$(sha256sum "$D/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" | cut -d' ' -f1)
sed -e "s#/root#$D/root#g" -e "s#df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf#$SA#" \
    -e "s#1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a#$SB#" "$L/day52-box.sh" > "$D/day52-box.sh"
S=$D/root/wt-c/research/spill-c-20260919
for f in day40-run-cell.sh day53-cell.sh day56-cell.sh; do
    printf '#!/usr/bin/env bash\necho "STUB %s $*" >> %s/calls.log\n' "$f" "$D" > "$S/$f"
done
for f in day40-attrib.py day52-views.py day51-decide.py day54-slice-reading.py day56-reading.py; do
    printf 'import sys\nopen("%s/calls.log", "a").write("STUB %s " + " ".join(sys.argv[1:]) + "\\n")\n' "$D" "$f" > "$S/$f"
done
printf 'import sys\nopen("%s/calls.log", "a").write("STUB tier-battery " + " ".join(sys.argv[1:]) + "\\n")\n' "$D" > "$D/root/wt-c/tools/tier-battery.py"
cat > "$S/day52-box-build.sh" <<STUB
#!/usr/bin/env bash
echo "STUB day52-box-build.sh \$*" >> $D/calls.log
OUT=\$2; mkdir -p \$OUT/bins; shift 2
for spec in "\$@"; do [ "\${spec%%=*}" = final ] && { touch \$OUT/bins/run-gen-final \$OUT/bins/run-spec-final; chmod +x \$OUT/bins/run-*-final; }; done
echo BUILDS-DONE >> \$OUT/builds.log
STUB
git -C "$D/root/wt-c" init -q; git -C "$D/root/wt-c" -c user.email=x -c user.name=x commit -q --allow-empty -m x
printf '#!/usr/bin/env bash\necho stand-in\n' > "$D/bin/nvidia-smi"; chmod +x "$D/bin/nvidia-smi"
cd "$D"
for run in 1 2; do
    : > calls.log
    [ "$run" = 2 ] && { rm -f "$D/root/spill-receipts/c-day52/builds.log"; B="base=1 i4=2 final=3"; } || B="base=1 i4=2"
    PATH=$D/bin:$PATH D52_BUILDS="$B" bash "$D/day52-box.sh" > "run$run.log" 2>&1 || echo "run $run rc=$?"
    echo "== run $run (D52_BUILDS=\"$B\") calls"
    sed "s#$D#D#g" calls.log
    echo "== run $run driver lines"
    grep -E "already done|final not built|box done|rc=" "run$run.log" | sed "s#$D#D#g"
done
