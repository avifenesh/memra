#!/usr/bin/env bash
# DAY80: a control-flow dry run of day80-box.sh in a sandbox, with the default machine label: the /root paths redirected under a temp dir, a tiny
# stand-in artifact with its SHA-256 substituted, every cell script, reader and the build replaced by a stub that
# logs its arguments, a stand-in nvidia-smi. Run 1 builds and runs every cell; run 2 reruns (every cell skipped by
# its .done marker, no build). No GPU, no build, nothing outside the temp dir. Prints both runs' calls.
set -euo pipefail
L=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d /tmp/c80dry.XXXXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/root/artifacts" "$D/bin" "$D/root/wt-c/research/spill-c-20260919" "$D/root/wt-c/tools"
echo a > "$D/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
SA=$(sha256sum "$D/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" | cut -d' ' -f1)
sed -e "s#/root#$D/root#g" -e "s#df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf#$SA#" \
    "$L/day80-box.sh" > "$D/day80-box.sh"
S=$D/root/wt-c/research/spill-c-20260919
# shellcheck disable=SC2016  # the stub expands its own arguments when it runs
printf '#!/usr/bin/env bash\necho "STUB day40-run-cell.sh $* script=$D40_CELL_SCRIPT" >> %s/calls.log\n' "$D" > "$S/day40-run-cell.sh"
for rd in day80-regtime-read.py day80-regpool-read.py; do printf 'import sys\nopen("%s/calls.log", "a").write("STUB %s " + " ".join(sys.argv[1:]) + "\\n")\n' "$D" "$rd" > "$S/$rd"; done
printf 'import sys\nopen("%s/calls.log", "a").write("STUB tier-battery " + " ".join(sys.argv[1:]) + "\\n")\n' "$D" > "$D/root/wt-c/tools/tier-battery.py"
cat > "$S/day63-box-build.sh" <<STUB
#!/usr/bin/env bash
echo "STUB day63-box-build.sh \$*" >> $D/calls.log
OUT=\$2; mkdir -p \$OUT/bins
for b in run-gen-p80; do touch \$OUT/bins/\$b; chmod +x \$OUT/bins/\$b; done
STUB
git -C "$D/root/wt-c" init -q; git -C "$D/root/wt-c" -c user.email=x -c user.name=x commit -q --allow-empty -m x
printf '#!/usr/bin/env bash\necho stand-in\n' > "$D/bin/nvidia-smi"; chmod +x "$D/bin/nvidia-smi"
cd "$D"
for run in 1 2; do
    : > calls.log
    PATH=$D/bin:$PATH D80_BUILDS="p80=1" bash "$D/day80-box.sh" > "run$run.log" 2>&1 || echo "run $run rc=$?"
    echo "== run $run calls"
    sed "s#$D#D#g" calls.log
    echo "== run $run driver lines"
    grep -E "already done|missing|box done|rc=|cpu cap|rig " "run$run.log" | sed "s#$D#D#g"
done
