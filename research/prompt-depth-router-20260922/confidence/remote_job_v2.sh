#!/usr/bin/env bash
set -euo pipefail

study_root=${STUDY_ROOT:?set STUDY_ROOT to the staged, pinned Qwen study}
cd "$study_root"
trap 'rc=$?; printf "exit=%s utc=%s\n" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > job-v2.exit; exit "$rc"' EXIT

test -z "${MEMRA_CAPTURE_DIR:-}"
test -f repo-confidence/PREFIX-ROUTING-SOURCE.json
test -x binaries-confidence/qwen-prefix-study
test -x binaries-confidence/run-spec
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

source = json.loads(Path("source-confidence.json").read_text())
model = json.loads(Path("models/qwen/artifacts.lock.json").read_text())[0]
for path, want in (
    (Path("runtime-source.tar.gz"), source["base_runtime_source_sha256"]),
    (Path(source["runtime_source_archive"]), source["runtime_source_sha256"]),
    (Path("repo-confidence/crates/memra-engine/src/spec.rs"), source["patched_spec_sha256"]),
    (Path("binaries-confidence/qwen-prefix-study"), source["binaries"]["qwen-prefix-study"]),
    (Path("binaries-confidence/run-spec"), source["oracle_binary_sha256"]),
    (Path("models/qwen") / model["local_file"], model["sha256"]),
):
    with path.open("rb") as stream:
        got = hashlib.file_digest(stream, "sha256").hexdigest()
    if got != want:
        raise SystemExit(f"{path}: expected {want}, got {got}")
    print(f"PIN_OK {path} {got}", flush=True)
PY

python3 - <<'PY'
from pathlib import Path

parts = Path("workloads/qwen-simple-qualification.txt").read_text().split("\n---TURN---\n")
if len(parts) != 8:
    raise SystemExit("qualification source has another request count")
Path("oracle-code-prompt.txt").write_text(parts[1])
PY

for arm in off c030 c030zero; do
    case "$arm" in
        off) pmin=0; pmin0=0 ;;
        c030) pmin=0.30; pmin0=0 ;;
        c030zero) pmin=0.30; pmin0=1 ;;
    esac
    MEMRA_CHAT=1 MEMRA_PROMPT_FILE="$study_root/oracle-code-prompt.txt" \
        MEMRA_NGEN=128 MEMRA_SPEC_K=3 MEMRA_SPEC_TEMP=0 \
        MEMRA_SPEC_ADAPT=0 MEMRA_SPEC_PMIN="$pmin" MEMRA_SPEC_PMIN0="$pmin0" \
        binaries-confidence/run-spec models/qwen/target.gguf \
        > "oracle-v2-$arm.log" 2>&1
    grep -q '^=== SELF-CONSISTENCY PASS ===$' "oracle-v2-$arm.log"
    printf "ORACLE_PASS %s\n" "$arm"
done

python3 harness/confidence/fixed_grid.py \
    --repo "$study_root/repo-confidence" \
    --models "$study_root/models" \
    --binaries "$study_root/binaries-confidence" \
    --source "$study_root/source-confidence.json" \
    --workloads "$study_root/workloads" \
    --out "$study_root/fixed-grid-v2"
python3 harness/confidence/report_fixed.py \
    --root "$study_root/fixed-grid-v2" \
    --out "$study_root/fixed-grid-v2-report.json"
tar -czf confidence-receipts-v2.tar.gz \
    fixed-grid-v2 fixed-grid-v2-report.json oracle-code-prompt.txt \
    oracle-v2-off.log oracle-v2-c030.log oracle-v2-c030zero.log \
    source-confidence.json source-patch.json runtime-source-confidence.tar.gz \
    confidence-build.log harness/confidence harness/prefix
echo CONFIDENCE_FIXED_GRID_COMPLETE
