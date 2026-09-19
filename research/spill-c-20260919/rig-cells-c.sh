#!/usr/bin/env bash
# CPU/dry-run orchestration is NOT a GPU qualification receipt.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
lane=research/spill-c-20260919
confirmed=0
dry=0
rig=5090
host_label=development
hy3=
hy3_manifest=
stub_fail=
while (($#)); do
  case "$1" in
    --non-serving-confirmed) confirmed=1; shift ;;
    --dry-run) dry=1; shift ;;
    --rig) rig=${2:?}; shift 2 ;;
    --host-label) host_label=${2:?}; shift 2 ;;
    --hy3-artifact) hy3=${2:?}; shift 2 ;;
    --hy3-manifest) hy3_manifest=${2:?}; shift 2 ;;
    --stub-fail) stub_fail=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ $rig == 5090 || $rig == pro-pair ]] || { echo 'rig must be 5090 or pro-pair' >&2; exit 2; }
[[ $host_label =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'host-label must be a public hardware alias, not an instance id' >&2; exit 2; }
[[ -z $stub_fail || $dry == 1 ]] || { echo '--stub-fail requires --dry-run' >&2; exit 2; }
((dry || confirmed)) || { echo "explicit --non-serving-confirmed required" >&2; exit 2; }
lock=/tmp/memra-5090.lock
[[ $rig != pro-pair ]] || lock=/tmp/memra-gpu.lock
mkdir -p "$lane/raw"
utc=$(date -u +%Y%m%dT%H%M%SZ)
# Atomic mkdir; repeat invocations never overwrite or reuse earlier evidence.
out=$lane/raw/$host_label-$utc
suffix=0
while ! mkdir "$out" 2>/dev/null; do
  suffix=$((suffix+1))
  ((suffix < 100)) || { echo 'cannot create evidence directory' >&2; exit 2; }
  out=$lane/raw/$host_label-$utc-$suffix
done
head=$(git rev-parse HEAD)
record() {
  python3 "$lane/record-cell.py" "$out" "$head" "$dry" "$@"
}
stub() {
  printf 'STUB ONLY, not executed:'; printf ' %q' "$@"; printf '\n'
  [[ $cell != "$stub_fail" ]] || { echo 'injected stub failure' >&2; return 23; }
}
run() {
  local cell=$1; shift
  local rc=0
  printf 'CELL %s:' "$cell"; printf ' %q' "$@"; printf '\n'
  if ((dry)); then
    stub "$@" 2>&1 | tee "$out/$cell.log" || rc=$?
  else
    "$@" 2>&1 | tee "$out/$cell.log" || rc=$?
  fi
  record "$cell" "$rc" "$out/$cell.log" "$@"
  if ((rc)); then
    if ((!dry)); then
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | tee "$out/$cell-processes.log" || true
    fi
    return "$rc"
  fi
}
blocked() {
  printf 'BLOCKED: %s\n' "$2" | tee "$out/$1.log"
  record "$1" 3 "$out/$1.log" BLOCKED "$2"
}
# Day-4 CPU/source checks precede native work; they do not qualify CUDA ownership.
run source-contract cargo test -p memra-tier --offline --test bank day4
run patch-check git apply --check "$lane/HY3-DISPATCH-PATCH.diff"
# Build before flock: sccache/cargo daemons must never inherit the GPU lock.
run build cargo build --release -p memra-engine --bin qwen4exp_gpu_gate --bin kernel-check --bin run-gen --bin run-spec
if ((dry)); then
  run lock flock "$lock" true
else
  command -v flock >/dev/null
  exec 9>"$lock"
  flock -n 9 || { blocked lock 'canonical whole-box lock held'; exit 3; }
fi
run topology nvidia-smi topo -m
run processes-before nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# Fits 32 GiB: tiny PLE oracle first, before any full Hy3 artifact load.
if ((!dry)); then
  python3 -c 'import pathlib,sys; lines=pathlib.Path(sys.argv[1]).read_text().splitlines(); sys.exit(0 if len(lines)==1 else 3)' "$out/processes-before.log" || { blocked occupied 'GPU compute processes present or inventory malformed; no interference authorized'; exit 3; }
fi
# PRE-STREAMING goldens must accompany the TSV; never mint them with this loader.
run ple-goldens-copy cp "$root/research/qwen4exp-bringup-20260829/gpu-eager/bank-bytes-goldens.tsv" "$root/$out/bank-bytes-goldens.tsv"
run ple-goldens-sha256 sha256sum "$root/$out/bank-bytes-goldens.tsv"
run ple-tiny "$root/target/release/qwen4exp_gpu_gate" "$root/$out/ple-tiny.tsv"
run kernel-check "$root/target/release/kernel-check"
# These baseline gates do not exercise the proposed bank/row GPU adapter.
# Refuse instead of treating absent tests as a successful empty test run.
blocked ple-adapter 'table_rows_gpu native target/owner binding is not implemented; day-4 source/SLRU CPU contracts do not qualify ReadyView or projection consumption'
if [[ $rig == 5090 ]]; then
  blocked hy3-pro-pair 'Hy3 full-artifact cells require a non-serving PRO pair; a 5090 needs a separately hash-bound host-spill fitting receipt'
elif [[ -z $hy3 || -z $hy3_manifest ]]; then
  blocked hy3-artifact 'supply immutable Hy3 artifact and byte manifest; no download, conversion, or CPU-expert fallback'
else
  if ((!dry)); then
    [[ -e $hy3 && -s $hy3_manifest ]] || { blocked hy3-artifact 'artifact/manifest absent'; exit 3; }
    # sha256sum manifest paths are relative to the artifact directory.
    # The caller must supply a standard byte manifest, not a summary score file.
    artifact_dir=$hy3
    [[ -d $artifact_dir ]] || artifact_dir=$(dirname "$hy3")
    manifest=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$hy3_manifest")
    run hy3-manifest bash -c 'cd "$1"; sha256sum --strict -c "$2"' _ "$artifact_dir" "$manifest"
  fi
  # kernel-check accepts GGUF only, unlike run-gen/run-spec's native directory loader.
  if [[ $hy3 == *.gguf ]]; then
    run hy3-kernel "$root/target/release/kernel-check" "$hy3" --require-cell d2-cache-bit-identity
  else
    blocked hy3-kernel-artifact 'kernel-check model reader is GGUF-only; retain generic kernel result, request compatible kernel fixture without substituting the Hy3 artifact'
  fi
  run hy3-gen "$root/target/release/run-gen" "$hy3" --prompt 'Compute 17 times 23 and explain briefly.'
  run hy3-spec env -u MEMRA_SPEC_K "$root/target/release/run-spec" "$hy3"
  blocked hy3-adapter 'banked_residency_gpu missing; native mixed/pruned/SLRU/staged/grouped engagement and before/after identity remain required'
fi
run processes-after nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
printf 'Receipt directory: %s (baseline/dry-run only; adapter qualification BLOCKED)\n' "$out"
# Dry-run success means harness flow only; real run refuses incomplete qualification.
((dry)) || exit 3
