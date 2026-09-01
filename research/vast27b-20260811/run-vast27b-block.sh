#!/usr/bin/env bash
# Run one bounded fused-aux verification block on the Vast PRO 6000 box.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
    build|gates|ab|final) ;;
    *) echo "usage: $0 build|gates|ab|final" >&2; exit 2 ;;
esac

export PATH="/root/.cargo/bin:/usr/local/cuda/bin:$PATH"
REMOTE_ROOT=${REMOTE_ROOT:-/workspace/cx-vast27b}
REPO=${REPO:-$REMOTE_ROOT/memra}
RAW_ROOT=${RAW_ROOT:-$REMOTE_ROOT/raw}
STAMP=${VAST27B_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${VAST27B_OUT:-$RAW_ROOT/$MODE-$STAMP}
MODEL=$REMOTE_ROOT/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DRAFT=$REMOTE_ROOT/models/draft-daily-owntrim-nvfp4head-q4blk.gguf
PROMPT=$REPO/research/e2e/prompts/p1-code-short.txt
KC_ALIAS=$REMOTE_ROOT/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
KERNEL=$REPO/target/release/kernel-check
RUN_GEN=$REPO/target/release/run-gen
RUN_SPEC=$REPO/target/release/run-spec
EXPECTED_SOURCE=c58ebd6257334c7b2628ec7367efd4713e8126c1
EXPECTED_MODEL=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
EXPECTED_DRAFT=b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581
EXPECTED_PROMPT=6e00d76296069277dc7717115f977aedcab502b610c95a042c63c30eefdb86b2
SOAK_STOPPED=0
SAMPLER_PID=0

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

compute_apps() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
        compute_apps | sed 's/^/[compute-app] /'
    } >"$path" 2>&1
}

capture_service() {
    local root=$1
    mkdir -p "$root"
    date -u +%FT%TZ >"$root/timestamp.txt"
    pgrep -af 'memra-server|/root/soak.py' >"$root/processes.txt" || true
    curl -fsS --max-time 10 http://127.0.0.1:8002/health >"$root/health.json" 2>"$root/health.err" || true
    curl -fsS --max-time 10 http://127.0.0.1:8002/readyz >"$root/readyz.txt" 2>"$root/readyz.err" || true
    curl -fsS --max-time 10 http://127.0.0.1:8002/v1/models >"$root/models.json" 2>"$root/models.err" || true
    tail -200 /var/log/memra-server.log >"$root/server-tail.log" 2>&1 || true
    tail -30 /var/log/soak.jsonl >"$root/soak-tail.jsonl" 2>&1 || true
    snapshot "$root/gpus.log" service-state
}

verify_stream() {
    local root=$1
    curl --no-buffer -fsS --max-time 300 \
        -H 'Content-Type: application/json' \
        -d '{"model":"stepfun/step-3.7-flash","messages":[{"role":"user","content":"Reply with exactly: VAST27B OK"}],"max_tokens":32,"temperature":0,"seed":3407,"stream":true,"stream_options":{"include_usage":true}}' \
        http://127.0.0.1:8002/v1/chat/completions >"$root/streamed-completion.sse"
    python3 - "$root/streamed-completion.sse" "$root/streamed-completion-summary.json" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
pieces = []
usage = {}
done = False
for line in source.read_text(errors="replace").splitlines():
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        done = True
        continue
    event = json.loads(payload)
    if event.get("error"):
        raise SystemExit(event["error"])
    usage = event.get("usage") or usage
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        pieces.append(
            (delta.get("content") or "")
            + (delta.get("reasoning") or "")
            + (delta.get("reasoning_content") or "")
        )
text = "".join(pieces)
receipt = {
    "done": done,
    "visible_text": text,
    "visible_bytes": len(text.encode()),
    "visible_sha256": hashlib.sha256(text.encode()).hexdigest(),
    "usage": usage,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
assert done and text.strip(), receipt
PY
}

source_preflight() {
    local source
    source=$(git -C "$REPO" rev-parse HEAD)
    echo "source_commit=$source"
    git -C "$REPO" status --short --branch --untracked-files=no
    [[ $source == "$EXPECTED_SOURCE" ]]
}

artifact_preflight() {
    local model_hash draft_hash prompt_hash
    model_hash=$(sha256sum "$MODEL" | awk '{print $1}')
    draft_hash=$(sha256sum "$DRAFT" | awk '{print $1}')
    prompt_hash=$(sha256sum "$PROMPT" | awk '{print $1}')
    echo "model_sha256=$model_hash"
    echo "draft_sha256=$draft_hash"
    echo "prompt_sha256=$prompt_hash"
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$DRAFT" "$PROMPT"
    [[ $model_hash == "$EXPECTED_MODEL" ]]
    [[ $draft_hash == "$EXPECTED_DRAFT" ]]
    [[ $prompt_hash == "$EXPECTED_PROMPT" ]]
}

preflight() {
    source_preflight
    artifact_preflight
}

prepare_kc_alias() {
    local resolved alias_hash
    # At c58ebd62 the DUAL-BATCHED-AUX cell is nested under the historical
    # nvfp4-batched 9B resolver even though the cell only needs any real NVFP4 gate/up pair.
    # Map that resolver to the already-pinned 27B bytes instead of changing the tested binary.
    if [[ -L $KC_ALIAS ]]; then
        resolved=$(readlink -f "$KC_ALIAS")
        [[ $resolved == "$MODEL" ]]
    elif [[ -e $KC_ALIAS ]]; then
        echo "refusing unexpected non-symlink kernel-check alias: $KC_ALIAS" >&2
        return 1
    else
        ln -s "$(basename "$MODEL")" "$KC_ALIAS"
        resolved=$(readlink -f "$KC_ALIAS")
        [[ $resolved == "$MODEL" ]]
    fi
    alias_hash=$(sha256sum "$KC_ALIAS" | awk '{print $1}')
    [[ $alias_hash == "$EXPECTED_MODEL" ]]
    echo "kernel_check_alias=$KC_ALIAS -> $resolved sha256=$alias_hash"
}

run_logged() {
    local label=$1 log=$2 rc
    shift 2
    echo "gate=$label start=$(date -u +%FT%TZ)"
    snapshot "$OUT/$label-before.log" "$label-before"
    set +e
    timeout 14400 "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    snapshot "$OUT/$label-after.log" "$label-after"
    echo "gate=$label end=$(date -u +%FT%TZ) rc=$rc"
    return "$rc"
}

stop_pid() {
    local pid=$1
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    return 1
}

stop_soak() {
    local pids pid
    pids=$(pgrep -f '^python3 /root/soak\.py$' || true)
    [[ -n $pids ]]
    SOAK_STOPPED=1
    for pid in $pids; do stop_pid "$pid"; done
    if pgrep -f '^python3 /root/soak\.py$' >/dev/null; then
        echo "soak process remained after stop" >&2
        return 1
    fi
    echo "soak_stopped=$(date -u +%FT%TZ) pids=${pids//$'\n'/,}"
}

start_sampler() {
    nvidia-smi \
        --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -lms 500 >"$OUT/gpu-500ms.csv" 2>"$OUT/gpu-500ms.err" &
    SAMPLER_PID=$!
}

stop_sampler() {
    if (( SAMPLER_PID > 0 )); then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
        SAMPLER_PID=0
    fi
}

restore_soak_and_verify() {
    local root=$OUT/service-restored before after soak_pid
    mkdir -p "$root"
    before=$(wc -l </var/log/soak.jsonl 2>/dev/null || echo 0)
    if ! pgrep -f '^python3 /root/soak\.py$' >/dev/null; then
        setsid nohup python3 /root/soak.py >>/var/log/soak.jsonl 2>&1 </dev/null &
        soak_pid=$!
        echo "$soak_pid" >"$root/soak-launch.pid"
        sleep 2
        kill -0 "$soak_pid"
    fi
    for _ in $(seq 1 45); do
        after=$(wc -l </var/log/soak.jsonl 2>/dev/null || echo 0)
        (( after > before )) && break
        sleep 2
    done
    after=$(wc -l </var/log/soak.jsonl 2>/dev/null || echo 0)
    (( after > before ))
    curl -fsS --max-time 10 http://127.0.0.1:8002/health >"$root/health.json"
    curl -fsS --max-time 10 http://127.0.0.1:8002/readyz >"$root/readyz.txt"
    curl -fsS --max-time 10 http://127.0.0.1:8002/v1/models >"$root/models.json"
    verify_stream "$root"
    capture_service "$root"
    pgrep -af '^python3 /root/soak\.py$' >"$root/soak-process.txt"
    touch "$root/restored.ok"
    SOAK_STOPPED=0
    echo "service_and_soak_verified=$(date -u +%FT%TZ)"
}

on_exit() {
    local rc=$? restore_rc=0
    trap - EXIT INT TERM
    set +e
    stop_sampler
    if (( SOAK_STOPPED )); then restore_soak_and_verify; restore_rc=$?; fi
    if (( rc == 0 && restore_rc != 0 )); then rc=$restore_rc; fi
    echo "block=$MODE out=$OUT exit=$rc restore_exit=$restore_rc done=$(date -u +%FT%TZ)"
    exit "$rc"
}
trap on_exit EXIT INT TERM

reduce_ab() {
    python3 - "$OUT" <<'PY'
import hashlib
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("r*-*-*.log")):
    match = re.fullmatch(r"r(\d+)-([AB])-(base|auxdual)\.log", path.name)
    if not match:
        continue
    text = path.read_text(errors="replace")
    speed = re.search(r"\[generate_spec K=3\].*?= ([0-9.]+) tok/s", text)
    tokens = re.search(r"^  tokens: (\[.*\])$", text, re.MULTILINE)
    stats = re.search(r"^\[spec-stats\] (.*)$", text, re.MULTILINE)
    acceptance = re.search(r"^  acceptance: (.*)$", text, re.MULTILINE)
    if not all((speed, tokens, stats, acceptance)):
        raise SystemExit(f"missing measurement field in {path}")
    rows.append({
        "rep": int(match.group(1)),
        "arm": match.group(2),
        "label": match.group(3),
        "spec_tok_s": float(speed.group(1)),
        "tokens_sha256": hashlib.sha256(tokens.group(1).encode()).hexdigest(),
        "spec_stats": stats.group(1),
        "acceptance": acceptance.group(1),
        "self_consistency_pass": "=== SELF-CONSISTENCY PASS ===" in text,
        "path": path.name,
    })
if len(rows) != 10:
    raise SystemExit(f"expected 10 scored runs, found {len(rows)}")
arms = {arm: [r["spec_tok_s"] for r in rows if r["arm"] == arm] for arm in "AB"}
medians = {arm: statistics.median(values) for arm, values in arms.items()}
paired = []
for rep in range(1, 6):
    values = {r["arm"]: r["spec_tok_s"] for r in rows if r["rep"] == rep}
    paired.append({"rep": rep, "A": values["A"], "B": values["B"],
                   "gain_pct": (values["B"] / values["A"] - 1.0) * 100.0})
summary = {
    "contract": {"A": "MEMRA_NVFP4_AUX_DUAL=0", "B": "default-on",
                 "order": "A,B,B,A,A,B,B,A,A,B", "N_per_arm": 5},
    "rows": rows,
    "median_A_tok_s": medians["A"],
    "median_B_tok_s": medians["B"],
    "median_gain_pct": (medians["B"] / medians["A"] - 1.0) * 100.0,
    "paired": paired,
    "all_self_consistent": all(r["self_consistency_pass"] for r in rows),
    "distinct_token_hashes": sorted({r["tokens_sha256"] for r in rows}),
    "distinct_spec_stats": sorted({r["spec_stats"] for r in rows}),
    "distinct_acceptance": sorted({r["acceptance"] for r in rows}),
}
(root / "ab-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["all_self_consistent"] or len(summary["distinct_token_hashes"]) != 1:
    raise SystemExit("A/B exactness failure")
PY
}

echo "block=$MODE start=$(date -u +%FT%TZ) out=$OUT"

case "$MODE" in
    build)
        source_preflight
        capture_service "$OUT/service-before"
        {
            rustc --version
            cargo --version
            nvcc --version
        } >"$OUT/toolchain.txt" 2>&1
        snapshot "$OUT/gpus-before.log" build-before
        run_logged cargo-build "$OUT/cargo-build.log" \
            cargo build --manifest-path "$REPO/Cargo.toml" --release \
                --bin kernel-check --bin run-gen --bin run-spec
        sha256sum "$KERNEL" "$RUN_GEN" "$RUN_SPEC" >"$OUT/binary-sha256.txt"
        snapshot "$OUT/gpus-after.log" build-after
        capture_service "$OUT/service-after"
        ;;
    gates)
        preflight
        prepare_kc_alias
        test -x "$KERNEL"
        test -x "$RUN_GEN"
        test -x "$RUN_SPEC"
        exec 9>/tmp/memra-gpu.lock
        flock -w 1200 9
        capture_service "$OUT/service-before"
        sha256sum "$KERNEL" "$RUN_GEN" "$RUN_SPEC" >"$OUT/binary-sha256.txt"
        run_logged kernel-check "$OUT/kernel-check.log" env \
            -u MEMRA_NVFP4_AUX_DUAL -u MEMRA_PROFILE_SPEC \
            CUDA_VISIBLE_DEVICES=0 MEMRA_KC_MODELS_DIR="$REMOTE_ROOT/models" \
            "$KERNEL" "$MODEL" --require-manifest "$REPO/tools/kernel-check-27b.cells"
        grep -q '^ALL GREEN ([0-9][0-9]* cells, [0-9][0-9]* skipped)$' "$OUT/kernel-check.log"
        run_logged run-gen "$OUT/run-gen.log" env \
            -u MEMRA_NVFP4_AUX_DUAL -u MEMRA_PROFILE_SPEC \
            CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 \
            "$RUN_GEN" "$MODEL"
        [[ $(grep -cE 'argmax=.* MATCH$' "$OUT/run-gen.log") -ge 2 ]]
        if grep -q 'MISMATCH' "$OUT/run-gen.log"; then
            echo "run-gen reported MISMATCH" >&2
            exit 1
        fi
        run_logged run-spec "$OUT/run-spec.log" env \
            -u MEMRA_NVFP4_AUX_DUAL -u MEMRA_SPEC_K -u MEMRA_PROFILE_SPEC \
            CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=64 \
            MEMRA_PROMPT_FILE="$PROMPT" MEMRA_CHAT=1 "$RUN_SPEC" "$MODEL"
        [[ $(grep -c 'self-consistency: PASS' "$OUT/run-spec.log") -eq 8 ]]
        if grep -q 'SELF-CONSISTENCY FAIL' "$OUT/run-spec.log"; then
            echo "run-spec reported SELF-CONSISTENCY FAIL" >&2
            exit 1
        fi
        capture_service "$OUT/service-after"
        echo "gates_result=PASS"
        ;;
    ab)
        preflight
        test -x "$RUN_SPEC"
        exec 9>/tmp/memra-gpu.lock
        flock -w 1200 9
        capture_service "$OUT/service-before"
        stop_soak
        for _ in $(seq 1 60); do
            curl -fsS --max-time 5 http://127.0.0.1:8002/health | grep -q '"phase":"idle"' && break
            sleep 1
        done
        curl -fsS --max-time 5 http://127.0.0.1:8002/health | grep -q '"phase":"idle"'
        start_sampler
        echo "contract=K3_NGEN64_CHAT1 order=A,B,B,A,A,B,B,A,A,B server=resident-idle soak=stopped"
        common=(
            CUDA_VISIBLE_DEVICES=0
            MEMRA_MTP_DRAFT="$DRAFT"
            MEMRA_SPEC_K=3
            MEMRA_NGEN=64
            MEMRA_PROMPT_FILE="$PROMPT"
            MEMRA_CHAT=1
            MEMRA_SPEC_STATS=1
        )
        run_arm() {
            local rep=$1 arm=$2 label log
            if [[ $arm == A ]]; then label=base; else label=auxdual; fi
            log=$OUT/r$(printf '%02d' "$rep")-$arm-$label.log
            if [[ $arm == A ]]; then
                run_logged "r$(printf '%02d' "$rep")-$arm" "$log" env \
                    -u MEMRA_NVFP4_AUX_DUAL -u MEMRA_DEBUG -u MEMRA_PROFILE_SPEC \
                    "${common[@]}" MEMRA_NVFP4_AUX_DUAL=0 "$RUN_SPEC" "$MODEL"
            else
                run_logged "r$(printf '%02d' "$rep")-$arm" "$log" env \
                    -u MEMRA_NVFP4_AUX_DUAL -u MEMRA_DEBUG -u MEMRA_PROFILE_SPEC \
                    "${common[@]}" "$RUN_SPEC" "$MODEL"
            fi
        }
        run_arm 1 A
        run_arm 1 B
        run_arm 2 B
        run_arm 2 A
        run_arm 3 A
        run_arm 3 B
        run_arm 4 B
        run_arm 4 A
        run_arm 5 A
        run_arm 5 B
        stop_sampler
        reduce_ab
        restore_soak_and_verify
        echo "ab_result=PASS"
        ;;
    final)
        preflight
        capture_service "$OUT/service-before"
        restore_soak_and_verify
        echo "final_service_result=PASS"
        ;;
esac
