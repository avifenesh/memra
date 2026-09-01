#!/usr/bin/env python3
"""Interleaved cold-prefill A/B on the capped local RTX 5090.

The caller must hold /tmp/memra-5090.lock and set MEMRA_5090_LOCK_HELD=1.
Each scored request gets a fresh server process and cache namespace. A short,
disjoint warmup request removes first-kernel initialization from the scored
request without creating a prefix-cache hit.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from profile_request import PROMPT_TOKENS, frozen_prompt_ids


ROOT = Path(__file__).resolve().parents[2]
PROFILE_REQUEST = Path(__file__).with_name("profile_request.py")
DEFAULT_OUTPUT = Path(__file__).with_name("raw") / "measurement"
BASE_TARGET = Path("/home/avifenesh/.cache/memra-targets/cx-fa3softmax-base-v0812")
CANDIDATE_TARGET = Path("/home/avifenesh/.cache/memra-targets/cx-fa3softmax-candidate")
MODELS = {
    "q27": Path(
        "/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/"
        "Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf"
    ),
    "q35": Path(
        "/data/ai-ml/hf-models/qwen36-35b-moe/"
        "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
    ),
}
TARGETS = {"baseline": BASE_TARGET, "candidate": CANDIDATE_TARGET}
TELEMETRY_QUERY = (
    "timestamp,pstate,temperature.gpu,clocks.current.sm,power.draw,"
    "memory.used,utilization.gpu"
)


@functools.cache
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_raw(path: Path, command: list[str]) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    path.write_bytes(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {shlex.join(command)}; raw={path}"
        )
    return completed


def compute_apps(path: Path) -> str:
    completed = run_raw(
        path,
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    return completed.stdout.decode(errors="replace").strip()


def gpu_snapshot(path: Path) -> None:
    run_raw(
        path,
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,"
            "clocks.current.sm,power.draw,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
    )


def wait_ready(base: str, server: subprocess.Popen[bytes], timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited before ready: rc={server.returncode}")
        try:
            with urllib.request.urlopen(base + "/readyz", timeout=2.0) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(0.2)
    raise RuntimeError(f"server readiness timeout: {last_error}")


def warmup(base: str, model: str, cache_salt: str, output: Path) -> None:
    body = {
        "model": model,
        "prompt_ids": frozen_prompt_ids()[:64],
        "max_ctx": 80,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 3_407,
        "stream": False,
        "cache_salt": cache_salt,
    }
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300.0) as response:
        raw = response.read()
        status = response.status
    output.write_bytes(raw + b"\n")
    if status != 200:
        raise RuntimeError(f"warmup returned HTTP {status}; raw={output}")


def stop_process(process: subprocess.Popen[bytes], label: str) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)
        raise RuntimeError(f"{label} ignored SIGTERM and required SIGKILL")


def parse_ttft(server_log: Path) -> dict[str, float | str | int]:
    matches: list[dict[str, str]] = []
    for line in server_log.read_text(errors="replace").splitlines():
        if not line.startswith("[ttft] "):
            continue
        values: dict[str, str] = {}
        for field in shlex.split(line.removeprefix("[ttft] ")):
            if "=" in field:
                key, value = field.split("=", 1)
                values[key] = value
        if values.get("prompt_tokens") == str(PROMPT_TOKENS):
            matches.append(values)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {PROMPT_TOKENS}-token TTFT trace, found {len(matches)}; raw={server_log}"
        )
    values = matches[0]
    numeric = (
        "prime_wait_ms",
        "prime_ms",
        "decode_wait_ms",
        "sse_handoff_ms",
        "first_sse_byte_ms",
        "total_ms",
    )
    parsed: dict[str, float | str | int] = {
        "request_id": values["id"],
        "model": values["model"],
        "prompt_tokens": int(values["prompt_tokens"]),
        "outcome": values["outcome"],
    }
    for key in numeric:
        parsed[key] = float(values[key])
    return parsed


def parse_request(request_log: Path) -> dict[str, object]:
    lines = [line for line in request_log.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one request JSON row, found {len(lines)}; raw={request_log}")
    result = json.loads(lines[0])
    usage = result.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    assertions = {
        "http_status": result.get("http_status") == 200,
        "done": result.get("done") is True,
        "finish_reason": result.get("finish_reason") == "length",
        "prompt_tokens": usage.get("prompt_tokens") == PROMPT_TOKENS,
        "cached_tokens": details.get("cached_tokens") == 0,
        "completion_tokens": usage.get("completion_tokens") == 60,
    }
    failed = [key for key, passed in assertions.items() if not passed]
    if failed:
        raise RuntimeError(f"request contract failed {failed}; raw={request_log}")
    return result


def server_environment(model_name: str, model_path: Path, port: int) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MEMRA_")
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MEMRA_MODELS": f"{model_name}={model_path}",
            "MEMRA_ADDR": f"127.0.0.1:{port}",
            "MEMRA_COMPAT": "openai",
            "MEMRA_TAG": "cx-fa3softmax-local-ab",
            "MEMRA_SERVE_SPEC": "0",
            "MEMRA_CTX": "8192",
            "MEMRA_PREFIX_CACHE_MB": "1024",
            "MEMRA_PREFIX_DEDUP": "1",
            "MEMRA_REUSE_POOL": "0",
            "MEMRA_AFFINITY": "0",
            "MEMRA_MAX_SESSIONS": "8",
            "MEMRA_TTFT_TRACE": "1",
        }
    )
    return environment


def run_arm(
    output_root: Path,
    sequence: int,
    model_name: str,
    arm: str,
    repetition: int,
    port: int,
) -> dict[str, object]:
    run_dir = output_root / f"{sequence:02d}-{model_name}-{arm}-r{repetition}"
    run_dir.mkdir()
    if compute_apps(run_dir / "compute-apps-before.log"):
        raise RuntimeError(f"GPU not idle before timed arm; raw={run_dir / 'compute-apps-before.log'}")
    gpu_snapshot(run_dir / "gpu-before.log")

    target = TARGETS[arm]
    server_binary = target / "release" / "memra-server"
    model_path = MODELS[model_name]
    base = f"http://127.0.0.1:{port}"
    server_log_path = run_dir / "server.log"
    telemetry_log_path = run_dir / "telemetry.csv"
    request_log_path = run_dir / "request.jsonl"
    environment = server_environment(model_name, model_path, port)

    with telemetry_log_path.open("wb") as telemetry_log, server_log_path.open("wb") as server_log:
        telemetry = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={TELEMETRY_QUERY}",
                "--format=csv,noheader,nounits",
                "--loop-ms=100",
            ],
            stdout=telemetry_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        server = subprocess.Popen(
            [str(server_binary)],
            cwd=ROOT,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        error: BaseException | None = None
        try:
            wait_ready(base, server)
            warmup(
                base,
                model_name,
                f"fa3softmax-warmup-{sequence}-{model_name}-{arm}",
                run_dir / "warmup-response.json",
            )
            command = [
                sys.executable,
                str(PROFILE_REQUEST),
                "--base",
                base,
                "--model",
                model_name,
                "--cache-salt",
                f"fa3softmax-scored-{sequence}-{model_name}-{arm}-r{repetition}",
            ]
            run_raw(request_log_path, command)
        except BaseException as caught:
            error = caught
        finally:
            try:
                stop_process(server, "memra-server")
            except BaseException as caught:
                error = error or caught
            if server.returncode != 0:
                error = error or RuntimeError(
                    f"memra-server exited rc={server.returncode}; raw={server_log_path}"
                )
            try:
                stop_process(telemetry, "nvidia-smi telemetry")
            except BaseException as caught:
                error = error or caught
        if error is not None:
            raise error

    for _ in range(50):
        apps = compute_apps(run_dir / "compute-apps-after.log")
        if not apps:
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(f"GPU process remained after arm; raw={run_dir / 'compute-apps-after.log'}")
    gpu_snapshot(run_dir / "gpu-after.log")

    request = parse_request(request_log_path)
    trace = parse_ttft(server_log_path)
    prime_ms = float(trace["prime_ms"])
    result: dict[str, object] = {
        "schema": "memra.fa3softmax.measurement.v1",
        "sequence": sequence,
        "model": model_name,
        "arm": arm,
        "repetition": repetition,
        "thermal_cap_mhz": [210, 1200],
        "server_sha256": sha256(server_binary),
        "model_sha256": sha256(model_path),
        "prompt_tokens": PROMPT_TOKENS,
        "cached_tokens": 0,
        "prime_ms": prime_ms,
        "prefill_tok_s": PROMPT_TOKENS / (prime_ms / 1000.0),
        "cold_ttft_ms_client": request["ttft_ms"],
        "cold_ttft_ms_server": trace["total_ms"],
        "text_sha256": request["text_sha256"],
        "trace": trace,
        "raw_dir": str(run_dir.relative_to(ROOT)),
    }
    (run_dir / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--port", type=int, default=18_813)
    args = parser.parse_args()
    if os.environ.get("MEMRA_5090_LOCK_HELD") != "1":
        raise SystemExit(
            "refusing timed run: acquire flock /tmp/memra-5090.lock and set "
            "MEMRA_5090_LOCK_HELD=1"
        )
    if args.repetitions < 5:
        raise SystemExit("at least five repetitions per arm are required")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite measurement output: {output}")
    output.mkdir(parents=True)

    for path in [PROFILE_REQUEST, *MODELS.values()]:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    for target in TARGETS.values():
        server = target / "release" / "memra-server"
        if not server.is_file() or not os.access(server, os.X_OK):
            raise SystemExit(f"missing server binary: {server}")

    provenance = {
        "schema": "memra.fa3softmax.measurement-provenance.v1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repetitions": args.repetitions,
        "thermal_cap_mhz": [210, 1200],
        "lock": "/tmp/memra-5090.lock",
        "prompt_tokens": PROMPT_TOKENS,
        "prompt_ids_sha256_canonical_json": hashlib.sha256(
            json.dumps(frozen_prompt_ids(), separators=(",", ":")).encode()
        ).hexdigest(),
        "servers": {
            arm: {
                "path": str(target / "release" / "memra-server"),
                "sha256": sha256(target / "release" / "memra-server"),
            }
            for arm, target in TARGETS.items()
        },
        "models": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in MODELS.items()
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    results_path = output / "measurements.jsonl"
    sequence = 0
    # Strict A/B alternation within each model. Reverse the starting arm for Q35
    # to balance which binary leads a model's thermal window.
    orders = {"q27": ("baseline", "candidate"), "q35": ("candidate", "baseline")}
    for model_name in ("q27", "q35"):
        for repetition in range(1, args.repetitions + 1):
            for arm in orders[model_name]:
                sequence += 1
                print(
                    f"ARM_START sequence={sequence} model={model_name} arm={arm} "
                    f"repetition={repetition}",
                    flush=True,
                )
                result = run_arm(
                    output,
                    sequence,
                    model_name,
                    arm,
                    repetition,
                    args.port,
                )
                with results_path.open("a") as results_file:
                    results_file.write(json.dumps(result, sort_keys=True) + "\n")
                print(
                    f"ARM_DONE sequence={sequence} model={model_name} arm={arm} "
                    f"repetition={repetition} prime_ms={result['prime_ms']:.3f} "
                    f"prefill_tok_s={result['prefill_tok_s']:.3f} "
                    f"cold_ttft_ms={result['cold_ttft_ms_client']:.3f}",
                    flush=True,
                )

    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    for model_name in MODELS:
        hashes = {row["text_sha256"] for row in rows if row["model"] == model_name}
        if len(hashes) != 1:
            raise RuntimeError(
                f"STOP: actual-shape greedy output differs for {model_name}: {sorted(hashes)}"
            )
    provenance["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    provenance["actual_shape_text_identity"] = "PASS"
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"MEASUREMENT_COMPLETE rows={len(rows)} actual_shape_text_identity=PASS", flush=True)


if __name__ == "__main__":
    main()
