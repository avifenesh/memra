"""Serialize a bounded, goal-linked native experiment on its research GPU."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit import audit_run, save, sha


class Runner:
    def __init__(self, repo, models, binaries, source, out):
        self.repo, self.models, self.binaries, self.out = repo, models, binaries, out
        self.source = json.loads(source.read_text())
        self.lock = open("/tmp/memra-gpu.lock", "a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base = repo / "research/mtp-context-depth-20260921"
        sys.path.insert(0, str(base))
        from audit_context import audit_run as context_auditor
        from loop_audit import loop_candidate
        from run_study import gpu_processes, metrics_from_log
        self.context_auditor, self.loop_candidate = context_auditor, loop_candidate
        self.gpu_processes, self.metrics_from_log = gpu_processes, metrics_from_log
        if gpu_processes():
            raise RuntimeError("research GPU has another workload; no process evicted")
        self.environment = os.environ.copy()
        settings = sorted(key for key in self.environment if key.startswith("MEMRA_"))
        if settings:
            raise ValueError("unregistered Memra settings: " + ", ".join(settings))
        for name, expected in self.source["binaries"].items():
            if sha(binaries / name) != expected:
                raise ValueError("native binary differs from the build artifact")
        binding = json.loads((repo / "PREFIX-ROUTING-SOURCE.json").read_text())
        for name, expected in binding["files"].items():
            if sha(repo / name) != expected:
                raise ValueError("prepared source differs from the compiled experiment")
        artifacts = {}
        for family in ("qwen", "gemma"):
            artifacts[family] = json.loads((models / family / "artifacts.lock.json").read_text())
            pinned = json.loads((base / f"{family}-artifacts.lock.json").read_text())
            if artifacts[family] != pinned:
                raise ValueError("model lock differs from the pinned runtime source")
            for row in artifacts[family]:
                path = models / family / row["local_file"]
                if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                    raise ValueError("model artifact differs from its lock")
        save(out / "identity.json", {
            "source": self.source, "artifacts": artifacts,
            "runtime_binding_sha256": sha(repo / "PREFIX-ROUTING-SOURCE.json"),
            "runner_sha256": sha(Path(__file__)),
            "context_auditor_sha256": sha(base / "audit_context.py"),
            "loop_auditor_sha256": sha(base / "loop_audit.py"),
            "gpu": subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,power.limit,power.max_limit",
                "--format=csv"], text=True),
            "clock": "native request: tokenization, forecast, cache allocation, prefill, sampled generation, detokenization; excludes model load, warmup and receipt I/O",
            "cache": "fresh native cache and one user message for every request",
            "heads": "full vocabulary, unchanged in every arm",
        })

    def close(self):
        self.lock.close()

    def run(self, family, phase, label, entry, workload, arm, max_new, gate=False, seed=None):
        seed = entry["seed"] if seed is None else seed
        if self.gpu_processes():
            raise RuntimeError("GPU contention before the next arm; stopped without eviction")
        if sha(workload) != entry["sha256"]:
            raise ValueError("workload changed after registration")
        parent = self.out / family / phase
        parent.mkdir(parents=True, exist_ok=True)
        root = parent / label
        if root.exists():
            raise ValueError("refusing to overwrite an earlier run")
        binary = self.binaries / f"{family}-prefix-study"
        target = self.models / family / "target.gguf"
        draft = str(self.models / family / "assistant.gguf") if family == "gemma" else "embedded"
        cmd = [str(binary), str(target), draft, str(workload), str(root),
               arm, str(seed), str(max_new), "32768", "0" if gate else "0.7"]
        if gate:
            cmd.append("gate")
        environment = {
            **self.environment, "MEMRA_SPEC_ADAPT": "1", "MEMRA_SPEC_ADAPT_FLOOR": "1",
            "MEMRA_SPEC_CAPMAX": "7" if family == "qwen" else "5",
            "MEMRA_SPEC_PMIN": "0", "MEMRA_SPEC_PMIN_INROUND": "0",
        }
        save(parent / f"{label}.command.json", {
            "argv": cmd, "settings": {k: v for k, v in environment.items() if k.startswith("MEMRA_")},
            "seed": seed, "max_new": max_new, "gate": gate, "arm": arm,
            "workload_sha256": entry["sha256"], "binary_sha256": sha(binary),
        })
        print(json.dumps({"started": f"{family}/{phase}/{label}", "arm": arm}), flush=True)
        began = time.monotonic()
        log_path = parent / f"{label}.log"
        contamination, child, monitor = None, None, None
        result = None
        try:
            with log_path.open("w") as log, (parent / f"{label}.gpu.csv").open("w") as telemetry:
                monitor = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm",
                    "--format=csv", "--loop-ms=250"], stdout=telemetry, stderr=subprocess.STDOUT)
                child = subprocess.Popen(cmd, env=environment, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    try:
                        result = child.wait(timeout=2)
                        break
                    except subprocess.TimeoutExpired:
                        processes = self.gpu_processes().splitlines()
                        if len(processes) > 1:
                            contamination = "\n".join(processes)
                            raise RuntimeError("concurrent GPU processes; current arm rejected")
                        if time.monotonic() - began > 2400:
                            raise TimeoutError("native arm exceeded its bounded runtime")
        finally:
            for process in (child, monitor):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            save(parent / f"{label}.exit.json", {
                "returncode": result, "process_seconds": time.monotonic() - began,
                "contamination": contamination,
            })
        if result:
            raise RuntimeError(f"native arm exited {result}; inspect {log_path}")
        text = log_path.read_text()
        vocabulary = 248320 if family == "qwen" else 262144
        if f"full_vocab={vocabulary}" not in text:
            raise ValueError("full vocabulary path engagement missing")
        if family == "qwen" and f"draft_vocab={vocabulary} mtp=embedded" not in text:
            raise ValueError("full embedded MTP path engagement missing")
        metrics = self.metrics_from_log(text)[arm]
        audit = audit_run(root, entry, arm, seed, metrics, self.context_auditor, self.loop_candidate)
        result = {"path": str(root.relative_to(self.out)), "family": family, "phase": phase,
                  "label": label, "arm": arm, "seed": seed, "max_new": max_new,
                  "gate": gate, "metrics": metrics, **audit}
        save(parent / f"{label}.audit.json", result)
        print(json.dumps({"completed": result["path"], "tokens": metrics["tokens"],
                          "format_covered": sum(r["format"]["requested_format_covered"]
                                                for r in result["requests"])}), flush=True)
        return result
