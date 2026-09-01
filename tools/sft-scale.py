#!/usr/bin/env python3
"""Scale the verified DeepSeek SFT trace corpus in resumable, committed batches.

This orchestrator reuses tools/sft-gen.py for isolated opencode execution, then
normalizes each exported session into the K3 trace contract and runs the
darklanes validator, outcome verifier, and Qwen converter before admission.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
GENERATOR_PATH = ROOT / "tools" / "sft-gen.py"
DEFAULT_CORPUS_REPO = Path("~/projects/sft-traces")
DEFAULT_PIPELINE_DIR = Path("~/projects/darklanes/sft-pipeline")
DEFAULT_PROGRESS = ROOT / "research" / "cx-sft-20260808" / "PROGRESS.md"
DEFAULT_STEERING = Path("~/.lanectl/inbox/cx-sft-scale.md")
PROOF_CORPUS = DEFAULT_CORPUS_REPO / "corpus" / "deepseek-v4-flash-20260808.jsonl"
PROOF_FAILURES = (
    DEFAULT_CORPUS_REPO / "corpus" / "deepseek-v4-flash-20260808-failures.jsonl"
)

MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
TASK_KINDS = ("bug_fix", "refactor", "test_writing", "explain")
SCALE_TASK_RE = re.compile(
    r"^scale-(bug-fix|refactor|test-writing|explain)-([a-z0-9-]+)-(\d{6})$"
)
PROGRESS_START = "<!-- SFT-SCALE:START -->"
PROGRESS_END = "<!-- SFT-SCALE:END -->"
UNKNOWN_FAILURE_RESERVE_USD = 0.10
ACTIVE_CALL_RESERVE_USD = 0.50
COORDINATION_BRANCHES = ("lane/cx-sft-tag",)

ANSWER_PATTERNS = {
    "cache-economics": (
        r"(?is)(?=.*\b(?:prompt|cached)\b)"
        r"(?=.*\b(?:computed|billed)\b)"
        r"(?=.*\b(?:invariants?|must|equal)\b).*"
    ),
    "fleet-deltas": (
        r"(?is)(?=.*\b(?:cumulative|counter)\b)"
        r"(?=.*\b(?:restart|reset)\b)"
        r"(?=.*\b(?:delta|difference)\b).*"
    ),
    "acceptance-parser": (
        r"(?is)(?=.*\b(?:percent|percentage)\b)"
        r"(?=.*\b(?:fraction|fractional|normalize|normalized|divided)\b)"
        r"(?=.*\b(?:error|evidence|tail)\b).*"
    ),
    "perf-markers": (
        r"(?is)(?=.*\b(?:marker|block)\b)"
        r"(?=.*\b(?:duplicate|single|one)\b)"
        r"(?=.*\b(?:backslash|callable|replacement)\b).*"
    ),
    "batch-divergence": (
        r"(?is)(?=.*\b(?:prefix|length)\b)"
        r"(?=.*\b(?:divergence|mismatch)\b)"
        r"(?=.*\b(?:hash(?:es|ing)?|fingerprint|tail|evidence|context)\b).*"
    ),
    "tier-plan": (
        r"(?is)(?=.*\b(?:pruned|active)\b)"
        r"(?=.*\b(?:router|original|position)\b)"
        r"(?=.*\b(?:Q2_K|Q3_K|NVFP4|BF16)\b).*"
    ),
}

REVIEW_LENSES = (
    "Preserve every public function signature and returned field.",
    "Keep exception text and validation order stable unless the task requires otherwise.",
    "Avoid new dependencies, classes, or framework-style abstractions.",
    "Keep the change local to the demonstrated behavior.",
    "Preserve input immutability and deterministic output ordering.",
    "Prefer a direct invariant check over a broad rewrite.",
    "Do not weaken an existing assertion to make the suite pass.",
    "Keep error evidence explicit enough for a future regression report.",
    "Retain the existing standard-library-only implementation.",
    "Treat boundary values as part of the public behavior.",
    "Keep helper extraction proportional to this small module.",
    "Preserve the fixture's current naming and data model.",
)

VERIFICATION_LENSES = (
    "In the final response, name the exact behavior the tests establish.",
    "Report the unittest command and the number of tests that passed.",
    "Explain why the existing green-path example remains valid.",
    "Call out the boundary or failure case covered by the change.",
    "Mention any file intentionally left unchanged.",
    "State the invariant in terms of concrete inputs and outputs.",
    "Keep the final report factual and limited to observed test evidence.",
    "Distinguish the production-code change from test-only coverage.",
    "Note whether the task changed behavior or only structure.",
    "Identify the smallest regression that the new check prevents.",
    "Describe the verification result without claiming broader coverage.",
    "Use the local fixture terminology in the final summary.",
)

IMPLEMENTATION_LENSES = (
    "Use straightforward control flow that another maintainer can audit quickly.",
    "Keep all arithmetic and comparisons explicit.",
    "Avoid catch-all exception handling.",
    "Do not add compatibility fallbacks for unspecified inputs.",
    "Keep parsing and validation fail-closed.",
    "Prefer small named helpers only when they clarify a real boundary.",
    "Do not mutate caller-owned mappings or sequences.",
    "Keep generated evidence byte-stable where the existing contract requires it.",
)

TOOL_DESCRIPTIONS = {
    "bash": "Run a shell command inside the isolated task repository.",
    "shell": "Run a shell command inside the isolated task repository.",
    "run_command": "Run a command inside the isolated task repository.",
    "read": "Read a file from the isolated task repository.",
    "write": "Write a file in the isolated task repository.",
    "edit": "Apply a targeted edit to a file in the isolated task repository.",
    "glob": "Find files in the isolated task repository.",
    "grep": "Search text in the isolated task repository.",
    "list": "List files in the isolated task repository.",
    "apply_patch": "Apply a patch inside the isolated task repository.",
}


def load_generator():
    spec = importlib.util.spec_from_file_location("sft_gen_scale_base", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GEN = load_generator()


@dataclass(frozen=True)
class Paths:
    corpus_repo: Path
    pipeline_dir: Path
    progress: Path
    steering: Path
    work_root: Path

    @property
    def raw(self) -> Path:
        return self.corpus_repo / "raw"

    @property
    def verified(self) -> Path:
        return self.corpus_repo / "verified"

    @property
    def converted(self) -> Path:
        return self.corpus_repo / "converted"

    @property
    def rejects(self) -> Path:
        return self.corpus_repo / "rejects"

    @property
    def manifests(self) -> Path:
        return self.corpus_repo / "manifests"

    @property
    def stats(self) -> Path:
        return self.corpus_repo / "stats"


@dataclass
class AttemptResult:
    task_id: str
    template: Any
    record: dict[str, Any] | None
    workspace: Path | None
    work_parent: Path
    failure: dict[str, Any] | None


@dataclass(frozen=True)
class CorpusSnapshot:
    verified: int
    raw: int
    rejected: int
    generation_failures: int
    known_cost_usd: float
    unknown_cost_reserve_usd: float
    kinds: Counter
    scenarios: Counter
    max_sequence: int

    @property
    def attempts(self) -> int:
        return self.raw + self.generation_failures

    @property
    def budget_accounted_usd(self) -> float:
        return self.known_cost_usd + self.unknown_cost_reserve_usd

    @property
    def keep_rate(self) -> float:
        denominator = self.verified + self.rejected
        return self.verified / denominator if denominator else 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(json_dumps(value) + "\n")
        target.flush()
        os.fsync(target.fileno())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno}: JSONL row must be an object")
            yield value


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def require_clean_repo(repo: Path, allowed: set[str] | None = None) -> None:
    result = run(["git", "status", "--short"], cwd=repo, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git status failed in {repo}")
    dirty = []
    for line in result.stdout.splitlines():
        relative = line[3:]
        if allowed is None or relative not in allowed:
            dirty.append(line)
    if dirty:
        raise RuntimeError(
            f"unexpected dirty files in {repo}:\n" + "\n".join(dirty)
        )


def git_head(repo: Path) -> str:
    result = run(["git", "rev-parse", "HEAD"], cwd=repo, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git rev-parse failed in {repo}")
    return result.stdout.strip()


def sync_coordination_branches(repo: Path) -> None:
    for branch in COORDINATION_BRANCHES:
        reference = f"refs/heads/{branch}"
        exists = run(
            ["git", "show-ref", "--verify", "--quiet", reference],
            cwd=repo,
            timeout=30,
        )
        if exists.returncode == 1:
            continue
        if exists.returncode != 0:
            raise RuntimeError(
                exists.stderr.strip() or f"cannot inspect coordination branch {branch}"
            )

        incorporated = run(
            ["git", "merge-base", "--is-ancestor", branch, "HEAD"],
            cwd=repo,
            timeout=30,
        )
        if incorporated.returncode == 0:
            continue
        if incorporated.returncode != 1:
            raise RuntimeError(
                incorporated.stderr.strip()
                or f"cannot compare coordination branch {branch}"
            )

        fast_forwardable = run(
            ["git", "merge-base", "--is-ancestor", "HEAD", branch],
            cwd=repo,
            timeout=30,
        )
        if fast_forwardable.returncode == 0:
            update = run(
                ["git", "merge", "--ff-only", branch],
                cwd=repo,
                timeout=120,
            )
            if update.returncode != 0:
                raise RuntimeError(
                    update.stderr.strip()
                    or f"cannot fast-forward from coordination branch {branch}"
                )
            continue
        if fast_forwardable.returncode != 1:
            raise RuntimeError(
                fast_forwardable.stderr.strip()
                or f"cannot compare coordination branch {branch}"
            )
        raise RuntimeError(
            f"coordination branch {branch} diverged from HEAD; "
            "wait for it to rebase before committing the next corpus batch"
        )


def git_commit(paths: Paths, files: list[Path], message: str) -> str:
    relative = [str(path.relative_to(paths.corpus_repo)) for path in files if path.exists()]
    if not relative:
        return git_head(paths.corpus_repo)
    sync_coordination_branches(paths.corpus_repo)
    add = run(["git", "add", "--", *relative], cwd=paths.corpus_repo, timeout=60)
    if add.returncode != 0:
        raise RuntimeError(add.stderr.strip() or "git add failed")
    staged = run(["git", "diff", "--cached", "--quiet"], cwd=paths.corpus_repo)
    if staged.returncode == 0:
        return git_head(paths.corpus_repo)
    if staged.returncode != 1:
        raise RuntimeError(staged.stderr.strip() or "git diff --cached failed")
    commit = run(["git", "commit", "-m", message], cwd=paths.corpus_repo, timeout=180)
    if commit.returncode != 0:
        raise RuntimeError(commit.stderr.strip() or "corpus git commit failed")
    return git_head(paths.corpus_repo)


def scale_environment() -> dict[str, str]:
    env = GEN.minimal_environment()
    env.update(
        {
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_DISABLE_MODELS_FETCH": "true",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
            "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "8192",
        }
    )
    return env


def isolated_opencode_environment(
    base_env: dict[str, str],
    work_parent: Path,
) -> dict[str, str]:
    source = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    try:
        credentials = json.loads(source.read_text(encoding="utf-8"))
        openrouter = credentials["openrouter"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("cannot isolate the OpenRouter opencode credential") from exc
    if not isinstance(openrouter, dict) or openrouter.get("type") != "api":
        raise RuntimeError("unexpected OpenRouter opencode credential shape")

    data_home = work_parent / "xdg-data"
    opencode_data = data_home / "opencode"
    opencode_data.mkdir(parents=True, mode=0o700)
    os.chmod(opencode_data, 0o700)
    auth_path = opencode_data / "auth.json"
    auth_path.write_text(
        json.dumps({"openrouter": openrouter}),
        encoding="utf-8",
    )
    os.chmod(auth_path, 0o600)

    cache_home = work_parent / "xdg-cache"
    cache_home.mkdir(parents=True, mode=0o700)
    env = dict(base_env)
    env["XDG_DATA_HOME"] = str(data_home)
    env["XDG_CACHE_HOME"] = str(cache_home)
    return env


def provider_receipt(config: Path) -> dict[str, Any]:
    return GEN.validate_opencode_config(config, MODEL)


def tool_schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS.get(
                name, "Tool invoked by opencode inside the isolated task repository."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": True,
            },
        },
    }


def tool_output(part: dict[str, Any]) -> str:
    state = part.get("state") if isinstance(part.get("state"), dict) else {}
    output = state.get("output")
    if isinstance(output, str) and output:
        return output
    evidence = {
        key: state[key]
        for key in ("status", "error", "metadata", "title")
        if key in state
    }
    return json.dumps(evidence or {"status": "completed", "output": ""}, ensure_ascii=False)


def assistant_parts(message: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    text = []
    reasoning = []
    tools = []
    for part in message.get("parts", []):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        value = part.get("text")
        if kind == "text" and isinstance(value, str) and value.strip():
            text.append(value)
        elif kind == "reasoning" and isinstance(value, str) and value.strip():
            reasoning.append(value)
        elif kind == "tool":
            tools.append(part)
    return text, reasoning, tools


def unique_call_id(raw: str | None, used: set[str], index: int) -> str:
    base = raw if isinstance(raw, str) and raw.strip() else f"call_{index}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def session_messages(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session = record["session_export"]
    messages = [{"role": "user", "content": record["prompt"]}]
    pending_text: list[str] = []
    pending_reasoning: list[str] = []
    tool_names: set[str] = set()
    used_call_ids: set[str] = set()
    call_index = 0

    for exported in session.get("messages", []):
        if not isinstance(exported, dict):
            continue
        info = exported.get("info") if isinstance(exported.get("info"), dict) else {}
        if info.get("role") != "assistant":
            continue
        text, reasoning, tools = assistant_parts(exported)
        pending_text.extend(text)
        pending_reasoning.extend(reasoning)
        if not tools:
            continue

        calls = []
        tool_turns = []
        for part in tools:
            call_index += 1
            name = str(part.get("tool") or "unknown_tool")
            tool_names.add(name)
            call_id = unique_call_id(part.get("callID"), used_call_ids, call_index)
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            arguments = state.get("input") if isinstance(state.get("input"), dict) else {}
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
            tool_turns.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": tool_output(part),
                }
            )
        assistant = {
            "role": "assistant",
            "content": "\n\n".join(pending_text),
            "tool_calls": calls,
        }
        if pending_reasoning:
            assistant["reasoning_content"] = "\n\n".join(pending_reasoning)
        messages.append(assistant)
        messages.extend(tool_turns)
        pending_text = []
        pending_reasoning = []

    if not pending_text:
        raise ValueError(f"{record['template_id']}: exported session has no final assistant text")
    final = {
        "role": "assistant",
        "content": "\n\n".join(pending_text),
    }
    if pending_reasoning:
        final["reasoning_content"] = "\n\n".join(pending_reasoning)
    messages.append(final)
    return messages, [tool_schema(name) for name in sorted(tool_names)]


def fixture_manifest(template: Any) -> dict[str, Any]:
    return {
        "files": dict(template.files),
        "agents_md": GEN.WORKSPACE_INSTRUCTIONS,
        "file_sha256": {
            path: GEN.sha256_text(content) for path, content in template.files.items()
        },
    }


def k3_record(
    record: dict[str, Any],
    template: Any,
    *,
    workspace: Path | None,
    proof: bool = False,
) -> dict[str, Any]:
    messages, tools = session_messages(record)
    verification = record["verification"]
    usage = record["usage"]
    task_id = (
        f"proof-{record['template_id']}" if proof else record["template_id"]
    )
    meta = {
        "model": record["model"],
        "ts": record["timestamp"],
        "turns": len(messages),
        "sub_session": record["opencode"]["session_id"],
        "task_kind": record["task_kind"],
        "scenario": record["scenario"],
        "provider_policy": record["provider_policy"],
        "usage": usage,
        "content_sha256": record["content_sha256"],
        "source": record["source"],
        "opencode": record["opencode"],
        "workspace_verification": verification,
        "fixture": fixture_manifest(template),
        "proof_migration": proof,
    }

    if proof:
        outcome = {
            "verified": True,
            "method": "manual",
            "detail": {
                "source": "proof batch workspace verification",
                "passed": verification["passed"],
                "initial_tests": verification["initial_tests"],
                "final_tests": verification["final_tests"],
                "changed_files": verification["changed_files"],
                "git_diff": verification["git_diff"],
            },
        }
        verify = None
    elif template.category == "explain":
        outcome = {
            "verified": False,
            "method": "answer_check",
            "detail": "pending pipeline verification",
        }
        verify = {
            "type": "answer",
            "source": "last_assistant",
            "pattern": ANSWER_PATTERNS[template.scenario],
        }
    else:
        if workspace is None:
            raise ValueError(f"{task_id}: code task requires a live verification workspace")
        outcome = {
            "verified": False,
            "method": "tests_pass",
            "detail": "pending pipeline verification",
        }
        verify = {
            "type": "patch",
            "repo": str(workspace),
            "rev": git_head(workspace),
            "patch": verification["git_diff"],
            "test_cmd": [sys.executable, "-m", "unittest", "-v"],
            "timeout": 120,
        }

    trace = {
        "task_id": task_id,
        "seed_source": "memra-proof-fixture" if proof else "memra-scale-fixture",
        "tools": tools,
        "messages": messages,
        "outcome": outcome,
        "meta": meta,
    }
    if verify is not None:
        trace["verify"] = verify
    return trace


def variant_prompt(base: Any, ordinal: int) -> str:
    review = REVIEW_LENSES[ordinal % len(REVIEW_LENSES)]
    verification = VERIFICATION_LENSES[
        (ordinal // len(REVIEW_LENSES)) % len(VERIFICATION_LENSES)
    ]
    implementation = IMPLEMENTATION_LENSES[
        (
            ordinal
            // (len(REVIEW_LENSES) * len(VERIFICATION_LENSES))
        )
        % len(IMPLEMENTATION_LENSES)
    ]
    return (
        f"{base.prompt}\n\n"
        "Additional review constraints for this instance:\n"
        f"- {review}\n"
        f"- {verification}\n"
        f"- {implementation}"
    )


def base_template_index() -> dict[tuple[str, str], Any]:
    return {(template.category, template.scenario): template for template in GEN.build_templates()}


def task_kind_slug(kind: str) -> str:
    return kind.replace("_", "-")


def task_id(kind: str, scenario: str, sequence: int) -> str:
    return f"scale-{task_kind_slug(kind)}-{scenario}-{sequence:06d}"


def scale_template(base: Any, sequence: int, ordinal: int) -> Any:
    return replace(
        base,
        template_id=task_id(base.category, base.scenario, sequence),
        prompt=variant_prompt(base, ordinal),
    )


def trace_kind(trace: dict[str, Any]) -> str | None:
    meta = trace.get("meta") if isinstance(trace.get("meta"), dict) else {}
    value = meta.get("task_kind")
    return value if value in TASK_KINDS else None


def trace_scenario(trace: dict[str, Any]) -> str | None:
    meta = trace.get("meta") if isinstance(trace.get("meta"), dict) else {}
    value = meta.get("scenario")
    return value if isinstance(value, str) else None


def trace_cost(trace: dict[str, Any]) -> float:
    meta = trace.get("meta") if isinstance(trace.get("meta"), dict) else {}
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    return float(usage.get("cost_usd", 0.0) or 0.0)


def trace_session_id(trace: dict[str, Any]) -> str | None:
    meta = trace.get("meta") if isinstance(trace.get("meta"), dict) else {}
    value = meta.get("sub_session")
    return value if isinstance(value, str) and value else None


def proof_spend() -> tuple[float, set[str]]:
    total = 0.0
    sessions: set[str] = set()
    proof_path = PROOF_CORPUS.expanduser()
    for row in read_jsonl(proof_path):
        if row.get("record_type") != "trace":
            continue
        session = row.get("opencode", {}).get("session_id")
        if isinstance(session, str):
            sessions.add(session)
        total += float(row.get("usage", {}).get("cost_usd", 0.0) or 0.0)
    for failure in read_jsonl(PROOF_FAILURES.expanduser()):
        text = failure.get("stdout")
        if not isinstance(text, str):
            continue
        match = re.search(r'"cost"\s*:\s*([0-9.]+)', text)
        if match:
            total += float(match.group(1))
    return total, sessions


def corpus_snapshot(paths: Paths) -> CorpusSnapshot:
    verified = raw = rejected = generation_failures = 0
    known_cost, seen_sessions = proof_spend()
    unknown_failures = 0
    kinds: Counter = Counter()
    scenarios: Counter = Counter()
    max_sequence = 0

    for path in sorted(paths.raw.glob("*.jsonl")):
        for trace in read_jsonl(path):
            raw += 1
            session = trace_session_id(trace)
            if session and session not in seen_sessions:
                known_cost += trace_cost(trace)
                seen_sessions.add(session)
            match = SCALE_TASK_RE.match(str(trace.get("task_id", "")))
            if match:
                max_sequence = max(max_sequence, int(match.group(3)))

    for path in sorted(paths.verified.glob("*.jsonl")):
        for trace in read_jsonl(path):
            verified += 1
            kind = trace_kind(trace)
            scenario = trace_scenario(trace)
            if kind:
                kinds[kind] += 1
            if kind and scenario:
                scenarios[(kind, scenario)] += 1

    for path in sorted(paths.rejects.glob("*.jsonl")):
        for row in read_jsonl(path):
            rejected += 1
            failure = row.get("generation_failure")
            if failure is not None:
                generation_failures += 1
                usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
                cost = float(usage.get("cost_usd", 0.0) or 0.0)
                session = row.get("session_id")
                if cost and row.get("seed_source") != "memra-proof-fixture":
                    cost_key = (
                        session
                        if isinstance(session, str) and session
                        else f"failure:{row.get('task_id', 'unknown')}"
                    )
                    if cost_key not in seen_sessions:
                        known_cost += cost
                        seen_sessions.add(cost_key)
                elif not cost:
                    unknown_failures += 1
                match = SCALE_TASK_RE.match(str(row.get("task_id", "")))
                if match:
                    max_sequence = max(max_sequence, int(match.group(3)))

    return CorpusSnapshot(
        verified=verified,
        raw=raw,
        rejected=rejected,
        generation_failures=generation_failures,
        known_cost_usd=known_cost,
        unknown_cost_reserve_usd=unknown_failures * UNKNOWN_FAILURE_RESERVE_USD,
        kinds=kinds,
        scenarios=scenarios,
        max_sequence=max_sequence,
    )


def existing_template_attempts(paths: Paths) -> Counter:
    counts: Counter = Counter()
    for directory in (paths.raw, paths.rejects):
        for path in sorted(directory.glob("*.jsonl")):
            for row in read_jsonl(path):
                value = str(row.get("task_id", ""))
                match = SCALE_TASK_RE.match(value)
                if not match:
                    continue
                kind = match.group(1).replace("-", "_")
                counts[(kind, match.group(2))] += 1
    return counts


def plan_templates(paths: Paths, count: int) -> list[Any]:
    snapshot = corpus_snapshot(paths)
    planned_kinds = Counter(snapshot.kinds)
    planned_scenarios = Counter(snapshot.scenarios)
    attempts = existing_template_attempts(paths)
    bases = base_template_index()
    scenarios = sorted({scenario for _, scenario in bases})
    sequence = snapshot.max_sequence
    result = []

    for _ in range(count):
        kind = min(TASK_KINDS, key=lambda item: (planned_kinds[item], TASK_KINDS.index(item)))
        kind_index = TASK_KINDS.index(kind)
        scenario = min(
            scenarios,
            key=lambda item: (
                planned_scenarios[(kind, item)],
                (scenarios.index(item) - kind_index) % len(scenarios),
            ),
        )
        sequence += 1
        ordinal = attempts[(kind, scenario)]
        attempts[(kind, scenario)] += 1
        result.append(scale_template(bases[(kind, scenario)], sequence, ordinal))
        planned_kinds[kind] += 1
        planned_scenarios[(kind, scenario)] += 1
    return result


def failure_usage(failure: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    candidate = failure.get("candidate_trace")
    if isinstance(candidate, dict):
        usage = candidate.get("usage")
        session = candidate.get("opencode", {}).get("session_id")
        if isinstance(usage, dict):
            return dict(usage), session if isinstance(session, str) else None
    text = failure.get("stdout")
    if isinstance(text, str):
        match = re.search(r'"cost"\s*:\s*([0-9.]+)', text)
        if match:
            return {"cost_usd": float(match.group(1))}, None
    return {"cost_usd": 0.0}, None


def generate_attempt(
    template: Any,
    *,
    paths: Paths,
    opencode: str,
    env: dict[str, str],
    timeout: float,
) -> AttemptResult:
    parent = paths.work_root / template.template_id
    if parent.exists():
        shutil.rmtree(parent)
    parent.mkdir(parents=True)
    task_env = isolated_opencode_environment(env, parent)
    try:
        record, workspace = GEN.generate_one(
            template,
            repo_root=ROOT,
            opencode=opencode,
            model=MODEL,
            env=task_env,
            timeout=timeout,
            work_parent=parent,
            keep_workspace=True,
            auto_approve=True,
        )
        return AttemptResult(
            task_id=template.template_id,
            template=template,
            record=record,
            workspace=workspace,
            work_parent=parent,
            failure=None,
        )
    except GEN.GenerationError as exc:
        return AttemptResult(
            task_id=template.template_id,
            template=template,
            record=None,
            workspace=None,
            work_parent=parent,
            failure=exc.failure,
        )


def pipeline_hashes(paths: Paths) -> dict[str, str]:
    names = ("validate_trace.py", "verify_outcome.py", "convert_k3_qwen.py")
    return {name: sha256_file(paths.pipeline_dir / name) for name in names}


def invoke_pipeline(
    trace: dict[str, Any],
    *,
    paths: Paths,
    allow_manual: bool,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    with tempfile.TemporaryDirectory(prefix="sft-scale-pipeline-", dir=paths.work_root) as tmp:
        root = Path(tmp)
        raw_path = root / "raw.jsonl"
        verified_path = root / "verified.jsonl"
        rejected_path = root / "rejected.jsonl"
        converted_path = root / "converted.jsonl"
        raw_path.write_text(json_dumps(trace) + "\n", encoding="utf-8")

        validate = run(
            [sys.executable, str(paths.pipeline_dir / "validate_trace.py"), str(raw_path)],
            timeout=120,
        )
        if validate.returncode != 0:
            raise RuntimeError(
                f"{trace.get('task_id')}: validate_trace.py failed:\n"
                f"{validate.stdout}{validate.stderr}"
            )

        verify_argv = [
            sys.executable,
            str(paths.pipeline_dir / "verify_outcome.py"),
            str(raw_path),
            "-o",
            str(verified_path),
            "--rejects",
            str(rejected_path),
            "--workdir",
            str(paths.work_root),
        ]
        if allow_manual:
            verify_argv.append("--allow-manual")
        verify = run(verify_argv, timeout=300)
        if verify.returncode == 2:
            raise RuntimeError(
                f"{trace.get('task_id')}: verify_outcome.py harness error:\n"
                f"{verify.stdout}{verify.stderr}"
            )
        if verify.returncode == 1:
            rows = list(read_jsonl(rejected_path))
            if len(rows) != 1:
                raise RuntimeError(
                    f"{trace.get('task_id')}: verifier rejected without one reject row"
                )
            return "rejected", rows[0], None
        if verify.returncode != 0:
            raise RuntimeError(
                f"{trace.get('task_id')}: verify_outcome.py exit {verify.returncode}:\n"
                f"{verify.stdout}{verify.stderr}"
            )

        rows = list(read_jsonl(verified_path))
        if len(rows) != 1:
            raise RuntimeError(
                f"{trace.get('task_id')}: verifier produced {len(rows)} verified rows"
            )
        verified = rows[0]
        convert = run(
            [
                sys.executable,
                str(paths.pipeline_dir / "convert_k3_qwen.py"),
                "--roundtrip",
                str(verified_path),
                "-o",
                str(converted_path),
            ],
            timeout=120,
        )
        if convert.returncode != 0:
            rejected = dict(verified)
            rejected["reject"] = {
                "reason": "conversion failed after outcome verification",
                "ts": utc_now(),
                "evidence": {
                    "stdout": convert.stdout,
                    "stderr": convert.stderr,
                    "exit_code": convert.returncode,
                },
            }
            return "rejected", rejected, None
        converted_rows = list(read_jsonl(converted_path))
        if len(converted_rows) != 1:
            raise RuntimeError(
                f"{trace.get('task_id')}: converter produced {len(converted_rows)} rows"
            )
        return "verified", verified, converted_rows[0]


def batch_paths(paths: Paths, batch: int) -> dict[str, Path]:
    stem = f"scale-{batch:04d}"
    return {
        "raw": paths.raw / f"{stem}.jsonl",
        "verified": paths.verified / f"{stem}.jsonl",
        "converted": paths.converted / f"{stem}.jsonl",
        "rejects": paths.rejects / f"{stem}.jsonl",
        "manifest": paths.manifests / f"{stem}.jsonl",
    }


def next_batch_number(paths: Paths) -> int:
    values = []
    for path in paths.manifests.glob("scale-*.jsonl"):
        match = re.match(r"scale-(\d{4})\.jsonl$", path.name)
        if match:
            values.append(int(match.group(1)))
    return max(values, default=0) + 1


def record_generation_failure(
    result: AttemptResult,
    *,
    files: dict[str, Path],
) -> dict[str, Any]:
    assert result.failure is not None
    usage, session = failure_usage(result.failure)
    rejected = {
        "task_id": result.task_id,
        "seed_source": "memra-scale-fixture",
        "task_kind": result.template.category,
        "scenario": result.template.scenario,
        "usage": usage,
        "session_id": session,
        "reject": {
            "reason": (
                f"generation failed at {result.failure.get('stage', 'unknown')}: "
                f"{result.failure.get('error', 'unknown error')}"
            ),
            "ts": utc_now(),
        },
        "generation_failure": result.failure,
    }
    append_jsonl(files["rejects"], rejected)
    manifest = {
        "task_id": result.task_id,
        "task_kind": result.template.category,
        "scenario": result.template.scenario,
        "status": "generation_failure",
        "usage": usage,
        "session_id": session,
        "completed_at": utc_now(),
    }
    append_jsonl(files["manifest"], manifest)
    return manifest


def process_attempt(
    result: AttemptResult,
    *,
    paths: Paths,
    files: dict[str, Path],
    hashes: dict[str, str],
) -> dict[str, Any]:
    try:
        if result.failure is not None:
            return record_generation_failure(result, files=files)
        assert result.record is not None
        trace = k3_record(
            result.record,
            result.template,
            workspace=result.workspace,
        )
        trace["meta"]["pipeline_sha256"] = hashes
        status, outcome_row, converted = invoke_pipeline(
            trace,
            paths=paths,
            allow_manual=False,
        )
        append_jsonl(files["raw"], trace)
        if status == "verified":
            assert converted is not None
            append_jsonl(files["verified"], outcome_row)
            append_jsonl(files["converted"], converted)
        else:
            append_jsonl(files["rejects"], outcome_row)
        manifest = {
            "task_id": result.task_id,
            "task_kind": result.template.category,
            "scenario": result.template.scenario,
            "status": status,
            "usage": result.record["usage"],
            "session_id": result.record["opencode"]["session_id"],
            "completed_at": utc_now(),
        }
        append_jsonl(files["manifest"], manifest)
        return manifest
    finally:
        shutil.rmtree(result.work_parent, ignore_errors=True)


def migrate_proof(paths: Paths) -> str | None:
    raw_path = paths.raw / "proof-20260808.jsonl"
    verified_path = paths.verified / "proof-20260808.jsonl"
    converted_path = paths.converted / "proof-20260808.jsonl"
    reject_path = paths.rejects / "proof-generation-failures-20260808.jsonl"
    manifest_path = paths.manifests / "proof-20260808.jsonl"
    targets = (raw_path, verified_path, converted_path, reject_path, manifest_path)
    if any(path.exists() for path in targets):
        expected = {
            raw_path: 24,
            verified_path: 24,
            converted_path: 24,
            reject_path: 3,
            manifest_path: 24,
        }
        actual = {path: count_jsonl(path) for path in targets}
        if actual != expected:
            detail = ", ".join(
                f"{path.name}={actual[path]} (expected {expected[path]})"
                for path in targets
            )
            raise RuntimeError(f"partial proof migration detected: {detail}")
        return None

    hashes = pipeline_hashes(paths)
    templates = {template.template_id: template for template in GEN.build_templates()}
    for row in read_jsonl(PROOF_CORPUS.expanduser()):
        if row.get("record_type") != "trace":
            continue
        template = templates[row["template_id"]]
        trace = k3_record(row, template, workspace=None, proof=True)
        trace["meta"]["pipeline_sha256"] = hashes
        status, verified, converted = invoke_pipeline(
            trace,
            paths=paths,
            allow_manual=True,
        )
        if status != "verified" or converted is None:
            raise RuntimeError(f"{trace['task_id']}: proof migration unexpectedly rejected")
        append_jsonl(raw_path, trace)
        append_jsonl(verified_path, verified)
        append_jsonl(converted_path, converted)
        append_jsonl(
            manifest_path,
            {
                "task_id": trace["task_id"],
                "task_kind": template.category,
                "scenario": template.scenario,
                "status": "verified",
                "usage": row["usage"],
                "session_id": row["opencode"]["session_id"],
                "completed_at": utc_now(),
            },
        )

    for index, failure in enumerate(read_jsonl(PROOF_FAILURES.expanduser()), 1):
        usage, session = failure_usage(failure)
        append_jsonl(
            reject_path,
            {
                "task_id": f"proof-generation-failure-{index:02d}",
                "seed_source": "memra-proof-fixture",
                "usage": usage,
                "session_id": session,
                "reject": {
                    "reason": (
                        f"proof generation failed at {failure.get('stage', 'unknown')}: "
                        f"{failure.get('error', 'unknown error')}"
                    ),
                    "ts": utc_now(),
                },
                "generation_failure": failure,
            },
        )

    commit = git_commit(
        paths,
        [raw_path, verified_path, converted_path, reject_path, manifest_path],
        "data: migrate SFT proof traces into verified pipeline layout",
    )
    return commit


def verified_task_ids(paths: Paths) -> set[str]:
    result = set()
    for path in sorted(paths.verified.glob("*.jsonl")):
        for trace in read_jsonl(path):
            value = trace.get("task_id")
            if isinstance(value, str):
                result.add(value)
    return result


def recover_answer_rejects(paths: Paths) -> tuple[list[Path], int]:
    verified_path = paths.verified / "answer-recovered-20260808.jsonl"
    converted_path = paths.converted / "answer-recovered-20260808.jsonl"
    manifest_path = paths.manifests / "answer-recovered-20260808.jsonl"
    admitted = verified_task_ids(paths)
    hashes = pipeline_hashes(paths)
    recovered = 0

    for reject_path in sorted(paths.rejects.glob("scale-*.jsonl")):
        for row in read_jsonl(reject_path):
            task_id_value = row.get("task_id")
            outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            scenario = meta.get("scenario")
            if (
                not isinstance(task_id_value, str)
                or task_id_value in admitted
                or outcome.get("method") != "answer_check"
                or scenario not in ANSWER_PATTERNS
            ):
                continue

            trace = dict(row)
            trace.pop("reject", None)
            trace["outcome"] = {
                "verified": False,
                "method": "answer_check",
                "detail": "pending unordered answer re-verification",
            }
            verify = dict(trace.get("verify") or {})
            verify.update(
                {
                    "type": "answer",
                    "source": "last_assistant",
                    "pattern": ANSWER_PATTERNS[scenario],
                }
            )
            trace["verify"] = verify
            trace_meta = dict(meta)
            trace_meta["answer_reverification"] = {
                "source_reject": str(reject_path.relative_to(paths.corpus_repo)),
                "reason": (
                    "replace brittle ordered or vocabulary-specific concept regex "
                    "with unordered semantic lookaheads"
                ),
                "pipeline_sha256": hashes,
                "checked_at": utc_now(),
            }
            trace["meta"] = trace_meta

            status, verified, converted = invoke_pipeline(
                trace,
                paths=paths,
                allow_manual=False,
            )
            if status != "verified" or converted is None:
                raise RuntimeError(
                    f"{task_id_value}: unordered answer re-verification still rejected"
                )
            append_jsonl(verified_path, verified)
            append_jsonl(converted_path, converted)
            append_jsonl(
                manifest_path,
                {
                    "task_id": task_id_value,
                    "task_kind": trace_kind(verified),
                    "scenario": scenario,
                    "status": "verified_after_answer_recheck",
                    "usage": meta.get("usage", {}),
                    "session_id": trace_session_id(verified),
                    "source_reject": str(reject_path.relative_to(paths.corpus_repo)),
                    "completed_at": utc_now(),
                },
            )
            admitted.add(task_id_value)
            recovered += 1

    return [verified_path, converted_path, manifest_path], recovered


def aggregate_rejects(paths: Paths, output: Path) -> None:
    with output.open("w", encoding="utf-8") as target:
        for path in sorted(paths.rejects.glob("*.jsonl")):
            with path.open(encoding="utf-8") as source:
                shutil.copyfileobj(source, target)


def update_stats(paths: Paths) -> Path:
    paths.stats.mkdir(parents=True, exist_ok=True)
    output = paths.stats / "current.md"
    with tempfile.TemporaryDirectory(prefix="sft-scale-stats-", dir=paths.work_root) as tmp:
        rejects = Path(tmp) / "rejects.jsonl"
        aggregate_rejects(paths, rejects)
        verified = [str(path) for path in sorted(paths.verified.glob("*.jsonl"))]
        if not verified:
            output.write_text("# SFT corpus stats\n\nTraces: **0**\n", encoding="utf-8")
            return output
        result = run(
            [
                sys.executable,
                str(paths.pipeline_dir / "corpus_stats.py"),
                *verified,
                "--rejects",
                str(rejects),
                "-o",
                str(output),
            ],
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "corpus_stats.py failed")
    return output


def steering_receipt(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "checked_at": utc_now(),
    }


def replace_progress_block(text: str, block: str) -> str:
    rendered = f"{PROGRESS_START}\n{block.rstrip()}\n{PROGRESS_END}"
    if PROGRESS_START not in text or PROGRESS_END not in text:
        return text.rstrip() + "\n\n" + rendered + "\n"
    before, rest = text.split(PROGRESS_START, 1)
    _, after = rest.split(PROGRESS_END, 1)
    return before + rendered + after


def progress_block(
    snapshot: CorpusSnapshot,
    *,
    budget: float,
    target: int,
    corpus_commit: str,
    steering: dict[str, Any],
    status: str,
) -> str:
    rejected_outcomes = snapshot.rejected - snapshot.generation_failures
    rows = "\n".join(
        f"| {kind} | {snapshot.kinds[kind]} |" for kind in TASK_KINDS
    )
    return f"""## Scale run

Status: **{status}**

Updated: `{utc_now()}`

Corpus commit: `{corpus_commit}`

| Measure | Running total |
|---|---:|
| Verified traces | {snapshot.verified} / {target} |
| Raw pipeline traces | {snapshot.raw} |
| Outcome/conversion rejects | {rejected_outcomes} |
| Generation failures | {snapshot.generation_failures} |
| Keep rate | {snapshot.keep_rate:.2%} |
| Provider-reported spend | ${snapshot.known_cost_usd:.6f} |
| Unknown-failure reserve | ${snapshot.unknown_cost_reserve_usd:.2f} |
| Budget-accounted spend | ${snapshot.budget_accounted_usd:.6f} / ${budget:.2f} |
| Budget headroom | ${budget - snapshot.budget_accounted_usd:.6f} |

### Verified task mix

| Task kind | Traces |
|---|---:|
{rows}

Steering checked at `{steering["checked_at"]}` with SHA-256
`{steering["sha256"]}`.

The budget total includes the 24 successful proof calls and the three original
export-failure calls. No training, GPU, rustup, tag, merge, or origin push is
part of this run."""


def commit_progress(
    paths: Paths,
    snapshot: CorpusSnapshot,
    *,
    budget: float,
    target: int,
    corpus_commit: str,
    steering: dict[str, Any],
    status: str,
) -> str:
    text = paths.progress.read_text(encoding="utf-8")
    updated = replace_progress_block(
        text,
        progress_block(
            snapshot,
            budget=budget,
            target=target,
            corpus_commit=corpus_commit,
            steering=steering,
            status=status,
        ),
    )
    paths.progress.write_text(updated, encoding="utf-8")
    diff_check = run(["git", "diff", "--check"], cwd=ROOT, timeout=60)
    if diff_check.returncode != 0:
        raise RuntimeError(diff_check.stdout + diff_check.stderr)
    add = run(
        ["git", "add", "--", str(paths.progress.relative_to(ROOT))],
        cwd=ROOT,
        timeout=60,
    )
    if add.returncode != 0:
        raise RuntimeError(add.stderr.strip() or "progress git add failed")
    staged = run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if staged.returncode == 0:
        return git_head(ROOT)
    commit = run(
        [
            "git",
            "commit",
            "-m",
            f"data: update SFT scale progress to {snapshot.verified} verified",
        ],
        cwd=ROOT,
        timeout=180,
    )
    if commit.returncode != 0:
        raise RuntimeError(commit.stderr.strip() or "progress git commit failed")
    return git_head(ROOT)


def ensure_directories(paths: Paths) -> None:
    for path in (
        paths.raw,
        paths.verified,
        paths.converted,
        paths.rejects,
        paths.manifests,
        paths.stats,
        paths.work_root,
    ):
        path.mkdir(parents=True, exist_ok=True)


def initialize_corpus_metadata(paths: Paths) -> str | None:
    gitignore = paths.corpus_repo / ".gitignore"
    if gitignore.exists():
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    else:
        lines = []
    changed = False
    if ".work/" not in lines:
        lines.append(".work/")
        changed = True
    if not changed:
        return None
    gitignore.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return git_commit(
        paths,
        [gitignore],
        "chore: ignore resumable SFT generation workspaces",
    )


def run_batch(
    templates: list[Any],
    *,
    batch: int,
    paths: Paths,
    opencode: str,
    env: dict[str, str],
    workers: int,
    timeout: float,
) -> tuple[dict[str, Path], Counter]:
    files = batch_paths(paths, batch)
    hashes = pipeline_hashes(paths)
    results: Counter = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        generated = executor.map(
            lambda template: generate_attempt(
                template,
                paths=paths,
                opencode=opencode,
                env=env,
                timeout=timeout,
            ),
            templates,
        )
        for index, result in enumerate(generated, 1):
            manifest = process_attempt(
                result,
                paths=paths,
                files=files,
                hashes=hashes,
            )
            results[manifest["status"]] += 1
            print(
                f"[batch {batch:04d} {index}/{len(templates)}] "
                f"{result.task_id}: {manifest['status']}",
                file=sys.stderr,
                flush=True,
            )
    return files, results


def validate_runtime(
    paths: Paths,
    config: Path,
    expected_steering: str,
    expected_pipeline: dict[str, str],
) -> dict[str, Any]:
    receipt = provider_receipt(config)
    steering = steering_receipt(paths.steering)
    if steering["sha256"] != expected_steering:
        raise RuntimeError(
            "steering file changed; stop for owner-direction review before more API calls"
        )
    current_hashes = pipeline_hashes(paths)
    if current_hashes != expected_pipeline:
        raise RuntimeError(
            "pipeline scripts changed; stop for contract review before more API calls"
        )
    print(
        "runtime gate PASS: "
        f"config={receipt['sha256']} steering={steering['sha256']}",
        file=sys.stderr,
        flush=True,
    )
    return steering


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-verified", type=int, default=1000)
    parser.add_argument("--budget-usd", type=float, default=150.0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument(
        "--opencode-config",
        type=Path,
        default=GEN.DEFAULT_OPENCODE_CONFIG,
    )
    parser.add_argument("--corpus-repo", type=Path, default=DEFAULT_CORPUS_REPO)
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--steering", type=Path, default=DEFAULT_STEERING)
    parser.add_argument(
        "--no-proof-migration",
        action="store_true",
        help="skip migration of the existing 24 proof traces",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="stop after this many newly generated batches",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate gates and print the next planned batch without API calls",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target_verified <= 0:
        raise SystemExit("--target-verified must be positive")
    if args.budget_usd <= 0:
        raise SystemExit("--budget-usd must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    paths = Paths(
        corpus_repo=args.corpus_repo.expanduser().resolve(),
        pipeline_dir=args.pipeline_dir.expanduser().resolve(),
        progress=args.progress.expanduser().resolve(),
        steering=args.steering.expanduser().resolve(),
        work_root=args.corpus_repo.expanduser().resolve() / ".work",
    )
    ensure_directories(paths)
    require_clean_repo(paths.corpus_repo, allowed={".gitignore"})
    branch = run(["git", "branch", "--show-current"], cwd=ROOT).stdout.strip()
    if branch != "lane/sft-corpus":
        raise SystemExit(f"wrong memra branch: {branch!r}")

    config = args.opencode_config.expanduser().resolve()
    expected_steering = steering_receipt(paths.steering)["sha256"]
    expected_pipeline = pipeline_hashes(paths)
    steering = validate_runtime(
        paths,
        config,
        expected_steering,
        expected_pipeline,
    )
    opencode = shutil.which(args.opencode) or args.opencode
    version = GEN.opencode_version(opencode, scale_environment())
    if version != "1.18.13":
        raise SystemExit(f"opencode version drift: expected 1.18.13, got {version}")

    if args.dry_run:
        snapshot = corpus_snapshot(paths)
        gap = max(0, args.target_verified - snapshot.verified)
        planned = plan_templates(paths, min(args.batch_size, gap or args.batch_size))
        print(
            json.dumps(
                {
                    "snapshot": {
                        "verified": snapshot.verified,
                        "raw": snapshot.raw,
                        "rejected": snapshot.rejected,
                        "known_cost_usd": snapshot.known_cost_usd,
                        "budget_accounted_usd": snapshot.budget_accounted_usd,
                        "task_kinds": dict(snapshot.kinds),
                    },
                    "planned": [template.template_id for template in planned],
                    "steering": steering,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    initialize_corpus_metadata(paths)
    if not args.no_proof_migration:
        proof_commit = migrate_proof(paths)
        if proof_commit:
            stats = update_stats(paths)
            proof_commit = git_commit(
                paths,
                [stats],
                "data: add SFT proof migration stats",
            )
            snapshot = corpus_snapshot(paths)
            commit_progress(
                paths,
                snapshot,
                budget=args.budget_usd,
                target=args.target_verified,
                corpus_commit=proof_commit,
                steering=steering,
                status="RUNNING",
            )

    recovery_files, recovered = recover_answer_rejects(paths)
    if recovered:
        stats = update_stats(paths)
        corpus_commit = git_commit(
            paths,
            [*recovery_files, stats],
            f"data: recover {recovered} order-independent SFT answer checks",
        )
        snapshot = corpus_snapshot(paths)
        commit_progress(
            paths,
            snapshot,
            budget=args.budget_usd,
            target=args.target_verified,
            corpus_commit=corpus_commit,
            steering=steering,
            status="RUNNING",
        )

    env = scale_environment()
    batches_run = 0
    while True:
        snapshot = corpus_snapshot(paths)
        if snapshot.verified >= args.target_verified:
            corpus_commit = git_head(paths.corpus_repo)
            steering = steering_receipt(paths.steering)
            commit_progress(
                paths,
                snapshot,
                budget=args.budget_usd,
                target=args.target_verified,
                corpus_commit=corpus_commit,
                steering=steering,
                status="PILOT TARGET MET",
            )
            print(
                json.dumps(
                    {
                        "status": "target_met",
                        "verified": snapshot.verified,
                        "rejected": snapshot.rejected,
                        "known_cost_usd": snapshot.known_cost_usd,
                        "budget_accounted_usd": snapshot.budget_accounted_usd,
                        "corpus_commit": corpus_commit,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.max_batches is not None and batches_run >= args.max_batches:
            print(
                json.dumps(
                    {
                        "status": "max_batches",
                        "verified": snapshot.verified,
                        "known_cost_usd": snapshot.known_cost_usd,
                    },
                    sort_keys=True,
                )
            )
            return 0

        remaining = args.budget_usd - snapshot.budget_accounted_usd
        if remaining <= ACTIVE_CALL_RESERVE_USD:
            raise SystemExit(
                f"budget hard stop: ${snapshot.budget_accounted_usd:.6f} accounted "
                f"against ${args.budget_usd:.2f}"
            )
        steering = validate_runtime(
            paths,
            config,
            expected_steering,
            expected_pipeline,
        )
        gap = args.target_verified - snapshot.verified
        count = min(args.batch_size, gap)
        max_parallel = int(remaining // ACTIVE_CALL_RESERVE_USD)
        count = min(count, max_parallel)
        if count <= 0:
            raise SystemExit("budget hard stop: no reserved call capacity remains")
        templates = plan_templates(paths, count)
        batch = next_batch_number(paths)
        print(
            f"starting batch {batch:04d}: {count} attempts, "
            f"{snapshot.verified}/{args.target_verified} verified, "
            f"${snapshot.budget_accounted_usd:.6f}/${args.budget_usd:.2f}",
            file=sys.stderr,
            flush=True,
        )
        files, results = run_batch(
            templates,
            batch=batch,
            paths=paths,
            opencode=opencode,
            env=env,
            workers=min(args.workers, count),
            timeout=args.timeout,
        )
        stats = update_stats(paths)
        commit_files = [*files.values(), stats]
        corpus_commit = git_commit(
            paths,
            commit_files,
            (
                f"data: add SFT scale batch {batch:04d} "
                f"({results['verified']} verified)"
            ),
        )
        snapshot = corpus_snapshot(paths)
        commit_progress(
            paths,
            snapshot,
            budget=args.budget_usd,
            target=args.target_verified,
            corpus_commit=corpus_commit,
            steering=steering,
            status="RUNNING",
        )
        batches_run += 1


if __name__ == "__main__":
    raise SystemExit(main())
