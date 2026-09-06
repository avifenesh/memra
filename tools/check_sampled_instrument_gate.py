#!/usr/bin/env python3
"""Check real sampled continuation evidence for an argmax-instrument-only push.

This does not certify a changed serving runtime or a comparative speed claim.
Raw workload files remain outside the public repository; only hashes are printed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

PROBE = "crates/memra-engine/src/bin/argmax_margin_probe.rs"
INSTRUMENT_FILES = {PROBE, "tools/argmax-margin-gate.sh", "tools/test_argmax_margin_gate.py",
                    "tools/check_sampled_instrument_gate.py", "tools/test_sampled_instrument_gate.py",
                    "tools/hooks/pre-push", ".github/workflows/ci.yml", "docs/TESTING.md", "docs/FLAGS.md"}
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def read(path, limit=8 << 20):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(),
            "canonical regular evidence file required")
    require(path.stat().st_size <= limit, "evidence file exceeds bound")
    raw = path.read_bytes()
    require(len(raw) <= limit, "evidence grew beyond bound")
    return raw


def reference(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "invalid evidence reference")
    require(isinstance(ref["sha256"], str) and SHA.fullmatch(ref["sha256"]), "invalid SHA256")
    raw = read(ref["path"])
    require(digest(raw) == ref["sha256"], "evidence SHA mismatch")
    return raw


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(repo, base, receipt, receipt_sha, now=None):
    require(isinstance(base, str) and COMMIT.fullmatch(base), "explicit base commit required")
    require(SHA.fullmatch(receipt_sha) is not None, "explicit receipt SHA required")
    raw = read(receipt)
    require(digest(raw) == receipt_sha, "receipt SHA mismatch")
    data = json.loads(raw)
    require(set(data) == {"schema", "tested_commit", "probe_sha256", "correctness_table", "identity", "inputs", "server_log", "rows", "window",
                          "request_root", "original_request_root"}, "unexpected receipt fields")
    require(data["schema"] == "memra.sampled-instrument-gate.v1", "wrong receipt schema")
    tested = data["tested_commit"]
    require(isinstance(tested, str) and COMMIT.fullmatch(tested), "invalid tested commit")
    # A serving/library change cannot borrow performance evidence from another runtime.
    changed = set(git(repo, "diff", "--name-only", base + "..HEAD").decode().splitlines())
    require({p for p in changed if p.startswith("crates/")} == {PROBE},
            "sampled instrument gate refuses serving/library changes")
    require(all(p in INSTRUMENT_FILES or p.startswith("research/hf-argmax-margin-20260905/") for p in changed),
            "change is outside the argmax instrument publication scope")
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", tested, "HEAD"], check=True)
    tested_source = git(repo, "show", tested + ":" + PROBE)
    require(digest(tested_source) == data["probe_sha256"], "tested probe SHA mismatch")
    require(git(repo, "show", "HEAD:" + PROBE) == tested_source
            and read(repo / PROBE) == tested_source, "probe changed after its correctness receipt")
    table = reference(data["correctness_table"]).decode()
    require("Engine/gate source: " + tested in table, "numeric table source differs")
    numeric = [line.split() for line in table.splitlines() if re.match(r"^\d+\s+\d+", line)]
    require(len(numeric) == 12 and len({r[0] for r in numeric}) == 12, "twelve unique margin rows required")
    flips = 0
    for row in numeric:
        require(len(row) >= 7 and row[1].isdigit() and row[3].isdigit(), "invalid margin row")
        mp, md, delta = map(float, (row[2], row[4], row[5]))
        require(all(math.isfinite(v) and v >= 0 for v in (mp, md, delta)), "invalid margin value")
        agree = row[1] == row[3]
        require(row[6] == ("yes" if agree else "NO"), "margin agreement flag contradicts IDs")
        if not agree:
            flips += 1
            require(delta > min(mp, md), "unexplained margin flip")
    require(flips <= 1, "margin flip budget exceeded")

    identity = json.loads(reference(data["identity"]))
    require(isinstance(identity.get("source_commit"), str) and COMMIT.fullmatch(identity["source_commit"])
            and isinstance(identity.get("binary_sha256"), str) and SHA.fullmatch(identity["binary_sha256"]),
            "sampled runtime identity missing")
    inputs = json.loads(reference(data["inputs"]))
    require(identity.get("inputs_sha256") == data["inputs"]["sha256"]
            and inputs.get("source_commit") == identity["source_commit"]
            and inputs.get("files_sha256", {}).get(inputs.get("binary")) == identity["binary_sha256"],
            "sampled runtime does not match its pinned input manifest")
    log = reference(data["server_log"]).decode()
    builds = re.findall(r"^\[server\] build: (memra-\S+) \(id: source-tree, git: ([0-9a-f]+)\)$", log, re.M)
    require(len(builds) == 1 and builds[0][1] == identity["source_commit"][:12], "runtime boot identity differs")
    fingerprint = builds[0][0]
    visit = identity.get("visit", {})
    require(visit.get("phase") == "sampled" and visit.get("speculative") is True,
            "vendor-sampled speculative visit required")
    window = json.loads(reference(data["window"]))
    start, end = window.get("start_unix"), window.get("end_unix")
    require(window.get("phase") == "sampled" and number(start) and number(end) and 0 < start < end,
            "invalid sampled measurement window")
    require(start >= int(git(repo, "show", "-s", "--format=%ct", tested).decode().strip())
            and end <= (time.time() if now is None else now) + 5, "measurement time does not cover tested source")
    rows = [json.loads(line) for line in reference(data["rows"]).splitlines() if line.strip()]
    require(len(rows) == 8, "exactly eight continuation turns required")
    local_root = Path(data["request_root"])
    original_root = Path(data["original_request_root"])
    require(local_root.is_absolute() and local_root.resolve() == local_root and local_root.is_dir()
            and original_root.is_absolute() and ".." not in original_root.parts, "invalid request root")
    history, salt, model, wall = [], None, None, 0.0
    for i, row in enumerate(rows):
        require(type(row.get("turn")) is int and row["turn"] == i + 1
                and row.get("strict_valid") is True and row.get("done") is True
                and row.get("error") is None and row.get("loop") is False,
                "invalid, failed or looping continuation turn")
        require(row.get("fingerprint") == fingerprint, "response fingerprint differs from captured runtime")
        require(row.get("route") == "dflash2" and type(row.get("spec_rounds")) is int
                and row["spec_rounds"] > 0, "speculation did not engage")
        cached = row.get("cached_tokens")
        require(type(cached) is int and (cached == 0 if i == 0 else cached > 0), "cache continuation missing")
        require(type(row.get("completion_tokens")) is int and 0 < row["completion_tokens"] <= 512,
                "invalid output length")
        require(number(row.get("wall_s")) and number(row.get("ttft_s"))
                and 0 <= row["ttft_s"] < row["wall_s"], "invalid timing")
        wall += row["wall_s"]
        text = row.get("text")
        require(isinstance(text, str) and text and digest(text.encode()) == row.get("text_sha256"),
                "generated text binding failed")
        relative = Path(row["request_receipt"]).relative_to(original_root)
        require(".." not in relative.parts, "request path escaped root")
        request = read(local_root / relative)
        require(digest(request) == row.get("request_sha256"), "request binding failed")
        body = json.loads(request)
        require(set(body) == {"model", "messages", "max_tokens", "stream", "stream_options", "cache_salt"}
                and type(body["max_tokens"]) is int and body["max_tokens"] == 512
                and body["stream"] is True and body["stream_options"] == {"include_usage": True},
                "sampled serving shape changed or sampling override supplied")
        require(isinstance(body["cache_salt"], str) and body["cache_salt"]
                and isinstance(body["model"], str) and body["model"], "missing model/cache identity")
        if i == 0:
            salt, model = body["cache_salt"], body["model"]
        require(body["cache_salt"] == salt and body["model"] == model, "conversation identity changed")
        messages = body["messages"]
        require(isinstance(messages, list) and len(messages) == 2 * i + 1, "not an eight-turn continuation")
        user = messages[-1]
        require(isinstance(user, dict) and set(user) == {"role", "content"}
                and user["role"] == "user" and isinstance(user["content"], str) and user["content"].strip(),
                "invalid user turn")
        require(messages == history + [user], "assistant history was not continued exactly")
        history = messages + [{"role": "assistant", "content": text}]
    require(wall <= end - start + 1, "row timings exceed measured window")
    return {"passed": True, "turns": 8, "tested_commit": tested,
            "runtime_source": identity["source_commit"],
            "scope": "argmax instrument only; no serving-runtime change or speedup claim"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=lambda s: Path(s).resolve(), required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--receipt", type=lambda s: Path(s).absolute(), required=True)
    p.add_argument("--receipt-sha256", required=True)
    a = p.parse_args()
    try:
        result = validate(a.repo_root, a.base, a.receipt, a.receipt_sha256)
        subprocess.run([sys.executable, "tools/test_argmax_margin_gate.py"], cwd=a.repo_root, check=True)
        print(json.dumps(result))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError) as e:
        print("sampled instrument gate refused: " + type(e).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
