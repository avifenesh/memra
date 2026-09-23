"""Post-result C cost diagnostic; never used as a serving-rate gate."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


ARMS = ("off", "prob", "cut")
PHASE = re.compile(
    r"\[spec-phase\] draft=([0-9.]+)ms .*?verify-issue=([0-9.]+)ms "
    r".*?verify-wait=([0-9.]+)ms .*?commit-host=([0-9.]+)ms .*?rounds=(\d+)"
)
HIST = re.compile(r"\[spec-stats\] rounds=(\d+) full_accept=\d+ len_hist=\[([0-9, ]+)\]")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def ids(path):
    return [int(token) for token in Path(path).read_text().split()]


def parse_log(path):
    text = path.read_text()
    phases = PHASE.findall(text)[-8:]
    hists = HIST.findall(text)[-8:]
    if len(phases) != 8 or len(hists) != 8:
        raise ValueError("phase diagnostic lacks eight complete native requests")
    parsed = []
    for phase, (rounds, values) in zip(phases, hists):
        draft, issue, wait, commit, phase_rounds = phase
        hist = [int(value.strip()) for value in values.split(",")]
        if len(hist) != 8 or sum(hist) != int(rounds) or int(rounds) != int(phase_rounds):
            raise ValueError("phase and draft-length rounds differ")
        parsed.append({
            "draft_ms": float(draft),
            "verify_issue_ms": float(issue),
            "verify_wait_ms": float(wait),
            "commit_host_ms": float(commit),
            "rounds": int(rounds),
            "draft_lengths": hist,
        })
    return parsed


def summarize(records):
    tokens = sum(row["output_tokens"] for row in records)
    seconds = sum(row["elapsed_s"] for row in records)
    fields = ("draft_ms", "verify_issue_ms", "verify_wait_ms", "commit_host_ms")
    return {
        "requests": len(records),
        "tokens": tokens,
        "native_request_seconds": seconds,
        "native_tokens_per_second": tokens / seconds,
        "rounds": sum(row["rounds"] for row in records),
        "confidence_shortened_rounds": sum(
            sum(row["draft_lengths"][:3]) for row in records
        ),
        "phase_ms": {field: sum(row[field] for row in records) for field in fields},
        "phase_ms_per_100_tokens": {
            field: 100 * sum(row[field] for row in records) / tokens
            for field in fields
        },
    }


def analyze(root, repo):
    if not (root / "exit.txt").read_text().startswith("exit=0 "):
        raise ValueError("phase diagnostic did not complete")
    source = json.loads((root.parent / "source-confidence.json").read_text())
    if sha(root.parent / "binaries-confidence/qwen-prefix-study") != source["binaries"]["qwen-prefix-study"]:
        raise ValueError("phase diagnostic binary differs from the measured research binary")
    if sha(root.parent / "runtime-source-confidence.tar.gz") != source["runtime_source_sha256"]:
        raise ValueError("phase diagnostic runtime source differs")
    manifest = json.loads((root.parent / "heldout-workloads-v2/manifest.json").read_text())
    entry = manifest["scenarios"]["0"]
    if sha(root.parent / "heldout-workloads-v2" / entry["file"]) != entry["sha256"]:
        raise ValueError("phase diagnostic prompt differs from its frozen source")
    sys.path.insert(0, str(repo / "research/mtp-context-depth-20260921"))
    from loop_audit import loop_candidate  # noqa: PLC0415

    records, exclusions, identity = {}, [], []
    for cycle in (0, 1):
        group = {}
        for arm in ARMS:
            name = f"r{cycle}-{arm}"
            directory = root / name
            if (root / f"{name}.exit").read_text().strip() != "0":
                raise ValueError("a phase diagnostic native arm failed: " + name)
            command = (root / f"{name}.command.txt").read_text()
            expected_pmin = {"off": "0", "prob": "0.00000001", "cut": "0.15"}[arm]
            required = {
                "arm=fixed:3", "seed=20760000", "max_new=512",
                "ctx=32768", "temperature=0.7", "top_k=20", "top_p=0.95",
                f"MEMRA_SPEC_PMIN={expected_pmin}",
                "MEMRA_SPEC_PMIN0=0", "MEMRA_SPEC_PHASE=1",
                "MEMRA_SPEC_PHASE_SYNC=1",
            }
            if not required <= set(command.splitlines()):
                raise ValueError("phase diagnostic command differs: " + name)
            if (root / f"{name}.gpu.csv").stat().st_size == 0:
                raise ValueError("phase diagnostic telemetry is missing: " + name)
            native = rows(directory / "turns.tsv")
            phases = parse_log(root / f"{name}.log")
            if len(native) != 8:
                raise ValueError("phase diagnostic has another request count")
            combined = []
            for turn, (record, phase) in enumerate(zip(native, phases), 1):
                output = ids(directory / f"turn-{turn}.output.ids")
                if len(output) != int(record["output_tokens"]):
                    raise ValueError("phase diagnostic output tape differs from timing")
                combined.append({
                    **phase,
                    "turn": turn,
                    "kind": entry["cells"][turn - 1]["kind"],
                    "length_target": entry["cells"][turn - 1]["length_target"],
                    "output_tokens": len(output),
                    "elapsed_s": float(record["elapsed_s"]),
                    "loop": loop_candidate(output),
                    "output_sha256": sha(directory / f"turn-{turn}.output.ids"),
                    "prompt_sha256": sha(directory / f"turn-{turn}.prompt.ids"),
                })
            group[arm] = combined
        for turn in range(1, 9):
            current = [group[arm][turn - 1] for arm in ARMS]
            if len({row["prompt_sha256"] for row in current}) != 1:
                raise ValueError("phase diagnostic compared different prompts")
            if any(row["loop"] for row in current):
                exclusions.append({"cycle": cycle, "turn": turn})
        # Off versus prob is a confidence-overhead control only when the
        # proposal lengths and actual sampled output IDs remain identical.
        prob_matches = all(
            group["off"][i]["output_sha256"] == group["prob"][i]["output_sha256"]
            and sum(group["prob"][i]["draft_lengths"][:3]) == 0
            and sum(group["off"][i]["draft_lengths"][:3]) == 0
            and sum(group["prob"][i]["draft_lengths"][4:]) == 0
            and sum(group["off"][i]["draft_lengths"][4:]) == 0
            for i in range(8)
        )
        identity.append({"cycle": cycle, "off_prob_equal_no_cut": prob_matches})
        records[cycle] = group

    scoped = {}
    for arm in ARMS:
        included = [
            records[cycle][arm][turn - 1]
            for cycle in (0, 1) for turn in range(1, 9)
            if records[cycle][arm][turn - 1]["kind"] == "code"
            and {"cycle": cycle, "turn": turn} not in exclusions
        ]
        if not included:
            raise ValueError("no nonlooped code requests remain for the diagnostic")
        scoped[arm] = summarize(included)
    isolated = all(row["off_prob_equal_no_cut"] for row in identity)
    result = {
        "status": "probability-overhead-isolated" if isolated else "diagnostic-no-isolated-overhead",
        "scope": "post-result sampled source-path diagnostic at max_new=512 with phase synchronization; not pooled into native serving-rate evidence",
        "source_binary_sha256": source["binaries"]["qwen-prefix-study"],
        "source_runtime_sha256": source["runtime_source_sha256"],
        "matched_loop_exclusions": exclusions,
        "off_prob_identity": identity,
        "code": scoped,
    }
    if isolated:
        off = scoped["off"]["phase_ms_per_100_tokens"]["draft_ms"]
        prob = scoped["prob"]["phase_ms_per_100_tokens"]["draft_ms"]
        result["prob_vs_off_draft_ms_per_100_tokens"] = prob - off
        result["prob_vs_off_draft_percent"] = 100 * (prob / off - 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.repo)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "prob_vs_off_draft_percent": result.get("prob_vs_off_draft_percent"),
        "code_requests_per_arm": result["code"]["off"]["requests"],
    }))


if __name__ == "__main__":
    main()
