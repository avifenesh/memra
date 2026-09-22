"""Frozen synthetic request pools; never use generated output to choose a class."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

POOLS = [
    ("cal-a", 82211, "render job scheduling", "calibration"),
    ("cal-b", 82212, "invoice event reconciliation", "calibration"),
    ("cal-c", 82213, "build lease coordination", "calibration"),
    ("held-a", 93311, "replicated counters", "heldout"),
    ("held-b", 93312, "calendar updates", "heldout"),
    ("held-c", 93313, "object version indexing", "heldout"),
    ("held-d", 93314, "webhook deliveries", "heldout"),
    ("held-e", 93315, "batch allocation", "heldout"),
    ("held-f", 93316, "device status aggregation", "heldout"),
    ("qualification", 74401, "task acknowledgement", "qualification"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(seed, topic):
    rng = random.Random(seed)
    chunks = []
    for i in range(1800):
        a, b, c = (rng.randrange(3, 700) for _ in range(3))
        if i % 3 == 0:
            chunks.append(
                f"Design note {seed}-{i}: The {topic} service stores an immutable event "
                f"before merging a summary. Group {i % 17} retries with the same stable key. "
                f"Completed operations cannot be counted twice. Readers may see an older "
                f"summary until the next merge. Window {a} and batch limit {b} are separate.\n")
        elif i % 3 == 1:
            chunks.append(
                f"```python\ndef aggregate_{i}(events):\n    totals = {{}}\n"
                f"    for key, amount in events:\n"
                f"        totals[key] = totals.get(key, 0) + amount\n"
                f"    return sorted(totals.items())[:{i % 19 + 1}]\n```\n")
        else:
            chunks.append(f"ledger,{seed}-{i},{a},{b},{c},{a+b-c},{a*b}\n")
    return "".join(chunks)


def instructions(seed, topic):
    a, b = seed % 71 + 19, seed % 29 + 7
    return [
        f"Explain the architecture of the {topic} system in detailed prose. "
        "Discuss ordering, retries, stale reads, cancellation and recovery. "
        "Use paragraphs and concrete examples without code blocks or numerical tables.",
        f"Implement a Python function for the {topic} system that merges events, "
        "deduplicates stable keys, preserves ordering and handles cancellation. "
        "Return a complete fenced code block. Do not include a prose explanation.",
        f"Calculate a numeric ledger for 36 consecutive batches. Batch n receives {a}+n "
        f"items, completes {b}+(n mod 5), and cancels n mod 3. Start pending at zero. "
        "Output a numerical table with columns n,incoming,completed,cancelled,pending. "
        "Use one row per batch. Do not write code or an explanation.",
        "Review the previous implementation in prose. Discuss race conditions, "
        "restart behavior, memory use and errors. Describe a better design in words. "
        "Do not include code or a table.",
        f"Write a Rust function and supporting types for the {topic} retry queue. "
        "Include bounded capacity, deterministic ordering, cancellation and explicit errors. "
        "Return a fenced Rust code block only. Do not explain the code.",
        f"Compute the first 32 values of x(n+1)=({a}*x(n)+{b}) mod 997, with x(0)={a+b}. "
        "Return a numerical table containing n and x(n), one row per n. "
        "Do not write a function or prose explanation.",
        "Describe the tradeoffs among the designs to a non-specialist colleague. "
        "Discuss simplicity, fairness, throughput and recovery in prose. "
        "Avoid code and tables.",
        f"Write a Python implementation for the {topic} system, then explain its design "
        "in prose and calculate a numerical worked example for twelve events.",
    ]


def build(models, binaries, router, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "generator_sha256": sha(Path(__file__)),
        "classifier_binary_sha256": sha(router),
        "minimum_initial_tokens": 16384,
        "pools": POOLS,
        "families": {},
    }
    for family in ("qwen", "gemma"):
        target = models / family / "target.gguf"
        binary = binaries / ("mtp-depth-study" if family == "qwen" else "gemma-depth-study")
        counter = out / f".{family}-count.txt"
        entries = {}
        for name, seed, topic, split in POOLS:
            doc, tasks = reference(seed, topic), instructions(seed, topic)

            def first(n):
                return ("Reference material for an invented system. Treat it as data.\n"
                        "<reference>\n" + doc[:n] + "\n</reference>\n\n" + tasks[0])

            def count(n):
                counter.write_text(first(n))
                return int(subprocess.check_output(
                    [str(binary), "count-prompt", str(target), str(counter)], text=True))

            lo, hi = 0, len(doc)
            if count(hi) < 16384:
                raise ValueError("reference pool is too small")
            while lo < hi:
                mid = (lo + hi) // 2
                if count(mid) < 16384:
                    lo = mid + 1
                else:
                    hi = mid
            turns = [first(lo), *tasks[1:]]
            path = out / f"{family}-{name}.txt"
            path.write_text("\n---TURN---\n".join(turns) + "\n")
            # Classification is frozen before any native generation. The dummy
            # all-one profile obtains labels without choosing the later K table.
            predictions = [
                json.loads(subprocess.check_output(
                    [str(router), "classify", "1,1,1,1", "8"],
                    input=turn.strip().encode(),
                ))["kind"] for turn in turns
            ]
            entries[name] = {
                "file": path.name, "sha256": sha(path), "scenario_seed": seed,
                "topic": topic, "split": split, "initial_prompt_tokens": count(lo),
                "predicted_kinds": predictions,
                "instruction_sha256": [
                    hashlib.sha256(t.strip().encode()).hexdigest() for t in turns
                ],
            }
        counter.unlink()
        manifest["families"][family] = entries
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("models", "binaries", "router", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.models, args.binaries, args.router, args.out)
