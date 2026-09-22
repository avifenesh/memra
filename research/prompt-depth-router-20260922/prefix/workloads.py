"""Prepare the length/type matrix with the actual native tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import selectors
import subprocess

LENGTHS = (256, 1024, 4096, 16384)
SCENARIOS = (
    ("bounded queue", "Accept jobs with stable identifiers, reject duplicate identifiers, enforce a capacity limit, and allow cancellation before completion."),
    ("configuration parser", "Parse named settings with explicit types, defaults and ranges; reject malformed or unknown settings and return precise validation errors."),
    ("event deduplication", "Consume ordered events with stable keys, count each key once, preserve first-seen order and support an explicit reset boundary."),
    ("inventory updates", "Apply additions and removals to named items, reject negative inventory, preserve transaction order and return a deterministic summary."),
    ("priority selection", "Select waiting jobs by priority and insertion order, support cancellation and bounded capacity, and keep ties deterministic."),
    ("interval merging", "Validate integer intervals, merge overlaps, preserve disjoint intervals, and report malformed or reversed endpoints clearly."),
)
CODE_OPENINGS = (
    "Implement a Python module for", "Write Python code implementing",
    "Create Python functions for", "Return a Python implementation of",
    "Build a Python module for", "Provide Python source code for",
)
PROSE_OPENINGS = (
    "Explain the design of", "Describe a reliable approach to",
    "Write a prose explanation of", "Discuss how to design",
    "Give a conceptual explanation of", "Compare design choices for",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Counter:
    def __init__(self, binary, target, stderr):
        self.errors = stderr.open("w")
        self.process = subprocess.Popen(
            [str(binary), "count-server", str(target)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors,
            text=True, bufsize=1,
        )
        self.ready = selectors.DefaultSelector()
        self.ready.register(self.process.stdout, selectors.EVENT_READ)

    def count(self, text):
        self.process.stdin.write(text.encode().hex() + "\n")
        self.process.stdin.flush()
        if not self.ready.select(timeout=45):
            raise TimeoutError("native tokenizer did not answer a preparation request")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("native tokenizer preparation stopped; inspect its retained stderr")
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 17:
            raise ValueError("native tokenizer returned an unexpected record")
        predictions = {}
        for index, budget in enumerate((64, 128, 256)):
            kind, k, used, size, ns = fields[2 + 5 * index:7 + 5 * index]
            predictions[str(budget)] = {
                "kind": kind, "k": int(k), "tokens_read": int(used),
                "decoded_bytes": int(size), "routing_ns": int(ns),
            }
            if int(used) > budget:
                raise ValueError("forecaster exceeded its token budget")
        return {"input_tokens": int(fields[0]), "user_tokens": int(fields[1]),
                "predictions": predictions}

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.ready.close()
        self.errors.close()


def reference(seed, topic):
    rng = random.Random(seed)
    notes = []
    for i in range(2200):
        amount, capacity, revision = (rng.randrange(10, 900) for _ in range(3))
        notes.append(
            f"Example record {i}: the {topic} example has identifier item_{i}, "
            f"amount {amount}, capacity {capacity}, and revision {revision}. "
            "These records illustrate data shapes only. The contract in the request "
            "defines behavior; the example list does not add new requirements.\n"
        )
    return "".join(notes)


def instruction(index, kind):
    topic, contract = SCENARIOS[index]
    if kind == "code":
        return (
            f"{CODE_OPENINGS[index]} {topic}. "
            "Use explicit types, input validation, clear errors, docstrings and usage examples. "
            "Return one fenced Python code block, without a prose explanation. "
            f"Contract: {contract} "
            "Implement the general contract; do not embed the entire reference dataset."
        )
    return (
        f"{PROSE_OPENINGS[index]} {topic} in prose for a software engineer. "
        "Discuss validation, edge cases, failures, tradeoffs and concrete worked examples. "
        "Use paragraphs, without source code or a numerical table. "
        f"Contract: {contract} "
        "Discuss the general contract; do not reproduce the entire reference dataset."
    )


def sized_prompt(counter, task, context, target, late=False):
    def text(n):
        block = "<reference>\n" + context[:n] + "\n</reference>"
        return (block + "\n\n" + task if late else task + "\n\n" + block).strip()

    base = counter.count(text(0))
    if base["user_tokens"] > target + 8:
        raise ValueError("task exceeds the shortest requested prompt length")
    lo, hi = 0, len(context)
    if counter.count(text(hi))["user_tokens"] < target:
        raise ValueError("reference pool is too short")
    while lo < hi:
        mid = (lo + hi) // 2
        if counter.count(text(mid))["user_tokens"] < target:
            lo = mid + 1
        else:
            hi = mid
    # Token counts can change locally at subword merges. Validate the actual
    # count instead of assuming character count is perfectly monotone.
    best = None
    for n in range(max(0, lo - 24), min(len(context), lo + 24) + 1):
        candidate = text(n)
        measured = counter.count(candidate)
        count = measured["user_tokens"]
        if target <= count <= target + 8:
            key = (count - target, n)
            if best is None or key < best[0]:
                best = key, candidate, measured
                if count == target:
                    break
    if best is None:
        raise ValueError("could not prepare a prompt within the recorded length tolerance")
    return best[1], best[2]


def build(models, binaries, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1, "generator_sha256": digest(Path(__file__)),
        "length_targets": LENGTHS, "prefix_budgets": (64, 128, 256),
        "scope": "instruction-first main matrix; late-task robustness is separate",
        "families": {},
    }
    for family in ("qwen", "gemma"):
        binary = binaries / f"{family}-prefix-study"
        counter = Counter(binary, models / family / "target.gguf", out / f"{family}-tokenizer.stderr")
        try:
            groups = {}
            for index, (topic, _) in enumerate(SCENARIOS):
                context = reference(940000 + index, topic)
                cells = [(length, kind) for length in LENGTHS for kind in ("prose", "code")]
                rotation = index % len(cells)
                cells = cells[rotation:] + cells[:rotation]
                if index % 2:
                    cells.reverse()
                prompts, records = [], []
                for length, kind in cells:
                    prompt, info = sized_prompt(counter, instruction(index, kind), context, length)
                    prompts.append(prompt)
                    records.append({
                        "kind": kind, "length_target": length, **info,
                        "user_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    })
                    print(json.dumps({"phase": "prepared-prompt", "family": family,
                                      "scenario": index, "kind": kind,
                                      "length_target": length,
                                      "user_tokens": info["user_tokens"]}), flush=True)
                name = f"{family}-scenario-{index}.txt"
                path = out / name
                path.write_text("\n---TURN---\n".join(prompts) + "\n")
                groups[str(index)] = {
                    "file": name, "sha256": digest(path), "topic": topic,
                    "cells": records, "seed": (20710000 if family == "qwen" else 20720000) + index,
                }
            robustness = []
            context = reference(950000, SCENARIOS[0][0])
            for length in LENGTHS:
                for kind in ("prose", "code"):
                    prompt, info = sized_prompt(counter, instruction(0, kind), context, length, late=True)
                    path = out / f"{family}-late-{kind}-{length}.txt"
                    path.write_text(prompt)
                    robustness.append({"file": path.name, "sha256": digest(path),
                                       "kind": kind, "length_target": length, **info})
            manifest["families"][family] = {
                "binary_sha256": digest(binary), "scenarios": groups,
                "late_task_robustness": robustness,
            }
        finally:
            counter.close()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("models", "binaries", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.models, args.binaries, args.out), indent=2))
