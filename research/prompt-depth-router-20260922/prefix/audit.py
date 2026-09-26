"""Reconstruct the bounded prefix and audit each independent native request."""

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re

K_BY_KIND = {"prose": 2, "code": 4, "numeric": 4, "mixed": 3, "unknown": 3}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def ids(path):
    return [int(value) for value in Path(path).read_text().split()]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def inspected_prefix(data, used, budget):
    """Mirror only UTF-8/partial-word framing, not the Rust classifier."""
    if len(data) > 16384:
        return b""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data":
            return b""
        text = data[:error.start].decode("utf-8")
    if used == budget and text and text[-1].isascii() and text[-1].isalpha():
        boundary = next((i for i in range(len(text) - 1, -1, -1) if text[i].isspace()), None)
        text = "" if boundary is None else text[:boundary]
    return text.encode()


def audit_prefix(root, turn, route, prompt):
    budget, used = int(route["budget"]), int(route["tokens_read"])
    user_tokens = int(route["user_tokens"])
    if budget not in (64, 128, 256) or used != min(budget, user_tokens):
        raise ValueError("prefix token count differs from the requested bound")
    span_rows = rows(root / f"turn-{turn}.prefix-span.tsv")
    if len(span_rows) != 1:
        raise ValueError("missing or duplicate prefix span")
    start, end, skip = (int(span_rows[0][key]) for key in ("start", "end", "leading_skip"))
    if not 0 <= start <= end <= len(prompt) or end - start != user_tokens:
        raise ValueError("prefix span differs from the user-token count")
    selected = ids(root / f"turn-{turn}.prefix.ids")
    if selected != prompt[start:start + used]:
        raise ValueError("forecaster did not receive the first user tokens")
    pieces = rows(root / f"turn-{turn}.prefix-bytes.tsv")
    if [int(row["id"]) for row in pieces] != selected:
        raise ValueError("prefix token/byte receipt order differs")
    mapping = {}
    for row in pieces:
        token, piece = int(row["id"]), bytes.fromhex(row["hex"])
        if token in mapping and mapping[token] != piece:
            raise ValueError("one token has inconsistent decoded bytes")
        mapping[token] = piece
    decoded = b"".join(mapping[token] for token in selected)
    if not 0 <= skip <= len(decoded):
        raise ValueError("invalid template overlap")
    prefix = decoded[skip:]
    if prefix != (root / f"turn-{turn}.prefix.bin").read_bytes():
        raise ValueError("prefix bytes do not reconstruct from the inspected tokens")
    inspected = inspected_prefix(prefix, used, budget)
    if len(prefix) != int(route["decoded_bytes"]) or len(inspected) != int(route["inspected_bytes"]):
        raise ValueError("prefix byte counts do not reconstruct")
    if K_BY_KIND.get(route["kind"]) != int(route["k"]):
        raise ValueError("selected K disagrees with the frozen prediction map")
    if not 0 < int(route["core_ns"]) <= int(route["routing_ns"]):
        raise ValueError("invalid forecaster timing")
    return {"prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "inspected_sha256": hashlib.sha256(inspected).hexdigest(),
            "prefix_ids_sha256": sha(root / f"turn-{turn}.prefix.ids")}


def coverage(root, turn, requested):
    public = ids(root / f"turn-{turn}.output.ids")
    mapping = {int(row["id"]): bytes.fromhex(row["hex"]) for row in rows(root / "token-bytes.tsv")}
    pieces = [mapping[token] for token in public]
    tape = b"".join(pieces)
    answer = (root / f"turn-{turn}.answer.txt").read_bytes()
    # Native Qwen removes the reasoning segment. Gemma's native answer is its
    # returned text. Empty special-token byte pieces are retained in token counts.
    position = tape.rfind(answer) if answer else -1
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    final_tokens = sum(
        bool(piece) and a < position + len(answer) and b > position
        for piece, a, b in zip(pieces, offsets, offsets[1:])
    ) if position >= 0 else 0
    blocks = []
    text = answer.decode("utf-8", errors="replace")
    # Closed, line-delimited fences only: a mention of backticks is not code.
    for match in re.finditer(r"(?m)^```(?:python|py)?[ \t]*\r?\n(.*?)^```[ \t]*$", text, re.S):
        body = match.group(1)
        try:
            tree = ast.parse(body)
            parses = bool(tree.body)
        except (SyntaxError, ValueError):
            parses = False
        blocks.append({"bytes": len(body.encode()), "python_parses": parses})
    code = any(block["python_parses"] and block["bytes"] >= 80 for block in blocks)
    prose = len(answer.strip()) >= 200 and not re.search(r"(?m)^```", text)
    return {
        "requested": requested, "answer_bytes": len(answer),
        "native_answer_located": position >= 0,
        "answer_start_byte": position,
        "final_tokens_with_bytes": final_tokens,
        "nonfinal_tokens_with_bytes": sum(bool(piece) for piece in pieces) - final_tokens,
        "empty_byte_tokens": sum(not piece for piece in pieces),
        "python_blocks": blocks, "code_covered": code, "prose_covered": prose,
        "requested_format_covered": code if requested == "code" else prose,
        "scope": "format coverage and Python syntax only; no task-correctness claim",
    }


def audit_run(root, entry, runtime_arm, seed, metrics, context_auditor, loop_candidate):
    root = Path(root)
    turns, routing, spans = (rows(root / name) for name in ("turns.tsv", "routing.tsv", "spans.tsv"))
    if len(turns) != 8 or len(routing) != 8 or len(entry["cells"]) != 8:
        raise ValueError("incomplete eight-request matrix group")
    if metrics["turns"] != 8 or metrics["tokens"] != sum(int(t["output_tokens"]) for t in turns):
        raise ValueError("summary token counts differ from raw requests")
    elapsed = sum(float(t["elapsed_s"]) for t in turns)
    if (not math.isfinite(elapsed) or elapsed <= 0
            or not math.isclose(elapsed, metrics["elapsed_s"], abs_tol=1e-6)
            or not math.isclose(metrics["tokens"] / elapsed, metrics["e2e_tok_s"], rel_tol=1e-7)):
        raise ValueError("complete-request times and throughput differ")
    if any(t["resumed"] != "false" or int(t["cached_tokens"]) or int(t["checkpoint_tokens"])
           for t in turns):
        raise ValueError("independent request unexpectedly reused a cache")
    if sum(int(t["drafted"]) for t in turns) <= 0:
        raise ValueError("speculative path did not engage")
    results = []
    for turn, (record, route, cell) in enumerate(zip(turns, routing, entry["cells"]), 1):
        if int(record["turn"]) != turn or int(route["turn"]) != turn or int(record["seed"]) != seed:
            raise ValueError("request identity changed")
        prompt = ids(root / f"turn-{turn}.prompt.ids")
        output = ids(root / f"turn-{turn}.output.ids")
        if (sha(root / f"turn-{turn}.user.txt") != cell["user_sha256"]
                or len(prompt) != cell["input_tokens"] or len(prompt) != int(route["input_tokens"])
                or int(route["user_tokens"]) != cell["user_tokens"]
                or len(prompt) != int(record["new_input_tokens"])
                or len(output) != int(record["output_tokens"])):
            raise ValueError("frozen input/output counts or user text changed")
        k = int(route["k"])
        if runtime_arm.startswith("prefix:"):
            budget = int(runtime_arm.split(":")[1])
            expected = cell["predictions"][str(budget)]
            if route["source"] != "prefix" or int(route["budget"]) != budget:
                raise ValueError("requested prefix policy did not run")
            if (route["kind"], k) != (expected["kind"], expected["k"]):
                raise ValueError("prediction differs from the pre-generation tokenizer preparation")
            bound = audit_prefix(root, turn, route, prompt)
        else:
            expected_k = (int(runtime_arm.split(":")[1]) if runtime_arm.startswith("fixed:")
                          else int(runtime_arm.split(":")[1].split(",")[turn - 1]))
            if k != expected_k or int(route["tokens_read"]) or int(route["budget"]):
                raise ValueError("fixed control or replay unexpectedly classified input")
            bound = None
        full_rounds = [s for s in spans if int(s["turn"]) == turn and s["eligible"] == "true"]
        if not full_rounds or any(int(s["k"]) != k for s in full_rounds):
            raise ValueError("chosen K did not engage throughout the request")
        results.append({
            "turn": turn, "kind": cell["kind"], "length_target": cell["length_target"],
            "user_tokens": cell["user_tokens"], "input_tokens": len(prompt),
            "output_tokens": len(output), "elapsed_s": float(record["elapsed_s"]),
            "drafted": int(record["drafted"]), "accepted": int(record["accepted"]),
            "k": k, "prediction": route["kind"], "routing_ns": int(route["routing_ns"]),
            "core_ns": int(route["core_ns"]), "prefix": bound,
            "format": coverage(root, turn, cell["kind"]), "loop": loop_candidate(output),
            "finish_reason": record.get("finish_reason", "not-recorded-by-this-driver"),
            "output_sha256": sha(root / f"turn-{turn}.output.ids"),
        })
    return {"requests": results,
            "context_accounting": context_auditor(root, "fixed", 7 if "qwen" in root.parts else 5)}


def same_tapes(first, second):
    for turn in range(1, 9):
        for kind in ("prompt", "output"):
            name = f"turn-{turn}.{kind}.ids"
            if (first / name).read_bytes() != (second / name).read_bytes():
                raise ValueError(f"token identity failed at request {turn}, tape {kind}")
