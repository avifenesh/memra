"""Freeze one mixed code/prose/math train, selection, and final split."""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re

import duckdb


MBPP_SHA = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
GSM_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
WILD_SHA = "e1fbdc5d46deca34e77f92991c23b0476860e26adf2bf28379a3bb3bf00d3ed6"
V9_SHA = "ebceeeffdf36128b98b303c124cf7459966877991d66afce15f7ecc6ffd4d6ee"
V10_FIRST_SHA = "2142c62394fbb3f90c684a942e09a227f24f8a32d0011f550dd7b6841ed32dc1"
V11_SHA = "71c538295aeb970856f5feb5d11c888aa2dea4d220f41ab844d6e0587108a3df"
DUCKDB_VERSION = "1.3.2"
TURNS = 8
DELIMITER = "\n---TURN---\n"
SPLITS = (
    ("qualification", 1),
    ("training", 16),
    ("validation", 8),
    ("final", 24),
)
DOMAINS = ("code", "prose", "math")
CODE_SEED = 26092611
MATH_SEED = 26092612
PROSE_SEED = 26092630
EXCLUDED_CODE_DIAGNOSTIC_IDS = {561, 729, 694, 720, 264, 467, 174, 613}
STRICT_TAGS = (
    "Advice seeking",
    "Brainstorming",
    "Creative Writing",
    "Editing",
    "Role playing",
)
ADDITIONAL_TAGS = ("Information seeking", "Planning", "Reasoning")
STRICT_QUOTA = {
    "qualification": {
        "Creative Writing": 5, "Editing": 2, "Brainstorming": 1,
    },
    "training": {
        "Creative Writing": 9, "Editing": 3, "Role playing": 1,
        "Brainstorming": 2, "Advice seeking": 1,
    },
    "validation": {
        "Creative Writing": 19, "Editing": 7, "Role playing": 2,
        "Brainstorming": 2, "Advice seeking": 2,
    },
    "final": {
        "Creative Writing": 76, "Editing": 27, "Role playing": 8,
        "Brainstorming": 10, "Advice seeking": 7,
    },
}
ADDITIONAL_QUOTA = {
    "qualification": {},
    "training": {"Information seeking": 42, "Planning": 34, "Reasoning": 36},
    "validation": {"Information seeking": 12, "Planning": 10, "Reasoning": 10},
    "final": {"Information seeking": 24, "Planning": 20, "Reasoning": 20},
}
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{12,}|api[_-]?key\s*[:=]\s*\S+|"
    r"password\s*[:=]\s*\S+)\b", re.I,
)
PHONE = re.compile(r"\b(?:\+?\d[\d(). -]{7,}\d)\b")
GOLD = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def locked(path, expected):
    if sha(path) != expected:
        raise ValueError(f"mixed-workload source changed: {path.name}")


def read_jsonl(path, expected_rows):
    rows = [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != expected_rows:
        raise ValueError(f"mixed-workload source count differs: {path.name}")
    return rows


def verify_lock(lock):
    expected_sources = {
        "mbpp_full": MBPP_SHA,
        "gsm8k_test": GSM_SHA,
        "wildbench_v2": WILD_SHA,
    }
    expected_exclusions = {
        "v9_code_manifest_sha256": V9_SHA,
        "v10_first_noncode_manifest_sha256": V10_FIRST_SHA,
        "v11_full_noncode_manifest_sha256": V11_SHA,
    }
    if (
        lock["schema"] != 1
        or {key: lock["sources"][key]["sha256"] for key in expected_sources}
        != expected_sources
        or lock["exclusions"] != expected_exclusions
        or lock["prose_selection"]["pinned_reader"]
        != f"duckdb=={DUCKDB_VERSION}"
        or tuple(lock["prose_selection"]["strict_tags"]) != STRICT_TAGS
        or tuple(lock["prose_selection"]["additional_training_and_mixed_tags"])
        != ADDITIONAL_TAGS
        or lock["split"] != {
            "turns_per_conversation": TURNS,
            **{
                f"{phase}_conversations_per_domain": count
                for phase, count in SPLITS
            },
        }
        or duckdb.__version__ != DUCKDB_VERSION
    ):
        raise ValueError("mixed-workload source lock differs")


def old_ids(v9_path, v10_path, v11_path):
    for path, digest in (
        (v9_path, V9_SHA), (v10_path, V10_FIRST_SHA),
        (v11_path, V11_SHA),
    ):
        locked(path, digest)
    v9 = json.loads(v9_path.read_text())
    v10 = json.loads(v10_path.read_text())
    v11 = json.loads(v11_path.read_text())
    code = {
        task for group in v9["groups"].values()
        for session in group for task in session["task_ids"]
    } | EXCLUDED_CODE_DIAGNOSTIC_IDS
    if len(code) != 400:
        raise ValueError("historical code exclusions differ")
    v10_math = {
        task for group in v10["groups"].values()
        for session in group["gsm8k"] for task in session["task_ids"]
    }
    v11_math = {
        task for group in v11["groups"].values()
        for session in group["gsm8k"] for task in session["task_ids"]
    }
    if (
        len(v10_math) != 136 or len(v11_math) != 328
        or v10_math & v11_math
    ):
        raise ValueError("historical math exclusions differ")
    return code, v10_math | v11_math


def code_rows(path, excluded):
    locked(path, MBPP_SHA)
    source = read_jsonl(path, 974)
    if len({row["task_id"] for row in source}) != 974:
        raise ValueError("full MBPP task IDs repeat")
    usable = [
        {
            "task_id": row["task_id"],
            "source_text": row["text"].strip(),
            "test_list": row["test_list"],
            "test_setup_code": row["test_setup_code"],
        }
        for row in source
        if row["task_id"] > 10
        and row["task_id"] not in excluded
        and not row["test_setup_code"].strip()
        and len(row["test_list"]) >= 3
        and DELIMITER.strip() not in row["text"]
    ]
    if len(usable) < sum(count for _, count in SPLITS) * TURNS:
        raise ValueError("too few fresh MBPP code tasks")
    usable.sort(key=lambda row: row["task_id"])
    random.Random(CODE_SEED).shuffle(usable)
    return usable


def math_rows(path, excluded):
    locked(path, GSM_SHA)
    source = read_jsonl(path, 1319)
    usable = []
    for index, row in enumerate(source):
        if index in excluded or DELIMITER.strip() in row["question"]:
            continue
        answer = row["answer"].split("#### ")[-1].strip()
        if not GOLD.fullmatch(answer):
            continue
        try:
            Decimal(answer.replace(",", ""))
        except InvalidOperation:
            continue
        usable.append({
            "task_id": index,
            "question": row["question"].strip(),
            "gold_answer": answer,
        })
    if len(usable) < sum(count for _, count in SPLITS) * TURNS:
        raise ValueError("too few fresh GSM8K math tasks")
    random.Random(MATH_SEED).shuffle(usable)
    return usable


def prose_rows(path):
    locked(path, WILD_SHA)
    rows = duckdb.connect().execute(
        "select id, session_id, conversation_input, checklist, primary_tag "
        "from read_parquet(?)", [str(path)],
    ).fetchall()
    if (
        len(rows) != 1024
        or len({row[0] for row in rows}) != 1024
        or len({row[1] for row in rows}) != 1024
    ):
        raise ValueError("WildBench source inventory differs")
    result = {tag: [] for tag in (*STRICT_TAGS, *ADDITIONAL_TAGS)}
    for task_id, session_id, conversation, checklist, tag in rows:
        if (
            tag not in result or len(conversation) != 1
            or conversation[0].get("role") != "user"
            or conversation[0].get("language") != "English"
            or conversation[0].get("redacted")
            or conversation[0].get("toxic")
            or not checklist
        ):
            continue
        prompt = conversation[0].get("content", "").strip()
        if (
            not 30 <= len(prompt) <= 5000
            or DELIMITER.strip() in prompt
            or any(not isinstance(item, str) for item in checklist)
            or any(
                pattern.search(prompt + "\n" + "\n".join(checklist))
                for pattern in (EMAIL, KEY, PHONE)
            )
        ):
            continue
        result[tag].append({
            "task_id": task_id,
            "source_session_id": session_id,
            "prompt": prompt,
            "checklist": checklist,
            "tag": tag,
        })
    expected = {
        "Advice seeking": 12,
        "Brainstorming": 18,
        "Creative Writing": 132,
        "Editing": 47,
        "Information seeking": 89,
        "Planning": 72,
        "Reasoning": 74,
        "Role playing": 12,
    }
    if {tag: len(group) for tag, group in result.items()} != expected:
        raise ValueError("screened WildBench category census differs")
    for ordinal, tag in enumerate(sorted(result)):
        result[tag].sort(key=lambda row: row["task_id"])
        random.Random(PROSE_SEED + ordinal).shuffle(result[tag])
    return result


def allocate_prose(by_tag):
    offsets = {tag: 0 for tag in by_tag}
    groups = {}
    for phase_index, (phase, count) in enumerate(SPLITS):
        chosen = []
        for quotas in (STRICT_QUOTA[phase], ADDITIONAL_QUOTA[phase]):
            for tag in sorted(quotas):
                size = quotas[tag]
                selected = by_tag[tag][offsets[tag]:offsets[tag] + size]
                if len(selected) != size:
                    raise ValueError(f"WildBench {tag}/{phase} quota differs")
                chosen.extend(selected)
                offsets[tag] += size
        if len(chosen) != count * TURNS:
            raise ValueError(f"WildBench {phase} arm count differs")
        random.Random(PROSE_SEED + 100 + phase_index).shuffle(chosen)
        groups[phase] = chosen
    if (
        len({row["task_id"] for group in groups.values() for row in group})
        != sum(count for _, count in SPLITS) * TURNS
        or sum(
            row["tag"] in STRICT_TAGS for row in groups["final"]
        ) != 128
    ):
        raise ValueError("WildBench mixed-domain split differs")
    return groups


def code_turn(row, number):
    prompt = (
        row["source_text"]
        + "\n\nThe function must satisfy this example:\n"
        + row["test_list"][0].strip()
        + "\n\nGive only one fenced Python code block, with no reasoning "
        "or prose. Earlier turns are context, not requests to repeat "
        "earlier code."
    )
    return prompt, {
        "turn": number, "task_id": row["task_id"],
        "test_setup_code": row["test_setup_code"],
        "test_list": row["test_list"], "visible_test_count": 1,
    }


def math_turn(row, number):
    prompt = row["question"] + "\n\nShow your work and end with #### <number>."
    return prompt, {
        "turn": number, "task_id": row["task_id"],
        "gold_answer": row["gold_answer"],
    }


def prose_turn(row, number):
    return row["prompt"], {
        "turn": number, "task_id": row["task_id"],
        "source_session_id": row["source_session_id"],
        "tag": row["tag"], "prompt": row["prompt"],
        "checklist": row["checklist"],
    }


def conversation(domain, rows, seed, out):
    make = {"code": code_turn, "prose": prose_turn, "math": math_turn}[domain]
    prepared = [make(row, number) for number, row in enumerate(rows, 1)]
    prompts = [prompt for prompt, _ in prepared]
    out.write_bytes((DELIMITER.join(prompts) + "\n").encode())
    turns = [
        {
            **item,
            "user_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        for prompt, item in prepared
    ]
    return {
        "domain": domain, "file": out.name, "sha256": sha(out),
        "seed": seed, "task_ids": [row["task_id"] for row in rows],
        "turns": turns,
    }


def build(args):
    lock_path = args.source_lock
    lock = json.loads(lock_path.read_text())
    verify_lock(lock)
    code_excluded, math_excluded = old_ids(
        args.v9, args.v10_first, args.v11,
    )
    rows = {
        "code": code_rows(args.mbpp, code_excluded),
        "math": math_rows(args.gsm, math_excluded),
    }
    prose = allocate_prose(prose_rows(args.wildbench))
    args.out.mkdir(parents=True, exist_ok=False)
    groups = {phase: {} for phase, _ in SPLITS}
    for domain_index, domain in enumerate(DOMAINS):
        offset = 0
        for phase_index, (phase, count) in enumerate(SPLITS):
            sessions = []
            for index in range(count):
                tasks = (
                    prose[phase][index * TURNS:(index + 1) * TURNS]
                    if domain == "prose" else
                    rows[domain][offset:offset + TURNS]
                )
                if domain != "prose":
                    offset += TURNS
                path = args.out / f"{phase}-{domain}-{index}.txt"
                seed = (
                    26093000 + domain_index * 1000
                    + phase_index * 200 + index
                )
                sessions.append(conversation(domain, tasks, seed, path))
            groups[phase][domain] = sessions
    counts = {
        domain: len({
            task for phase in groups.values()
            for session in phase[domain] for task in session["task_ids"]
        })
        for domain in DOMAINS
    }
    needed = sum(count for _, count in SPLITS) * TURNS
    if counts != {domain: needed for domain in DOMAINS}:
        raise ValueError("mixed-domain task IDs overlap")
    manifest = {
        "schema": 1,
        "scope": "one-policy mixed code/open-prose/math native conversations",
        "source_lock_sha256": sha(lock_path),
        "generator_sha256": sha(Path(__file__)),
        "sources_sha256": {
            "mbpp_full": MBPP_SHA, "gsm8k_test": GSM_SHA,
            "wildbench_v2": WILD_SHA,
        },
        "exclusions_sha256": {
            "v9_code_manifest": V9_SHA,
            "v10_first_noncode_manifest": V10_FIRST_SHA,
            "v11_full_noncode_manifest": V11_SHA,
        },
        "assignment_seeds": {
            "code": CODE_SEED, "math": MATH_SEED,
            "prose": PROSE_SEED,
        },
        "split_conversations": dict(SPLITS),
        "turns_per_conversation": TURNS,
        "prose_strict_final_turns": 128,
        "prose_source_attribution": (
            "AllenAI WildBench v2, CC BY 4.0; derived from public WildChat"
        ),
        "groups": groups,
    }
    (args.out / "manifest.json").write_bytes(encoded(manifest))
    return {
        "manifest_sha256": sha(args.out / "manifest.json"),
        "conversations_per_domain": dict(SPLITS),
        "task_ids_per_domain": counts,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "source-lock", "mbpp", "gsm", "wildbench",
        "v9", "v10-first", "v11", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "source_lock", "mbpp", "gsm", "wildbench",
        "v9", "v10_first", "v11", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    print(json.dumps(build(args), sort_keys=True))


if __name__ == "__main__":
    main()
