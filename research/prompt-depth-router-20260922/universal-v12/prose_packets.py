"""Freeze blinded, reversed-order WildBench checklist comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
import re


FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
VALIDATION_SHA = "bf920b82e0176c4304bcc562ccce34a28384082c888355a280620163a45e5313"
TEMPLATE_SHA = "ccd57bd8c4c73f4f83cf8963ef3c2697c1c7b9e907ead91e0d0512cca4ae7a11"
REFERENCE = "fixed-k20-d3-c0"
PLACEHOLDER = re.compile(
    r"\{\$(history|user_query|candidate_A|candidate_B|checklist)\}"
)
HISTORY = (
    "Each source query is a standalone benchmark request. "
    "Score the current response to the current query. "
    "Earlier native turns test cached conversation performance."
)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def render(template, query, checklist, a, b):
    replacements = {
        "history": HISTORY,
        "user_query": query,
        "candidate_A": a,
        "candidate_B": b,
        "checklist": "\n".join("- " + item for item in checklist),
    }
    found = set(PLACEHOLDER.findall(template))
    if found != set(replacements):
        raise ValueError("pinned WildBench judge template changed")
    return PLACEHOLDER.sub(lambda match: replacements[match.group(1)],
                           template)


def freeze(args):
    expected = {"validation": VALIDATION_SHA, "final": FULL_SHA}[args.phase]
    if (
        sha(args.workloads / "manifest.json") != expected
        or sha(args.template) != TEMPLATE_SHA
    ):
        raise ValueError("prose judge source or phase differs")
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    arms = json.loads(args.arms.read_text())
    if (
        manifest["schema"] != 1
        or arms["schema"] != 1
        or arms["phase"] != args.phase
        or arms["source_manifest_sha256"] != FULL_SHA
        or len({item["label"] for item in arms["arms"]})
        != len(arms["arms"])
    ):
        raise ValueError("prose judge arm inventory differs")
    by_label = {item["label"]: item for item in arms["arms"]}
    if (
        REFERENCE not in by_label
        or by_label[REFERENCE]["role"] != "fixed"
    ):
        raise ValueError("prose judge lacks fixed reference")
    if args.phase == "validation":
        pairs = [
            (item["label"], REFERENCE)
            for item in arms["arms"]
            if item["role"] in ("fixed", "learned")
            and item["label"] != REFERENCE
        ]
    else:
        selected = json.loads(
            args.arms.with_name("shared-selected.json").read_text()
        )
        if (
            selected["status"] != "selected"
            or arms["selected_from_validation"]
            != sha(args.arms.with_name("shared-selected.json"))
        ):
            raise ValueError("final prose judge lacks shared selection")
        candidate = selected["selected_policy"]["label"]
        controls = {
            selected["global_fixed"],
            selected["domain_best_fixed_diagnostic"]["prose"],
        }
        if candidate not in by_label or not controls.issubset(by_label):
            raise ValueError("final prose comparison arm missing")
        pairs = [(candidate, control) for control in sorted(controls)]
    return manifest, arms, pairs


def packets(args):
    manifest, arms, pairs = freeze(args)
    template = args.template.read_text()
    entries = manifest["groups"][args.phase]["prose"]
    produced = []
    for index, entry in enumerate(entries):
        for turn, task in enumerate(entry["turns"], 1):
            answers = {}
            for label in {value for pair in pairs for value in pair}:
                name = f"{args.phase}-prose-{index}-{label}"
                native = json.loads(
                    (args.root / f"{name}.result.json").read_text()
                )
                if (
                    native["name"] != name
                    or native["task_ids"] != entry["task_ids"]
                    or native["cached_later_turns"] != 7
                ):
                    raise ValueError("prose judge lacks native continuation")
                answer = (
                    args.root / name / f"turn-{turn}.answer.txt"
                ).read_text()
                if not answer or len(answer) > 80000:
                    raise ValueError("prose judge response size differs")
                answers[label] = answer
            for candidate, control in pairs:
                for order in (0, 1):
                    a_label, b_label = (
                        (candidate, control) if order == 0
                        else (control, candidate)
                    )
                    prompt = render(
                        template, task["prompt"], task["checklist"],
                        answers[a_label], answers[b_label],
                    )
                    if len(prompt) > 200000:
                        raise ValueError("prose judge prompt exceeds cap")
                    produced.append({
                        "schema": 1,
                        "phase": args.phase,
                        "domain": "prose",
                        "conversation": index,
                        "turn": turn,
                        "task_id": task["task_id"],
                        "candidate": candidate,
                        "control": control,
                        "order": order,
                        "response_a": a_label,
                        "response_b": b_label,
                        "template_sha256": TEMPLATE_SHA,
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode()
                        ).hexdigest(),
                    })
    ids = [
        (item["conversation"], item["turn"], item["candidate"],
         item["control"], item["order"])
        for item in produced
    ]
    if len(ids) != len(set(ids)) or len(produced) != (
        len(entries) * 8 * len(pairs) * 2
    ):
        raise ValueError("reversed prose comparison inventory differs")
    return {
        "schema": 1,
        "phase": args.phase,
        "scope": "standalone source query, blind A/B with reversed order",
        "arms_sha256": sha(args.arms),
        "workloads_sha256": sha(args.workloads / "manifest.json"),
        "template_sha256": TEMPLATE_SHA,
        "pairs": pairs,
        "packet_count": len(produced),
        "packet_inventory_sha256": digest([
            item["prompt_sha256"] for item in produced
        ]),
    }, produced


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "workloads", "arms", "template", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "final"),
                        required=True)
    args = parser.parse_args()
    for name in ("root", "workloads", "arms", "template", "out"):
        setattr(args, name, getattr(args, name).resolve())
    manifest, entries = packets(args)
    args.out.mkdir(exist_ok=False)
    with (args.out / "packets.jsonl").open("x") as output:
        for entry in entries:
            output.write(json.dumps(entry, sort_keys=True) + "\n")
    manifest["packets_sha256"] = sha(args.out / "packets.jsonl")
    with (args.out / "manifest.json").open("x") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "phase": args.phase, "packets": len(entries),
        "packet_inventory_sha256":
        manifest["packet_inventory_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
