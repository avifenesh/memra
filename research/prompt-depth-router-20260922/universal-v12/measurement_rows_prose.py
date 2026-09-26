"""Turn fresh randomized prose sessions into K/D/C training labels."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "joint-v11"))
import measurement_rows as prior


def load_parent_collect():
    path = BASE / "joint-v11/collect.py"
    spec = importlib.util.spec_from_file_location(
        "v12_prose_parent_collect", path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parent_collect = load_parent_collect()
TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
SOURCE = "v12-prose"
FIXED_VARIANTS = {
    "k3-fixed-d3", "k10-fixed-d3", "k20-fixed-d3",
}


def verified(root, name, entry, label, k):
    recorded = json.loads((root / f"{name}.result.json").read_text())
    observed = prior.v9_collect.verify_session(
        root / name, entry, label, k, "training",
    )
    if recorded != observed or recorded["task_ids"] != entry["task_ids"]:
        raise ValueError(f"fresh prose native session differs: {name}")
    return recorded


def retag(rows, assignment=None):
    for row in rows:
        yield {
            **row, "source": SOURCE, "domain": "prose",
            **({"assignment": assignment} if assignment else {}),
        }


def extract(native, workloads):
    if prior.sha(workloads / "manifest.json") != TRAIN_SHA:
        raise ValueError("fresh prose training projection changed")
    manifest = json.loads((workloads / "manifest.json").read_text())
    if (
        manifest["schema"] != 1
        or manifest["source_full_manifest_sha256"] != FULL_SHA
        or set(manifest["groups"]) != {"qualification", "training"}
    ):
        raise ValueError("fresh prose training phase differs")
    wrapper = prior.Directory(native)
    classes = {}
    fixed_records = []
    explore_records = []
    random_records = []
    skipped = []
    skipped_random = []
    entries = manifest["groups"]["training"]["prose"]
    if len(entries) != 16:
        raise ValueError("fresh prose training conversation count differs")
    for index, entry in enumerate(entries):
        group = []
        for k in (3, 10, 20):
            for label in ("fixed-d3", "explore-d"):
                name = f"training-{index}-k{k}-{label}"
                group.append(verified(native, name, entry, label, k))
        random_name = f"training-{index}-random-k"
        random_record = json.loads(
            (native / f"{random_name}.result.json").read_text()
        )
        observed_random = parent_collect.verify_random(
            native / random_name, entry, index,
        )
        if random_record != prior.canonical(observed_random):
            raise ValueError("fresh prose randomized K receipt differs")
        if random_record["loops"]:
            skipped_random.append(index)
        else:
            random_records.append(random_record)
        if any(row["loops"] for row in group):
            skipped.append(index)
        else:
            fixed_records.extend(
                row for row in group
                if row["variant"].endswith("fixed-d3")
            )
            explore_records.extend(
                row for row in group
                if row["variant"].endswith("explore-d")
            )
    if len(skipped) > 2 or len(skipped_random) > 2:
        raise ValueError("fresh prose training lost too many conversations")
    prior.add_classes(
        classes, wrapper,
        fixed_records + explore_records + random_records,
    )
    k20 = [row for row in fixed_records if row["k"] == 20]
    reference_tok_s = (
        sum(row["tokens"] for row in k20)
        / sum(row["seconds"] for row in k20)
    )
    k = list(retag(prior.old_rows.k_rows(
        wrapper, {"records": fixed_records}, SOURCE, FIXED_VARIANTS,
    )))
    k += list(retag(prior.old_rows.k_rows(
        wrapper, {"records": random_records}, "v6", {"random-k"},
    ), assignment="randomized-turn"))
    d = []
    c = []
    prefixes = {}
    for kind, row in prior.old_rows.round_rows(
        wrapper, {"records": explore_records}, "v9",
        lambda record: record["variant"].endswith("explore-d"),
    ):
        if kind == "c":
            key = row["conversation"], row["turn"]
            if key not in prefixes:
                value = wrapper.ids(
                    f"native/{key[0]}/turn-{key[1]}.user-prefix.ids"
                )
                if not 1 <= len(value) <= 32:
                    raise ValueError("fresh prose C prefix differs")
                prefixes[key] = value
            c.append({
                **row, "source": SOURCE, "domain": "prose",
                "first32_user_ids": prefixes[key],
            })
        else:
            d.append({
                **row, "source": SOURCE, "domain": "prose",
            })
    if not k or not d or not c:
        raise ValueError("fresh prose lacks randomized K/D/C labels")
    return {
        "k": k, "d": d, "c": c,
    }, classes, {
        "fixed_and_depth": skipped, "random_k": skipped_random,
        "reference_tok_s": reference_tok_s,
    }


def build(native, workloads, out):
    data, classes, excluded = extract(native, workloads)
    out.mkdir(exist_ok=False)
    manifest = {
        "schema": 1, "use": "training-only",
        "scope": "fresh open-prose randomized native K/D/C",
        "training_workloads_sha256": TRAIN_SHA,
        "source_full_manifest_sha256": FULL_SHA,
        "expected_training_sessions": 112,
        "excluded_looped": {
            key: value for key, value in excluded.items()
            if key != "reference_tok_s"
        },
        "reference_tok_s": {SOURCE: excluded["reference_tok_s"]},
        "rows": {},
    }
    for kind, rows in data.items():
        path = out / f"{kind}.jsonl.gz"
        prior.write_gzip(path, rows)
        manifest["rows"][kind] = {
            "count": len(rows), "sha256": prior.sha(path),
            "by_k": {
                str(k): sum(row["draft_k"] == k for row in rows)
                for k in (3, 10, 20)
            },
        }
    prior.save(out / "token-classes.json", {
        str(token): kind for token, kind in sorted(classes.items())
    })
    manifest["token_classes_sha256"] = prior.sha(
        out / "token-classes.json"
    )
    prior.save(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for name in ("native", "workloads", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.native.resolve(), args.workloads.resolve(),
        args.out.resolve(),
    )
    print(json.dumps({
        "rows": {
            key: item["count"] for key, item in result["rows"].items()
        },
        "reference_tok_s": result["reference_tok_s"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
