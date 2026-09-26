"""Make K/D/C training rows from replayed v10 and randomized v11 receipts."""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile

import training_replay


V9 = Path(__file__).resolve().parent.parent / "joint-v9"
def load_v9(name):
    spec = importlib.util.spec_from_file_location(
        "v9_training_" + name, V9 / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v9_collect = load_v9("collect")
old_rows = load_v9("measurement_rows")


V10_WORKLOAD_SHA = "dd9fc45646931f66fee8a3b328b404d7da227765c58ce78f57536b3605fcbeb8"
V11_WORKLOAD_SHA = "655223c8e4ca61fab179f7a4708b152ec35cf351209ba5210c10956080075a58"
DOMAINS = ("ifeval", "gsm8k")
FIXED = tuple(f"fixed-k{k}-d3-c0" for k in (3, 10, 20))
FIXED_D4 = "fixed-k20-d4-c0"


class V10Archive(old_rows.Archive):
    def read(self, name):
        if name.startswith("native/"):
            name = "native/results/" + name.removeprefix("native/")
        return super().read(name)

    def receipt(self, name):
        return super().read(name)


class Directory:
    def __init__(self, root):
        self.root = root

    def read(self, name):
        return (self.root / name.removeprefix("native/")).read_bytes()

    def table(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode()),
                                   delimiter="\t"))

    def ids(self, name):
        return [int(value) for value in self.read(name).split()]


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def add_classes(classes, wrapper, records):
    for record in records:
        for row in wrapper.table(f"native/{record['name']}/token-bytes.tsv"):
            token = int(row["id"])
            kind = old_rows.classify(bytes.fromhex(row["hex"]))
            if token in classes and classes[token] != kind:
                raise ValueError("token byte class changed across model receipts")
            classes[token] = kind


def tagged(rows, source, domain):
    for row in rows:
        yield {**row, "source": source, "domain": domain}


def v10_rows(archive, custody, classes):
    receipt = json.loads(custody.read_text())
    if (
        receipt["status"] != "verified-training-projection"
        or receipt["archive_sha256"] != archive.manifest["archive_sha256"]
        or receipt["parent_archive_sha256"]
        != archive.manifest["parent_archive_sha256"]
        or receipt["model_sha256"] != archive.manifest["model_sha256"]
        or receipt["parent_replay"]["status"]
        != "archive-native-IFEval-GSM8K-noops-and-E2E-match"
        or receipt["parent_replay"]["heldout_arms"] != 320
        or receipt["sessions"] != 256
        or any(name.startswith("inputs/datasets/")
               for name in archive.manifest["members"])
    ):
        raise ValueError("v10 native archive lacks matching replay custody")
    source = json.loads(archive.receipt("inputs/workloads/manifest.json"))
    if sha_from_bytes(archive.receipt("inputs/workloads/manifest.json")) != (
        V10_WORKLOAD_SHA
    ):
        raise ValueError("v10 non-code split changed")
    output = {"k": [], "d": [], "c": []}
    excluded = {}
    reference = {}
    for domain in DOMAINS:
        records = []
        skipped = []
        for index, entry in enumerate(source["groups"]["heldout"][domain]):
            group = []
            for label in (*FIXED, FIXED_D4):
                name = f"heldout-{domain}-{index}-{label}"
                row = archive.json(f"native/{name}.result.json")
                if row["name"] != name or row["task_ids"] != entry["task_ids"]:
                    raise ValueError("v10 task-to-result match differs")
                group.append(row)
            if any(row["loops"] for row in group):
                skipped.append(index)
            else:
                records.extend(group)
        if len(skipped) > 2:
            raise ValueError(f"too few unlooped v10 {domain} conversations")
        excluded[domain] = skipped
        add_classes(classes, archive, records)
        reference_rows = [
            row for row in records if row["variant"] == FIXED[-1]
        ]
        reference[f"v10-{domain}"] = (
            sum(row["tokens"] for row in reference_rows)
            / sum(row["seconds"] for row in reference_rows)
        )
        k_records = [row for row in records if row["variant"] in FIXED]
        d_records = [
            row for row in records
            if row["variant"] in (FIXED[-1], FIXED_D4)
        ]
        output["k"].extend(tagged(
            old_rows.k_rows(
                archive, {"records": k_records}, f"v10-{domain}", set(FIXED)
            ), f"v10-{domain}", domain
        ))
        output["d"].extend(tagged(
            (row for kind, row in old_rows.round_rows(
                archive, {"records": d_records}, f"v10-{domain}",
                lambda record: record["variant"] in (FIXED[-1], FIXED_D4)
            ) if kind == "d"),
            f"v10-{domain}", domain
        ))
    return output, excluded, reference


def sha_from_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def v11_rows(native, workloads, classes):
    if sha(workloads / "manifest.json") != V11_WORKLOAD_SHA:
        raise ValueError("v11 frozen training split changed")
    source = json.loads((workloads / "manifest.json").read_text())
    wrapper = Directory(native)
    output = {"k": [], "d": [], "c": []}
    excluded = {}
    reference = {}
    spec = importlib.util.spec_from_file_location(
        "v11_training_collect", Path(__file__).resolve().parent / "collect.py"
    )
    v11_collect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v11_collect)
    for domain_index, domain in enumerate(DOMAINS):
        records = []
        skipped = []
        random_records = []
        skipped_random = []
        for index, entry in enumerate(source["groups"]["training"][domain]):
            global_index = domain_index * 16 + index
            group = []
            for k in (3, 10, 20):
                for label in ("fixed-d3", "explore-d"):
                    name = f"training-{global_index}-k{k}-{label}"
                    row = json.loads((native / f"{name}.result.json").read_text())
                    observed = v9_collect.verify_session(
                        native / name, entry, label, k, "training"
                    )
                    if row != observed or row["task_ids"] != entry["task_ids"]:
                        raise ValueError("v11 randomized native receipt differs")
                    group.append(row)
            random_name = f"training-{global_index}-random-k"
            random_row = json.loads(
                (native / f"{random_name}.result.json").read_text()
            )
            if random_row != canonical(v11_collect.verify_random(
                native / random_name, entry, global_index
            )):
                raise ValueError("v11 randomized K receipt differs")
            if random_row["loops"]:
                skipped_random.append(index)
            else:
                random_records.append(random_row)
            if any(row["loops"] for row in group):
                skipped.append(index)
            else:
                records.extend(group)
        if len(skipped) > 2 or len(skipped_random) > 2:
            raise ValueError(f"too few unlooped v11 {domain} training sessions")
        excluded[domain] = {
            "fixed_and_depth": skipped,
            "random_k": skipped_random,
        }
        add_classes(classes, wrapper, records + random_records)
        fixed = [
            row for row in records if row["variant"].endswith("fixed-d3")
        ]
        randomized = [
            row for row in records if row["variant"].endswith("explore-d")
        ]
        k20 = [row for row in fixed if row["k"] == 20]
        reference[f"v11-{domain}"] = (
            sum(row["tokens"] for row in k20)
            / sum(row["seconds"] for row in k20)
        )
        output["k"].extend(tagged(
            old_rows.k_rows(wrapper, {"records": fixed},
                            f"v11-{domain}",
                            {"k3-fixed-d3", "k10-fixed-d3", "k20-fixed-d3"}),
            f"v11-{domain}", domain,
        ))
        output["k"].extend(tagged(
            old_rows.k_rows(
                wrapper, {"records": random_records}, "v6", {"random-k"}
            ),
            f"v11-{domain}", domain,
        ))
        user_prefixes = {}
        for kind, row in old_rows.round_rows(
            wrapper, {"records": randomized}, "v9",
            lambda record: record["variant"].endswith("explore-d")
        ):
            extra = {}
            if kind == "c":
                key = row["conversation"], row["turn"]
                if key not in user_prefixes:
                    ids = wrapper.ids(
                        f"native/{key[0]}/turn-{key[1]}.user-prefix.ids"
                    )
                    if not 1 <= len(ids) <= 32:
                        raise ValueError("C label lacks bounded user prefix")
                    user_prefixes[key] = ids
                extra["first32_user_ids"] = user_prefixes[key]
            output[kind].append({
                **row, **extra,
                "source": f"v11-{domain}", "domain": domain,
            })
    return output, excluded, reference


def write_gzip(path, rows):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as z:
            for row in rows:
                z.write((json.dumps(row, sort_keys=True) + "\n").encode())


def sealed_training(archive, manifest_path, replay_path, root):
    manifest = json.loads(manifest_path.read_text())
    replay = json.loads(replay_path.read_text())
    if (
        manifest["schema"] != 1
        or sha(archive) != manifest["archive_sha256"]
        or replay["status"] != "training-native-random-K-D-C-KV-replay-match"
        or replay["archive_sha256"] != manifest["archive_sha256"]
        or replay["sessions"] != 224
        or manifest["workloads_sha256"] != V11_WORKLOAD_SHA
    ):
        raise ValueError("v11 training lacks matching sealed replay")
    training_replay.extract(archive, manifest, root)
    return root / "native/training-results", root / "inputs/workloads"


def build(v10_archive, v10_manifest, v10_custody,
          v11_archive, v11_manifest, v11_replay, out):
    classes = {}
    archive = V10Archive(v10_archive, v10_manifest)
    try:
        old, old_excluded, old_reference = v10_rows(
            archive, v10_custody, classes
        )
    finally:
        archive.close()
    with tempfile.TemporaryDirectory(prefix="mtp-v11-rows-") as temp:
        native, workloads = sealed_training(
            v11_archive, v11_manifest, v11_replay, Path(temp)
        )
        new, new_excluded, new_reference = v11_rows(
            native, workloads, classes
        )
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": 1,
        "use": "training-only",
        "v10_archive_sha256": sha(v10_archive),
        "v10_parent_archive_sha256": json.loads(
            v10_custody.read_text()
        )["parent_archive_sha256"],
        "v11_training_archive_sha256": sha(v11_archive),
        "v11_workload_sha256": V11_WORKLOAD_SHA,
        "excluded_looped_v10": old_excluded,
        "excluded_looped_v11": new_excluded,
        "reference_tok_s": {**old_reference, **new_reference},
        "rows": {},
    }
    for kind in ("k", "d", "c"):
        rows = old[kind] + new[kind]
        if not rows:
            raise ValueError(f"no {kind} training measurements")
        path = out / f"{kind}.jsonl.gz"
        write_gzip(path, rows)
        result["rows"][kind] = {
            "count": len(rows),
            "sha256": sha(path),
            "by_source": {
                source: sum(row["source"] == source for row in rows)
                for source in sorted({row["source"] for row in rows})
            },
        }
    save(out / "token-classes.json", {
        str(token): kind for token, kind in sorted(classes.items())
    })
    result["token_classes_sha256"] = sha(out / "token-classes.json")
    save(out / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "v10-archive", "v10-manifest", "v10-custody",
        "v11-archive", "v11-manifest", "v11-replay", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.v10_archive, args.v10_manifest, args.v10_custody,
        args.v11_archive, args.v11_manifest, args.v11_replay, args.out,
    )
    print(json.dumps(result["rows"], sort_keys=True))


if __name__ == "__main__":
    main()
