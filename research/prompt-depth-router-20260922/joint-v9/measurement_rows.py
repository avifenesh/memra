"""Reclassify sealed v6/v8 measurements as follow-up training observations.

This reads archive members without extracting files or treating any old
selection/heldout conversation as a new evaluation result.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def classify(raw):
    if b"\n" in raw or b"\r" in raw:
        return 3
    if raw and all(byte in b" \t\v\f" for byte in raw):
        return 2
    if any(48 <= byte <= 57 for byte in raw):
        return 1
    if any(65 <= byte <= 90 or 97 <= byte <= 122 for byte in raw):
        return 0
    if any(byte in b"`{}[]()=+-*/:;,.<>|" for byte in raw):
        return 4
    return 5


class Archive:
    def __init__(self, archive, manifest_path):
        self.manifest = json.loads(manifest_path.read_text())
        if (
            self.manifest["schema"] != 1
            or digest(archive) != self.manifest["archive_sha256"]
        ):
            raise ValueError(f"archive differs from manifest: {archive}")
        self.stream = tarfile.open(archive, "r:gz")
        names = self.stream.getnames()
        if len(names) != len(set(names)) or set(names) != set(self.manifest["members"]):
            raise ValueError("archive member inventory differs")

    def read(self, name):
        expected = self.manifest["members"][name]
        member = self.stream.getmember(name)
        if not member.isfile() or member.size != expected["bytes"]:
            raise ValueError(f"member shape differs: {name}")
        with self.stream.extractfile(member) as source:
            data = source.read()
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"member hash differs: {name}")
        return data

    def json(self, name):
        return json.loads(self.read(name))

    def table(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode()), delimiter="\t"))

    def ids(self, name):
        return [int(item) for item in self.read(name).split()]

    def close(self):
        self.stream.close()


def k_rows(archive, summary, source, variant):
    for record in summary["records"]:
        if record["variant"] not in variant:
            continue
        session = f"native/{record['name']}"
        turns = archive.table(f"{session}/turns.tsv")
        if len(turns) != 8 or record["loops"]:
            raise ValueError(f"ineligible K training conversation: {session}")
        previous = None
        for index, row in enumerate(turns, 1):
            prefix = archive.ids(f"{session}/turn-{index}.user-prefix.ids")
            k = int(row["draft_top_k"])
            if (
                k not in (3, 10, 20) or int(row["sampler_top_k"]) != 20
                or not 1 <= len(prefix) <= 32 or float(row["elapsed_s"]) <= 0
            ):
                raise ValueError(f"invalid K decision receipt: {session}/{index}")
            yield {
                "source": source,
                "conversation": record["name"],
                "turn": index,
                "draft_k": k,
                "first32_user_ids": prefix,
                "previous_turn_acceptance": previous,
                "output_tokens": int(row["output_tokens"]),
                "complete_request_seconds": float(row["elapsed_s"]),
                "assignment": (
                    "randomized-turn" if source == "v6" else "fixed-arm"
                ),
            }
            drafted = int(row["drafted"])
            previous = int(row["accepted"]) / drafted if drafted else None


def round_rows(archive, summary, source, include):
    for record in summary["records"]:
        if not include(record):
            continue
        session = f"native/{record['name']}"
        if record["loops"]:
            raise ValueError(f"ineligible D training conversation: {session}")
        rounds = archive.table(f"{session}/rounds.tsv")
        spans = archive.table(f"{session}/spans.tsv")
        turn_k = {
            int(row["turn"]): int(row["draft_top_k"])
            for row in archive.table(f"{session}/turns.tsv")
        }
        if len(rounds) != len(spans):
            raise ValueError(f"round and span counts differ: {session}")
        confidence = (
            {
                (turn, int(row["round"])): row
                for turn in range(1, 9)
                for row in archive.table(f"{session}/turn-{turn}.confidence.tsv")
            }
            if source in ("v6", "v9") else None
        )
        output = {
            turn: archive.ids(f"{session}/turn-{turn}.output.ids")
            for turn in range(1, 9)
        }
        previous = None
        for round_row, span in zip(rounds, spans):
            turn = int(span["turn"])
            key = turn, int(span["round"])
            d = int(round_row["draft_depth"])
            emitted = int(round_row["emitted"])
            elapsed_ns = int(round_row["elapsed_ns"])
            eligible = round_row["eligible_for_learning"] == "true"
            if (
                key != (int(round_row["turn"]), int(round_row["round"]))
                or d != int(span["k"]) or elapsed_ns != int(span["elapsed_ns"])
                or span["eligible"] != round_row["eligible_for_learning"]
                or int(span["output_end"]) - int(span["output_start"]) != emitted
            ):
                raise ValueError(f"round and span fields differ: {session}/{key}")
            if not eligible:
                continue
            start = int(span["output_start"])
            accepted = emitted - 1
            if not (1 <= d <= 4 and 0 <= accepted <= d and elapsed_ns > 0):
                raise ValueError(f"invalid D outcome: {session}/{key}")
            row = {
                "source": source,
                "conversation": record["name"],
                "turn": turn,
                "round": key[1],
                "draft_k": turn_k[turn],
                "draft_depth": d,
                "committed_history_ids": output[turn][:start][-16:],
                "previous_round_acceptance": (
                    previous[0] if previous is not None else None
                ),
                "previous_round_ms": (
                    previous[1] if previous is not None else None
                ),
                "accepted_prefix": accepted,
                "emitted": emitted,
                "round_ms": elapsed_ns / 1e6,
                "assignment": (
                    "randomized-round" if source in ("v6", "v9")
                    else "selected-policy" if record["variant"] in (
                        "joint-learned", "cd-fixedk20", "cd-fixedk3",
                    ) else "fixed-arm"
                ),
            }
            if confidence is not None:
                offered = confidence[key]
                if (
                    int(offered["drafted"]) != d
                    or int(offered["accepted_prefix"]) != accepted
                    or int(offered["elapsed_ns"]) != elapsed_ns
                    or offered["eligible"] != "true"
                ):
                    raise ValueError(f"confidence trace differs: {session}/{key}")
                q_bits = [int(item) for item in offered["q_bits"].split(",")]
                if len(q_bits) != d:
                    raise ValueError(f"proposal count differs: {session}/{key}")
                for position, bits in enumerate(q_bits):
                    if position > accepted:
                        break
                    q = struct.unpack("<f", struct.pack("<I", bits))[0]
                    if not 0 < q <= 1:
                        raise ValueError(f"invalid chosen probability: {session}/{key}")
                    yield "c", {
                        **row,
                        "offer_position": position,
                        "chosen_probability": q,
                        "accepted_offer": position < accepted,
                    }
            yield "d", row
            previous = accepted / d, elapsed_ns / 1e6


def build(v6, v8, out):
    out.mkdir(parents=True, exist_ok=False)
    development = v6.json("native/development-summary.json")
    heldout = v8.json("native/heldout-summary.json")
    if (
        development["phase"] != "development"
        or heldout["phase"] != "heldout"
    ):
        raise ValueError("training parent or retired heldout summary changed")
    counts = {"k": {}, "d": {}, "c": {}}
    classes = {}
    for archive, summary, include in (
        (v6, development, lambda row: row["arm"] == "explore-d"),
        (v8, heldout, lambda row: row["variant"] in {
            "topk3-d3-c0", "topk10-d3-c0", "topk20-d3-c0",
            "joint-learned", "cd-fixedk20", "cd-fixedk3",
        }),
    ):
        for record in summary["records"]:
            if not include(record):
                continue
            for row in archive.table(f"native/{record['name']}/token-bytes.tsv"):
                token = int(row["id"])
                klass = classify(bytes.fromhex(row["hex"]))
                if token in classes and classes[token] != klass:
                    raise ValueError("historical tokenizer class differs")
                classes[token] = klass
    (out / "token-classes.json").write_text(
        json.dumps(classes, sort_keys=True) + "\n"
    )
    with (
        gzip.open(out / "k.jsonl.gz", "xt") as k_stream,
        gzip.open(out / "d.jsonl.gz", "xt") as d_stream,
        gzip.open(out / "c.jsonl.gz", "xt") as c_stream,
    ):
        for archive, summary, source, names in (
            (v6, development, "v6", {"random-draftk"}),
            (v8, heldout, "v8", {
                "topk3-d3-c0", "topk10-d3-c0", "topk20-d3-c0",
            }),
        ):
            for row in k_rows(archive, summary, source, names):
                k_stream.write(json.dumps(row, sort_keys=True) + "\n")
                key = f"{source}:K{row['draft_k']}"
                counts["k"][key] = counts["k"].get(key, 0) + 1
        for archive, summary, source, include in (
            (v6, development, "v6", lambda r: r["arm"] == "explore-d"),
            (v8, heldout, "v8", lambda r: r["variant"] in {
                "topk3-d3-c0", "topk10-d3-c0", "topk20-d3-c0",
                "joint-learned", "cd-fixedk20", "cd-fixedk3",
            }),
        ):
            for kind, row in round_rows(archive, summary, source, include):
                stream = c_stream if kind == "c" else d_stream
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                key = f"{source}:K{row['draft_k']}:D{row['draft_depth']}"
                counts[kind][key] = counts[kind].get(key, 0) + 1
    result = {
        "schema": 1,
        "use": "training-only",
        "v6_archive_sha256": v6.manifest["archive_sha256"],
        "v8_archive_sha256": v8.manifest["archive_sha256"],
        "counts": counts,
        "outputs": {
            **{
                name: digest(out / f"{name}.jsonl.gz")
                for name in ("k", "d", "c")
            },
            "token-classes": digest(out / "token-classes.json"),
        },
        "token_classes": len(classes),
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6", type=Path, required=True)
    parser.add_argument("--v8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parent = Archive(args.v6 / "native-data.tar.gz", args.v6 / "manifest.json")
    final = Archive(args.v8 / "native-data.tar.gz", args.v8 / "manifest.json")
    try:
        result = build(parent, final, args.out)
    finally:
        parent.close()
        final.close()
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
