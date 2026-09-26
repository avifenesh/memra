#!/usr/bin/env python3
"""Header-only, optimistic two-card placement bound for the pinned MiMo source.

This reads no tensor payload in remote mode. A positive result is not a runtime
fit claim: CUDA context, KV scales, workspaces, activations, and fragmentation
are excluded. Use the measured card MiB from the eventual target box.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

MODEL = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
REVISION = "3b38d063180c3e4aed9691fdc735f3d10b266ee4"
INDEX_SHA256 = "09d9b96a77ed9765fa4e02a6a45f92797702eef432da426efc6b45c1131b1812"
CONFIG_SHA256 = "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621"
HEADER_LIMIT = 64 * 1024 * 1024
FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "crates/memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def remote_bytes(filename, first=None, last=None):
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{quote(filename)}"
    headers = {} if first is None else {"Range": f"bytes={first}-{last}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
        if first is not None and response.status != 206:
            raise ValueError(f"{filename}: HTTP Range refused ({response.status})")
        data = response.read() if first is None else response.read(last - first + 1)
    if first is not None and len(data) != last - first + 1:
        raise ValueError(f"{filename}: short HTTP Range response")
    return data


def read_header(filename, checkpoint):
    if checkpoint is None:
        prefix = remote_bytes(filename, 0, 7)
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= HEADER_LIMIT:
            raise ValueError(f"{filename}: header exceeds bound")
        header = remote_bytes(filename, 8, 7 + header_size)
    else:
        path = checkpoint / filename
        with path.open("rb") as file:
            prefix = file.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{filename}: short prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= HEADER_LIMIT:
                raise ValueError(f"{filename}: header exceeds bound")
            header = file.read(header_size)
            if len(header) != header_size:
                raise ValueError(f"{filename}: short header")
    return filename, prefix + header, json.loads(header)


def category(name, layer_count):
    match = re.match(r"^model\.layers\.(\d+)\.", name)
    if match:
        layer = int(match.group(1))
        if layer >= layer_count:
            raise ValueError(f"{name}: layer outside pinned config")
        return "text_layers", layer
    if name == "model.embed_tokens.weight":
        return "text_embedding", None
    if name == "model.norm.weight":
        return "text_norm", None
    if name == "lm_head.weight":
        return "text_head", None
    for prefix, label in [
        ("model.mtp.", "mtp"),
        ("visual.", "visual"),
        ("audio_encoder.", "audio_encoder"),
        ("speech_embeddings.", "speech_embeddings"),
    ]:
        if name.startswith(prefix):
            return label, None
    raise ValueError(f"{name}: unclassified tensor")


def placement(config, categories, layers, card_bytes, sessions, kv_bytes):
    context = config["max_position_embeddings"]
    pattern = config["hybrid_layer_pattern"]
    if len(pattern) != len(layers) or set(pattern) != {0, 1}:
        raise ValueError("pinned attention pattern changed")
    per_layer_kv = [
        sessions
        * kv_bytes
        * (context * config["num_key_value_heads"] if kind == 0 else config["sliding_window"] * config["swa_num_key_value_heads"])
        * (config["head_dim"] + config["v_head_dim"])
        for kind in pattern
    ]
    cuts = []
    for cut in range(1, len(layers)):
        stage0 = categories["text_embedding"] + sum(layers[:cut]) + sum(per_layer_kv[:cut])
        stage1 = categories["text_norm"] + categories["text_head"] + sum(layers[cut:]) + sum(per_layer_kv[cut:])
        cuts.append(
            {
                "cut_before_layer": cut,
                "stage0_base_bytes": stage0,
                "stage1_base_bytes": stage1,
                "minimum_headroom_bytes": min(card_bytes - stage0, card_bytes - stage1),
            }
        )
    cuts.sort(key=lambda row: row["minimum_headroom_bytes"], reverse=True)
    return {
        "sessions": sessions,
        "kv_bytes_per_element": kv_bytes,
        "context_tokens_per_session": context,
        "total_kv_bytes": sum(per_layer_kv),
        "best_contiguous_cuts": cuts[:4],
        "cuts_fitting_before_workspace": sum(row["minimum_headroom_bytes"] >= 0 for row in cuts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, help="Read full local shards instead of pinned remote headers")
    parser.add_argument("--card-mib", type=int, required=True, help="Memory per card in MiB; use measured memory.total for target qualification")
    parser.add_argument("--output", type=Path, help="Create a JSON receipt; refuses overwrite")
    args = parser.parse_args()
    if args.card_mib <= 0:
        parser.error("--card-mib must be positive")
    fixture = FIXTURE.read_bytes()
    if digest(fixture) != CONFIG_SHA256:
        raise ValueError("MiMo config fixture differs from pinned source")
    config = json.loads(fixture)
    if args.checkpoint_dir is None:
        index_bytes = remote_bytes("model.safetensors.index.json")
    else:
        index_bytes = (args.checkpoint_dir / "model.safetensors.index.json").read_bytes()
        if digest((args.checkpoint_dir / "config.json").read_bytes()) != CONFIG_SHA256:
            raise ValueError("local MiMo config differs from pinned source")
    if digest(index_bytes) != INDEX_SHA256:
        raise ValueError("source index differs from pinned revision")
    index = json.loads(index_bytes)
    files = sorted(set(index["weight_map"].values()))
    categories = defaultdict(int)
    layers = [0] * config["num_hidden_layers"]
    seen = set()
    header_hash = hashlib.sha256()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for filename, raw, entries in pool.map(
            lambda name: read_header(name, args.checkpoint_dir), files
        ):
            header_hash.update(filename.encode())
            header_hash.update(len(raw).to_bytes(8, "little"))
            header_hash.update(raw)
            for name, info in entries.items():
                if name == "__metadata__":
                    continue
                if index["weight_map"].get(name) != filename or name in seen:
                    raise ValueError(f"{filename}: index or duplicate mismatch at {name}")
                seen.add(name)
                first, last = info["data_offsets"]
                if first < 0 or last < first:
                    raise ValueError(f"{name}: bad data offsets")
                label, layer = category(name, len(layers))
                size = last - first
                categories[label] += size
                if layer is not None:
                    layers[layer] += size
    if seen != set(index["weight_map"]):
        raise ValueError("source index and shard headers differ")
    if sum(categories.values()) != index["metadata"]["total_size"]:
        raise ValueError("physical byte sum differs from pinned index")
    expected = set(range(len(layers)))
    if {position for position, size in enumerate(layers) if size > 0} != expected:
        raise ValueError("missing text layer")
    text_bytes = sum(layers) + sum(categories[key] for key in ("text_embedding", "text_norm", "text_head"))
    card_bytes = args.card_mib * 1024 * 1024
    report = {
        "source": MODEL,
        "revision": REVISION,
        "index_sha256": INDEX_SHA256,
        "header_sha256": header_hash.hexdigest(),
        "physical_tensor_count": len(seen),
        "file_count": len(files),
        "total_index_bytes": index["metadata"]["total_size"],
        "text_only_bytes": text_bytes,
        "excluded_modal_and_mtp_bytes": index["metadata"]["total_size"] - text_bytes,
        "categories": dict(categories),
        "card_mib_input": args.card_mib,
        "card_capacity_bytes": card_bytes,
        "capacity_scope": "optimistic weight plus KV bound; excludes CUDA, workspaces, activations, KV metadata, and fragmentation",
        "scenarios": [
            placement(config, categories, layers, card_bytes, sessions, kv_bytes)
            for sessions in (1, 2)
            for kv_bytes in (1, 2)
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x") as file:
            file.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
