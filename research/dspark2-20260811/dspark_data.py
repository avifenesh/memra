"""Read memra's chunked, columnar DSpark anchor corpus without rewriting it."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


HIDDEN_SIZE = 4096
BLOCK_SIZE = 5
TOP_K = 64


@dataclass(frozen=True)
class RecordRef:
    chunk: int
    record: int
    anchor_position: int


class CorpusChunk:
    def __init__(self, path: Path):
        self.path = path
        extracted = path / "extracted"
        tokens_path = extracted / "tokens.u32"
        manifest_path = path / "sha256.txt"
        if not manifest_path.exists():
            raise ValueError(f"{path} lacks sha256.txt")
        if tokens_path.stat().st_size % ((BLOCK_SIZE + 1) * 4):
            raise ValueError(f"{tokens_path} has a partial record")
        self.records = tokens_path.stat().st_size // ((BLOCK_SIZE + 1) * 4)
        shapes = {
            "hiddens.bf16": self.records * HIDDEN_SIZE * 2,
            "top_ids.u32": self.records * BLOCK_SIZE * TOP_K * 4,
            "top_probs.f32": self.records * BLOCK_SIZE * TOP_K * 4,
            "top_logits.f32": self.records * BLOCK_SIZE * TOP_K * 4,
            "tail_probs.f32": self.records * BLOCK_SIZE * 4,
        }
        for name, expected in shapes.items():
            actual = (extracted / name).stat().st_size
            if actual != expected:
                raise ValueError(f"{extracted / name} has {actual} bytes, expected {expected}")
        expected_hashes = {}
        for line in manifest_path.read_text().splitlines():
            digest, relative = line.split(maxsplit=1)
            expected_hashes[relative.strip()] = digest
        required = [
            "extracted/hiddens.bf16",
            "extracted/index.tsv",
            "extracted/tail_probs.f32",
            "extracted/tokens.u32",
            "extracted/top_ids.u32",
            "extracted/top_probs.f32",
        ]
        for relative in required:
            expected = expected_hashes.get(relative)
            if expected is None:
                raise ValueError(f"{manifest_path} does not cover {relative}")
            actual = hashlib.sha256((path / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"{path / relative} hash mismatch")
        self.hiddens = np.memmap(
            extracted / "hiddens.bf16", mode="r", dtype="<u2", shape=(self.records, HIDDEN_SIZE)
        )
        self.tokens = np.memmap(
            tokens_path, mode="r", dtype="<u4", shape=(self.records, BLOCK_SIZE + 1)
        )
        self.top_ids = np.memmap(
            extracted / "top_ids.u32",
            mode="r",
            dtype="<u4",
            shape=(self.records, BLOCK_SIZE, TOP_K),
        )
        self.top_probs = np.memmap(
            extracted / "top_probs.f32",
            mode="r",
            dtype="<f4",
            shape=(self.records, BLOCK_SIZE, TOP_K),
        )
        self.tail_probs = np.memmap(
            extracted / "tail_probs.f32",
            mode="r",
            dtype="<f4",
            shape=(self.records, BLOCK_SIZE),
        )
        self.index_rows = []
        with (extracted / "index.tsv").open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                self.index_rows.append(row)
        if len(self.index_rows) != self.records:
            raise ValueError(
                f"{extracted / 'index.tsv'} has {len(self.index_rows)} rows, expected {self.records}"
            )
        for expected, row in enumerate(self.index_rows):
            if int(row["record"]) != expected:
                raise ValueError(f"{path} index record is not contiguous at {expected}")


class DSparkCorpus(Dataset):
    def __init__(
        self,
        chunks_root: Path,
        split: str,
        *,
        pair_start: int = 0,
        pair_end: int = 2000,
        label: str = "pilot",
    ):
        if split not in {"train", "heldout"}:
            raise ValueError("split must be train or heldout")
        paths = sorted(chunks_root.glob(f"{label}-*"))
        self.chunks = []
        self.records = []
        self.chunk_manifest_hashes = []
        covered_pairs = set()
        for path in paths:
            bounds = path.name.rsplit("-", 2)
            if len(bounds) != 3:
                continue
            begin, end = int(bounds[-2]), int(bounds[-1])
            if begin >= pair_end or end <= pair_start:
                continue
            chunk = CorpusChunk(path)
            chunk_index = len(self.chunks)
            self.chunks.append(chunk)
            covered_pairs.update(range(max(begin, pair_start), min(end, pair_end)))
            self.chunk_manifest_hashes.append(
                (path.name, hashlib.sha256((path / "sha256.txt").read_bytes()).hexdigest())
            )
            for local_record, row in enumerate(chunk.index_rows):
                pair_id = int(row["pair_id"])
                if not pair_start <= pair_id < pair_end:
                    continue
                if row["split"] == split:
                    self.records.append(
                        RecordRef(chunk_index, local_record, int(row["anchor_pos"]))
                    )
        expected_pairs = set(range(pair_start, pair_end))
        if covered_pairs != expected_pairs:
            missing = sorted(expected_pairs - covered_pairs)
            raise ValueError(
                f"corpus does not cover exact pair range [{pair_start},{pair_end}); "
                f"missing first ids {missing[:8]}"
            )
        if not self.records:
            raise ValueError(f"no {split} records in exact pair range")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.records[index]
        chunk = self.chunks[ref.chunk]
        return {
            "hidden_bits": torch.from_numpy(chunk.hiddens[ref.record].copy()),
            "tokens": torch.from_numpy(chunk.tokens[ref.record].astype(np.int64)),
            "top_ids": torch.from_numpy(chunk.top_ids[ref.record].astype(np.int64)),
            "top_probs": torch.from_numpy(chunk.top_probs[ref.record].copy()),
            "tail_probs": torch.from_numpy(chunk.tail_probs[ref.record].copy()),
            "anchor_position": torch.tensor(ref.anchor_position, dtype=torch.long),
        }

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, manifest_hash in self.chunk_manifest_hashes:
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(manifest_hash.encode())
            digest.update(b"\n")
        return digest.hexdigest()


def collate_records(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = {key: torch.stack([row[key] for row in records]) for key in records[0]}
    batch["hidden"] = batch.pop("hidden_bits").contiguous().view(torch.bfloat16)
    return batch


def load_d2t(path: Path, expected: int = 32768) -> torch.Tensor:
    if path.stat().st_size != expected * 4:
        raise ValueError(f"{path} does not contain exactly {expected} u32 ids")
    values = np.fromfile(path, dtype="<u4")
    return torch.from_numpy(values.astype(np.int64))
