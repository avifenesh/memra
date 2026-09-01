#!/usr/bin/env python3
"""One real-corpus full-size forward/backward gate on the provisioned L40S."""

import json
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dspark_data import DSparkCorpus, collate_records, load_d2t
from dspark_loss import build_target_to_draft, compute_dspark_loss
from dspark_model import DSparkConfig, DSparkModel, expected_trainable_parameters, load_shared_artifact


def main() -> None:
    device = torch.device("cuda", 0)
    gpu = torch.cuda.get_device_name(device)
    if "L40S" not in gpu:
        raise RuntimeError(f"refusing preflight on {gpu}")
    torch.manual_seed(20260811)
    config = DSparkConfig()
    shared = Path("/home/ubuntu/dspark2/artifacts/pilot-02000/shared")
    data = DSparkCorpus(
        Path("/home/ubuntu/dspark2/corpus/chunks"), "train", pair_end=2000
    )
    loader = DataLoader(
        data, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_records
    )
    d2t = load_d2t(shared / "d2t.u32", config.draft_vocab)
    inverse = build_target_to_draft(d2t, config.target_vocab).to(device)
    embedding, head = load_shared_artifact(shared, config)
    model = DSparkModel(config, embedding, head).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable == expected_trainable_parameters(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.0, fused=True)
    batch = {key: value.to(device) for key, value in next(iter(loader)).items()}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch["hidden"], batch["tokens"], batch["anchor_position"], inverse)
    loss = compute_dspark_loss(
        output.logits,
        output.confidence_logits,
        batch["tokens"][:, 1:],
        batch["top_ids"],
        batch["top_probs"],
        batch["tail_probs"],
        inverse,
    )
    loss.total.backward()
    grad_norm = clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    result = {
        "gate": "PASS",
        "gpu": gpu,
        "batch_size": batch["tokens"].shape[0],
        "trainable_parameters": trainable,
        "corpus_fingerprint": data.fingerprint(),
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "grad_norm": float(grad_norm),
        **loss.metrics(),
    }
    if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
        raise FloatingPointError("non-finite gradient")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
