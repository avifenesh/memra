"""Passive vLLM V1 logits capture for the HY3 offline qualification oracle.

The processor returns the input tensor unchanged.  It exists because vLLM's public logprobs
return path can hang under pipeline parallelism; capturing on the final sampling stage avoids that
transport path while preserving the model's raw logits and greedy result.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor


class RawLogitsCapture(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        del vllm_config, device, is_pin_memory
        output = os.environ.get("HY3_VLLM_RAW_LOGITS_FILE")
        vocab = os.environ.get("HY3_VLLM_RAW_LOGITS_VOCAB")
        if not output or not vocab:
            raise RuntimeError("HY3 vLLM raw-logit capture environment is incomplete")
        self.output = Path(output)
        self.vocab = int(vocab)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < self.vocab:
            raise RuntimeError(
                f"unexpected raw-logit shape {tuple(logits.shape)} for vocab {self.vocab}"
            )
        row = logits[0, : self.vocab].detach().to(dtype=torch.float32, device="cpu").contiguous()
        payload = row.numpy().tobytes()
        if len(payload) != self.vocab * 4:
            raise RuntimeError(f"captured {len(payload)} bytes, expected {self.vocab * 4}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(f".{self.output.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, self.output)
        return logits

    def is_argmax_invariant(self) -> bool:
        # Conservatively opt out of vLLM's argmax-only elision: the tensor is unchanged, but the
        # capture side effect is the entire purpose of this offline processor.
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        del batch_update
