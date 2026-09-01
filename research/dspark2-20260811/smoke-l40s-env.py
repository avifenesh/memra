#!/usr/bin/env python3
"""Receipt-producing CUDA/backend smoke test for the DSpark L40S venv."""

from __future__ import annotations

import importlib.metadata
import json
import time

import torch
import torch.nn.functional as F


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
assert torch.cuda.get_device_capability(0) == (8, 9), torch.cuda.get_device_capability(0)

device = torch.device("cuda:0")
torch.manual_seed(20260811)
torch.cuda.manual_seed_all(20260811)

left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
right = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
torch.cuda.synchronize()
started = time.perf_counter()
product = left @ right
torch.cuda.synchronize()
matmul_ms = (time.perf_counter() - started) * 1_000.0
assert torch.isfinite(product).all().item()

query = torch.randn((2, 8, 64, 64), device=device, dtype=torch.bfloat16)
key = torch.randn_like(query)
value = torch.randn_like(query)
sdpa = F.scaled_dot_product_attention(query, key, value, is_causal=True)
torch.cuda.synchronize()
assert torch.isfinite(sdpa).all().item()

try:
    import flash_attn  # type: ignore[import-not-found]

    flash_version = flash_attn.__version__
    attention_backend = "flash-attn"
except (ImportError, OSError) as exc:
    flash_version = None
    attention_backend = "sdpa"
    flash_import_error = f"{type(exc).__name__}: {exc}"
else:
    flash_import_error = None

receipt = {
    "attention_backend": attention_backend,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "device_capability": list(torch.cuda.get_device_capability(0)),
    "device_name": torch.cuda.get_device_name(0),
    "flash_attn": flash_version,
    "flash_attn_import_error": flash_import_error,
    "matmul_bf16_ms": matmul_ms,
    "matmul_checksum": float(product.float().sum().item()),
    "packages": {
        name: package_version(name)
        for name in (
            "accelerate",
            "bitsandbytes",
            "datasets",
            "numpy",
            "safetensors",
            "scikit-learn",
            "torch",
            "transformers",
        )
    },
    "sdpa_checksum": float(sdpa.float().sum().item()),
    "torch": torch.__version__,
}
print(json.dumps(receipt, indent=2, sort_keys=True))
