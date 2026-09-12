# DeepSeek V4 Flash

> **Serving bring-up PAUSED (owner decision, 2026-09-12).** This lane continues as research and
> correctness work only; nothing in it is a serving claim or a roster commitment, and no serving
> qualification cell is scheduled for it until the owner reopens it. Receipts and gates below stay
> valid as engine records.

| | Recommended use |
|---|---|
| **Status** | Experimental engine support; functional and gated, not serving-grade |
| **Path** | Exact-source sharded Safetensors checkpoint through the dedicated two-card implementation; no GGUF conversion |
| **Hardware** | Two-card target used by the experimental gate |
| **Use this when** | You are evaluating or contributing to the experimental path, not deploying it as a supported server |

Read the [DeepSeek section](../MODELS.md#deepseek-v4-checkpoint-dirs-serve-through-their-own-door-lanedsv4-flash-revival-20260822)
and follow [issue #4](https://github.com/avifenesh/memra/issues/4). Do not infer production support
from a successful load or prompt.
