# Qwen3.6 35B-A3B

| | Recommended use |
|---|---|
| **Status** | Supported |
| **Best starting path** | IQ4_XS GGUF with the MTP and own-generation trimmed drafter path |
| **Hardware** | RTX 5090-class Blackwell for the tracked resident MoE path |
| **Use this when** | You want the supported Qwen3.6 MoE route rather than the dense 27B route |

Treat the exact artifact as part of the support contract. See [models and hardware](../MODELS.md),
then use [serving](../SERVING.md) for deployment controls.

**When the experts spill** (they do not fit the card, or `MEMRA_MOE_RESIDENT=0`), `run-gen` and `run-spec` serve
the approved IQ4_XS artifact through the MoE slot cache door by default: the whole expert bank in a registered
pinned host tier, leased through its owner thread, with the in-token prefetch on. `MEMRA_EXPERTS_VIA_TIER=0` rolls
back to the legacy slot cache, `MEMRA_MOE_PREFETCH=0` turns the prefetch off. The door holds the bank only when it
fits three quarters of the host's available memory; otherwise the legacy runs and says why. `memra-server` keeps
the legacy slot cache until its installer lands. The decision and its receipts:
[MOE-SPILL-DOOR-DEFAULT.md](../decisions/MOE-SPILL-DOOR-DEFAULT.md).
