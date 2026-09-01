# Decision records

What was chosen, what was rejected, and the measurement that settled it. Kept permanently:
these are the rationale half of this repo's record, and a rejection is often the more useful
half — "we tried X and it lost by Y% on Z" cannot be reconstructed from the code that shipped.

| Record | Decides |
|---|---|
| [SAFETENSORS-DECISION.md](SAFETENSORS-DECISION.md) | safetensors as the semantic source, and what that does not mean for the compute format |
| [FORMAT-DECISION.md](FORMAT-DECISION.md) | which artifact formats the engine imports and serves |
| [QUANT-GEMM-DECISION.md](QUANT-GEMM-DECISION.md) | the quantized GEMM path |
| [RIG-NATIVE-DECODE.md](RIG-NATIVE-DECODE.md) | decoding in rig-native layouts rather than the source layout |
| [PHASE1-HYBRID.md](PHASE1-HYBRID.md) | the hybrid phase-1 shape |
| [BEST-OF-ALL-WORLDS.md](BEST-OF-ALL-WORLDS.md) | per-device arm selection instead of one compromise default |
| [VISION-LANE.md](VISION-LANE.md) | in-engine vision tower, and the parity oracle that gates it |
| [PUBLIC-BOUNDARY-DETECTION.md](PUBLIC-BOUNDARY-DETECTION.md) | what the public-boundary gate matches, which candidate rules were rejected as too noisy, and why published refs need their own scan |
| [ORNITH-PAIR-OWNER.md](ORNITH-PAIR-OWNER.md) | why source-verbatim pair-owner MoE ordering stays out of the runtime while its receipts and exact candidate remain banked |

Adding one: a decision that changes a default, a format, a target or an arm belongs here, with
the measurement that settled it. Superseded records get a banner naming what replaced them —
they are not deleted. See `CLAUDE.md` § "Measurements and decisions are a corpus".
