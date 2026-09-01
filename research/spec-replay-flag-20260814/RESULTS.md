# Literal Spec-Replay Flag Gate

Commit under test: `5cd91d574` (`fix(spec): parse replay flag literally`).

The local RTX 5090 gate ran five balanced repetitions of three arms against the
same Qwen3.5 9B NVFP4 model, prompt, K=3, 32-token greedy workload, and binary:

| Arm | Replay selected | Median tok/s | Acceptance | Token SHA-256 |
|---|---:|---:|---:|---|
| unset | no | 226.05 | 0.815 | `acd81945ffd01e756f97eaaff043449d8852d8764010e4ae4abbe43bb09c9b08` |
| `MEMRA_SPEC_REPLAY=0` | no | 226.43 | 0.815 | `acd81945ffd01e756f97eaaff043449d8852d8764010e4ae4abbe43bb09c9b08` |
| `MEMRA_SPEC_REPLAY=1` | yes | 146.88 | 0.815 | `acd81945ffd01e756f97eaaff043449d8852d8764010e4ae4abbe43bb09c9b08` |

Orders were rotated across repetitions:

1. unset, zero, one
2. zero, one, unset
3. one, unset, zero
4. one, zero, unset
5. zero, unset, one

Every arm passed spec/plain self-consistency. Literal zero reproduced unset
within 0.17% median throughput, while legacy replay remained an explicit,
slower diagnostic program. The private controller receipt includes all 15 raw
logs, input hashes, and 250 ms telemetry; public evidence intentionally omits
host identity and absolute artifact paths.

Reproduction:

```bash
MEMRA_NGEN=32 \
MEMRA_SPEC_K=3 \
MEMRA_SPEC_TEMP=0 \
MEMRA_SPEC_STATS=1 \
MEMRA_DEBUG_SPEC=1 \
MEMRA_PROMPT_FILE=/path/to/prompt.txt \
MEMRA_SPEC_REPLAY=0 \
target/release/run-spec /path/to/model.gguf
```
