# Gemma 4 12B QAT: bind the oracle's batch and fusion program

Issue [#398](https://github.com/avifenesh/memra/issues/398). Four raw prompts on one
RTX 5090 Laptop GPU (24,463 MiB), CUDA 13.1. No Memra runtime change or support
promotion. The original 600-item HTTP score table remains unchanged.

## Result

The pinned llama.cpp HTTP server splits these prompts into `N-4` and `4` tokens
for SWA checkpointing, even with `cache_prompt:false`. The old direct API probe
used a single decode call. Matching that boundary removes the interface gap:

| Same-build comparison | Prompts | Token IDs | Raw top-five logits |
|---|---:|---|---|
| Default HTTP vs direct `N-4/4` | 4 | exact | exact |
| HTTP `--ctx-checkpoints 0` vs direct monolithic | 4 | exact | exact |
| Checkpoint-disabled HTTP vs direct output-limit=1 | 4 | exact | exact |
| Checkpoint-disabled HTTP vs common initialization | 4 | exact | exact |

The checkpoint split is part of the numerical contract. All 600 historical raw
HTTP captures report `tok_idx=3`, consistent with this mechanism. The new native
run is four prompts, not a 600-prompt rerun.

A second controlled ablation isolates the local oracle's fusion effect to the
**RMS-norm fusion family**. Disabling only RMS-norm fusions equals disabling all
CUDA fusion, byte-for-byte over all 262,144 logits in all eight direct cells
(four prompts, two batch programs). Disabling matmul fusions or the remaining
fusion families leaves the default logits byte-identical in all 16 cells.
The diagnostic selector's default first reproduced all eight original full-logit
files exactly. Its dispatch/capture logs identify RMS_NORM and SCALE fusions;
they are not invocation counters for subsequent CUDA graph replays.

This identifies a numerical program difference. It does not establish which
program is an absolute correctness oracle, or prove a Memra kernel defect.

## First-token observations

Token IDs: 107 = newline, 562 = ` A`, 565 = ` C`, 603 = ` B`, 622 = ` D`.

| Prompt | Historical Memra / HTTP | Rebuilt Memra | Local HTTP (split) | Local API (monolithic) | RMS fusion off, either batch program |
|---|---|---:|---:|---:|---:|
| clinical_knowledge/test/52 | 107 / 565 | 107 | 565 | 107 | 107 |
| computer_security/test/50 | 565 / 565 | 565 | 565 | 565 | 565 |
| high_school_psychology/test/16 | 622 / 622 | 622 | 562 | 562 | 622 |
| high_school_biology/test/97 | 603 / 603 | 603 | 603 | 603 | 603 |

All four Memra top-five rows reproduce the historical values at their saved
nine-decimal precision. The rebuilt laptop oracle differs from the historical
desktop oracle. For the clinical prompt, its default HTTP top-two margin is
0.221508, versus the historical 4.068754. The compiler and hardware also differ
from the historical run, so these are separate receipts. One historically
agreeing control now disagrees under the local fused oracle.

Turning RMS fusion off gives 4/4 argmax agreement with Memra on this selected
set; local default HTTP gives 2/4. This is a diagnostic agreement result, not
an accuracy estimate or a default recommendation.

## Localization and the rejected instrument

The ordinary evaluation callback changed full logits in **8/8** default-fusion
cells and changed argmax in 3/8. Those captures are retained as a failed
instrument control and cannot locate a production first-bad operator.

With fusion disabled on both the plain and traced processes, all eight complete
logit files match byte-for-byte. In this transparent diagnostic class, the
last-token normalized input matches exactly between monolithic and split
execution. The first observed numerical difference is at layer-zero Q/K/V:

| Clinical prompt, last token | Max absolute difference | Relative L2 |
|---|---:|---:|
| attn_norm-0 | 0 | 0 |
| Qcur-0 | 1.857883 | 0.009436 |
| Kcur-0 | 2.234862 | 0.008342 |
| Vcur-0 | 1.980259 | 0.008313 |

The pinned source selects narrow-batch MMVQ versus wider-batch MMQ. Their Q8_1
activation representations differ: matvec stores a half-precision scale/sum;
Q4_0 MMQ uses float D4 scales. This is a source-level explanation for different
projection programs, not proof that either approximation is faulty.

A separate replay of Memra's layer-zero public operations, after its normal
prime, has numerically equal normalized input to the unfused monolithic oracle
(signed-zero bytes differ). Q/K/V relative L2 differences are respectively
0.000181, 0.000166 and 0.000166. This replay does not instrument the complete
Memra forward path and does not attribute the final discrepancy to one kernel.

## Distribution comparison boundary

Memra suppresses IDs 258882 and 258883. The raw oracle does not. Unconditioned
`KL(oracle || Memra)` is therefore infinite. The comparator reports this and
separately labels KL conditioned on the shared support. For the clinical case,
that conditional KL is 0.343876 and the oracle mass on the two suppressed IDs is
0.003007. No NLL or base-model quality claim is made.

## Pins and checks

- GGUF SHA256: `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.
- Memra source: `55d83a7cf3948d6d173d28e3e1fd18008fdeb9bf` (v0.136.0), plus the standalone probe.
- llama.cpp source: `f3f1a8f2760f28325a5ec20c05b171e5b7c83a29`, plus the recorded diagnostic patches.
- Raw prompts, BOS 2, context 8192, batch 4096, microbatch 2048, all 49 oracle layers offloaded, two CPU threads, one output token, temperature zero, seed zero, no request prompt cache.
- Sequential processes and fresh caches; no simultaneous runtimes. GPU cells were bounded to 600 or 1200 seconds, with 14 GiB free-VRAM admission and no full FP16 mirrors. A failed preparatory helper link was corrected before execution; no native cell failed or timed out.
- Actual native log selects INT8 MMQ without FP16 mirrors. Compiler, library and probe hashes are recorded alongside the raw outputs.
- `python3 verify_receipts.py` verifies the raw archive, eight reciprocal HTTP/API pairs, the rejected/transparent trace controls, fusion-family controls, and four historical Memra top-five rows.

Current Memra main uses a different Gemma prime program after PRs #561 and #564.
This report investigates the issue's original pin and does not qualify current
main. No kernel, dispatch, default, support record or qualification tolerance
changed. No full release battery was required for these standalone diagnostics.
A development topic push carries no native release qualification.

## Reproduction and remaining decision

`build_oracle.sh` builds the pinned oracle with the original pre-sampler capture
patch. `run_cell.py` captures the reciprocal controls. `oracle_probe_v2.cpp`
accepts `aligned`, `split4`, `output-limit` and `common-init`; an optional fourth
argument enables stage capture. `ORACLE_LOGITS_DIR` records complete logits.
Decompress and apply `fusion-family-instrumentation.patch.gz`, then rebuild for the family ablation;
`DIAG_FUSION_BLOCK` accepts `rms`, `matmul`, `other` or `none`, and
`DIAG_FUSION_LOG` records selected fusions. The native probe is compiled against
the stated Memra pin as `gemma-numerical-probe`, with `MEMRA_Q4F16=0`.

The raw f32 files are deduplicated by SHA256 in `f32-receipts.tar.gz`, with their
original paths in `f32-index.json`. `unpack_f32.py DESTINATION` materializes them
for `compare_logits.py` and `compare_traces.py`. Text logs are in `logs.tar.gz`; their hashes are in `log-index.json`. They retain
their numerical content; machine-local paths are normalized. Source patches are
gzip-wrapped without changing their uncompressed hashes. Original queue receipts and
unsanitized files are also banked locally.

Before a native-fidelity fix or a revised evaluation table, the owner must name
the intended reference class: batch boundaries, fusion policy, suppression mask,
compiler and hardware. The specific RMS fusion variant and any semantic error
inside it remain unproven. No serving default is changed, and #398 stays open.
