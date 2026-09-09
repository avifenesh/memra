# Qwen FA2 prefill qualification

`MEMRA_PRIME_ATTN_FA2` is a default-OFF experiment on sm_120a, decide-by
2026-09-23. It preserves the 24 Q / 4 KV / d256 geometry and shares staged K/V
across the six query heads. The numerical program changes through direct FP32
PV accumulation and the BF16-rounded softmax denominator. No external kernel
is linked or shipped.

Standalone attention reaches 144.09 TF/s at 1024 x 32768 and 146.15 TF/s at
1024 x 131070, against 111-112 TF/s for the existing kernel. The 8k shape reaches
139.24 TF/s, accepted by the owner. See `fa2/CHECKPOINT.md` for the original
microbench and its numerical boundary.

The owner-selected chunk-1024/512 final-chunk calibration is byte-identical.
A second isolation control retains main's denominator and scale but changes
only PV accumulation order: max/RMS logit delta 13.7485/1.6584, alongside FA2's
13.8527/1.5964. The new staging/layout with main's complete arithmetic order is
byte-identical through final logits. These controls support accumulation-order
amplification; serving quality is decided by the independent gates.

The first integration failed cold/restored greedy identity because t<128 and
widened tails fell back to the old numerical class. The corrected dispatch
covers all qualified prefill chunks, including t=16..1039. Decode is unchanged.
The corrected ON program passes the four-turn cold/restored greedy twin and
cold/restored boundary capture. Its frozen-prompt margin gate has zero flips.
All remaining gates now pass. The receipt is `fa2-receipt.json`. Proposed flip:
enable the door for the qualified Qwen/5090 profile in the owner's next batched
release. The code default stays OFF; no deployment or release occurs here.

| Measurement | OFF | ON |
| --- | ---: | ---: |
| Cold TTFT 8k, three boots/arm | 2.465124 s | 2.450418 s |
| Cold TTFT 32k, three boots/arm | 10.999672 s | 10.622078 s |
| Cold TTFT 131k, three boots/arm | 66.489175 s | 60.375959 s |
| 131k attention slice | 30.923492 s | 24.580813 s |
| 131k idle share | 0.3876% | 0.4503% |
| Mean NLL, three paired boots | 0.8587744443 | 0.8577161673 |
| Warm cache median TTFT, 21 turns/arm | 0.186281 s | 0.184511 s |

The attention slice misses the 20-23 s design target. The 131k cold improvement is
9.19%; 32k is 3.43%, 8k is 0.60%. All 24 sampled campaign boots have unique nonces,
the same binary hash, dspark-acc engagement and no OOM. One repetitive OFF cache
initializer is excluded from performance aggregates; it is not part of the cold
matrix or warm-cache medians.

Corrected margin: zero flips across 24 teacher-forced positions, max logit delta
1.0574546. Same-binary ON run-spec K=1..8 matches plain. All four ON greedy restore
turns match their cold twins; cold/restored boundary state hashes match at 8160.
The 64-layer diagnostic has final/max logit delta 0.6003599 on the board prompt.

The paired NLL delta is -0.0010583; diagnostic paired token-block bootstrap 95%
interval [-0.0028498, +0.0005642] includes zero. Each arm reproduces its per-token
loss vector exactly over three boots. This estimates sampling noise without a
new acceptance tolerance.

Sampled c1 output streams differ with the prefill class, so their rates do not
isolate decode. The fixed-state control primes OFF in both arms, then toggles
only for decode. All 192 steps have bit-identical logits/tokens; median ON/OFF rate
ratio is 0.99584. Every warm cache turn retains at least 8160 cached tokens.

Twelve live-depth graph replay byte comparisons pass, including 16-row suffixes
and 1034-row widened tails. Memcheck and synccheck report zero errors. All 99
baseline CUDA entry symbols remain, with only the two FA2 entries added. Remote
Clippy, fmt, flags and fatbin checks pass; engine/server/ModelPlan/reference/runtime
libraries report 467/652/229/30/1 passing tests. Ignored GPU tests are not counted.

## Post-qualification review narrowing

Self-review of the PR head found the dispatch guard admitting t=16..1039 against a
hardcoded 1024, independent of the deployment's actual prime chunk. At
`MEMRA_PRIME_CHUNK=2048` or 4096 every full chunk would stay on the legacy class
while only the folded final tail took FA2, so one prime would mix numerical classes
and a restored suffix would stop matching its cold twin: the same defect class as
the integration v1 t<128 failure. `prime_attn_fa2_enabled` now also requires
`MEMRA_PRIME_CHUNK=1024`.

This narrowing is a no-op on every sealed arm. `fa2/gate_cli.py` builds each gate's
environment from `requal-profile.json`, which pins `MEMRA_PRIME_CHUNK=1024`, and the
serving campaign derives the same profile. No ON arm in this receipt loses its FA2
dispatch, and no OFF arm changes. The change touches only `lib.rs`; the kernel source
SHA256 `d54c1061a548ffedcff86df02050946f0943ea070f2967af078fb19fd7dffb92` is unchanged
and the fatbin is unchanged. The same review moved the door read behind the shape
predicates so a non-Qwen prefill on a 120a build no longer reads the environment.

Server binary SHA256:
`989d8f618a7f4d4567e4c976ddd2e9799904afa39a892f4e576f342b38c4b460`.
The qualified Qwen runtime is `cad2b21ba`; subsequent formatting, diagnostic-only
changes and the DSV4 HC-default merge do not change that Qwen program.

All GPU gates and builds run on the designated non-serving 5090, architecture
120a, with the canonical lock and an empty compute list before each job.
RUSTC_WRAPPER is empty for CUDA rebuilds. Local rig gates are prohibited by the
owner: pushes set MEMRA_SKIP_PERF_CI=1; hosted CI still gates the merge.
