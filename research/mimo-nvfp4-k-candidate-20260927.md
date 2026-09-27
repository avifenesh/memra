# MiMo NVFP4-K candidate, 2026-09-27

Decision: **do not advance this codec and kernel pair**.

The pinned `XiaomiMiMo/MiMo-V2.6-Flash-RL` source at
`3b38d063180c3e4aed9691fdc735f3d10b266ee4` was tested on one Vast
2×NVIDIA RTX PRO 6000 Blackwell Server Edition development box. The tested
Memra code was `342600139b7359da3914a75a391b521c35a3e3b5`. This was
a split-attention component experiment with synthetic Q/K/V, not a model
request or customer-serving run.

The control stored global K as Q8_0 and V as GGUF NVFP4. The candidate
stored both K and V as GGUF NVFP4. Both used a 256-token split, F32 Q and
attention arithmetic, and the same NVFP4 V bytes. The candidate GPU
encoder matched the CPU NVFP4 bytes for widths 128 and 192. GPU attention
agreed with the independent oracle over dequantized K/V to a maximum
absolute error below `2.10e-8` in the measured short patterns.

| Measure | Q8-K control | NVFP4-K candidate |
| --- | ---: | ---: |
| Largest short-pattern K-only attention relative L2 vs uncompressed K | 0.00414656 | 0.05552477 |
| GPU 0 median 1M packed-cache attention, 3 runs | 13.5424 ms | 20.0824 ms |
| GPU 1 median 1M packed-cache attention, 3 runs | 14.1192 ms | 20.8547 ms |

The candidate's largest measured K-only output error was 13.39× the
control's. Its 1M component time was 1.483× the control on GPU 0 and
1.477× on GPU 1. Runs reversed arm order across cards and repeats.
The 1M case used one constant-cache query, so these times are not
end-to-end decode throughput or a full model quality result.

NVFP4 K would save 384 K bytes per token per global layer. At the
1,048,576-token allocation, the five/four global layers per stage imply
2,013,265,920/1,610,612,736 fewer persistent K bytes. This candidate's
larger 256-token-split workspace costs another 68,689,920 bytes per
stage versus the current deep Q8 program. The net projection is
1,944,576,000/1,541,922,816 bytes per stage. It is an allocation
calculation, not a demonstrated full-model capacity gain.

The private raw archive is
`~/.local/state/mimo-memra-vast-k-candidate/20260927/qualification-receipts.tar.gz`
with SHA-256
`7c3792cc9cd7bdef3089b9cc0928cde72464fba53cfd2a0def3e030fc17eed58`.
All three recorded jobs exited zero. The task-owned VM and timer were
closed, and no customer route changed.
