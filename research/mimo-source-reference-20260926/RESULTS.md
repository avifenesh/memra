# MiMo source checkpoint on Memra: reference and component verdict

Date: 2026-09-26. Decision: **NO-GO for Memra customer or trial serving.**
The DeepSeek trial and all customer routes remain unchanged. Tiyuvta's product
is help for companies that self-deploy open models; this was a non-production
engine qualification run.

## Pinned run

- Model: `XiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4`.
  The previously published `tiyuvta` NVFP4 mint is a separate development
  seed with `input_scale=1.0`; it has no activation or model-quality gate.
- Hardware: one Verda FIN-02 `2RTXPRO6000.60V` VM, two RTX PRO 6000
  Blackwell Server Edition cards, 176 GiB host RAM, 600 GB requested disk.
- Source verification: HF checked 90 files at the pinned revision.
  The source header digest matched
  `5ebbdd27e45716b805fc2bdf115c8345b4bfc03c6012b860f76b6b222c758aee`;
  73,081 physical header entries bound to 36,922 semantic tensors.
- CUDA build image ID:
  `sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d`.
  The full logs and report files are held off-box in a private archive with
  SHA-256 `0780e10c804fe5a51b9be29ca7553acb2ea4cf357c22cd98cf840a28e9574db5`.

| Evidence | Exact program and result | Boundary |
| --- | --- | --- |
| Streamed text reference | Memra `63eb79c0a054d1cd3ff754fb5cfb823c2ec598d7`, binary SHA-256 `0803f8743f066651854ad67d2037eedccb5a40711f78ff70ac0451714f9ed50f`. Token ID 42 traversed all 48 trunk layers in 20m 40s; 152,576 finite last-position logits, argmax 220, raw TSV SHA-256 `1c467a0dc033bc3087fe229233806cc97bc68f57158690e0bbae76af2018c1eb`. Peak RSS was 177,970,964 KiB, about 170 GiB. | One Memra f32-accumulation diagnostic, not independent numerical parity. Reference RAM margin on this host is small. |
| FP8 fused QKV | Memra `23edb61d3868be864f89815967ca3112ac501041`, binary SHA-256 `7918623369604ecb1e7ae640a45d8d6035c1b54ae054e2e3d64e9d75ea22cf97`. Each card ran the four original checkpoint shards from one full and one sliding layer on a fixed `-1/0/1` input. All 16 shard outputs were finite. Worst per-row absolute error versus Memra's CPU source decoder: `2.6226e-05`. Minimum reported cosine: `1.0` at nine decimal places. | Projection component, not a complete attention step or a model request. |
| MXFP4 expert decode | Memra `27f35b31ee2ba3ed3aa3bf0cded8b276f7f0a6cc`, binary SHA-256 `5fe688ed3a1b491cf03cfc3322addac9227c9b242e2d817ff33a8e228b8f727e`. On each card, expert 0 and expert 255 each decoded gate/up/down. `100,663,296` BF16 values were checked across both cards with zero bit mismatches against Memra's scalar OCP decoder. | Weight-dequant component, not routing, matmul, activation, or a complete request. |

The verified lease quote was $25.3902 at a six-hour on-demand ceiling,
including the declared disk bound and cleanup margin. The VM ran from
06:51:34 to 07:59:28 UTC, about 1.13 hours. A prorated compute-plus-disk
estimate is $4.6812; the provider invoice is not verified. The direct Verda
instance read reports `discontinued`, and no owned instance, volume, script
or SSH key remains. The controller reaper is inactive. The sanitized
[teardown receipt](VERDA-LEASE.json) records those reads. No customer data
or traffic used this VM.

## Remaining admission work

1. Execute the pinned source through a plan-driven Memra GPU text path, including
   sharded QKV projection, both attention types, sliding-layer sink softmax,
   value scaling before the KV write, and the source MXFP4 routed experts.
2. Compare a complete native request with an independent model oracle. Repeat
   the fixed code and math gates on the exact two-card Memra profile.
3. Measure cold and warm customer-shaped requests, continuing-session reuse,
   fan-out, 1M admission/concurrency, long output, and the supported modality
   surfaces on one two-card box.
4. Only after engine and workload gates pass, verify the customer route,
   request-linked accounting, tenant isolation, provider controls, fleet pin,
   canary, and rollback.
