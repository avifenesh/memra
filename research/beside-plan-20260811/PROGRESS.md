# 27B-beside-Step protocol progress — 2026-08-11

Lane: `lane/cx-besideplan`

Status: protocol complete and ready for the lane's single commit; no GPU, build, server launch,
or measurement occurred in this lane.

## Scope and stop line

Produce one executable measurement protocol for comparing Step-3.7-Flash alone with
Step plus the Qwen3.8-27B release target on the 2x RTX PRO 6000 serving pair. This lane
does not choose pricing, deployment, or hardware; it does not merge, tag, push, or run
`cargo fmt`.

## Inputs checked

- [x] Lane inbox and `/home/avifenesh/projects/bw24/CLAUDE.md` read first.
- [x] `crates/memra-server/src/` load, model-list, request-model, and routing seams inspected.
- [x] `ECONOMY-20260810.md` Q3, `ECONOMY-VERDICT-20260810.md`, and `TRIAL-PLAN-V2.md` read.
- [x] Existing Step box1 receipts and the later `research/27bab-20260810/` evidence reconciled.
- [x] `PROTOCOL.md` drafted and every future result cell marked `PENDING box1 execution`.
- [x] Intended two-file diff reviewed and ready for the lane's single commit.

## Findings that constrain the protocol

- memra already supports multiple resident models in one process through `MEMRA_MODELS`,
  a per-name `LoadedModel` map, request `model` lookup, `/models`, and `/v1/models`. It is
  not a single-model-only server.
- That in-process option shares one CUDA-owner worker and one interleaving scheduler. The
  measurement arm therefore keeps the economy document's two-process shape: Step remains
  PP-2 on physical devices 0 and 1, while Q27 is restricted to physical device 0 in its
  own process and port. This is deliberate contention, not an exclusive card partition.
- The Q3 `+18.7%` figure is sourced, but only to a stopped, incomplete training-co-location
  campaign: one observed cell, not a campaign median, with an exactness mismatch. It is a
  proxy, not a Q27 serving result.
- The repository now contains a direct Qwen3.6-27B campaign that postdates the first-pass
  economy note. It found resident-idle neutrality but severe co-active interference. Those
  receipts are prior evidence only; they do not fill any Qwen3.8-27B result cell.
- The lane inputs do not yet pin an exact Qwen3.8-27B artifact. The protocol must
  refuse execution until the exact target, quantization, draft, runtime commit, prompt
  corpus, and hashes are frozen. The Qwen3.6 15--16 GiB weight estimate is a planning proxy,
  not a Qwen3.8 memory receipt.

## Deliverable

`PROTOCOL.md` is the only result artifact. It pins the two arms, documents the already
implemented multi-model seam, defines N=5 interleaving and receipt capture, separates Step
interference from Q27 productive throughput, and provides an economics-shaped decision equation
without producing a verdict.
