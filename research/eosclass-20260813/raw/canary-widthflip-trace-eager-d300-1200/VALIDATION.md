# Validation

Verdict: **PASS as a diagnostics-only reproduction of the retired eager-to-batched
transition.** This is not a repaired-default gate: the server explicitly ran with
`MEMRA_SERVE_B1FAST=1`, `MEMRA_SERVE_DEVSAMPLE=0`, and `MEMRA_EOSCLASS_TRACE=1`.
`MEMRA_SERVE_GS=1` was recorded too, but the frozen request has a 60-token generation
budget, below the default `MEMRA_GS_MIN=384`, so the exercised solo program was eager B1.

- Client exit: 0; verifier verdict: PASS with no errors.
- Frozen delay sweep: 37 target cells, 300--1,200 ms in 25 ms increments.
- Accounting: 153/153 admitted/completed; 149/149 post-seed full-prefix hits; 4 cold
  seeds; 8,748 emitted tokens; zero cache evictions, admission/session/VRAM defers,
  OOM parks, or protocol failures.
- Four targets (450, 600, 625, and 675 ms) returned HTTP 200,
  `finish_reason=stop`, and exactly 11 tokens with the historical SHA-256
  `ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73`.
- Each terminal receipt selected the declared model EOS id `248046` at generated index
  10 from a complete 248,320-value host logit vector. EOS was rank 1/top 1 in all four
  cells. The top-1 margins were 1.1117076874 at 450 ms and 0.4370250702 at
  600/625/675 ms. This rules out stale device sampling and HTTP-side truncation for the
  reproduced 11-token outcome: the model logits genuinely favored EOS.
- Every failing target crossed numerical programs before EOS. At 450 ms the receipt ran
  boundary sample -> eager B1 for generated indices 1--2 -> B2 for index 3 -> B4 for
  indices 4--10. At 600/625/675 ms it ran boundary sample -> eager B1 for indices 1--7
  -> B4 for indices 8--10.
- `trace-verification.json` joins each client result to its exact sampler receipts by
  `trace_id`; it reports 8,748 sampler receipts for 8,748 output tokens.
- The failure-signature scan's only match is the startup gpu-watch banner enumerating
  fatal Xid codes; no Xid, CUDA error, OOM, panic, illegal address, or protocol failure
  occurred.

The diagnostic binary SHA-256 is
`17a222026e08b65f9344407ba9108cb554688c0431365932bb4e78de1033597d`.
The provenance snapshot also records a pre-existing ColBERT Python process using
1,390 MiB. This run makes no throughput claim; all conclusions are exact token/logit
and state receipts.
