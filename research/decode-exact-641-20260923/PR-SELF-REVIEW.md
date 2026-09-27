# Self-review: memra#641 (a request primed inside a fresh batch took a second numeric program)

Author's review of the full diff `main..lane/decode-exact-641-20260923`, posted as a PR comment per the owner rule.
The lane was run by lane E and reviewed by the lead.

## What the diff is
- `crates/memra-engine/src/hybrid_forward.rs`: `prime_cache_batch_inner` runs `full_attn_prime_core_inner` per
  sequence for every batch, fresh or carried. The per-sequence core is the solo prime's program: quantize-then-attend
  through the cache view. The fresh varlen arm (`use_favl`, task #18, `MEMRA_FA_VL`) attended bf16 copies of the
  pre-quantization K/V, so a request primed inside a fresh `[A,B,C]` batch got different logits from the same request
  primed alone. That arm is deleted. The batched per-sequence o_proj f16 path now takes the solo
  `fa.wo_pqs.is_none()` AWQ guard.
- Kernels and bindings deleted: the VL kernels in `cu/flash_attn.cu`, the Hopper `memra_fa3_vl` twin in
  `cu/fa3_prefill.cu`, `Engine::fa_prefill_vl8` / `attn_pre_vl8` and the `FaSeqVl`/`FaVl8`/`AttnPreVl`/`AttnPreVl8`
  structs in `lib.rs`. `MEMRA_FA_VL` moves to the FLAGS "Removed doors, 2026-09-23" ledger, and the KERNELS.md rows
  follow.
- Gates: `tools/prime-tick-exact-gate.sh` (the serving shape, fast-gate rows `ptick`/`ptickc`) and
  `tools/prime-batch-exact-gate.sh` (the standing `prime-batch-gate --exact` on the 9B, rows `pbg9`/`pbg9c`), each with
  a canary, in a local-ci stage behind `MEMRA_CI_PRIME_EXACT`. `concat_prime_probe` is the engine replay the gates
  drive, and `prime-batch-gate --canary` is new.
- `research/decode-exact-641-20260923/` (RESULTS, the pre-registrations, raw receipts for both cards and both main
  merges), the INDEX row, and one line in `research/spec-ctx-edge-20260923/RESULTS.md`: #668's owed PRO 6000 run,
  which lane E's box sitting produced.

## What I checked
- Every reference to the deleted symbols is gone from `crates`, `tools` and `docs` (`git grep` for `attn_pre_vl8`,
  `fa3_vl_raw`, `FaSeqVl`, `AttnPreVl`, `memra_fa3_vl`, `MEMRA_FA_VL`). The only remaining hits are the explanatory
  comment, the Removed-doors ledger, the KERNELS.md note and the gate's red-arm header.
- The replacement loop is the carried arm's own per-sequence dispatch, now unconditional. `carried` had no other
  reader, so its binding goes too.
- A retired door in a code-owned family refuses memra-server boot through the env audit (#483), so a launcher
  still exporting `MEMRA_FA_VL` fails loudly instead of silently serving the per-sequence core.
- One numeric program per request: the ptick gate reads logits, hidden and 66 cache digests bitwise against the solo
  prime, with 0 differences on the fix. Its canary proves the gate can see a difference.
- Red and green on both cards. 5090 9B: the pbg exact rows fail on main (`6 FAIL(s)`, `8 FAIL(s)`) and pass on the
  fix, and the ptick gate is red at token 8 on main and green on the fix. PRO 6000 27B: the three exact rows fail on
  main and pass on the fix, and ptick is green. Both main merges were re-gated on the 5090 (windows 9 and 10), with
  identical lines.
- Cost: flat on both cards. 5090 +0.19% (N=6 interleaved). PRO 6000 -0.22% (N=6 per arm, alternating, fix faster in
  all 6 pairs; batched prime median 790.772 ms against 792.487 ms). The removed arm was not a PRO win at this shape.
- serve-smoke, decode-batch-gate (config B=8 and strict B=4), spec-ctx-edge and the fairness cell are green. On the
  CPU side: clippy `-D warnings`, fmt, check-flags, diff-check and shellcheck.

## Limits
- No H100 re-measure: Hopper batched primes now run the per-sequence core too. A varlen twin over the dequantized
  view may come back only with its own bit-identity receipt against the solo prime.
- The AWQ plus f16-mirror pairing has no receipt, because neither rig has a scaled artifact. The guard only makes the
  batched path refuse what the solo path refuses.
- The PRO cells ran on the first main merge. The second merge (#674 to #676) touches no prime or spec path.

## Push regime
Engine-source pushes used `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged). Every other hook ran.
Revuto: if capped or unavailable, this comment is the review.
