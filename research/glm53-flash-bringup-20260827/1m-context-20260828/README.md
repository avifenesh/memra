# GLM-5.3-Flash: making the 1,048,576-token context reachable

**Lane** `lane/glm53-1m-context` (branched from `lane/glm53-flash-bringup` @ `bd68c3681b`,
merged `lane/glm53-ring-sizing` @ `52732cef75`) · opened 2026-08-28.

The ring lane is a DEPENDENCY, not a coincidence: without `index_ring_take` draining the DSA
index ring inside the call, any chunk size this lane picks would be coupled back to ring sizing,
which is the exact coupling that lane killed.

## The blocker, and it is not one term

The reported blocker was the DSA indexer's score plane: `t * n_pools` f32 with
`n_pools = t_kv / index_kpool`, which at a monolithic prime (`t == t_kv == N`) is N-SQUARED
bytes, per MLA layer, per call — 1099.5 GB at 1M.

That is the largest term. It is not the only one, and REMOVING IT IS NOT SUFFICIENT.
`prime_cache_hyper` is documented "deliberately UNCHUNKED", so `t` is the admission limit itself
and EVERY per-call transient is proportional to the whole context. Measured from the banked
config of the real artifact (`glm-config.json`, hidden 4096, 4 mHC streams, 64 heads,
kv_lora_rank 512, qk/v head dim 256, indexer 32x128 pool 4 topk 2048, 11 trunk MLA layers):

| term (one call, t = N = 1,048,576) | bytes | site |
|---|---:|---|
| DSA score plane | **1099.51 GB** | `mla_kpool_indices`, `t * n_pools` f32 |
| MLA `q_lat` | 137.44 GB | `mla_attn_core`, `t * nh * kv_lora_rank` |
| MLA `o_lat` | 137.44 GB | `mla_attn_core` |
| mHC stream state `x` | 68.72 GB | `hyper::expand`, `t * streams * hidden` |
| MLA `q_b` / `q_nope` | 68.72 GB | `t * nh * qk_head_dim` |
| MLA `attn` | 68.72 GB | `t * nh * v_head_dim` |
| trunk per-site transients (y, h, mixed, z, ffn_out) | 85.90 GB | `prime_chunk_hyper` layer loop |
| indexer `q_index` | 17.18 GB | `t * index_n_heads * index_head_dim` |
| `embedded` | 17.18 GB | `self.embed` |
| selected-row index list | 8.60 GB | `t * (topk + pool - 1)` i32 |
| k_raw / k_norm / gate | 1.61 GB | `t * index_head_dim` x3 |
| **per-call total** | **1711.02 GB** | (the capacity gate's own number) |

**Zero the score plane and 611 GB remains** — 3.2x past a 2x96 GB box. So a query-tiled
score/select inside `mla_kpool_indices` is neither necessary nor sufficient. `t` has to leave the
requirement one altitude UP, the same move the ring lane made one altitude down: the prime is
SPLIT into calls, and every term above becomes proportional to a bounded working set instead of
to the context.

Also true, and load-bearing for why the split is free rather than a numeric change:

* **The mHC residual is strictly per token.** `crate::hyper`'s own contract: `mixes[t,:]`, the
  RMS rescale over the token's own `streams*hidden` slab, the Sinkhorn, the collapse and the
  post are all indexed by one token. There is no cross-token coupling to break.
* **KDA prefill is a SEQUENTIAL scan**, not a chunked UT transform (`kda.rs` header:
  `memra_kda_scan_s128` runs prefill and decode alike). There is no fold grid, so the GDN grid
  law (`align_prime_ranges_to_gdn`) has no KDA analogue to violate. The conv ring already carries
  across calls — it is the seam decode uses every step.
* **The latent KV plane is f32** (`LatentKvLayer::rows`, `e.zeros(max_ctx * width)`), so a later
  chunk reads the earlier chunks' rows in the SAME numeric class it would have computed them in.
  There is no analogue of the serial trunk's f32-vs-quantized-KV class edge that made
  `MEMRA_PRIME_CHUNK` steer arithmetic in 2026-08-05.
* **The indexer's pool keys are already incremental** (`index_pools_ready`): a pool key is a pure
  function of its own `pool` state rows and the constant `kpool_ape`, final the instant the
  pool's last row lands. A split cannot move one.

## Gate first, RED (receipt 01)

`crates/memra-engine/tests/glm5_prime_capacity.rs`, host-only, four arms, pinned against the
BANKED config of the real artifact through the real `HfConfig`/`ModelConfig` parser — no typed
geometry constant in the file:

* `the_mhc_prime_schedule_covers_the_prompt_exactly` — GREEN today and must stay green: the
  schedule is a partition (starts at 0, ends at t, gapless, no empty call), so "bound the
  per-call rows" cannot be bought by dropping tokens.
* `the_mhc_prime_carries_a_bounded_number_of_rows_at_every_context` — RED: 16384 rows at ctx
  16384, against the `PRIME_CHUNK_MAX_TOKENS` bound of 4096. Four context values, 16k to 1M.
* `the_mhc_prime_transient_is_sub_quadratic_in_context` — RED: 42.5 GB at ctx 65536 to 221.6 GB
  at ctx 262144, a 5.2x step over a 4x context step. Two bands, adjacent-rung (<=5x per 4x) and
  END TO END (<=80x over the ladder's 64x of context, against 4096x for a quadratic), so a
  PARTIAL reintroduction of the N^2 term fails too.
* `the_mhc_prime_transient_at_the_native_context_fits_the_activation_budget` — RED: 1711.02 GB
  against the 8 GB the placement receipt reserves for CUDA context, activations and workspace.

RATIO, NOT A MAGIC BYTE COUNT, on the scaling arm — the fit arm is allowed its budget because it
compares against real hardware, and it names the receipt the budget comes from.

PLUMBING on the red commit, all of it inert so the red is the RULE and not the shape of the code:
`prime_cache_hyper` splits into a driver walking `hyper_prime_ranges` and a `prime_chunk_hyper`
body; `hyper_prime_ranges` today returns ONE range covering the whole prompt, which is exactly
today's behaviour; `hyper_prime_call_rows` names the number the gate asserts on; `seq_end` is
computed ONCE before the walk with `queued_after` (the tick-seg law the serial loop carries).

## The fix, GREEN (receipts 04, 05)

`hyper_prime_ranges` delegates to `prime_chunk_ranges`, so the mHC prime walks the same schedule
the serial trunk does. Nothing else changed: no kernel, no allocation site, no numeric path.

| ctx | prime calls | rows/call | per-call transient | whole-prime stack | peak |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 4 | 4096 | 2.456 GB | 0.268 GB | 2.724 GB |
| 65,536 | 16 | 4096 | 2.657 GB | 1.074 GB | 3.731 GB |
| 262,144 | 64 | 4096 | 3.462 GB | 4.295 GB | 7.757 GB |
| **1,048,576** | **256** | **4096** | **6.684 GB** | **17.180 GB** | **23.864 GB** |

**1711.02 GB -> 6.684 GB per call at the native context, a 256x cut.** Growth across the ladder's
64x of context is 2.7x (linear would be 64x, quadratic 4096x). The per-call row count is 4096 at
every context, which is the whole claim: `t` is out of the requirement.

### The term the split does NOT remove, stated rather than hidden

`prime_cache` returns the full pre-output_norm hidden stack `[t, n_embd]` (`generate_spec`'s
`prompt_h`). That is **17.18 GB at 1M**, linear in context, and no split can bound it — each
call's rows are copied into the same full-length buffer. It is now a named term in the capacity
gate (`prime_lifetime_bytes`) and it is the largest single follow-up available: only spec/MTP
consumes `prompt_h`, and glm5_next refuses every speculative entry point today, so making it
optional would give back 16 GiB. That is a RETURN-CONTRACT change and is deliberately not
smuggled into this lane.

### Placement at 1M, corrected

`PLACEMENT-RECEIPT.md` budgeted 8 GB for "CUDA context + activations + workspace" and a 40.42 GB
KV plane. Both numbers moved and this lane owns the correction:

* KV at 1M is now **27.6 GB**, because `lane/glm53-ring-sizing` landed the tail ring that receipt
  lists as an unimplemented saving (12.88 GB indexer plane -> 63 MB);
* 8 GB was never an activation measurement. The prime's **23.86 GB (22.2 GiB)** peak is, and it is generated by
  the gate rather than transcribed.

Box usable 191.2 GiB, minus non-expert weights 13.66, minus KV 25.70, minus a 4 GiB CUDA reserve,
minus the prime's 22.2 leaves **~125 GiB for routed experts against a 163.27 GiB bank — about 77%
resident under the existing SLRU**, above the 15% hot mass the receipt shows already fits and
below the 81% its (pre-ring) 1M row promised. **1M closes on 2x96 GB.** The capacity gate asserts
this as a derived criterion (expert residency >= 75%), not as a byte literal.

## Bit identity is NOT the bar on this trunk, and that is measured (receipt 02)

`MEMRA_PRIME_CHUNK` is documented a pure memory knob and the serial trunk's `chunkinv` gate holds
it to byte identity. That cannot be held here, and the cause is not the split:

* the chunked and monolithic arms diverge at **row 0**, at every chunk size, worst 3.815e-6
  absolute. Row 0 cannot be reached by any cross-token state — not the KDA conv ring, not the
  recurrent state, not the latent plane, not the indexer's incremental pool keys;
* isolated directly: **`Engine::linear` (cuBLASLt f32), the mHC `mixes` GEMM in `hyper::pre`, is
  not m-invariant.** m=32 vs m=200 moves 9601/12288 output bits at worst 3.815e-6 — the same
  number — while m=128 and m=199 vs m=200 are bit-identical. cuBLASLt reselects its algorithm by
  shape and the reduction order goes with it.

`hyper.rs`'s own header already conceded this GEMM is "a serving trunk, not a byte-parity oracle".
So the split EXPOSES a near-tie that was always there. Consequences, written into the
`MEMRA_PRIME_CHUNK` FLAGS row rather than discovered later:

* **prompts <= 4096 are byte-for-byte the pre-change binary** (`ranges.len() == 1` short-circuits
  to the old body), so every banked short-prompt receipt for this model survives;
* **prompts > 4096 change bits by default.** Any banked long-prompt greedy transcript is not
  byte-stable across this change and an argmax near-tie can flip;
* the k-pool selection sorts on ReLU'd scores where exact-0.0 ties are ORDINARY, so a last-ulp
  move can flip which zero-scoring pool enters the budget. The reference arm is the instrument
  that covers that; a sibling-only comparison would not.

## The correctness gate, and how it was proven able to fail (receipts 03, 05)

`crates/memra-engine/tests/glm5_chunked_prime_gpu.rs`, over a fixture carrying **both mixers and a
live indexer** (one KDA layer + one DSA layer), because the three things a split could break live
in three places: the KDA conv ring and recurrent state, the MLA latent plane, and the indexer's
resident pool keys with `index_pools_ready`.

TWO ARMS, and the truth arm is the primary one — chunked-vs-monolithic alone is comparing
SIBLINGS, and two arms sharing one corrupted input both pass (LAW:pin-against-truth):

* `a_chunked_mhc_prime_matches_the_reference_executor` and
  `a_chunked_prime_then_decode_matches_the_reference_executor` — against
  `memra_reference::execute` at 2e-5 relative, three prompt lengths x five chunk sizes including
  lengths that are not a multiple of the chunk and chunks that do not divide `index_kpool`;
* `a_chunked_mhc_prime_stays_inside_the_near_tie_band_of_the_monolithic_prime` — the sibling arm,
  kept only to isolate THE SPLIT as the variable, over every row of the hidden stack.

PROVEN ABLE TO FAIL, twice:

* **mutation (receipt 03):** walking the chunk ranges in REVERSE — a real state-carry defect —
  lands at **1.346e0**, five orders above the band and in the range of the serial trunk's actual
  2026-08-05 chunk-invariance defect (1.813e0). Both truth arms and the sibling arm go red; the
  negative control stays green, which is what says the control is measuring something else.
* **negative control:** `the_parity_comparison_has_teeth` asserts a one-token prompt change
  escapes the band. Its FIRST version failed and the failure was informative: it perturbed the
  middle of the prompt, and with KDA's -5.0 decay floor and a two-pool fixture budget a token 100
  positions back genuinely has no reach to the last row. It now perturbs the LAST token for the
  logits half and asserts the hidden stack moves at the perturbed row itself for the decay-proof
  half.

**A mutation that did NOT fire, and it is a finding.** Making `pos_d` call-local instead of
session-absolute — the classic chunked-prime bug — moved nothing at all. glm5_next is NoPE end to
end (`qk_rope_head_dim = 0`, `mla_use_nope`) and KDA is positionless, so `pos_d` reaches no kernel
on this path. Recorded so nobody reads the gate as covering a positional contract it cannot.

## Nothing else moved (receipt 05)

Rig 5090 under `flock`, `NVIDIA_TF32_OVERRIDE=0`, correctness only, no timing quoted:
**4/4 chunked-prime, 12/12 kpool indexer, 6/6 hyper-connections, 5/5 MLA forward, 3/3 KDA
fixture** — 30/30. Host: 5/5 capacity, 18/18 memra-kv.

## What is NOT demonstrated here, and what it needs

* **The real artifact at 1M has not been primed.** Everything above is the arithmetic (host gate,
  real banked config, real schedule function) plus fixture-scale correctness on the rig. The
  real-artifact capacity arms need the two-card box; both bench boxes belong to other lanes and
  this lane did not take one. **This is an ask to the owner, not a take.**
* The stateless `HybridModel::forward` / `mla_attn` arm keeps its monolithic score plane. It is
  sized to the request rather than to a session and is not the serving prefill path, so it is out
  of scope here — but it is the same N^2 shape and would need the same treatment if it were ever
  put on a long-context path.
* **DEBT, named in code:** if the chunked KDA UT twin ever replaces the sequential scan as the
  prefill path, it acquires a fold grid and this schedule's internal boundaries must be snapped to
  it exactly as the GDN ones are (`align_prime_ranges_to_gdn`), or chunked prime stops matching.
  The `gdn_grid` argument on `hyper_prime_ranges` is the seam that change lands on.

## Receipt provenance

Receipt 02 (`02-diag-chunk-divergence.txt`) was produced by two diagnostic tests —
`diag_first_divergent_row` in the chunked-prime gate and a standalone `diag_m_invariance.rs` —
that were deleted once their numbers were banked, because a diagnostic that only ever answers one
question is not coverage. **Both are reproducible at commit `e179a7c7f4`.**
