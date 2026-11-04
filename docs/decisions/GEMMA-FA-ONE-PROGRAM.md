# Gemma 4 SWA prefill runs one attention program (2026-11-04)

**Chosen:** dense and MoE Gemma 4 SWA layers prime through the windowed attention arm at every
prompt length. `MEMRA_GEMMA_FA_ONE_PROGRAM` is deleted.

**Rejected:** the previous default, which switched program at `t > sliding_window` (unwindowed
f32 operands below the window, windowed bf16 above). Under that default the published prefix
KV could depend on the total prompt length: rows `[0..932)` of a 932-row and a 1048-row prime
differed in 59 of 60 layers (lane/gemma-restore-exactness-20260819). That breaks the
one-numeric-program-per-request law for prefix restore.

**Why the rejected arm had kept the default:** the windowed arm changes the bytes short prompts
get, so the flip was left to the owner. Owner, 2026-11-03: flip ON once the gemma4 dense and MoE
run-gen, run-spec and acceptance rows pass on the 5090, then delete the switch.

**Measurement that settled it:** `research/gemma-one-program-20261104/RESULTS.md`, RTX 5090, the
31B dense and 26B-A4B QAT Q4_0 GGUFs. Argmax was `flips=1 bad=0` (31B) and `flips=2 bad=0` (26B)
on both arms. Spec K=1..8 stream agreement was 128/128 at every K on both arms, and the
drafted/accepted counts matched at every K. The prefix KV, whole-prompt KV and logits digests
matched between arms. Single sequential runs, so throughput is read as flat.

**Scope:** on this card both arms wrote the same bytes, so the receipt shows the flip is safe
there, not that it fixes a live divergence there. Elsewhere it removes the second program by
construction.
