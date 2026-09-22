# Continuing-session depth qualification

Status: completed. The recorded short/long protocol gates, all 100 timed runs,
offline audits and exact archive reproduction passed. See [RESULTS.md](RESULTS.md).
The measured runtime is retained as a source archive; this publication applies
no engine change.

The completed matrix uses the stable prompt-checkpoint rule below, with suffixes
at or above the prime floor. Arbitrary shorter suffixes and broader HTTP/default
admission are not established by this dataset. The protocol and coverage in the
results define the claims; the wider API edge checklist below is not a serving
qualification.

Owner correction, 2026-09-21: preserving conversation text and learned D while rebuilding
KV every turn does not measure the intended continuing session. The earlier cold-prefill
receipts remain valid for their stated protocol, but cannot establish this result.

The cold study used public base `7326f0e176326bb9b445720068cc502ea132ffad`. After
recovering the interrupted session, this candidate was advanced to public main
`ea08bc7f8`, including Gemma's continuation-capable prime (#561), its fast attention
implementation (#564), and the carried sub-floor suffix correction (#578,
`758538a47647b67b1341cfd8b15202d626f7cfea`). The latter fixes the server's prime
walker; this runner exercises the direct session API. Its Qwen3.5-9B receipt does
not qualify our Qwen3.8-27B artifact.

The candidate also routes Gemma's direct restored-session API through the newly supported
prime for sufficiently long suffixes when the plan's capability permits it. That API still
used the older verify-trunk suffix routine on fetched main. This is an unqualified research
change until the cold/resumed and policy identity gates pass; merely compiling it is not
evidence of numerical equivalence. Short suffixes retain their existing path.

## Required behavior

- One eight-message conversation, actual sampled replies retained, learning state persists
  within and across replies and can move in both directions. Independent conversations reset.
- Full heads and unchanged cost learner; fixed and native controls use the same cache policy.
- Reuse a stable prompt checkpoint before the live assistant header, aligned to the native
  GDN prime grid and leaving at least PRIME_MIN_T suffix rows. This mirrors the serving
  checkpoint rule. Re-rendered generated replies are processed in the suffix; cached rows
  below the checkpoint are retained. This is prompt-prefix reuse, not a claim that every
  generated KV row survives.
- Exact token-prefix comparison before every resume. Missing checkpoint, overwritten ring,
  mismatched prefix or a zero-token hit on turns 2-8 fails the run. No cold fallback.
- Record cached input tokens, newly processed input tokens, checkpoint position, output
  tokens, request E2E, chosen depths and learner updates per turn. Report the actual reuse
  fraction so a boolean hit cannot conceal rebuilding most of the prompt.
- Request E2E includes rendering/tokenization, checkpoint restore, suffix prefill, generation,
  synchronization and output detokenization. Setup/warmup and receipt I/O remain excluded.
- Return to the requested roughly 16K-token initial prompt and up-to-2K-token replies, with
  eight varied messages. Record actual token counts; context reservation is not input length.
- Freeze one eight-turn conversation per run for both families. Include every completed
  run regardless of sampled output length or elapsed time; a duration threshold must not
  select different conversations for different policies.

## Qualification before timing

1. Compile and pin all runtime source, binary, checkpoint and tokenizer identities.
2. Check the exact target checkpoints: Qwen3.8-27B NVFP4+Q5_K embedded MTP, then
   Gemma4-12B Q4_0 with its Q8_0 assistant. Earlier BF16/SGLang results are separate.
3. Greedy warm-session tapes must agree across original fixed, native, instrumented native,
   learned and candidate fixed depths for all eight turns. Check cold/segmented reference
   against resumed state separately; do not infer this from text-history equality.
4. Include length-cap overshoot, EOS, assistant-header rewriting, grid-aligned boundaries,
   and suffixes below and above the prime floor. Exercise exact cache bytes and recurrent
   state where relevant. A prime-walker fix is not automatically proof for the direct API.
5. Sampled native/instrumented tapes must match, and all scored turns must prove reuse.
6. Freeze calibration on disjoint warm-session prompts before held-out comparisons. The
   cold study's K=2/K=3 choices are historical controls, not warm-session winners.
7. Use balanced paired orders and pooled returned tokens / summed request seconds.

Do not pool this study with the cold-prefill study or promote a runtime default from a
preparation-only patch. Pending gates stay pending.
