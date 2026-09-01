# CX spec/plain identity fix — progress

- Branch: `lane/cx-specfix`
- Base: `4f80a0882f2c36572a386c6dc29831069c435ca9`
- Trusted bisection receipt: `research/affinity-20260809` at `f7e04bc2`
- Regression boundary: `349be208` (honesty merge), parent `5c585a16`
- Status: COMPLETE — fixed, raw before/after receipts captured, required gates green

## Required contracts

- Spec text must be byte-identical to plain text for the serving smoke prompt.
- `n_tokens == tokens.len()`.
- `max_tokens` remains exact.
- `tokens_emitted == generated.len()` remains asserted.
- Spec `tokens_out` accounting remains correct.
- The sampled truncation matrix, cache meter (23/23), server tests, and q9 K=1..8
  self-consistency must stay green.

## Work log

1. Accepted the existing three-binary bisection; no re-bisection will be run.
2. Confirmed a clean worktree at the requested base and that
   `~/.lanectl/inbox/cx-specfix.md` is currently absent.
3. Located the exact serving harness, captured plain/spec text and token ids under bounded
   RTX 5090 lock holds, and classified the fork as token divergence.
4. Reproduced on the tip binary (`sha256
   0cb88b54b2d6063301a0833c291745140232fbc06fd73519c9c1b33a815dee95`) with the exact
   serve-smoke q9 mutex prompt and 64-token budget. Text first differs at zero-based character
   110: plain `Constraint: in`, external-draft spec `Constraint: it in`.
5. Captured native token receipts for the same chat-templated prompt. Both responses satisfy
   `n_tokens == tokens.len() == 64`; the ids first differ at zero-based token 32, exactly the
   default 32-token scheduler-burst boundary:
   - common tail before the fork: `[262, 348, 256, 42794, 25]`
   - plain continuation: `[303, 348, 588, 11316, 19315, ...]`
   - external-draft continuation: `[424, 303, 348, 588, 11316, ...]`
   - plain ends `[..., 3437, 1503, 424]`; external-draft ends `[..., 3437, 1503]` at the
     same exact cap.
6. Classification: case (b), ids diverge. The honesty path uses the burst-local `room`
   (`request_remaining.min(32)`) as the public clamp. A session-mode engine burst may return
   one or more cache-authoritative surplus ids past that scheduler target; hiding those on a
   non-final burst advances `SpecSession` without advancing the worker's generated/sampler/fed
   history. This selected separating the scheduler burst target from the request-owned visible
   room and pinning the intermediate-surplus case with a unit test.
7. Fixed in `8392e9a5`: `burst_target = request_room.min(burst_t)` is passed only to the
   engine, while emission/public-vector clamping uses the full request-owned `request_room`.
   Thus a cache-authoritative surplus id remains visible on an intermediate burst, and only a
   surplus beyond the final request cap stays engine-private. `spec.rs` is untouched.
8. Post-fix q9 A/B at binary SHA-256
   `cd4b41ed2775ffb10e30ed306b0fe013687948fe699499389abf99367f98ed63`:
   OpenAI chat text MATCH and native token ids MATCH. Both native responses report
   `n_tokens == tokens.len() == 64` and `MaxNew`; both OpenAI usage blocks report exactly 64
   completion tokens and retain per-request spec accounting.
9. Required gates:
   - focused spec-emission tests: 3/3 PASS
   - `cargo test -p memra-server`: 154/154 PASS (153 baseline plus the new regression)
   - full `tools/serve-smoke.sh`: 0 failed; spec/plain MATCH; sampled truncation 4/4;
     cache-meter internal checks 23/23; affinity 4/4
   - q9 `run-spec`: K=1..8 self-consistency 8/8 PASS
