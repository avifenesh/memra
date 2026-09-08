# Independent static review, 2026-09-08

Reviewed `origin/main` 001c09e5d451798ef8d570f6087dc11527cbcc19 through code head
d49ec5e2bdca3c530b9c637ef11f102cf31c2e6b. The 15 lane commits replayed without conflicts;
range-diff reports every patch unchanged. This review used source and banked receipts,
not the previous worker's conclusions. No test, build, gate, GPU process or server was
started. Findings below are static, with diagnostics still required. Runtime code is
unchanged in this pass because no applicable passing regression receipt covers the fixes.

## P1: admission gives resident-slab credit to requests that may allocate fresh concat buffers

Location: `crates/memra-server/src/worker.rs:13899-13926`;
`crates/memra-engine/src/hybrid_forward.rs:6382-6423`.

After a plain request has populated the slab pool, the admission loop discounts each
new plain request based on that pool's capacity and lack of current borrowers. It does
not bind the credited request to a slab-consuming execution route. Later, two compatible
fresh requests can enter `prime_cache_batch` (`worker.rs:15764-15786`, default batch max
6 at `:15674-15680`). That route allocates fresh concatenated h/hx16 and FFN buffers
(`hybrid_forward.rs:8201-8204`, and the Step/Spark batch walker at `:7574-7575`), without
borrowing `prime_slabs_get`. The original resident pool remains allocated.

Thus reusable bytes are subtracted from estimated new workspace even though that route
will allocate them again. Under pressure this can admit a batch into CUDA OOM. The
12/12 historical pressure result serialized requests, so it does not disprove this
route mismatch. The prefix-composition formula correctly restores the old credit before
subtracting full workspace; the remaining risk is what execution the credit promises.

Required diagnostic: warm the resident pool, retain it, then admit two distinct fresh
prompts on a dense model with batching enabled, prefix/affinity capture excluded, and
headroom between credited and uncredited cost. Record `[prime-batch] B>=2`, pool/live
allocation deltas and every admission line. Compare serial and concat, including the
composed prefix arm where applicable. Fix by binding credit to actual slab reuse or
charging concat before scheduling; cover route selection in CPU tests and validate
allocation safety remotely. No speculative code fix made here.

## P2: Spark name normalization changes malformed GLM text into executable calls

Location: `crates/memra-server/src/toolcall.rs:730-736`, with shared dispatch at `:502-508`
and GLM parser selection at `crates/memra-server/src/lib.rs:7907`.

The new code strips a trailing `>` even when `strip_prefix("<function=")` did not match.
For example, `<tool_call>get_weather><arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>`
used to fail the name-markup validation and remain content. It now emits `get_weather`
with `city=Paris`. This applies to existing GLM routes as well as Spark. The complete
Qwen-body fallback is also enabled for every GLM parser instance, although its motivating
receipt is Spark-specific. No evidence here establishes that expanded GLM grammar.

Required diagnostic: CPU parser tests for bare `name>`, unknown markup, incomplete
function wrappers, legitimate Spark wrappers and full Qwen bodies, with every stream
split and both think-open states. Preserve GLM malformed-text passthrough; restrict the
normalization to a matched wrapper and scope Spark-only grammar if GLM compatibility is
not intended. Existing added tests cover valid wrappers and `<fn>`, not the bare `>` case.
No parser change was made here, and the sampled unterminated-think failure stays negative.

## P2: non-finite oracle tolerance inputs can mint a false passing rewrite receipt

Location: `crates/memra-engine/src/bin/rewrite_receipt_oracle.rs:70-79`.

The new environment readers accept every parsed f32, including `NaN` and infinity.
`ExecutionRewrite::verify_logits` (`crates/memra-gguf/src/execution_manifest.rs:195-204`)
uses `absolute > allowed` without validating the policy. With NaN allowance the
comparison is false even for out-of-band logits; matching argmax can therefore produce
`passed=true`. `validate_for` at `execution_manifest.rs:331-348` checks identity and the
passed bit, not finite tolerances, so the new producer can write that false pass.
This does not establish that any banked receipt used invalid inputs.

Required diagnostic: CPU tests rejecting NaN, +/-infinity and negative tolerances before
GPU initialization, plus a valid default-policy out-of-band same-argmax negative control.
The fix must retain the strict 0.05 lane policy. No tolerance was changed or exercised.
FLAGS rows now document the inputs and this open validation defect.

## P2: the new oracle reader accepts duplicate indices as a complete logit vector

Location: `crates/memra-engine/src/bin/rewrite_receipt_oracle.rs:42-52`.

The reader checks that the number of `logit` rows equals `vocab`, then assigns each
index into a zero-initialized vector. It never verifies a unique, complete index set.
For example, vocab 2 and two rows for index 0 passes the count check, overwrites index 0,
and silently invents a zero logit at index 1. An index >=vocab panics instead of refusing
the oracle. A damaged capture can therefore compare against fabricated values and mint
a misleading receipt if that fabricated vector passes. This is separate from the
observed numerical failure and is not a claim that its saved oracle was damaged.

Required diagnostic: CPU parser tests for duplicate, missing, out-of-range, non-finite
and reordered valid logit rows. Require exactly one finite value per vocabulary index
and reject invalid input before `Engine::new`. No fix without that test coverage.

## Other requested surfaces

- Dynamic FP8: source code/scale slices preserve aligned 128-row blocks and the format
  requirement survives lookup refusal. `model.rs:861` refuses requantized fallback;
  `lib.rs:17629` selects native dynamic MMQ before legacy dispatch;
  `lib.rs:17999,18043` disables q8_1 sharing and uses the original f32 operand;
  `lib.rs:18291` routes decode-exact through that same path. Component receipts cover
  m=1/2/5/9 with a nonzero q8_1 negative control. They do not prove BF16-container FP8
  oracle equivalence: intermediate rounding and KV format remain different, strict
  optimized parity is failed, and non-Spark declared-FP8 artifacts need regression proof.
  Next diagnostic: capture activation values/codes/scales and per-layer logits against
  the pinned FP8 oracle, keeping the 0.05 band. No claim of NativeQualified.
- hd256 window: the new single/double-buffer kernels instantiate the same 256-wide
  template, wrapper dispatch keys on head dimension, and portable builds retain the
  quantized f32 fallback. Banked cells cover CPU/f32 bands, a biting mask, odd tile
  starts, SB/DB identity and zero-window identity. No additional P1/P2 defect established
  by static inspection. Current-head runner and sampled long-context gates remain pending.
- Prefix arithmetic: `cost_after_prefix_restore` adds cold resident credit back before
  removing cold workspace, then charges suffix workspace minus suffix credit. Checked
  operations and credit bounds preserve fixed/context/draft/reserve terms. The remote
  CPU composition test covers those terms; GPU integration is still pending. This is
  distinct from the concat scheduling defect above.

Disposition: no-go for engine PR or merge until GATES.md requirements and these defects
are resolved with evidence. Documentation omissions were fixed; runtime findings remain
open. No band, parser behavior, GPU ownership or runtime file changed in this review.
