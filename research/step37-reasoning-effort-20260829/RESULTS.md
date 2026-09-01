# step37 reasoning_effort arms: diagnosis, verdict, and the door that was corrupting every answer

Lane: `lane/step37-reasoning-effort-20260829`, branched from `origin/main` @ `fb6e5abf77`.
Model: Step-3.7-Flash official NVFP4 safetensors. Machine: the dev box (2x RTX PRO 6000, TP
across both cards), GPU lock held per boot-block and released between cells.

## What was asked, and what was actually wrong

The lane was opened as "step37's `reasoning_effort` arms are broken/unverified": on the
production endpoint, `reasoning_effort: "high"` answered `What is 17*23? Reply with the number
only.` with 48 characters of reasoning and then `` ```json[{"bbox_2d": [19, 4...`` — bounding-box
JSON to an arithmetic question. The suspicion on entry was a mismapped effort level injecting a
wrong template segment, possibly a vision-grounding arm leaking in.

**The effort mapping was never the defect.** Two independent checks, both cheap, both first:

1. **Render parity against the vendor jinja.** `chat_template.jinja` in the served artifact is
   byte-identical to the pinned copy (`md5 48e5f5a97fc12290fe8fb5346396ea37`,
   `research/step37-bringup-20260802/raw/chat_template.jinja`). Rendering it under jinja2 3.1.6
   with `trim_blocks`/`lstrip_blocks` (`raw/render_vendor.py`, outputs in `raw/vendor-*.txt`)
   reproduces exactly what `apply_step35_template` emits: `absent` -> no `Reasoning:` line at
   all, `low|medium|high` -> `<|im_start|>system\nReasoning: {level}\n\n<|im_end|>\n` ahead of
   the user turn. There is no vision arm and no `<im_patch>` on any text path; the bbox JSON was
   the model confabulating, not a template segment.
2. **The symptom is not effort-specific.** It reproduces on the DEFAULT path (no
   `reasoning_effort` at all) and through `/v1/completions` fed the vendor-rendered bytes
   directly — same corrupt output, same leading fragment. A defect that survives removing the
   parameter is not in the parameter.

The effort ladder's mapping, refusals and jinja goldens were already tested on main
(`reasoning_effort_maps_to_effort_level_on_step35_class_templates`,
`step35_reasoning_effort_renders_in_the_system_turn`, goldens in
`research/step37-p2-20260806/raw/step35-template-goldens.txt`). Nothing there needed fixing.

## Root cause: `MEMRA_NVFP4_BANK_V2=1`

The corruption is a serving **door**, not a template, and it is one door out of the 24 the live
serving launcher sets. Bisected over 9 boots on one binary (`md5 dd62e8cbb97ff143283279f61ea582e0`),
four real short prompts x 2 greedy reps per arm, all arms sharing the qualified spec-on policy
(`MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1
MEMRA_SERVE_SPEC=1 MEMRA_AFFINITY=1 MEMRA_CTX=262144`):

| arm | doors added to the TP-only baseline | 17*23 | capital | leap year | verdict |
|---|---|---|---|---|---|
| baseline | none (TP essentials only) | **391** | Paris | 366 | CLEAN |
| halfA | 12 doors (OPROJ/ROUTER/ASYNC/MOE_DIRECT/DECODE_V2/QKV_FUSED/BF16_MMV/...) | **391** | Paris | 366 | CLEAN |
| halfB | 11 doors (DCW/RMS_BLOCK/BANK_V2/SIG_EXPF/HEAD_SPLIT/MEMSET/SHADOW/ROPE/SEL_*/FA_COMBINE) | *40* | Paris | 366 | CORRUPT |
| B1 | DCW, RMS_BLOCK, BANK_V2, SIG_EXPF, HEAD_SPLIT | *40* | Paris | 366 | CORRUPT |
| B2 | MEMSET, SHADOW, ROPE, SEL_DOWN8, SEL_MIRROR, FA_COMBINE | **391** | Paris | 366 | CLEAN |
| B1a | DCW, RMS_BLOCK | **391** | Paris | 366 | CLEAN |
| B1b | BANK_V2, SIG_EXPF, HEAD_SPLIT | *40* | *echo* | 366 | CORRUPT |
| **BANK_V2 alone** | `MEMRA_NVFP4_BANK_V2=1` | *no answer* | *echo* | 366 | **CORRUPT** |
| SIG_EXPF alone | `MEMRA_SIG_EXPF_DEV=1` | **391** | Paris | 366 | CLEAN |
| HEAD_SPLIT alone | `MEMRA_HEAD_SPLIT=1` | **391** | Paris | 366 | CLEAN |

`MEMRA_NVFP4_BANK_V2=1` reproduces the corruption **alone**, and reproduces the exact leading
fragment the very first battery of this lane recorded (`Assistance with the number only. Reply
with the number only...`). Every other door of the live serving env, armed together without it,
answers correctly. Arms were reproduced within themselves (r0 == r1 byte-for-byte on every row)
and across boots (`MEMRA_SWA_RING=0` gave byte-identical corrupt output, exonerating the ring).

### The falsified claim

The `MEMRA_NVFP4_BANK_V2` FLAGS row claimed the slot-major bank is a *"Pure storage permutation:
per-slot dp4a and scale order unchanged, outputs BIT-IDENTICAL to v1 (gated against the banked
tape)"*, plus a *"12-boot battery 12/12 at the full serving env"*. **That is false on the current
serving path.** First-token A/B, one binary, one boot recipe, greedy, same env but the door
(`raw/cell11-first-token-ab.txt`):

| prompt tokens | `max_tokens` 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 25, door **ON** | `Ass` | `Assistance` | `Assistance with the` | `Assistance with the number only. Reply` |
| 25, door **OFF** | `Got` | `Got it` | `Got it, let` | `Got it, let's calculate 17` |
| 613, door **ON** | `Got` | `Got it` | `Got it, let` | `Got it, let's tackle this.` |
| 613, door **OFF** | `Got` | `Got it` | `Got it, let` | `Got it, let's start by responding` |

The divergence is at **token 1** on the short prompt — before any decode step, i.e. already in
the prime's logits — and later on the long prompt. The 12-boot battery counted boots and
throughput; it never compared answer text.

### Why qualification missed it, and why the soak's own receipts contained it

The damage is **margin-dependent, not length-dependent**: at 613 prompt tokens the first token
still agrees, so every gate prompt in this family's history (613, 1480, 4k, 30k, 39.5k) sat on
the safe side, and the soak metered request success, TTFT, tok/s and spec engagement — never
answer quality. The pre-deploy soak's own banked 248-token replies carry the corrupt leading
fragments (` Stable.` — which then degenerates into a repetition loop, ` Else:`, ` Some of the
major issues encountered.`) while its 613-token twins are clean and identical across rounds. The
defect was inside the receipts the whole time; nothing in the stack was looking at the text.

This is also the honest reading of the "sampled deep-context degeneration" and "task derails"
that earlier lanes attributed to the model: at least part of that signal is this door.

### What is NOT accused

The host-side permutation itself. `nvfp4_matrix_v2_permute` had **no test at all** while its row
carried a bit-identity claim and the launcher pinned it on; it now has one
(`bank_v2_layout_tests` in `tp.rs`) pinning the documented slot-major mapping — slot `g`'s 16 qs
bytes at `g*16`, its two UE4M3 scale bytes at `nslots*16 + g*2` — and proving each row is a byte
permutation. It passes, which is the point: a host-side permutation test cannot see this bug.
The mismatched **reader** is a v1-vs-v2 layout disagreement somewhere downstream and is **not yet
localized to a kernel**. Several readers are v2-aware (`moe_f16g` direct lane and its workspace
dequant via `dequant_nvfp4v2_f16_kernel`, `qmatvec_nvfp4_dp4a_sel_v2`, the grouped prime's
`bank_qt` selection whose own comment records an earlier "feeding v2 bytes to the v1 kernel was
the garbage-output bug"), so the survivor is a rarer path. Closing that needs a device-side
v1-vs-v2 bank oracle — the named follow-up lane.

## The fix that landed

Fail closed, mirroring the precedent one function above it in the same file
(`dead_prime_kill_switch_refusal`, which refuses `MEMRA_STEP_GEMM_PRIME=0` in serving for the
same family):

- `unqualified_bank_v2_refusal(is_sliding_gated_moe, env)` (worker.rs) refuses to boot a
  step37-class model in the server with `MEMRA_NVFP4_BANK_V2=1`, with a named error that carries
  the receipt and the remedy — including that `MEMRA_SEL_DOWN8` must come off with it, since the
  routed down8 sweep requires the v2 banks. Scoped to the family that has the receipt and to the
  server load path; offline diagnostics (run-gen, engine bins, kernel_check) are unaffected.
- The FLAGS rows for `MEMRA_NVFP4_BANK_V2` and `MEMRA_SEL_DOWN8` now state the refusal, retain
  the falsified bit-identity claim *as* falsified, and name the cost (the +1.0 tok/s the door
  was bought for) and the follow-up lane.
- Unit tests: the refusal decision (`the_unqualified_v2_bank_layout_cannot_boot_a_serving_step37`,
  asserting the remedy text so a future edit cannot drop `MEMRA_SEL_DOWN8` from it) and the
  permutation mapping.

A door whose ON arm serves fluent WRONG answers is worse than one that fails, because every
counter stays green. That is why this is a boot refusal and not a default flip.

### Gates on the landed binary (`md5 31e00ec4678840f37bbe7a5076e5b373`)

- **The refusal fires** (`raw/cell13-refusal-gate.txt`): exit nonzero, `[server] FATAL: worker
  init failed: MEMRA_NVFP4_BANK_V2=1 is refused in serving...`, the message names
  `MEMRA_SEL_DOWN8`, `/health` unanswered (curl `000`), no server process left behind. The
  failure path was EXECUTED, not argued.
- **The corrected env still boots and serves**, and the default path is unchanged: 4/4 real
  prompts correct, and the reasoning/answer text is byte-identical to the pre-change baseline
  arms (`prodlike`, `B1a`) on all four. The change touches no numeric path when the flag is
  unset, so identity is by construction as well as measured.
- Zero `ILLEGAL`, zero `#87`, zero panics in every boot of this lane (14 cells).
- `cargo test -p memra-server --bins` and `-p memra-engine --lib` green, including all 11
  pre-existing reasoning-effort mapping tests.

### One named follow-up, measured not assumed

Both refusals exit **139 (SIGSEGV, core dumped)** rather than the `std::process::exit(1)` the
code asks for: the GPU worker thread is still tearing down its CUDA context when main exits. I
measured the pre-existing `MEMRA_STEP_GEMM_PRIME=0` arm on the same binary to attribute this
rather than assume it — it also exits 139 (`raw/cell14-*` in the lane transcript). The
operational contract still holds (nonzero exit, named FATAL on stderr, no service, no leaked
process), so this is a separate teardown lane, not a blocker: **worker-init FATAL should exit 1
deterministically instead of racing CUDA teardown.**

## The reasoning_effort ladder, measured on the corrected config

Battery (`raw/cell12-effort-ladder.txt`, rows in `cell12-effort-ladder-rows.json`): the full
serving door set MINUS `MEMRA_NVFP4_BANK_V2`/`MEMRA_SEL_DOWN8`, qualified spec-on policy, three
fixed real prompts, n=8 vendor-default sampled (temp 0.5 / top_p 0.9) per level plus one greedy,
`max_tokens` 1024. Depth metric = reasoning characters (this model emits all thinking into
`message.reasoning`).

| prompt | ABSENT | low | medium | high | xhigh |
|---|---|---|---|---|---|
| `17*23?` | **229** | 80 | 83 | 84 | 79 |
| train arrival | **1531** | 272 | **373** | 175 | 190 |
| python one-liner | **2723** | 263 | **992** | 529 | 818 |

(medians of n=8 sampled; greedy rep in the raw file.)

Read honestly, three findings:

1. **Every level is honoured and every level answers sanely.** No level produces malformed
   output on the corrected config; the arms differ in the rendered prompt by exactly the
   `Reasoning: {level}` line, as the jinja specifies.
2. **The default (no `reasoning_effort`) is the DEEPEST arm by a wide margin** — 229 vs ~80, 1531
   vs 175-373, 2723 vs 263-992. Naming any level *constrains* this model relative to its own
   template default. That is a product-relevant surprise: a client sending `high` gets
   **less** reasoning than a client sending nothing.
3. **The ladder is monotone low -> medium and then INVERTS at high**: medium > low on both
   non-trivial prompts (373 vs 272; 992 vs 263), but high < medium on both (175, 529), landing at
   or below `low`. `xhigh` canonicalizes to `high` as designed and tracks it.

**Is the ladder publishable? No — not as a depth ladder.** The engine side is verified and
correct: the mapping is byte-exact against the vendor template, unsupported values are named
400s, and the levels measurably change behaviour. But the vendor's three-level control does not
produce monotone depth on this checkpoint at n=8 per level: `high` does not buy more reasoning
than `medium`, and no level buys more than the default. Publishing "low/medium/high trades speed
for cognitive depth" would be a claim the receipts contradict. What IS publishable, if the
coordinator wants it: all three levels are honoured, refusals are named rather than silently
accepted, and the default path reasons most. A depth claim needs a bigger cell (more prompts,
n>=16, token counts rather than characters) and would still have to explain the inversion, which
is model behaviour and not something memra can render its way out of.

## Refusal arms, verified on the corrected config

All named 400s, never silent accepts (`raw/cell12-effort-ladder.txt`):

| request | result |
|---|---|
| `reasoning_effort: "none"` | 400 `model "step37" cannot disable reasoning: its chat template opens a think tail unconditionally...` |
| `reasoning_effort: "minimal"` | 400, same named refusal |
| `reasoning_effort: "banana"` | 400 `bad reasoning_effort "banana" (none|minimal|low|medium|high; xhigh/max/ultra clamp...)` |
| `reasoning: {enabled: false}` | 400, cannot-disable-reasoning |
| `enable_thinking: false` | 400, cannot-disable-reasoning |

This is correct behaviour and was already correct on entry: the template opens `<think>` with no
`enable_thinking` switch, so an off-request cannot be honoured and is refused by name.

## Action required outside this repo

The serving launcher for this model pins `MEMRA_NVFP4_BANK_V2=1` and `MEMRA_SEL_DOWN8=1`. Both
must come out of the serving env; with this binary a stack that still sets the first one will
refuse to boot rather than serve wrong text. Every answer served under the door — every prompt,
every `reasoning_effort` value, the default path included — carried this corruption, with short
prompts worst.
