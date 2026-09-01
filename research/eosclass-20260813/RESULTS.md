# Early-EOS numeric class — results

Date: 2026-08-13

## Verdict

The dense-Q27 11-token EOS and the earlier Q35-MoE 15/17/25-token EOS are the same served
numeric-program transition, not corrupted prefix-cache state and not an HTTP/accounting stop.
A request could begin on the eager B=1 fusion program (or eager-equivalent GraphSession) and move
to the generic batched program when peers arrived. Those programs deliberately use different
floating-point composition. The transition changes greedy tokens; on the frozen Q27 prompt it can
make the real EOS id `248046` win after 11 generated tokens.

The class-level repair is to use the generic batched program at B=1 and B>=2 by default. Both
solo-only programs now require exact explicit `=1` opt-in and are fixed-solo measurement doors.
This is a global serving policy, not another model/trigger fence. Q35-MoE's existing eager-path
exclusion remains as defense in depth.

The deterministic local Q27 regression gate passes the repaired binary. Unit/build checks pass.
The diagnostics-only EOS-logit canary confirms that the terminal EOS is genuinely rank 1 in full
host logits, rather than a stale device-sampler read. A current-tip grouped-Q35 A/B also establishes
that grouped dispatch retains its separate 25-token correctness failure and remains fenced. The
cleaned two-model local GPU battery passes end to end. No box1/Vast pre-release battery has run in
this lane; the orchestrator must allocate the global `/tmp/memra-gpu.lock` slot before shipping.

## Deterministic reproduction and repair proof

The frozen harness serially seeds four distinct cache namespaces with the same 4,860 prompt-token
sequence, runs a solo restored-hit control, then starts one restored target before three
already-restored peers. The only cell variable is peer-arrival delay.

| arm | target cells | restored post-seed hits | target hashes | early EOS | result |
|---|---:|---:|---:|---:|---|
| pre-fix default, delays 0--600 ms / 25 ms | 25 | 101/101 | 5 | 2 | exact historical 11-token EOS at 50 and 225 ms |
| repaired default, same delay grid | 25 | 101/101 | 1 | 0 | all targets equal the 60-token solo control |

Both runs admitted/completed 105/105 requests with zero cache evictions, admission defers, OOM
parks, or protocol failures. The pre-fix target rows at 50 and 225 ms are HTTP 200,
`finish_reason=stop`, 11 completion tokens, and exact SHA-256
`ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73` — the same bytes as the
independent cachesize failures. See
[`pre-widthflip-default-d25/client.jsonl`](raw/pre-widthflip-default-d25/client.jsonl) lines 15,
43, and 107.

The repaired run's 25 target rows all produce 60 tokens and the sole solo/target hash
`5790654979cb98bfacf6d3593b6a5d3def7a5f4bd2a1b8b65e4a6fabe1a72f66`. Its counter and per-delay
receipt is [`post-widthflip-default-d25/client.jsonl`](raw/post-widthflip-default-d25/client.jsonl)
line 107; the human-readable check is
[`VALIDATION.md`](raw/post-widthflip-default-d25/VALIDATION.md).

A same-binary, trace-free N=5 control independently changes only `MEMRA_SERVE_B1FAST=0`. The
pre-fix default yields two restored-target hashes; B1-off yields one hash across all five restored
hits. See [`pre-b1-default-r5b/VALIDATION.md`](raw/pre-b1-default-r5b/VALIDATION.md) and
[`post-b1off-r5/VALIDATION.md`](raw/post-b1off-r5/VALIDATION.md).

## Quoted mechanism

The decisive old dispatch predicates in `decode_step_batch` were:

```rust
if b_n == 1
    && Self::b1_fast_on()
```

After the architecture/config eligibility guards, that branch returns exactly:

```rust
return self.decode_step_b1_fast(e, tokens[0], caches, samp, masks, lean);
```

The eager trunk folds add/norm/quantize and SwiGLU/gate-up operations differently from the generic
body. The source itself records that the programs are not bit-identical and that the different FP
composition cannot be a load-changing default; see
[`decode_batch.rs`](../../crates/memra-engine/src/decode_batch.rs) lines 256--268 and 576--605.
GraphSession has the same numerical class and must degrade when concurrency arrives; see
[`worker.rs`](../../crates/memra-server/src/worker.rs) lines 106--115 and 4110--4114.

Current NVIDIA documentation independently confirms only the general premise: finite-precision
operations are non-associative and changing operation/FMA composition can change rounded results
([CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html),
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)).
That premise is not the attribution: the controlled peer-arrival reproduction and same-binary B1-off
control above establish this repository-specific mechanism.

## Why this is not restored-state corruption

The diagnostics run perturbed timing and did not reproduce EOS, so it is not counted as another
failure trial. It still checked the restore boundary exhaustively:

- all 53 restores matched their serial seed in token count/hash, position, and persisted boundary
  logits;
- all 48 recurrent layers matched in logical conv/SSM hashes;
- all 16 populated KV layers matched in logical K/V hashes and lengths; and
- the only aggregate KV digest difference was empty layer 64: a one-byte allocation sentinel in
  the source snapshot versus zero logical bytes after restore, with `len=0` on both sides.

Every target cell then sampled the persisted boundary, ran one eager B=1-produced tick, and ran 58
generic B=4-produced ticks. All targets moved to the same loaded hash. This localizes the differing
output to the program transition after a correct restore and falsifies the steering hypothesis that
the Q27 snapshot omitted recurrent/conv/KV/position state.

The sampler question is closed by a second diagnostics-only canary. It explicitly restored the
retired eager B1 path, disabled device sampling, and swept 37 later peer-arrival delays. Four target
cells (450, 600, 625, and 675 ms) reproduced the exact historical 11-token hash. In all four, the
terminal token at generated index 10 was EOS id `248046`, rank 1/top 1 in the complete 248,320-value
host logit vector. Its top-1 margin was 1.1117076874 in the 450 ms cell and 0.4370250702 in the other
three. Each target had crossed from eager B1 to a batched width before EOS: B1 -> B2 -> B4 at 450 ms,
and B1 -> B4 at 600/625/675 ms. The verifier joined all 8,748 emitted tokens to 8,748 sampler receipts
with no errors; see
[`trace-verification.json`](raw/canary-widthflip-trace-eager-d300-1200/trace-verification.json) and
[`VALIDATION.md`](raw/canary-widthflip-trace-eager-d300-1200/VALIDATION.md).

## Scope across the six reported triggers

- **Dense Q27 full-prefix restored hits:** closed by the one-program default; deterministic pre/post
  gate above.
- **Q35-MoE mixed serving:** prior independent evidence is decisive. With the transition live, EOS
  id `248046` appeared at lengths 15/17/25. Changing only `MEMRA_SERVE_B1FAST=0` made B=1 and B=2
  use the generic trunk and passed 5/5 cells: 100/100 mixed requests plus 40/40 serial seeds, one
  token-id hash and no early EOS (`research/q35bug-20260812/RESULTS.md` lines 127--145).
- **Step35 load-history drift:** the same eager/batched transition was already isolated by its
  architecture-specific exclusion; the global default removes the class rather than extending that
  exception.
- **Grouped-Q35 25/60 failure:** separate and still live. On the same repaired binary, changing only
  `MEMRA_MOE_GROUPED` made grouped ON fail all 8 serial seeds and all 20 mixed-c4 requests at exactly
  25 tokens with one hash; grouped OFF passed 8/8 and 20/20 at 60 tokens with one hash. Both arms had
  zero defers, OOM parks, carried-prime violations, or error signatures. Grouped therefore remains
  off for correctness independently of its target-PRO resident-KAT transfer result of -67.2% (N=5,
  0/5 wins). See [`post-q35-grouped-ab/VALIDATION.md`](raw/post-q35-grouped-ab/VALIDATION.md).
- **Routed-MoE carried prime and qwen3next alias:** separate priming/program-shape class. It remains
  fenced; this lane neither changes nor requalifies carried continuation priming.
- **Partial LCP restore:** separate unresolved class. Commit `6249b0096` proves source-to-destination
  transferred bytes match at splits 64/512/2048/4374, but candidate output differs from genuinely
  cold at 512 and 2048. The candidate also spent roughly 22--34 seconds suffix-priming those failing
  splits versus about 0.95 seconds for the cold reference. This repair changes decode selection only
  and does not unfence partial restore. Its request-3 whole-entry hit is a harness expectation issue:
  request 2 has already computed and published the full 4,860-token entry in that namespace, so a
  subsequent request 3 correctly finds the longer entry.

The evidence therefore rejects one universal “prefix cache is corrupt” explanation. At least two
classes remain: the now-closed live decode-program transition, and distinct prime/partial-restore
paths that retain their existing gates.

## Fix and gates

- `MEMRA_SERVE_B1FAST` is OFF unless its value is exactly `1`; unset, `0`, `true`, and other values
  stay on the generic program. Pure parser tests pin the contract.
- `MEMRA_SERVE_GS` follows the same exact opt-in rule; default serving cannot silently enter an
  eager-equivalent graph and later demote on peer arrival.
- `decode-batch-gate` config mode now treats B=1 generic versus B=N as the live bit-strength
  contract. Strict and PP gates explicitly opt into eager to retain its separate identity checks.
- `tools/local-ci.sh` asserts the default global/effective B1 policy is OFF; H100 strict validation
  opts into eager explicitly.
- The lane-only restore/logit tracing used to settle the state-corruption and sampler hypotheses was
  removed after its raw receipts were sealed. The clean shipping candidate contains only the two
  exact opt-in policy seams and their tests.

Completed checks:

| check | result |
|---|---|
| cleaned repaired `memra-server` release builds | PASS; pre-battery SHA-256 `06b264df7ee7c1e4b1982508f573c7ef299d4ed95bc98efc2a4d3e6c322527d9`; mandatory smoke rebuild/executed SHA-256 `e63f9fad6553820a7944687dcf1a8a45326ece039f3384536964b6c560e3594f` |
| repaired release gate-binary build | PASS: `decode-batch-gate`, `kernel-check`, `run-gen`, `run-spec` |
| engine B1 policy unit test | PASS |
| server GraphSession policy unit test | PASS |
| cleaned-source `DOCS_RS=1 cargo test -p memra-engine --lib` | PASS: 83 passed, 1 GPU-only ignored |
| cleaned-source `DOCS_RS=1 cargo test -p memra-server` | PASS: 221 passed |
| generated perf surfaces | `python3 tools/update-perf-board.py --check` PASS; no board source or marker block changed |
| deterministic Q27 post-fix width gate | PASS: 25/25 targets full/stable, 101/101 restored hits |
| host-logit EOS canary | PASS as diagnostic reproduction: 4 exact 11-token failures; EOS 248046 rank 1/top 1 in full host logits |
| repaired grouped-Q35 ON/OFF cell | discriminating A/B PASS; grouped ON 0/8 seeds + 0/20 mixed, grouped OFF 8/8 + 20/20; grouped remains NO-GO |
| cleaned local kernel/run-gen/run-spec/serve-smoke battery | PASS; all 11 stage exits and aggregate exit zero |
| Vast 2x RTX PRO 6000 pre-release battery | **NOT RUN — requires orchestrated global slot** |

The final local battery is sealed under
[`raw/post-local-gates-clean/`](raw/post-local-gates-clean/VALIDATION.md). Q27/Q35 kernel checks are
`ALL GREEN` (107/113 cells); both default config gates report B1 fast globally and effectively OFF;
both `run-gen` arms report the two required argmax matches; and both `run-spec` arms pass all eight
K=1--8 cells. The integrated serving battery reports zero failures. Its frozen Q35 mixed-c4 arm
completed 20/20 requests at 60 tokens (18 full-prefix hits, 2 cold misses) with zero defers,
evictions, OOM parks, golden mismatches, or integrity failures, while carried prime remained gated.
The failure scan contains only a configured-Xid banner and passing test-description text, not an
observed runtime fault.

The smoke script deliberately rebuilds before serving. Native clean rebuilds in these receipts do
not have stable ELF hashes, but the executed smoke binary embeds the same cleanup source fingerprint
`memra-7d41113dd3f4`; each phase's exact SHA is retained rather than treating one build hash as a
stable source identifier.

The broader `DOCS_RS=1 cargo test -p memra-engine -p memra-server` command is not a failed test
battery: Cargo attempted to link GPU gate binaries without CUDA objects and stopped before tests on
captured undefined CUDA FFI symbols. The supported library/server invocations above pass; the raw
logs and exclusion are sealed under [`raw/cargo-test/`](raw/cargo-test/).

## Performance and shipping boundary

The correctness cost is the inverse of the historical fixed-solo eager gain on the 82-SM 5090:
about 8.33% q9 and 5.19% q27 decode-only at c=1 (N=5); c>1 was flat except when a batch drains to
one. Those are historical paired measurements, not new board numbers. No generated board source or
published number was changed.

This lane does not unfence grouped prefill, routed-MoE carried priming, or partial prefix restore.
It does not merge, tag, push, edit the perf board, or touch live serving. The orchestrator owns final
target-PRO verification and shipment.
