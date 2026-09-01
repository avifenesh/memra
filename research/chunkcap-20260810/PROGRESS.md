# cx-chunkcap lane progress

## Contract

- Branch/worktree: `lane/cx-chunkcap` in `/home/avifenesh/projects/wt-cx-chunkcap`
- Base: `fda2d970`
- Rig: box1 cloud pair, two RTX PRO 6000 Server Edition GPUs
- Question: does lifting the Step35 decode chunk cap above 8 raise the c>=16 ceiling, and where does it break?
- Stop condition: if cap 8 is correctness-motivated for this Step35 IQ4_XS + MTP PP2 path, document the receipt and stop without running wider chunks.
- Discipline: bounded lock holds, N=3 interleaved, no origin push, no rustup, no nsys; raw logs and `research/chunkcap-20260810/RESULTS.md` are the deliverable.

## 2026-08-10 — initialization

- Read `research/throughput-20260810/RESULTS.md` before tracing code.
- Baseline receipt: all c=64 rows are ready, but one outer tick executes eight B=8 chunks; grouped-on step p50 grows from 48.30 ms at c=8 to 381.68 ms at c=64 while aggregate output stays nearly flat (128.38 -> 129.70 tok/s).
- The receipt says `MEMRA_DECODE_BATCH_CAP` is clamped to 8 for this model because it does not qualify for an exact-16 tier. Whether that is an active correctness guard or a stale tuning tier remains to be established from code, gates, history, and lane receipts.
- Working tree was clean on `lane/cx-chunkcap`; `~/.lanectl/inbox/cx-chunkcap.md` was absent at initialization.
- No runtime/source change has been made and box1 has not been touched.

## Next bounded block

1. Locate the cap and Step35 batched decode dispatch in source and quote current file:line references here.
2. Find the Step35 B>1-over-PP2 corruption fix, its fail-closed B=1 pin, and any promotion receipts/canaries.
3. Decide whether a cap>8 experiment is correctness-permitted before editing or running the rig.

## 2026-08-10 — source and receipt audit

### Cap mechanism (current source)

The scheduler forms per-model chunks at
`crates/memra-server/src/worker.rs:3688-3694`:

```text
// batched steps in per-model chunks (chunk_cap_for: exact-16 tier models chunk
// at 16, everything else 8; MEMRA_DECODE_BATCH_CAP is the explicit door).
for chunk in group_chunks(&active, &ready, &chunk_caps) {
```

The Step35 clamp is explicit at `crates/memra-server/src/worker.rs:5859-5875`:

```text
// Chunk cap 8: the exactness-tier width (IQ4_XS trunk + 288-expert MoE refuse
// exact16 by predicate — `decode_batch_exact16_ok` requires non-MoE — so 16 is
// structurally out).
...
return cap.clamp(1, 8);
```

The engine contract at `crates/memra-engine/src/decode_batch.rs:10-22` admits B=2..8 as the
per-row bit-identical tier, admits B=9..16 only through `decode_batch_exact16_ok`, and states
that B>16 has no exact kernel class. The predicate documents measured non-exact classes at
`decode_batch.rs:138-149` and rejects every MoE FFN at `decode_batch.rs:228-233`. Both the
unsplit and PP-N bodies enforce the width policy; the PP-N assert is
`decode_batch.rs:706-716`.

The old PP2 geometry hole is not the current reason for cap 8. The stage-scoped Step35 path is
selected instead of the generic path at `decode_batch.rs:786-817` and `decode_batch.rs:844-855`.
Its retained promotion battery covered B=1/2/4/8 with zero differing bits and byte-identical
serving (`research/step35-batch-20260808/PROGRESS.md:171-183`). It did not qualify B>8.

### Correctness history

- `research/step-sku-20260807/PROGRESS.md:108-119` records the original HTTP-200 garbage:
  B>1 over PP2 entered the generic Full arm, and the fail-closed repair pinned Step35 to B=1.
- Commit `c5cd6a35` added the dedicated Step35 batched walk and lifted the pin only to cap 8.
  Its ledger explicitly says `min(MEMRA_DECODE_BATCH_CAP, 8)` because exact16 is structurally
  refused by MoE (`research/step35-batch-20260808/PROGRESS.md:129-141`).
- The exact-16 lane measured why the boundary is a correctness policy rather than a tuning
  constant. `research/batched-tick-inc3-20260801/increments.md:17-25` records non-exact B=16
  step-0 differences of 1.3e-1 to 2.1e-1 without the exact tier and B=32 differences of
  1.3e-1 to 2.3e-1; `:45-55` makes MoE a disqualifier and keeps B>8 refused otherwise.
  These are general decode-tier receipts, not a claim that this Step35 checkpoint was run at
  B>8.

### Stop decision

The cap is correctness-motivated. The exactness guarantee first becomes unavailable at B=9;
the code does not claim that visible Step35 corruption necessarily starts at B=9, but it does
refuse to admit that width without a model-wide exact tier, and this MoE checkpoint cannot pass
the predicate. Lifting the Step35 clamp would turn a deliberately blocked measurement door into
a serving path and would trade the byte/bit-isolation contract for throughput.

Per the lane instruction, the work stops here. No source/runtime cap change was made, no build
was provisioned, no GPU lock was acquired, and no cap-16/32 traffic or QoS run was started.
`research/chunkcap-20260810/RESULTS.md` contains the final keep verdict and the explicitly
unrun A/B matrix.
