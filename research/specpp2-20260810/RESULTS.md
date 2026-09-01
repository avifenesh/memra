# spec-on-PP-2: serial verify hold

Date: 2026-08-10

Branch: `lane/cx-specpp2`

Starting tip: `e874528a`

Rig: box1, 2x RTX PRO 6000 Blackwell Server 96 GB

Artifact: Step-3.7-Flash IQ4_XS + external MTP Q8_0

## Verdict

**HOLD. Keep cross-device PP-2 speculative admission disabled.** No measured
configuration beats plain serving at c=1 or c=2, so there is no placement-policy
diff and the promotion exactness battery is not triggered.

The direct answer is:

- Spec loses because each request executes a whole MTP draft plus a whole
  `T=K+1` verify as **stage 0, then stage 1/head**, and the worker finishes one
  speculative session's burst before issuing the next session. Verify consumes
  95.13% of a K=1 round. A second request queues; it does not fill the idle PP
  stage.
- The draft head is already on the correct device: the last/head stage. Moving it
  to stage 0 recreates the measured wrong-head collapse. Replication cannot close
  the gap because draft is only 0.70 ms/round.
- K=1 is the least-bad depth, but still loses 18.81% at c=1 and 42.76% at c=2.
  K=2 and K=3 lose more.
- Local verify reshaping is flat. The schedule that could plausibly recover c=2
  is a **stage-resident, multi-session speculative pipeline plus batched
  speculative prefill**. c=1 has no independent request to occupy the other
  stage; winning there needs multiple speculative rounds in flight (or a
  similarly fundamental schedule), not a placement flag.

The current policy is therefore correct: `LOW=0/HIGH=1` for sharded cross-device
PP-2 (`worker.rs:1255-1257`), and `choose_spec_k` returns K=0 with source
`pp2-placement` for the first projected active request (`worker.rs:1377-1397`).

## Measurement contract

Every new performance cell used the requested serve shape:

```text
MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1
MEMRA_CTX=262144
MEMRA_MOE_GROUPED=1
MEMRA_PREFILL_TICK=2048
```

Model paths were the pinned three-shard Step-3.7-Flash IQ4_XS source plus
`Step3.7-flash-mtp-Q8_0.gguf`. All four blocks used the same remote binary:

```text
sha256 44ba51d82098e88387e22bc4159d1dab8616733c4b1b491ab13a76c3835b7e3d
```

Each block held `/tmp/memra-gpu.lock` once, saw no competing compute process
before or after, and released both cards at 0 MiB. Performance arms were cold
server cells interleaved within one lock hold. Loaded samples were 35-41 C and
2280-2362 MHz for the K sweep, 36-41 C and 2287-2362 MHz for c=2, and 36-40 C
and 2295-2362 MHz for the verify-shape A/B. Every published performance median
below is N=5. Error scans are empty and all requests completed without errors.

The anatomy block deliberately synchronized natural PP boundaries. It is a
timing decomposition, not a throughput cell.

## Round anatomy

Forced K=1, c=1, one 128-token request produced four burst summaries covering
74 rounds. Medians across those four summaries:

| Round phase | Device/path | Median ms/round | Share | Finding |
|---|---|---:|---:|---|
| MTP draft | device 1, primary/head stage | 0.7045 | 3.89% | Too small to explain the loss |
| PP verify | device 0 stage 0 -> device 1 stage 1/head | 17.222 | 95.13% | Dominant serial tax |
| Verify accept | device 1/primary | 0.024 | 0.13% | Negligible |
| Commit/rollback | stage-owned cache + device 1/primary | 0.147 | 0.81% | Negligible |
| Other | host | 0.007 | 0.04% | Negligible |
| **Total** | | **18.1045** | **100%** | |

The 73 steady T=2 verify observations decompose further:

| Verify subphase | Median ms | Notes |
|---|---:|---|
| Reverse safety fence | 0.013 | #87 correctness ordering, not the loss |
| Stage 0 | 8.450 | Full first trunk half |
| Peer TX | 0.013 | `[T,n_embd]` residual |
| Local RX | 0.014 | Copy into stage-1 working buffer |
| Stage 1 + output head | 8.720 | Full second trunk half + lm head |
| End-to-end PP verify | 17.224 | Stages are balanced but serial |

The peer boundary is about 0.16% of verify time. Removing the draft, accept,
rollback, and all other non-verify work perfectly would improve round rate by
only 5.12%; c=1 needs 23.16% more throughput merely to tie plain.

Raw receipt: [`raw/anatomy/`](raw/anatomy/).

## Code anatomy: placement and serialization

The serving worker is created on the last PP device, which is the head stage:

```text
crates/memra-server/src/worker.rs:2341-2375
```

For the requested `MEMRA_PP_DEVICES=0,1`, the observed boot line is
`Engine ready (device=1, MEMRA_FAST=true)`. The external drafter log identifies
`nextn.shared_head_head.weight`; `mtp_head_forward_dev` runs the whole draft on
the engine passed to it and applies that shared head (or the model output head)
there:

```text
crates/memra-engine/src/spec.rs:745-758
crates/memra-engine/src/spec.rs:865-879
```

The PP loader contract also maps MTP/NextN blocks and the output head to the last
stage:

```text
crates/memra-engine/src/pp.rs:1073-1086
```

The verify function documents and constructs one whole `[T,n_embd]` batch per
stage (`T=K+1`):

```text
crates/memra-engine/src/spec.rs:1634-1687
```

Its issued order is explicit: complete stage 0 and TX, then RX and complete the
last stage/head. There is no same-round cross-stage microchunk pipeline:

```text
crates/memra-engine/src/spec.rs:1763-1791
crates/memra-engine/src/spec.rs:1805-1832
```

Finally, serving says that speculative sessions burst solo and loops through
them one by one, while only plain sessions enter batched decode:

```text
crates/memra-server/src/worker.rs:3010-3014
crates/memra-server/src/worker.rs:3388-3415
crates/memra-server/src/worker.rs:6014-6044
```

That outer worker loop is why c=2 remains flat even though the two stage devices
have separate streams and balanced 8.45/8.72 ms work.

## Lever results

### 1. Draft depth, c=1

Four 128-token measured requests followed one warmup per cell. Acceptance below
excludes the warmup; it was identical across all five repeats.

| Arm | Median agg tok/s, N=5 | Delta vs plain | Measured-request acceptance |
|---|---:|---:|---:|
| Plain | **81.188** | reference | n/a |
| K=1 | 65.918 | -18.81% | 72.97% |
| K=2 | 58.357 | -28.12% | 51.59% |
| K=3 | 49.591 | -38.92% | 36.61% |

K=1 is acceptance-healthy enough to test the hypothesis, but target work still
costs more than the accepted-token gain. Every deeper arm lengthens the serial
verify bubble and lowers acceptance.

Raw receipt: [`raw/k-sweep/`](raw/k-sweep/).

### 2. Best depth at c=2

Eight 128-token measured requests followed two warmups per cell.

| Arm | Median agg tok/s, N=5 | Delta vs plain |
|---|---:|---:|
| Plain c=2 | **115.230** | reference |
| K=1 c=2 | 65.953 | -42.76% |

Plain scales 41.93% from c=1 to c=2. K=1 scales only 0.05%. This is the measured
serial-burst queue signature, not an inference from utilization.

Raw receipt: [`raw/c2-k1/`](raw/c2-k1/).

### 3. Draft-head placement

This lever was already resolved on the current train, so rerunning a knowingly
wrong primary placement would not test a new candidate. The committed v0.72
receipt measured q9 on the same PP-2 topology:

| Placement | c | Aggregate tok/s | N | Result |
|---|---:|---:|---:|---|
| Primary pinned to stage 0, head remote | 1 | 17.4 | 1 | Collapse |
| Primary pinned to stage 0, head remote | 2 | 17.5 | 1 | Flat collapse |
| Primary follows head stage | 1 | 111.7 / 112.0 / 111.9 | 3 | Restored |
| Head-affine reversed device order | 1 | 111.0 | 1 | Same fast class |

Source receipt:
[`../v072-fix2-20260808/PROGRESS.md`](../v072-fix2-20260808/PROGRESS.md), with
raw points under [`../v072-fix2-20260808/raw/`](../v072-fix2-20260808/raw/).

Current Step-3.7 anatomy confirms the fixed topology directly: device 1 is the
worker primary/head stage and drafting costs 0.7045 ms/round. Even deleting it
entirely gives only about 4.1% round-rate improvement. Replicating or moving the
head cannot close an 18.8% c=1 deficit; moving it to stage 0 is specifically the
old losing topology. Keep the drafter on stage 1/head, where a future
multi-session pipeline can overlap it with another session's stage-0 verify.

### 4. Verify shape, c=1 K=1

The default T=2 batched linear-attention verify was compared with the existing
`MEMRA_SPEC_M2=0` sequential-column fallback, one variable changed per cell.

| Verify shape | Median agg tok/s, N=5 | Delta vs default |
|---|---:|---:|
| Whole-T batched (default) | **65.955** | reference |
| Sequential columns | 65.905 | -0.08% |

Both remain about 18.8% below plain. Local column shape is flat; it does not
alter the outer stage serialization.

Naive cross-stage token microchunking is also bounded by measured stage times.
For K=1, two T=1 microchunks on two stages have an optimistic no-overhead
pipeline length of:

```text
5.557 + 6.179 + max(5.557, 6.179) = 17.915 ms
```

That is already 4.0% slower than the measured 17.224 ms whole-T verify because
the existing T=2 kernels amortize weight reads and launches inside each stage.

Raw receipt: [`raw/verify-shape/`](raw/verify-shape/).

## Mechanism bill

### c=2: stage-resident multi-session rounds plus batched spec prefill — large

The plausible schedule keeps each whole-T stage batch intact but interleaves
independent sessions:

| Pipeline interval | Stage 0 / device 0 | Stage 1 + head / device 1 |
|---|---|---|
| Fill | idle | draft A |
| Next | verify A | draft B |
| Steady | verify B | verify A, accept/commit, draft A-next |
| Steady | verify A-next | verify B, accept/commit, draft B-next |

At the measured medians, stage 1's steady burden is approximately
`8.720 + 0.7045 + 0.024 + 0.147 + 0.007 = 9.6025 ms` per round, versus
8.450 ms on stage 0. The ideal round interval therefore falls from 18.1045 ms
to about 9.60 ms, a 1.89x upper bound for the round phase.

That alone is not sufficient for this exact 228-prompt/128-output receipt. The
c=2 K=1 median takes 15.526 s and must beat plain's 8.887 s. Applying the
optimistic 9.60 ms interval to the observed 592 measured rounds saves about
5.03 s, while the measured gap is 6.64 s. Approximately 1.61 s still has to
come from setup/prompt work. This estimate mixes intrusive anatomy medians with
uninstrumented end-to-end medians and is a bound, not a projected score.

The implementation therefore needs both:

1. Split `generate_spec_session_*` into resumable draft, stage-0 verify, and
   stage-1 accept/commit phases with per-session round state.
2. Replace the worker's solo-session burst loop with stage-ready queues, using
   independent session caches and the PP runtime's double-buffered boundary
   slots (`pp.rs:800-878`).
3. Move first-turn speculative suffix priming out of `step_session`'s solo burst
   (`worker.rs:6014-6044`) and through the existing batched PP prefill schedule.
4. Preserve #87 reverse publication, stage-owned cache rollback, round-cadence
   SSE, and admission-yield semantics; then run ppspec bit identity, `run-spec`
   K=1..8, spec/plain byte identity, and N=5 c=2 A/B.

This is a scheduler/engine API refactor, not a flag flip: **effort class L**.
It is plausible for c=2, but the margin must be measured after both decode and
prefill composition exist.

### c=1: multiple unverified rounds in flight — research/XL

c=1 has no independent session. The next round's draft depends on the current
verify/accept result, so the c=2 schedule has nothing to place on the idle stage.
Token microchunks are already bounded negative, and perfect removal of all
non-verify work is far short of the required 23.16% throughput gain.

A c=1 win therefore requires a zero-bubble speculative pipeline that allows
future rounds (or branches) to be drafted before the previous target result is
known, with exact rollback/reconciliation, or an equivalently fundamental
model execution change. It touches draft semantics, cache checkpoints, routing,
and acceptance ordering: **effort class research/XL**. It is not justified as a
placement-policy change from this receipt.

## Decision and gates

- Policy diff: **none**.
- Keep c>=4 plain: unchanged.
- Keep c=1 and c=2 plain on sharded cross-device PP-2: confirmed by new N=5
  Step-3.7 receipts.
- Promotion gates (`run-spec` K=1..8 and spec/plain byte identity): **not run**, as
  required by the decision rule, because no candidate beat its plain denominator.
- Probe verification: local `cargo check -p memra-engine` passed; every remote
  harness returned rc=0; all request/error/GPU-state receipts are committed.

This lane stops at the hold receipt. No origin push, merge, tag, or release was
performed.
