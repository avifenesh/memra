# Draft composed cadence + dense model gate

Status: source composition and build checkpoint only. This protocol is pending
root review and a separately scheduled model slot. No combined model result or
serving admission is claimed. Reviewed source provenance is in COMPOSE-PIN.md.

## One binary, two fixed programs

A future standalone `dsv4_compose_cadence_dense_gate` should adapt the reviewed
`dsv4_dense_exact_tail_gate.rs` Arm/state/identity/census driver and the cadence
variant checks in `dsv4_full_token_replay_gate.rs`. Reuse public Dsv4Gpu APIs and
the existing dense selector FFI; do not change kernel math, reviewed helpers,
shared runtime, capture owner, or the legacy sampled CLI. The checkpoint builds
the two existing helpers, not this proposed adapter. Root review must cover the
adapter's source and new binary SHA before a model launch.

One immutable source/binary/model/input tuple, one model load, independently
allocated states with a shared pinned prime snapshot:

| Arm | Full replay | Cadence variants | Dense exact-tail | Sampler / diet | Split-K / GU N32 |
| --- | --- | --- | --- | --- | --- |
| A | ON | OFF, original full forward | OFF | device / ON | OFF / OFF |
| B | ON | ON, ordinary/C4/C4+C128 | ON | device / ON | OFF / OFF |
| eager identity oracle | OFF | OFF | OFF | device / ON | OFF / OFF |

Use `arm_full_token_replay_for_gate` for A and
`arm_full_token_replay_cadence_for_gate` for B. Select dense OFF/ON on the owner
thread after both-rank drain and before each arm's first capture. Keep the
choice fixed for that state's whole captured program, including all three B
forward variants and commit/head. Retained graphs select their captured kernel
functions and are never mutated or recaptured to change semantics. Host selector
changes between distinct states must not change any retained graph. Prime both
scored states from the same OFF prefix; qualification states are separate.

Keep the existing plain TP2 attention/expert-ID EP device-cache topology,
capacity520 (PRIME + OUTPUT + 8, matching both helpers) with replay input position below512, 256 prime and 256 sampled
outputs, source prompt/seed20260907, vendor-default temperature1/top-p1/top-k0,
MEMRA_DSV4_FMAD=0. Pin artifact revision, full weight manifest, tokenizer/config,
source tape, launcher environment, binary, source and per-file provenance.
The CPU build does not load a model or regenerate any artifact.

## Qualification before timing

Reuse the dense helper's changing-input sample loop: eagerly sample OFF and
compare A and B at every one of the 256 output positions. Assert next token,
full finite logits raw bits, both-rank cache/hidden digests, position, refusal
words, RNG/absolute-position freshness and all72 AR epochs/rank. Preserve
first-changing-token checks and final next-draw identity. Run both retained
reset proofs from the same prime snapshot; compare full graph hashes before
and after restoration and enforce stable allocation/capture ownership.

Adapt dense `dense_census` to inspect all retained cadence slots, using the
cadence variant dump/census APIs and the existing per-kernel DOT record parser.
Required actual captured functions, on both ranks:

- A forward slot0: 3240 kernel nodes, 86 AR kernels, 1 embedding, 86 HC posts;
  494 control FP8 and253 control dot nodes, no dense candidates. Slots2/3 absent.
- B forward slots0/2/3: respectively2741/3140/3240 kernel nodes, each preserving
  86 AR kernels, 1 embedding, 86 HC posts. Each has494 candidate FP8 and253
  candidate dot nodes, with zero control twins. No unsupported node types.
- Commit slot1 retains the original sampler/commit topology; rank1 head has
  two candidate dot nodes in B, two control dots in A. Rank0 has no head dots.
- A capture census is[1,1]; B is[3,1]. A row's variant device increments per
  rank must be[256,256,0,0]; B must be[192,256,62,2], slots ordered
  [ordinary/full forward,commit,C4,C4+C128]. Check deltas each row, not only
  cumulative end totals. Host enqueue counts are capture evidence only, never
  replay engagement; expected B first capture is[2964,1520] versus A[0,0].
  Retained rows must have zero new candidate enqueues/captures.

These kernel counts are source-derived admission expectations from the two
reviewed helpers, not measured composition evidence. Refuse timing if they do
not match; inspect the exact graph rather than weakening the census.

Use the union of both helpers' live refusal cases in both A and B: positions
258/layer0 (ordinary),259/layer0 (C4),383/layer21 (C128),511/layer42 (wrap), both
ranks, for16 distinct cases. This includes all12 dense cases plus ordinary
coverage from cadence, without counting duplicated cases twice. Inject only
after successful captures/replays with prior emissions live. Require unchanged
cache/position, no commit, expected refusal words, ordinary retry quarantine,
reset refusal and no retry execution. Check which variant advanced on the
refused forward. Preserve paired drain/fail-stop semantics; no waiver on an
unprovable peer completion. Timing starts only after every assertion passes.

## Bounded score and receipts

One A5/B5/B5/A5 run,20 rows total, all sampled with the same pins. Both scored
states begin fresh and uncaptured after qualification. A's first full-forward
and commit capture and B's first three forward variants plus commit capture
are inside their respective first measured rows (A row0, B row5). Record
first_capture=true once per arm. Do not warm A while timing B's capture, omit
B's extra capture cost, or time only retained rows.

Use the reviewed `sample_plus_forward_envelope`: restore outside timer,
initial carry draw outside timer, all256 sampled forward/commit steps plus the
final next draw inside timer. Preserve per-row eligibility, loop/EOS refusals,
position512, identities, epochs, capture counts, actual per-variant device
counts and DOT function census. Report all20 rows, arithmetic mean and pooled
tokens/total-wall rates per arm, percentage delta, block rates and first-capture
rows/costs. Failure means no accepted rate. No extra sweep or reverse without
root assignment. A repeatable small gain is useful; no arbitrary minimum floor.

Execution is remote-only on the assigned development pair after exact source
review and root slot relay: ROLE, both UUIDs, empty compute census, fd9
nonblocking shared lock, fresh namespace. Lock contention exits75 immediately;
no idle waiter or peer kill. Retain controller PID/start/exit, process census,
source/binary/model/launcher pins, raw rows and terminal lock readback. Release
for the next owner immediately; compact raw DOT banks to representative A/B
pairs and complete hash/size manifests with originals retained remotely.
