# DeepSeek-V4.1 route: the fixes of PR #1281's two reviews (2026-11-13)

Verdict: every finding of both independent reviews of #1281 is fixed on the route branch, rebased onto main
`c2d2388bd0`, except the re-verification of the support record on one RTX PRO 6000 (M1) and a long-context gate on
the pair, which need wrx cards and are written down below, not run. The cross-stream use-after-free in
`ensure_rows` was real and deterministic under a fault injection: the gate case below fails 12 of its 12 byte
checks on the tree before the fix (every carried row of the grown buffers is the injected fill) and passes all 12
after it, on two captures. The GB10 gate set of the fixed head is in [Gates](#gates).

Hardware: one DGX Spark GB10 (sm_121a, `121a` build, CUDA 13.1), MEMRA_FULL_PREC=1; the CPU checks on an x86_64
build box (`120a`, CUDA 13.1, toolchain 1.97.1). Every time below is a single run, not timed for publication.

## The cross-stream use-after-free (engine review, item 1)

A decode call of at most `SMALL_M` rows runs its keys side on the card's second stream (`Ctx::attn_keys`). When
that call grew the compressed KV or the index keys (a buffer first sized by a longer prefill, on the main stream),
`ensure_rows` copied the old buffer on the second stream and then dropped it on the issuing thread. Event
tracking is off on the ranks and `Buf::p` lets its guard go before the launch, so nothing ordered the drop after
the copy: a buffer under 1 MiB went back to the main stream's free list, a larger one to the device pool in the
main stream's order, and the heads' next allocation on the main stream (`q`, `o`) could take it while the copy
still read it.

Fix (`fix(dsv41): grow a call's compressed KV ...`): `Ctx::grow_keys` grows both buffers on the main stream
before the fork, to the sizes `attn_keys_rest` gives them (the rows a call completes end at `(sp + s) / ratio`), so
the branch never grows anything. The batched step already sized them on the host before any launch
(`plan_keys_step`). A second-stream buffer that leaves the branch's free list (over 1 MiB) is now freed behind an
event of the main stream (`Fence`): ranks that share a card share its pool, and a free on the second stream alone
let the other rank take memory the main stream still read. The keyed table of a decode call comes from the
stream's free list (`Ops::from_bytes`), not a fresh allocation on the second stream. The comment in `tp.rs` that
said every rank buffer lives on one stream now says what orders the two.

### The red arm

`dsv41-gate decode` gains a growth case on every layer that owns its compressed KV and index keys (2, 8, 14, 20):
a prefill of more than 8 rows sizes both for exactly its rows (`rows_hint`), then a one-token call (a branch: one
row, no trace) completes one more compressed row, so both grow inside the call. The same two calls with room for
the row are the reference: the call's output and both buffers' rows must be the same bytes. Both arms run under a
fault injection of the second stream (gate-set statics, no env read): every branch first holds the second stream
20 ms (`SIDE_DELAY_NS`, `v41_sleep_kernel`), and a main-stream buffer is filled with 0xFF on its stream as it drops
(`POISON_FREE`), which is what that stream's next allocation may do at the earliest. A grown buffer copied on the
branch from a buffer the call already let go then reads the fill instead of racing it.

| binary | capture | decode gate | growth checks |
|---|---|---|---|
| before the fix (`0f0abcd330`, the case alone) | straddle126 (prefill 125) | 148 of 160 | 0 of 12: every carried row of both buffers differs (62 of 63 rows at ratio 2, 125 of 126 at ratio 1; the call's new row matches), and the call's output differs on all four layers |
| before the fix | gsm (prefill 41) | 148 of 160 | 0 of 12, the same pattern |
| the head | straddle126 | 160 of 160 | 12 of 12 |
| the head | gsm | 160 of 160 | 12 of 12 |

Builds: the GB10 binaries were made from these commits as they stood before the route's last rebase onto main
`c2d2388bd0` (local build commits `05ba46224b` for `0f0abcd330`, `5e71a5929b` for the head before the control fix
below, `01cf01f2cf` for the pushed head); the rebase brought only main's own engine changes, none in a dsv41 file, so
each binary's dsv41 sources are the named commit's. Their sha256 are in `receipts/build/` and in each run's log.

The `grow/rows` check passes in every run: the prefill's compressed KV holds exactly its rows (62 at ratio 2, 125
at ratio 1) and the call grows it (124, 250). The byte checks print the gate's shared rule text ("byte-identical to
the single-card block_forward"); here the reference is the arm with room. Raw: `receipts/redarm/`.

Without the injection the race needs the main stream to reuse the buffer within the copy's few microseconds, which
no run of the lane's gates happened to show; the case is kept as a regression gate.

## The route flag (engine item 2, server M2)

`route_bad` was set by a gather whose peer never arrived and cleared only when the `Ctx` was made, so one timeout
failed every later step while the route's health kept admitting. It is now two words. The call word is cleared when
a serving step opens (`Runner::begin_step`; the first push cleared it on the read that reported it, which the fix
check below changed), so a flag fails that step's calls only; it carries the MoE plan's out-of-range count (bit 0),
which `Ctx::moe` read only under a trace before. The transport word keeps the gather timeout: the gathers' sequence words can no longer be trusted, so the
group is lost (`Group::route_verdict`) and every later forward, batched step and draft refuses (`Group::live`).

The server books a step fault as failed (`RouteRun::fail`, `RouteHealth::note_request_fault`; they were booked as
cancellations). When the route stops, with its reason in the health record and the waiting requests answered: only
on a lost transport or a failed probe of the ranks (the fix check below; the first push also stopped it on two
faulting steps in a row).

## The other findings

- Router NaN re-pick: a row with fewer orderable scores than top-k names no expert (-1, weight 0) in the slots past
  them; it picked an already chosen expert, and the plan's per-token order then left a slot unwritten. The plan
  counts and skips the -1 and sets the call flag. `dsv41-gate kernels` gains a NaN router row and a plan with an
  out-of-range id.
- `memra_v41_candidates` past 1.03M positions: the kernel has the top-k's BIG instance, which raises its own
  shared-memory limit, for emit tables past 48 KB with the static shared memory (more than 128,000 blocks).
- `copy_2d` checks both extents and a row against its pitch; the top-k and candidate launchers refuse k < 0.
- 16-byte alignment: `memra_v41_gemm_ffma` takes its 16-byte row forms only when both bases are aligned;
  `memra_v41_gemm_f8` and the tile list refuse an unaligned base, which every body would misread.
- `__restrict__` off the aliased pointers of `act_quant` (input) and `engram_gate` (the stream, written in place).
- `v41_wide` is an atomic: a gate or bench thread sets it while ranks launch.
- `ExpertCache::resolve` maps a slot only after its copy is queued, and the stream waits for the copies queued
  before a failure.
- The push-gather consumer fences after its spin before it reads the payload.
- The staged N-rank one-token MoE was unreachable (N ranks with the experts on the card always split them): it goes,
  with `memra_v41_moe_owner` and its KERNELS.md row. The `tp` module doc no longer promises a cross-rank routing
  check.
- Draft door: a request whose draft block would pass the runner's context decodes plainly (`len + block <=
  max_pos`) instead of failing every request of the step.
- Graph door: `Group::can_capture` requires no expert cache and no route log, so the served layout keeps every span
  launch by launch, as `docs/FLAGS.md` says.
- A `make_room` error with nothing running answers the head request with the engine's error (it was a wait that
  spun the serving thread at 100% CPU).
- A session's state is allocated at admission (`Runner::seq_alloc`, the bytes the forwards would allocate at first
  use), so the expert cache's regrowth between steps cannot take what its first emissions need.
- `catch_unwind` covers the admission door, `choose`, `commit`, `draft_commit`, the finishes and `grow_caches`.
- `rayon` is gone from memra-engine (the issue threads are `tp::Crew`); the Cargo.lock allowlist entry is re-pinned
  and the `MEMRA_DSV41_TP_ISSUE` row names the threads.
- Nits: prompts render on a thread of their own; the DSv4 and V4.1 shutdown joins share one deadline; the V4.1 close
  stop is the tool parser's tag; a doc comment sits on its function again.
- `ignore_eos: true` on a non-V4.1 model answers 400 (it was ignored): deliberate, under the clean-400 policy.

## Gates

The fixed head's dsv41 gate set on one GB10 (`dsv41-gate` with the pushed head's dsv41 sources, `da710634...`; see
Builds above), MEMRA_FULL_PREC=1, `receipts/gates-gb10/`:

| gate | checks | note |
|---|---|---|
| kernels (`fixtures-gpu/rand`) | 151 of 151 | 149 on the k3 tree plus the NaN router row and the out-of-range plan id |
| layer (needle, layers 1, 2, 14, 20, 21, 24, prefill + 2 decode steps, against the vendor floor) | 161 of 161 | the 111 saved tensors byte-identical to the k3 set (`perf/receipts/bytecheck-k3-gb10/save-k3.sha256`): the fixes move no byte |
| tp at 1 rank, default / peer-pull / host-canonical | 242, 242, 242 of 242 | |
| tp at 0,0 (two ranks on one card): default (device-push) / peer-pull / host-canonical | 480 of 480, 478 of 478, 478 of 478 | the push-gather stress passes 20,000 gathers; its control (`word_first`) catches 1,066,896 words |
| tp at 0,0,0,0: default / peer-pull / host-canonical | 916 of 916, 914 of 914, 914 of 914 | |
| decode straddle126 / gsm (layers 0, 1, 2, 3, 8, 14, 20, 21, 24) | 256 of 256, 256 of 256 | 240 as on the k3 tree, plus the growth case's 16 (all four layers grow and match) |
| decode capital (layers 1, 2, 3, 14, 20, 21, 24) | 72 of 72 | 60 plus the growth case's 12 (layers 2, 14, 20) |
| tp with `MEMRA_DSV41_DECODE_GRAPH=1` at 1 rank / at 0,0 (device-push) | 555 of 555, 1,105 of 1,105 | every decode step repeated through the captured spans: 6 captures, 42 replays, 12 self-checked, 0 refused at 0,0 |

The first run of this set, on the head before the control fix (`8e9e2c77...` is the red binary, `458a28da...` this
one), is kept in
`receipts/first-run-5e71a5929b/`: kernels 151 of 151, layer 161 of 161 with the same saved tensors, tp at 1 rank
242 of 242, at 0,0,0,0 916 of 916, and at 0,0 479 of 480: with the consumer's new fence after its spin, the
push-gather control (the producer stores the word before the data) stopped catching anything (0 words differ in
2,000 gathers). The control now holds its data back 50 us after the word (`v41_gather_kernel`, gate-only branch), and
catches the inverted order again.

CPU, at the pushed head's crates (`ab3f3ed3bb`, the build box): the workspace's release build with all targets,
`cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` pass; memra-engine's dsv41 lib tests
17 of 17, memra-server's lib tests 1,245 passed (14 ignored). The CI `gates` job, every `run:` step, at the receipts commit
before this text was final: 47 steps, 45 pass; the two that need bubblewrap, which the build box lacks and the hosted
runner installs ("Compiler input view and filesystem boundary", "Local development selection and immutable reuse
controls"), pass on a machine that has it, on this commit's tree less these two logs and this paragraph
(`receipts/cpu/`).

## The fix check (2026-11-13)

The fix check of both reviews at `f3d7086a57` approved the engine scope with nits and asked for one change in the
shared scope, made in `04873a7ba9` (engine nits) and `bef6ed28f7` (the route's fault rule):

- **Route death.** The first push stopped the route after any two faulting steps in a row, so two prompts that drive
  a router row non-finite, or two allocation failures under a co-tenant, took it down for good. Now a step fault
  fails that step's requests, then every rank is probed (`Runner::probe`: both streams drain and the context holds
  no recorded error), and the route stops only when the engine lost its transport or the probe fails. Call-flag
  faults, refused allocations and host panics fail their step and nothing else. Tests: two consecutive call-flag
  faults leave the route serving; a failed probe or a lost transport stops it.
- **The call flag's lifetime.** Cleared by the read that reported it, a call fault in an all-greedy batched step was
  read (and cleared) by the device pick, which refused; the logits read that `pick_rows` falls back to then found it
  clear and served the step. The flag is now cleared when a step opens (`Runner::begin_step`), so every read in the
  step reports it and the step fails.
- `MEMRA_DSV41_CALL_FAULT=<steps>`, the fault door of the serve smoke below: the listed steps run with the call flag
  set on every rank (`Runner::inject_call_fault`).
- Engine nits: a second stream's free list hands a buffer it cannot keep back to be freed behind the fence (it was
  freed on the second stream alone); `memra_v41_gemm_bf16`, `memra_v41_swiglu_tiles` and `memra_v41_sparse_attn(_keys)`
  refuse an unaligned base; Engram `Stage::load` checks the table's dtypes and scale shape; a forward checks a call's
  tokens (Engram hashes, embedding rows) before the sequence takes them (`EngramConsts::hash_after`, unit-tested
  against `hash_span` at every split); a leftover doc line on `Group::can_capture`.

Evidence at `bef6ed28f7` (`receipts/fix-check/`):

- **Serve smoke on GB10** (`memra-server` built at the head, `e8342cbd...`): the route on a 2-layer trunk
  (`MEMRA_DSV41_TRUNK_LAYERS=2`, a rehearsal of the serving loop, not the checkpoint's answers), MEMRA_CTX=8192,
  `MEMRA_DSV41_CALL_FAULT=1,2`, three chat requests one after another. Request A's decode step (1) and request B's
  prefill step (2) fail with the route flag (HTTP 500, `engine_error`), two call-flag faults in a row; after each the
  log reads `the ranks probe clean after the fault: serving on`. Request C is served (HTTP 200, 8 tokens), and the
  route's health record reads phase idle, 3 requests, 2 request faults, no dead reason. A first try failed before
  readiness because the page cache of the checkpoint left 1.3 GiB the card could count as free (GB10 memory is the
  host's); the second try drops the checkpoint's cached pages first (`smoke-try1/` keeps the first).
- **GB10 gates** (`dsv41-gate` at the head, `2d12d171...`): kernels 151 of 151, layer 161 of 161 with the 111 saved
  tensors byte-identical to the k3 set (the fixes move no byte), tp at 0,0 default (device-push) 480 of 480.
- **CPU** (the build box, 120a): the workspace's release build with all targets, `cargo clippy --workspace
  --all-targets -- -D warnings` and fmt pass at the head, and clippy over memra-engine and memra-server passes at
  `04873a7ba9` alone; memra-engine's dsv41 lib tests 18 of 18 (the new `hash_after` test), memra-server's lib tests
  1,246 passed, 14 ignored (the new route-death and fault-door tests).

## Not run here

- M1, the support record on one RTX PRO 6000: it pins `87bbe3bfc8`, before batched decode, device argmax and k3's
  kernels. It needs `memra model verify serve --against deepseek_v41` (and CheckpointParity on the residency) on one
  PRO 6000 at the landing head, then the record's commit, evidence and notes re-pinned; it runs in a wrx window.
- A long-context gate on the pair: every two-card long-context result ran the peer-pull program (`tp/README.md`
  carries the banner and the command, not run).
