# A DFlash request that does not fit with its drafter plane is refused the drafter and served plain (2026-11-12)

Date: 2026-11-12. Issue memra#1173, branch `fix/1173-drafter-kv-charge`, receipts
`research/drafter-admission-20261112/`. Card: one RTX 5090 Laptop GPU (24 GB; enforced power
limit 150 or 160 W, recorded per run and not changed), Qwen3.8-27B NVFP4 + Q5_K.

## Problem

The VRAM gate charges a speculating request its session, its per-session draft state `D` and
the transient reserve: `required = cost + D + reserve`. When that does not fit, a second arm
(the eager arm, `research/step37-vram-admission-20260830/`) reads headroom again against
`cost + reserve` and admits when that fits.

The arm is honest on the MTP route. There `D` is the measured high-water of the draft-graph
capture, and the pre-capture gate (`capture_headroom_refusal`, `MEMRA_SPEC_CAPTURE_GATE`,
default on) refuses a capture before it allocates unless effective free covers `2 x D` plus two
1,536 MiB floors. A request admitted on the eager arm found `free < cost + D + reserve`, so
after its own allocations at most `D + reserve` is left, which is under `2 x D + 3,072 MiB`
whenever `reserve < D + 3,072 MiB`. The capture is refused, the session speculates eagerly,
and the state it was not charged for is never allocated. If memory frees later, a capture runs
only when it would leave `D + 3,072 MiB` free behind it.

On the DFlash route `D` is the drafter KV plane (`DflashDraft::resident_kv_bytes`), which every
drafter session allocates when it is created (`dspark_prime_start`, `dspark_spec_session_new`,
the prefix restore and the cold-drafter conversion). Nothing later can refuse it. The same arm
therefore admitted a request, allocated its plane and charged nothing for it:

```text
[admission] eager-arm admit: model="..." ctx=N draft-state DMB uncharged (capture would be
refused at current headroom; the pre-capture gate enforces it)
```

The plane is about 170 MB for a DFlash2 drafter with a sliding window, constant in the cap. For
a drafter with no sliding layer it is `(cap + block) x 40,960 B`: 1,229 MB at a 30k cap and about
10.7 GB at a 262k cap.

## Chosen

The gate's verdict is two pure functions (`vram_gate_first_read`, `vram_gate_drained`). A
second arm exists only where something enforces it (`draft_state_arm`):

- **MTP route:** unchanged. The eager arm, while the pre-capture gate is armed.
- **DFlash route:** the gate refuses the request its drafter. It sets `Request::drafter_refused`,
  prints `[admission] drafter refused:` with the plane, the available memory, both requirements
  and the arm, puts the request back at the front of the queue and prices it again in the same
  admission pass as a plain request (`admission_request_may_spec` is false: no draft plane in
  its cost). If the plain charge fits it is admitted in that pass, in its queue position. If it
  does not, it defers or is refused like any plain request. The four arms:

  | arm | when | why the drafter goes |
  |---|---|---|
  | `busy-fits` | peers active; the charge without the plane fits now | main admitted here with the plane uncharged |
  | `busy-unfundable` | peers active; the full charge cannot fit even after every peer on the device retires and the evictable entries go | no wait can fund the plane; before, the request waited as a DFlash request with no head-of-line hold |
  | `idle-fits` | idle card, after the fence and trim re-read; the charge without the plane fits | main admitted here with the plane uncharged |
  | `idle-short` | idle card, after the re-read; not even the charge without the plane fits | the plain charge is smaller still (a DFlash walker is charged the whole prompt's stack, a plain prime one take: 948 MB against 623 MB at a 29,110-token prompt), so the plain pass decides; main answered HTTP 400 |

- **An idle card refuses only on its drained read.** A retiring session's frees are
  stream-ordered and may not have landed when the next request is priced. On main that order
  was harmless, since the eager arm admitted and the request kept its drafter. A refusal is for
  the request's life, so an idle card fences, trims and reads again first; when the drained
  card holds the full charge the request keeps its drafter.
- **Beside peers, a request the peers' retirement can fund still waits** as a DFlash request,
  as on main, when the charge without the plane does not fit either.
- **`admit` builds a drafter session only for a request whose plane the gate charged**
  (`Request::drafter_funded`, written by the gate on every pass; `route_ready` reads it, and
  the cold prime, the pool resume, the prefix restore and the cold-drafter conversion all read
  `route_ready`). This closes a second uncharged route: a step-OOM park replay is priced from
  its K, which is 0 for every DFlash session (`[spec-k] ... K=0 source=eligibility-fallback`),
  while `admit`'s DFlash route never read that K, so the replay primed a drafter on a plain
  charge. A replay now runs plain. No token is emitted before a park, so no stream mixes two
  programs.
- The DFlash arm does not read `MEMRA_SPEC_CAPTURE_GATE`: that flag is the MTP capture gate.
- `/metrics` gains `admission_drafter_refusals`, once per refused request.

**The refusal is sticky.** It holds for the request's life, a step-OOM park replay and a choice
group included. The request does not regain its drafter when a peer retires and memory frees: a
request served by two numeric programs is the crossing AGENTS.md forbids ("One numeric program
per request"), and a sampled DFlash session cannot change program mid-request at all. The cost
is decode speed for that request only: on short chat prompts at c = 1 the plain program decodes
at 38.2 tok/s against 70.6 with the DFlash2 drafter on the same binary (Measured). The next
request is judged on the memory it finds.

**What the client sees.** A greedy refused request writes the same text as its drafter run. A
seeded sampled one writes its plain stream, the text the same seed gives with speculation off,
which is not the drafter run's text (in every DFlash gate run the two sampled references differ
at seed 1234 while the two greedy references are equal; on the MTP route both pairs agree). That gap is not new and not closed here:
on the DFlash and DSpark routes a sampled round is still the rejection walk, so a seeded
request's text already depends on whether it speculated, for every request the route declines
under load or for its shape. memra#1154 tracks it (its status comment of 2026-11-10 lists
DFlash2 and DSpark as open after the MTP route was fixed in memra#1172). Until that lands,
memory headroom is one more input that decides which of the two texts a seeded sampled request
writes, and so is a step-OOM park.

**The reclaim ladder still runs against the full charge first.** Before any verdict the gate
evicts unleased prefix entries and parked sessions while the full charge does not fit. A
request that is then refused its drafter has flushed them for memory its plain program does
not all need. That is kept: the ladder is how the gate learns whether the plane can be funded
(a parked session carries no byte count, and an evicted entry's bytes return through the pool
only after the settle), and when it does fund the plane the request keeps its drafter. What it
freed stays free for the plain pass. With a drafter whose plane grows with the cap this
happens on every long request on a tight card.

**The cost model moves after the first plain run.** On a boot that had only speculated, the
first plain run teaches the fixed residual (`fixed residual high-water ... 362MB ... path=plain`),
and every later charge, a speculating one included, rises by it: the same request's
requirement with its plane went from 5,170 to 5,532 MB and without it from 3,973 to 4,335 MB
between the first and second refusal receipts of `gates/agentic-fix` (`gates/dflash2-fix`: 3,638
and 3,469 MB to 4,000 and 3,831 MB; both logs print the 362 MB line). The full-admission window
is that much narrower for every later request on that boot. Not changed here.

## Measured

Receipts: `research/drafter-admission-20261112/` (its README names every binary). "Main" is
`46920874c0` and "this change" is this PR's code on it, unless a heading says otherwise. The
PR head carries the same patches on `cdfdfd6006`, whose `worker.rs` changes (the tick's
phase (a) order, `spec_burst_pre`/`spec_burst_post`) do not touch the gate, `admit` or the
verdict; the CPU checks ran again there. Every gate run is a single run.

### Served gate

`tools/drafter-admission-gate.py` at this PR's head, one run per binary and route. One request
of 29,110 prompt tokens and 48 new tokens (MTP route: 52,579 tokens, ctx 65,536). A ballast
process sets effective free (driver free + pool cached) for each step: two idle steps in the
window (greedy, then sampled at seed 1234), an idle band step under the charge without the
plane, a busy window step and busy band steps beside a streamed peer, then a replay boot whose
request's first step reports a synthetic CUDA OOM.

| run | exit | failed verdicts | `eager-arm admit` lines | refusals (idle-fits / idle-short / busy-fits / busy-unfundable) | lowest effective free in an idle window step, against the reserve |
|---|---|---|---|---|---|
| windowless drafter, main | 1 | V2 V3 V4 V7 V8 | 3 | none | 2,120 MB against 2,156 MB |
| windowless drafter, this change | 0 | none | 0 | 2 / 1 / 1 / 1 | 3,340 MB against 2,156 MB |
| DFlash2 drafter, main | 1 | V2 V3 V4 V7 V8 | 3 | none | 2,129 MB against 1,653 MB |
| DFlash2 drafter, this change | 0 | none | 0 | 2 / 2 / 1 / 1 | 2,323 MB against 1,653 MB |
| MTP route, this change | 0 | none | 2 (the arm that stays) | not applicable | 1,825 MB against 1,611 MB |

- **The plane.** A refused request's peak pool use was 1,229 MB, the plain session, in every
  idle step of both drafters. A full admission peaked at 2,449 MB with the windowless drafter
  (plane 1,197 MB) and 1,423 MB with DFlash2 (plane 169 MB). On main every window step of the
  windowless drafter peaked at 2,449 MB: the plane was allocated with nothing charged for it,
  and the two window requests left 2,120 and 2,122 MB of effective free, under the 2,156 MB
  reserve the gate had promised to keep. No out-of-memory line and no 5xx in any gate run on
  either binary: at this size the reserve absorbed the plane. The plane grows with the cap and
  the reserve does not.
- **Served plain.** Every refused request answered 200 with 48 tokens and no `usage.spec`,
  printed `[dspark] declined: drafter refused at admission`, and moved `/metrics`
  `served_plain` and `admission_drafter_refusals` by one.
- **The bands.** Beside the peer, where not even the charge without the plane fit, main
  answered HTTP 400 at all four busy band steps (two per drafter); this change took
  `busy-unfundable` and served the request plain. On an idle card in the same band main
  answered HTTP 400 (DFlash2, 3,319 MB effective free; the windowless run could not place that
  step, since the driver could not give its ballast); this change took `idle-short` (3,552 and
  3,680 MB with DFlash2, 4,183 MB with the windowless drafter) and served it plain.
- **Replay.** Both binaries priced the replay `path=spec` 1,816 MB on its first attempt and
  `path=plain` 1,491 MB after the park. Main's replay then speculated, a drafter plane on a
  plain charge; this change's ran plain.
- **Text.** An idle refused greedy request wrote the text of its drafter run and of the plain
  reference boot; an idle refused sampled one wrote the plain reference's text.
- **MTP route.** Both window requests were admitted on the eager arm, and each capture was
  refused before it allocated (`effective free 1741MB (driver 672MB + pool-cached 1068MB) <
  required 3146MB (2x appetite 37MB + floor 3072MB); capture skipped pre-attempt`); both
  answered 200 with the reference text. Peak pool use is information only on this route:
  3,143 and 3,301 MB in the window against 3,143 MB for a full admission, while the draft
  state is 39 MB (the eager and captured draft programs' transients differ by more than that).

### Normal serving (before the rebase: main `340a5e5bfa`, this change on it)

DFlash2 drafter, closed-loop clients at c = 1, 2, 4, 8, 6 requests per client, greedy chat,
160 new tokens, streamed; four fresh boots in the order main, change, change, main; enforced
power limit 160 W on the first boot and 150 W on the other three. No request took either arm
or failed, and the four boots wrote the same text per prompt at every c.

| c | main tok/s per boot | this change tok/s per boot |
|---|---|---|
| 1 | 70.23, 70.63 | 70.65, 70.60 |
| 2 | 74.62, 77.75 | 77.82, 77.79 |
| 4 | 116.09, 120.77 | 120.98, 119.13 |
| 8 | 140.65, 146.19 | 148.90, 147.64 |

VRAM defer lines per boot: main 35, 35; this change 1, 35.

### What a refusal costs (before the rebase)

The same cell at c = 1 with `MEMRA_SERVE_SPEC=0` on this change's binary, two boots: 38.23 and
38.16 tok/s (ITL p50 24.4 ms), against 70.65 and 70.60 with the DFlash2 drafter. A refused
request decodes at 54% of its drafter rate on this shape.

### Agentic replay

The 2026-11-05 decision cell's config (`research/c8-workspace-20261108/harness/`): 29 agentic
turns of 1.4k to 17k prompt tokens, DFlash2 drafter, ctx 32,768, 8 sessions, prefix cache
4,000 MB, T = 1 seed 1234, 768 tokens; boots in the order main, change, change, main.

At c = 4 (before the rebase) every boot answered 29 of 29 with the same 10,293 tokens: main
44.56 and 46.37 tok/s, this change 46.34 and 46.90. Every request was priced plain (58
`path=plain` cost lines and none `path=spec` per boot): at c = 4 the wave is above the spec
gate's low mark, so no request carries a draft-state term and neither arm is reachable. 0
refusals, 0 eager-arm lines, 460 to 462 VRAM defer lines per boot on both binaries, 0
out-of-memory lines.

At c = 2 (the rebased pair, 150 W) every boot answered 29 of 29, and every request was priced
as speculating (58 `path=spec` cost lines per boot, each with its plane charged). The card held
the full charge every time: 0 refusals, 0 eager-arm lines and 0 VRAM defer lines on both
binaries, so on this workload neither arm fires at c = 2 either. tok/s in boot order: main
35.27, change 34.41, change 37.28, main 38.02.

Two of the four boots, one per binary, hit a step OOM at the same point of the replay
(`[admit-oom] step OOM parked session back to queue ... 2 active) [dev0 free 36MB pool-res
23328MB ...]`), during the cold prime of a second request whose cost line was `path=spec` with
its plane charged. That OOM is on both binaries with full charges and is not this change's
subject (memra#1255). Its replay is the route this change closes, seen in serving: main priced the replay
`path=plain` (`K=0 source=oom-replay`) and then decoded it with the drafter (`[dspark-acc]
ctx=4842`); this change priced it plain and served it plain (`[dspark] declined: admission
priced this request plain (no draft plane charged, a step-OOM park replay)`). The texts show
the sampled-text gap above: three boots wrote 10,818 tokens, and the change's OOM boot, whose
replay ran the plain stream, wrote 9,765.

On this replay the refusal costs nothing at c = 2 and c = 4: no request was refused. A refused
request pays its own decode rate (the cell above). Whether waiting for memory would serve such
a request faster was not measured: no request of this workload reached either arm, and the
gate's refusals were made with a ballast.

## Rejected

- **Charge the plane in full and keep the request a DFlash request** (the fix sketched in the
  issue). It never undercounts either, and it is a two-line change. But a request whose plane
  does not fit then waits for memory that may never come, and on an idle card it is refused
  with HTTP 400 although the trunk can serve it plain. With a windowless drafter that is every
  long request: the plane is 10.7 GB at a 262k cap, on top of the session. It also printed a
  wrong hint, since the largest-admissible-context estimate has no term for a plane that grows
  with the cap.
- **Wait for memory, then admit with the drafter.** Same wait as above while peers are active,
  for a decode-speed gain the K policy already gives up under load. The MTP arm made the same
  call: serve now on the program that fits.
- **Re-price in place** (compute the plain charge inside the same gate iteration). The gate's
  inputs (prime hold, PP plan, restore plan, reserve) all key on the spec estimate; running the
  pass again for that request reuses every one of them instead of a second copy of the
  arithmetic.
- **Reuse `spec_k_replay = Some(0)`** to mark the request plain. `admit`'s DFlash route does not
  read the K decision, and the replay field also switches off the serve-time K re-decision and
  the GLM5 cold arm.

## Not changed, and checked

- The estimate and the serve-time route can no longer disagree toward an uncharged plane:
  `admit` reads the gate's own answer. They can still disagree the other way (the gate prices
  a request as speculating and `admit` serves it plain: a greedy request with a penalty, a
  constrained one, a pinned K with a peer active). That over-charges.
- A GLM5 model is not in `dspark_drafts`, so the gate's draft-state term for it is the MTP gauge
  or zero and the DFlash arm never applies to it. How that route's drafter memory is charged was
  not examined here; this change does not touch it.
