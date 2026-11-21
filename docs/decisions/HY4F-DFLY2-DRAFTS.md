# Hy4-flash: DFly2 drafts on the serve route, on by default (2026-11-21)

**Status:** decided with step 10 of the Hy4-flash onboarding: the serve route (`hy4f_serve`) drafts
with the checkpoint's DFly2 drafter by default, 5 tokens a round
(`speculative_config.num_speculative_tokens`), and `MEMRA_HY4F_DRAFT` sets the count (`0`: no
drafter). Scope: Hy4-flash on one GB10.

## Question

The artifact ships a drafter that reads the target's own state: its final hidden row, and the KV and
indexer top-k of five of its layers. Does drafting change what the route serves, and does it pay at
one request on this box?

## Measurement

Identity. Every row of a route call computes on its own, so a verify call's rows are the one-token
decode rows bit for bit. `hy4f-spec gen` measured that: plain and drafted greedy give equal ids and
bit-equal logits on every emitted row, 128 tokens of `chat` and 64 of `long`. With drafts on, the
step 9 served-answer gates pass: served greedy equals `hy4f-gpu --gen`, 23 teacher-forced positions
equal the verified tapes, concurrent requests answer as alone, and seeded sampling answers as
alone. Every answer of that gate equals the same binary's with `MEMRA_HY4F_DRAFT=0`, 51 answers:
greedy, seeded sampled, penalized, stopped, streamed, tool calls
(research/modelplan-onboarding-hy4flash-20261119/, step 10, `dfly/receipts/`).

Speed, one GB10, through the server (`serve/scripts/bench.py`, N = 3 a cell, drafts on and off in
the same window):

| workload | drafts off | drafts on |
|---|---:|---:|
| chat, one request, 36-token prompt, 128 tokens (`ignore_eos`) | 5.95 tok/s | 15.6 tok/s |
| long, one request, 2,104-token prompt, 64 tokens | 5.40 tok/s | 8.2 tok/s |
| chat, four requests at once, aggregate | 5.88 tok/s | 14.9 tok/s |
| long, four requests at once, aggregate | 2.00 tok/s | 2.25 tok/s |
| six real chat turns, greedy, each to its own end, total wall | 169.8 s | 113.8 s |

One GB10, the binaries of `ef45222ac6`, 2026-11-21; decode tokens a second of a request, medians of
N = 3 after a warm-up; the second drafts-on run of the window read 15.7 and 8.2. Each of the six
turns is faster with drafts on, from 46.5 s to 39.5 s on a list (0.22 of its drafts kept) to 36.8 s
to 16.8 s on code (0.62 kept). The `chat` bench row decodes past its answer into a repeated token
and overstates a chat turn; the turns are the fair reading.

## Decision

Drafts are on by default at the artifact's 5. The tokens are the undrafted route's, greedy or
sampled, because a draft is kept only when it equals the route's own choice for its row. Each
session books one round's cache positions, and the drafter's 0.70 GiB of weights and 7.3 MiB of
scratch are resident before the memory book's ceiling is read. A draft round or a verify call's
ring backup that is refused turns drafting off for that request, which is then served by the
undrafted route's calls; a device fault stops the route as it does without drafts. The served context, read from
the free memory at boot, goes from 532,944 positions without the drafter to 509,886 with it in
the same window.

## Rejected

- Drafts off by default: the tokens are the same either way, and one request decodes faster with
  them on every workload measured.
- Sampled drafts at the request's temperature (the checkpoint's `draft_sample_method:
  probabilistic`): keeping a sampled draft needs rejection sampling against the target's
  distribution, another program with its own distribution gate. Drafts are greedy here and the
  route keeps them only on equality, so sampled requests stay exact at a lower acceptance.
- The drafter's band set from the drafter-only backend: over the f64 target's own inputs, the pinned
  program's `bf16x2a` drafter spreads 1.6e-4 and the route's drafter 9.3e-4. That is the same
  fourfold ratio the target class shows on `chat` in step 8, where the gate was banded from the
  `long` request. The class gate is composite, the route's target and drafter together against the
  f64 program, banded from the composite `bf16x2a` backend (0.0398).
