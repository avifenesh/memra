# WP-B day 43: O13, the spec pool's exact-extension miss after an overshooting final burst

OWED.md O13. DAY41 2.1 and 2.2 read the default spec route resuming 34 of 60 exact-extension turns on the 27B (R2 FAIL
on `keep`, both sittings, both orders). Placed in the code: `SpecSession::committed` keeps the final round's accepted
drafts past the request budget ("INCLUDING overshoot: spec commits accepted drafts past max_new"), so the parked
session is longer than the public stream the client sends back, and the exact probe
(`prompt.starts_with(&e.sess.committed)`) misses whenever the last round overshot. The DSpark engine already clamps its
commits at the budget. This day builds the MTP twin behind a default-OFF door and measures what it recovers. Every cell
is `executed-not-qualified`; no default moves.

## 1. Pre-registration

Committed and pushed before any day-43 code and before any day-43 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 The arm (`MEMRA_SPEC_BUDGET_CLAMP`, default unset; decide-by 2026-10-10)

Unset: today's program, byte for byte. `1`, on the Qwen MTP session route (`step_spec`'s qwen arm, `SpecSession`):

- **The worker** sets `SpecSession::budget_room = Some(request_room)` before each burst (the request's remaining public
  budget, `s.budget - s.generated.len()`, not the burst's cadence target, which may be 1 under prime fairness). Unset,
  the field stays `None`.
- **The engine**, in session mode, greedy, unconstrained, after the accept decision of a round (after the grammar
  truncation, which it never meets because it requires no constraint): with `room = budget_room - out.len()`, when
  `n_acc + 1 > room`, `room >= 1` and `base + room - 1 >= 1`, it truncates the round at slot `na = room - 1` exactly
  as the grammar truncation does: `n_acc = na`, `bonus = draft[na]` (the verify's own argmax at that column, because
  the draft there was accepted). The round then commits through the ordinary partial-accept path (KV truncated to
  `pos + base + na`, the recurrent state rebuilt from the round's `VerifyCkpt`, the bonus pending), which is the path
  every rejection takes. The public stream is the same tokens; the session's `committed` after the park's pending flush
  is exactly the prompt plus the public stream.
- **Receipt:** one line per firing, `[spec] budget clamp: round truncated at <na> of <n_acc> accepted (<room> of the
  request's budget left; model <m>)`.
- **Outside the arm, stated:** sampled spec (the rejection sampler's Philox counters would differ from today's for the
  truncated columns), constrained requests, the round-stream arm (`MEMRA_SPEC_STREAM=1`, experimental; the clamp does
  not apply there), a first round with no pending token and `room == 1` (`base + room - 1 == 0`), and the Gemma, GLM,
  DSpark (already clamped) and step35 tensor-parallel spec engines.

### 1.2 One numeric program per request

Within the request: the drafts, the verify batch (its tokens and width) and the accept decision are unchanged; the
clamp only declines to commit accepted columns past the budget, which were never public. The committed rows are the
same verify columns a rejection at that slot keeps (the partial-accept path), so the parked state is a state today's
program already produces on every rejected round. Across requests: a later turn that resumes the clamped session
resumes on the verify-produced rows through the pool's exact probe, the same program `keep` resumes on today whenever
the last round did not overshoot; its residual against a cold prime is the near-tie class DAY41 prices (O11), not a
new one.

### 1.3 CPU before the cards

A census test: the door is read only where the worker sets `budget_room` on the qwen spec arm; the engine reads the
field only in the session-mode greedy unconstrained accept path; unset leaves `budget_room` `None`. A pure-function
test of the truncation rule (`room`, `n_acc`, `base` over the edge cases of 1.1).

### 1.4 The cell (`day41-client.py` RX shape, unchanged)

The spec route only (the plain route does not overshoot), `MEMRA_PREFIX_CACHE_MB=0`, one binary, arms `unset` and
`clamp` (`MEMRA_SPEC_BUDGET_CLAMP=1`), orders O1 (unset then clamp) and O2 (clamp then unset), plus `offprev` (the tip
with the door's commit reverted, spec route, RX): 5 boots per card. Lengths 6,144 and 30,720 on both cards plus 122,880
on the target card; G in {32, 256}; N = 5 conversations per (L, G). Runner `day43-run.sh` (the day-41 runner with the
door as the arm), reader `day43-read.py`.

### 1.5 Clauses

- **C1 exactness within a request.** Every turn-1 row and every cold-twin row has the same completion digest on both
  arms in each order (fresh prompts: no resume can differ).
- **C2 the clamp fired and the parks match.** On `clamp`, at least one `[spec] budget clamp:` line per boot, and every
  resumed turn's `cached_tokens` equals the previous turn's prompt plus its completion tokens.
- **C3 resumes.** On `clamp`, at least 80% of RX turns 2 and 3 resume (`cached_tokens > 0`).
- **C4 health.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no 503, no crash line on any boot.
- **C5 door OFF.** Every `unset` RX row equals `offprev`'s.

Readings, no bound: resumed fraction per arm; resumed-turn TTFT p50 and p95 per arm (turns 2 and 3); resumed-against-cold
flips per arm (the near-tie residual, reading only); generated tokens over boot wall time; clamp firings per boot.

### 1.6 Price

Code: about 0.5 agent-day. Cells: the 5090 about 1.5 h, the target card about 3 h (5 boots, the 122,880-token cold twins
dominate).

## 2. Results

Written after the runs. Section 1 is unchanged.
### 2.1 The target card (the eighth sitting, one RTX PRO 6000 Blackwell Workstation Edition, 2026-09-26 05:42 to 09:29Z)

Binaries built on the box from `87d9e00d1`: `tip` sha256 `8aa9fcd6...77a5621d`, `offprev` (the tip with
`day43-nodoor.patch`) `be6dca87...ccb0e24`; receipts at `pro-single-day43/box/` (134 files, the box manifest checked,
binaries by hash only). Verbatim:

```
DAY43 C4 card=pro6000 boot=offprev oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY43 C4 card=pro6000 boot=rx-spec-O1-clamp oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY43 C4 card=pro6000 boot=rx-spec-O1-unset oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY43 C4 card=pro6000 boot=rx-spec-O2-clamp oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY43 C4 card=pro6000 boot=rx-spec-O2-unset oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY43 C1 card=pro6000 order=O1 rows=120 differ=['RX-30720-g256-r1-t3-cold', 'RX-30720-g256-r3-t3-cold', 'RX-30720-g32-r0-t3-cold', 'RX-6144-g256-r1-t3-cold', 'RX-6144-g256-r2-t3-cold'] -> FAIL
DAY43 C2 card=pro6000 boot=rx-spec-O1-clamp clamp_lines=120 resumed=60 cached_mismatch=[] -> PASS
DAY43 C3 card=pro6000 boot=rx-spec-O1-clamp turns=60 resumed=60 frac=1.00 -> PASS
DAY43 READING card=pro6000 boot=rx-spec-O1-unset turns=60 resumed=34 flips_vs_cold=6 later_turn_ttft_ms N=60 p50=1622.5 p95=9148.8 generated=25920 wall_s=2773.8 tok_per_s=9.34 clamp_lines=0
DAY43 READING card=pro6000 boot=rx-spec-O1-unset L=6144 G=32 turns=10 resumed=5 ttft_ms p50=882.9 p95=1644.8
DAY43 READING card=pro6000 boot=rx-spec-O1-unset L=6144 G=256 turns=10 resumed=3 ttft_ms p50=1729.2 p95=1804.7
DAY43 READING card=pro6000 boot=rx-spec-O1-unset L=30720 G=32 turns=10 resumed=2 ttft_ms p50=9028.8 p95=9068.5
DAY43 READING card=pro6000 boot=rx-spec-O1-unset L=30720 G=256 turns=10 resumed=4 ttft_ms p50=9118.1 p95=9246.6
DAY43 READING card=pro6000 boot=rx-spec-O1-unset L=122880 G=32 turns=10 resumed=10 ttft_ms p50=419.3 p95=3511.8
DAY43 READING card=pro6000 boot=rx-spec-O1-unset L=122880 G=256 turns=10 resumed=10 ttft_ms p50=439.8 p95=6661.4
DAY43 READING card=pro6000 boot=rx-spec-O1-clamp turns=60 resumed=60 flips_vs_cold=24 later_turn_ttft_ms N=60 p50=211.2 p95=417.0 generated=25847 wall_s=2607.9 tok_per_s=9.91 clamp_lines=120
DAY43 READING card=pro6000 boot=rx-spec-O1-clamp L=6144 G=32 turns=10 resumed=10 ttft_ms p50=161.8 p95=162.3
DAY43 READING card=pro6000 boot=rx-spec-O1-clamp L=6144 G=256 turns=10 resumed=10 ttft_ms p50=161.4 p95=162.5
DAY43 READING card=pro6000 boot=rx-spec-O1-clamp L=30720 G=32 turns=10 resumed=10 ttft_ms p50=211.6 p95=215.9
DAY43 READING card=pro6000 boot=rx-spec-O1-clamp L=30720 G=256 turns=10 resumed=10 ttft_ms p50=211.2 p95=215.7
DAY43 READING card=pro6000 boot=rx-spec-O1-clamp L=122880 G=32 turns=10 resumed=10 ttft_ms p50=416.8 p95=417.1
DAY43 READING card=pro6000 boot=rx-spec-O1-clamp L=122880 G=256 turns=10 resumed=10 ttft_ms p50=415.7 p95=416.4
DAY43 C1 card=pro6000 order=O2 rows=120 differ=['RX-30720-g256-r1-t3-cold', 'RX-30720-g256-r3-t3-cold', 'RX-30720-g32-r0-t3-cold', 'RX-6144-g256-r1-t3-cold', 'RX-6144-g256-r2-t3-cold'] -> FAIL
DAY43 C2 card=pro6000 boot=rx-spec-O2-clamp clamp_lines=120 resumed=60 cached_mismatch=[] -> PASS
DAY43 C3 card=pro6000 boot=rx-spec-O2-clamp turns=60 resumed=60 frac=1.00 -> PASS
DAY43 READING card=pro6000 boot=rx-spec-O2-unset turns=60 resumed=34 flips_vs_cold=6 later_turn_ttft_ms N=60 p50=1674.4 p95=9136.9 generated=25920 wall_s=2776.1 tok_per_s=9.34 clamp_lines=0
DAY43 READING card=pro6000 boot=rx-spec-O2-unset L=6144 G=32 turns=10 resumed=5 ttft_ms p50=909.9 p95=1695.9
DAY43 READING card=pro6000 boot=rx-spec-O2-unset L=6144 G=256 turns=10 resumed=3 ttft_ms p50=1752.3 p95=1826.0
DAY43 READING card=pro6000 boot=rx-spec-O2-unset L=30720 G=32 turns=10 resumed=2 ttft_ms p50=9059.9 p95=9090.6
DAY43 READING card=pro6000 boot=rx-spec-O2-unset L=30720 G=256 turns=10 resumed=4 ttft_ms p50=9128.6 p95=9246.7
DAY43 READING card=pro6000 boot=rx-spec-O2-unset L=122880 G=32 turns=10 resumed=10 ttft_ms p50=419.5 p95=3509.4
DAY43 READING card=pro6000 boot=rx-spec-O2-unset L=122880 G=256 turns=10 resumed=10 ttft_ms p50=439.4 p95=6655.8
DAY43 READING card=pro6000 boot=rx-spec-O2-clamp turns=60 resumed=60 flips_vs_cold=24 later_turn_ttft_ms N=60 p50=211.6 p95=416.7 generated=25847 wall_s=2608.4 tok_per_s=9.91 clamp_lines=120
DAY43 READING card=pro6000 boot=rx-spec-O2-clamp L=6144 G=32 turns=10 resumed=10 ttft_ms p50=161.8 p95=162.2
DAY43 READING card=pro6000 boot=rx-spec-O2-clamp L=6144 G=256 turns=10 resumed=10 ttft_ms p50=161.5 p95=162.8
DAY43 READING card=pro6000 boot=rx-spec-O2-clamp L=30720 G=32 turns=10 resumed=10 ttft_ms p50=215.8 p95=216.2
DAY43 READING card=pro6000 boot=rx-spec-O2-clamp L=30720 G=256 turns=10 resumed=10 ttft_ms p50=211.3 p95=216.0
DAY43 READING card=pro6000 boot=rx-spec-O2-clamp L=122880 G=32 turns=10 resumed=10 ttft_ms p50=416.6 p95=417.6
DAY43 READING card=pro6000 boot=rx-spec-O2-clamp L=122880 G=256 turns=10 resumed=10 ttft_ms p50=415.7 p95=416.3
DAY43 C5 card=pro6000 unset=rx-spec-O1-unset rows=180 differ=[] -> PASS
DAY43 C5 card=pro6000 unset=rx-spec-O2-unset rows=180 differ=[] -> PASS
```

**Clauses, as they read:**

- **C4 PASS on all 5 boots. C5 PASS in both orders** (180 of 180 `unset` rows equal `offprev`'s). **C2 PASS and C3 PASS
  in both orders**: 120 clamp lines per boot, 60 of 60 later turns resumed, every `cached_tokens` equal to the previous
  turn's prompt plus its completion.
- **C1 FAIL in both orders** (`differ=['RX-30720-g256-r1-t3-cold', ...]`, 5 rows). Placed from the rows: the clause
  assumed every turn-1 and cold-twin row has the same prompt on both arms, but a turn-3 prompt (and its cold twin's) is
  built from turn 2's completion, and turn 2 differs where the clamp arm resumed it and the resumed turn flipped against
  cold. By prompt hash: 113 of the 120 rows have the same prompt on both arms and every one of them has the same digest;
  7 have different prompts, 5 of those differ. No same-prompt row differs. The line reads FAIL as registered.

**Readings (N per line; the 250 ms regime in each boot's `samples.csv`).**

| reading | `unset` (today) | `clamp` |
|---|---|---|
| later turns resumed | 34 of 60 | 60 of 60 |
| later-turn TTFT p50 / p95 | 1,622 and 1,674 / 9,149 and 9,137 ms | 211 and 212 / 417 ms |
| TTFT p50 at 6,144 / 30,720 / 122,880 (G=32) | 883 to 910 / 9,029 to 9,060 / 419 ms | 162 / 212 to 216 / 417 ms |
| throughput over the boot wall | 9.34 tokens/s | 9.91 tokens/s |
| resumed turns that flip against cold | 6 of 34 | 24 of 60 |

The clamp turns every spec-route later turn into a resume at keep speed. It does not remove the near-tie residual: a
resumed turn keeps the previous reply's decoded rows, so 24 of 60 flip against cold on the spec route as on the plain
route (DAY41 2.1). That residual is O11's, now owed as an exact and fast resume (DAY44).
