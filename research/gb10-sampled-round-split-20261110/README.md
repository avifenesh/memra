# A sampled speculative round against a greedy one on GB10: where the 7 to 10 ms sits (memra#1186, 2026-11-10)

Verdict: the 7 to 10 ms is not inside the round. It sits at the burst boundary. A sampled burst
ends by committing its pending bonus token with one T=1 trunk pass, and the next burst starts
with a second T=1 trunk pass, the init feed of the boundary token. A greedy burst carries the
pending token into the next burst's first verify and runs neither pass. On GB10 a T=1 trunk pass
costs 75 to 76 ms, 97 to 98% of the T=4 verify pass of a round (77.3 ms), and the default
burst is 32 tokens, 12 to 14 rounds. The two passes and the work around them are 10.4 to 12.0 ms
of wall per round (the passes alone 9.7 to 11.4). Inside the round the three sampled classes cost
0.0, 0.9 and 2.0 ms more than greedy.

Per class at the 1.2k-token prompt, ms per round, sampled less greedy in the same boot (mean of
two profiled boots; both boots' values are in `nsys1/delta.tsv` and `nsys2/delta.tsv` and differ
by at most 0.4 ms on a component row, the verify pass row being the one that moves):

| component | greedy, ms per round | temperature 1 | temperature 0.7, top_p 0.95 | penalised |
|---|---:|---:|---:|---:|
| burst boundary: the tail's T=1 trunk pass (commit of the pending bonus) | 0.44 | +5.10 | +5.66 | +4.86 |
| burst boundary: after it (1 MB logits readback, boundary token draw, draft KV fill, burst exit and entry) | 0.04 | +0.28 | +0.24 | +0.25 |
| burst boundary: the init feed's T=1 trunk pass | 0.45 | +5.59 | +5.70 | +4.86 |
| burst boundary: after it (1 MB logits readback, host argmax, 1 MB upload of the retained row, burst setup, the first round's snapshot) | 0.00 | +0.39 | +0.37 | +0.45 |
| **burst boundary, sum** | 0.93 | **+11.35** | **+11.97** | **+10.42** |
| in the round: draft forward (3 steps of the MTP block and its full-vocabulary head) | 19.33 | +0.05 | +0.06 | +0.21 |
| in the round: draft draw and the 4-byte token readback | 0.20 | +0.01 | +0.65 | +0.97 |
| in the round: verify pass (T=4 trunk, output head) | 77.28 | +0.09 | -0.07 | +0.39 |
| in the round: verify draw (filter stats, Gumbel perturb and argmax over the K+1 rows) and the K+1-token readback | 0.06 | +0.11 | +0.50 | +0.59 |
| in the round: commit (partial-accept restore, draft KV fill, next snapshot) | 3.35 | -0.28 | -0.22 | -0.18 |
| **in the round, sum** | 100.22 | **-0.02** | **+0.92** | **+1.98** |
| round wall (decode window / rounds) | 101.16 | +11.33 | +12.90 | +12.40 |
| the boundary's tokens (one per burst, emitted outside the rounds) | | -3.66 | -3.14 | -3.12 |
| **the issue's figure** (1000 x tokens per round / decode rate) | | **+7.68** | **+9.76** | **+9.28** |
| the issue's table at 1.2k (N = 6 boots) | | +7.5 | +9.8 | +9.2 |

The greedy column's boundary rows are its two one-off T=1 passes: the cold init feed after the
prime and the flush of the pending token when the request ends. A sampled request has the same
cold init feed plus one tail pass and one init feed per burst. The commit row is negative for
the sampled classes because the first round of each sampled burst takes its snapshot before its
first draft step, which the split counts after the init feed.

The two figures. The issue's round time is 1000 x tokens per round / decode rate with tokens per
round = 1 + K x acceptance. That is the wall per round times (rounds + accepted) / completion
tokens. A greedy request emits every token from a round. A sampled burst emits one boundary
token outside its rounds (11 to 13 of 400 tokens here), so its figure is 96.8 to 97.3% of its
wall per round. In wall time a sampled round costs 11.3 to 12.9 ms more than a greedy one; the
issue's figure credits the boundary tokens and reads 7.7 to 9.8 ms.

## The boundary, as the device ran it

`nsys1/boundary-t1.txt` is one burst boundary of the temperature 1 request, every interval in
order. `nsys1/boundary.tsv` and `nsys2/boundary.tsv` count the passes.

| class | bursts | tail T=1 passes | ms each | init-feed T=1 passes | ms each |
|---|---:|---:|---:|---:|---:|
| greedy | 12 | 1 (the closing flush) | 75.0 / 75.2 | 1 (cold) | 77.2 / 76.9 |
| temperature 1 | 13 | 12 | 75.2 / 75.3 | 13 | 76.0 / 75.7 |
| temperature 0.7, top_p 0.95 | 13 | 13 | 75.3 / 74.9 | 13 | 75.7 / 75.8 |
| penalised | 12 | 12 | 75.6 / 75.5 | 12 | 75.7 / 75.8 |

Values are boot 1 / boot 2. Bursts are the `[spec-setup]` and `[spec-phase]` lines of the
unprofiled boots (`setup1/doors.txt`, `phase1/doors.txt`).

One T=1 pass (the 13 init feeds of the temperature 1 request, boot 1): 496 launches of
`qmatvec_nvfp4_mmvq_mr2_rp` at 133 us, 66.0 ms (the trunk's weights read once, as the T=4 verify
reads them once in 66 ms); the full-vocabulary output head `qmatvec_q5_K_mmvq_il`, 5.17 ms;
norms, quantize and glue 1.7 ms; `gdn_scan_s128_b` and conv 1.1 ms; attention 0.5 ms; launch
gaps 1.35 ms. One row costs what four cost; that the trunk is bandwidth bound on this device
is the reading of that, not a separate measurement.

Launch sites (`crates/memra-engine/src/spec.rs`; line numbers are in the measured tree, commit
9ca50c3556, the head of memra#1180, and move once that change is on main; the function names
do not):

- The tail pass: `spec_burst_end`, line 14646, `self.spec_target_step_h(e, b, &mut *cache)`, in
  the branch commented "SAMPLED tail (unchanged): commit the bonus (one T=1 pass) + draft fill".
  `spec_target_step_h` (line 11745) runs `decode_step_t_core` on one token (line 11756) and reads
  the whole logits row back to the host (line 11757, 0.99 MB).
- The boundary token draw: line 14653, `sample_boundary_token` (line 2189) uploads the row
  again (line 2198) and calls `sample_boundary_token_dev` (line 2085): one `filter_stats` launch
  (line 2123), two scalar readbacks (line 2127), `gumbel_perturb_filtered_f32` (line 2129), the
  argmax (line 2131) and the token readback (line 2133). About 0.2 ms per burst, 0.01 to 0.03 ms
  per round.
- The init feed: `spec_burst_begin`, line 13932, `self.spec_target_step_h(e, last_token, &mut
  *cache)`, then a host argmax over the 248,320-entry row (line 13934, 0.3 ms of idle device) and
  the upload of the row as `last_col_logits` (line 13940).
- What greedy does instead: the tail stashes the pending bonus on the session (lines 14623 to
  14640) and the next burst takes it as round 0's verify column 0 (line 13300, "no init feed, no
  tail commit pass"). `SpecSession.pending_tok` (line 1614) says the carry is for "the next
  empty-suffix greedy burst" and that sampled turns flush first.

## Inside the round

Phase tables: `nsys1/summary.txt`, `nsys2/summary.txt` (every kernel class per phase, launches
per round, us each). `nsys1/round-t1.txt` is one whole temperature 1 round.

**Verify side** (spec.rs lines 16403 to 16479; greedy lines 16328 to 16331).

| per round | greedy | temperature 1 | filtered | penalised |
|---|---:|---:|---:|---:|
| verify draw phase, ms | 0.06 | 0.17 | 0.56 | 0.65 |
| `filter_stats_coop_f32`, launches x us | | 2 x 23 | 2 x 225 | 2 x 223 |
| `gumbel_perturb_filtered_col_f32`, launches x us | | 4 x 5.9 | 4 x 3.4 | 4 x 4.2 |
| `argmax_partial_f32` + `argmax_final_f32`, launches x us | 4 x 8.0 | 4 x 4.4 | 4 x 4.4 | 4 x 4.3 |
| `penalize_logits_rows_inc_f32`, launches x us | | | | 1 x 83 |
| row copies into the K+1-row buffer (device copies, 3.97 MB) | | 4, 24 us | 4, 23 us | 4, 24 us |
| the K+1-token readback (the copy itself) | 3 us | 3 us | 4 us | 4 us |

Filter statistics are most of the verify side's cost: 0.05 ms with no filter, 0.45 ms with
top_p or top_k on. The K+1 = 4 rows take two cooperative launches because `Engine::filter_stats`
chunks rows to `sm_count / 16` per launch (`lib.rs` line 4671), 3 rows on this 48-SM device; a
launch with a filter on costs 0.22 ms whether it holds three rows or one. The Gumbel perturb and
the argmax over the four rows are 0.04 ms. The readback is one 16-byte copy. The host is ahead
of the device at that point: it blocks in the readback while the verify pass runs
(`hostcalls.tsv`), so the readback adds no wall.

**Draft side** (graph launch at line 15268 sampled, 15175 greedy; capture body lines 5532 to
5555; eager chain lines 15384 to 15428).

| per round | greedy | temperature 1 | filtered | penalised |
|---|---:|---:|---:|---:|
| draft forward phase, ms | 19.33 | 19.39 | 19.39 | 19.54 |
| draft draw phase, ms | 0.20 | 0.21 | 0.86 | 1.17 |
| the draw per drafted token | argmax, 7.9 us | `memra_sctr_inc` 1.5 us, `gumbel_perturb_ctr_f32` 6.2 us, argmax 3.7 us | `memra_sctr_inc`, `filter_stats_coop_f32` 219 us, `gumbel_perturb_filtered_ctr_f32` 4.2 us, argmax | `penalize_logits_f32` 85 us, `filter_stats_coop_f32` 212 us, two scalar readbacks, `gumbel_perturb_filtered_f32` 3.3 us, argmax |
| readbacks per round | 3 | 3 | 3 | 9 |
| idle between launches in the draft forward, ms | 0.06 | 0.08 | 0.07 | 0.24 |

- Pure temperature: the in-graph draw costs what the greedy argmax costs. Nothing to recover.
- Filtered: the in-graph filter statistics are 0.66 ms per round, three launches of 0.22 ms.
- Penalised: the eager chain costs 0.2 ms per round more than graph replay in launch gaps, 0.64
  ms in filter statistics, 0.25 ms in the three penalty passes and 0.1 ms in host time around
  six more readbacks.

**Host time between launches.** Idle between intervals over the five round phases, ms per
round: greedy 2.30, temperature 1 2.29, filtered 2.29, penalised 2.57 (`nsys*/kernels.tsv`, the
`idle (host)` rows). 1.70 ms of it is the verify pass's launch gaps, the same in every class.
The penalised class's extra 0.27 ms is the eager draft chain. The accept walk is host arithmetic
on four tokens and does not show. Synchronisation points (`nsys*/hostcalls.tsv`): readbacks
(`cuMemcpyDtoHAsync`), 4.0 per round greedy, 4.4 for the two graph-drafted sampled classes, 10.4
penalised; and `cuStreamSynchronize`, 5.1 to 5.5 calls per round (11.4 penalised), 1.1 to 1.2 ms
per round inside the call in every class. The host spends 93 to 102 ms of each round inside a
readback, waiting on a busy device: it runs ahead of the device, so none of these waits is
wall of its own.

## The same answer without the profiler

Three unprofiled boots of the same six requests (`cells.txt` has every cell on the client's
clock; texts are the same in all seven boots):

| cell | greedy | temperature 1 | filtered | penalised |
|---|---:|---:|---:|---:|
| `plain1`, no door: wall ms per round (client) | 99.26 | 110.54 | 112.49 | 111.56 |
| `phase1`, `MEMRA_SPEC_PHASE=1`: the round loop on the engine's clock, ms per round | 99.40 | 99.84 | 100.66 | 101.48 |
| `phase1`: client decode time outside the round loops, ms per round | 0.05 | 11.19 | 11.87 | 10.23 |
| `phase1`: the same per burst, ms | 0.8 | 140 | 146 | 146 |
| `setup1`, `MEMRA_SPEC_SETUP_TRACE=1`: init feed, ms per continuation burst | 0.00 | 75.8 | 75.7 | 75.6 |
| `setup1`: tail, ms per burst | 0.18 | 69.7 | 76.0 | 76.0 |
| `setup1`: round loop, ms per round | 99.34 | 99.57 | 100.75 | 101.72 |

The temperature 1 tail mean is lower because its last burst ends with no pending token (12
tail passes in 13 bursts). `MEMRA_SPEC_PHASE` books a sampled round's verify wait under
commit-host, because the sampled readback sits after the phase mark; the sum is unaffected.

The burst length probe. `MEMRA_SPEC_BURST` is the documented tokens-per-burst parameter (32 by
default). One boot each, no profiler, client clock, wall ms per round and the change against
the greedy request of the same boot:

| burst length | bursts in 400 tokens | greedy | temperature 1 | filtered | penalised |
|---:|---:|---:|---:|---:|---:|
| 32 (`plain1`) | 13 | 99.26 | 110.54 (+11.29) | 112.49 (+13.24) | 111.56 (+12.30) |
| 128 (`burst128`) | 4 | 99.15 | 102.70 (+3.54) | 103.74 (+4.59) | 104.48 (+5.33) |
| 512 (`burst512`) | 1 | 99.47 | 99.79 (+0.31) | 100.94 (+1.47) | 101.83 (+2.35) |

| decode tok/s | greedy | temperature 1 | filtered | penalised |
|---:|---:|---:|---:|---:|
| 32 | 23.71 | 22.20 | 22.22 | 20.97 |
| 128 | 23.73 | 23.90 | 23.51 | 21.88 |
| 512 | 23.65 | 24.15 (+8.8%) | 24.02 (+8.1%) | 22.45 (+7.1%) |

With one burst per request the sampled round is the greedy round plus 0.3, 1.5 and 2.4 ms. Part
of that is one more T=1 pass and not in-round cost: every sampled request, at every burst
length and in all seven cells, ends 75 to 92 ms after its last chunk (the closing tail pass that
commits the last pending token), where a greedy request ends 0.1 ms after its last chunk. At
one burst per request that pass is 0.43 to 0.46 ms per round, which leaves -0.1, +1.0 and +1.9
ms, the split's in-round sums. Every sampled text is byte-identical at the three burst lengths (same sha256 per
prompt): the tokens that are boundary draws at 32 are round draws at 512, and they are the same
tokens. Single boots: the probe is a causal check, not a default decision.

## What a fix could target

Named, not built. Each needs the seeded-invariance gate and the timed cell on GB10 and the RTX
5090.

1. **Carry the pending bonus across the burst boundary for sampled requests**, as the greedy
   path does. Since memra#1154 the sampled walk draws row 0 from verify column 0 whenever a
   pending token rides (base = 1), at the position's own event key, so the boundary needs no
   retained row and no separate draw. This is a reading of the code, not a test: no run here
   carried a pending token across a sampled boundary, and the comments on the sampled tail and
   on the entry flush give a reason for the pass that predates that walk. It would remove the
   tail pass (line 14646), the boundary draw (line 14653), the init feed (line 13932), the
   entry flush of a pending token, the argmax arm of the boundary door, and both 1 MB logits
   round trips: the burst boundary rows above, 10.4 to 12.0 ms of wall per round, 7 to 9% of
   decode rate (the one-burst probe). The identical texts across burst lengths are the
   evidence that the round's draw equals the boundary's draw on these prompts; the gate has to
   show it in general, with the pooled pair and the resume cells. What stays: the cold init
   feed, and the closing pass at the end of a request (75 to 92 ms after the last chunk, on
   the client's clock for the whole request, not for its decode rate) unless the carry also
   leaves the last token pending as greedy does.
2. **Burst length on this device**, if the carry is not taken or until it lands. 128 tokens
   leaves 3.5 to 5.3 ms of the gap. It moves the scheduler cadence for concurrent requests,
   which this lane did not measure.
3. **Filter statistics, filtered and penalised classes**: five launches of 0.22 ms per round
   (two in the verify, three in the draft). The verify's second launch holds one row because of
   the three-row chunk on a 48-SM device; one four-row launch would save 0.22 ms per round if
   the co-residency rule allows it. The per-launch cost with a filter on is 10 times the
   no-filter cost (225 us against 23 us); why was not profiled.
4. **Penalised draft**: the penalty pass (85 us x 3) and the eager chain's launch gaps (0.2 ms)
   are what a graph-captured penalised draft would touch. About 0.5 ms per round; the capture
   rule refuses penalties today because the history changes every round.

Not a target: the pure-temperature in-graph draw, the Gumbel perturb and argmax over the K+1
rows, the K+1-token readback, the accept walk. Together under 0.15 ms per round.

## Method

Device: one DGX Spark (GB10, sm_121a, 48 SMs). Model: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP
head (full-vocabulary draft head), context 8,192, two sessions, thinking off, every switch at
the device default: the boot of the timed cell in
`research/spec-draft-accept-leftovers-20261110/` (`tools/ab_cell.py`; that directory arrives
with memra#1180). Binary: `memra-server` built from the sources of commit 9ca50c3556 (the head
of memra#1180) on the device with `tools/gb10-build.sh`
(`cargo build --release --locked -j 12 -p memra-server --bin memra-server`), sha256
`cbc259e708068d76ee461ad958944aee9d9f8e1af4b6ea8cdebe9d48b742e4d7` in every `guard.log`. The
binary carries no git stamp and the build log is not committed, so the tie to that commit is
this statement. No engine change and no instrumentation patch.

K: not pinned by an operator variable. The device's spec cost model chose K=3 for every request,
as it did in the timed cell: all 42 requests log `[spec-k] ... K=3 source=cost-model`
(`*/server.log`).

Requests (`tools/req_driver.py`): one boot serves six cold requests, one at a time, 3 s apart:
a short greedy warm-up (48 tokens), then the timed cell's four 1.2k-token prompts with its
sampler settings and seeds, 400 tokens each (greedy; temperature 1; temperature 0.7 with top_p
0.95; temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5), then a second greedy 1.2k
prompt as the closing control. `tools/check_shapes.py` checks the four against `ab_cell.py`'s
rows. The temperature 1 text has the timed cell's sha256 (`e12ce14478d9edba`).

Cells (`tools/prof_cell.sh MODE NAME`, one boot per device lease, in this order):

| cell | mode |
|---|---|
| `nsys1`, `nsys2` | `nsys profile -t cuda,nvtx -f true --sample=none --cpuctxsw=none --cuda-graph-trace=node --cuda-flush-interval=1000 -o prof /usr/bin/env MEMRA_COMPAT=openai MEMRA_MODELS=q=<gguf> MEMRA_ADDR=127.0.0.1:18196 MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=2 memra-server`, then `nsys export --type sqlite` |
| `plain1` | the same env, no profiler |
| `phase1` | plus `MEMRA_SPEC_PHASE=1 MEMRA_SPEC_STATS=1` (wall marks only, no added synchronisation) |
| `setup1` | plus `MEMRA_SPEC_SETUP_TRACE=1` (one added stream synchronize per burst, at its tail) |
| `burst128`, `burst512` | plus `MEMRA_SPEC_BURST=128`, `=512` |

The split (`tools/split.py prof.sqlite reqs.jsonl OUT`): the method of
`research/gb10-decode-20261110/` (kernels, device copies and memsets from the sqlite export, by
name) with phases added. A request is the device work between two idles of more than 1 s. The
device timeline is one stream, so every interval is serial and the idle between two intervals
is host time. A state machine gives each interval and each idle to a phase by marker kernels
(the tool's header lists them): the T=1 passes, what follows each, the draft forward, the draft
draw, the verify pass, the verify draw up to the accept readback, and the commit. The phases
sum to the decode window exactly (the tool asserts it). A round is one launch of the verify
head `qmatvec_q5_K_mmvq_b4_r2`; the count equals the request's `usage.spec.rounds`. The decode
window opens at the cold init feed and closes at the request's last device work.

N: per class one request per boot, 160 to 171 rounds and 12 to 13 bursts each. Two profiled
boots for the split; three unprofiled boots at the default burst length and one boot per probe
length for the client and engine clocks. Seven boots, 42 requests, 0 errors.

Thermal regime: steady in all seven boots. Per request the SM clock median is 2,177 to 2,184
MHz, the temperature median 62 to 68 C (72 C maximum), power 31 to 33 W, and no sample inside a
request window carries a throttle reason (`thermal.txt`, 250 ms `nvidia-smi` telemetry in each
cell; the warm-up requests run at 2,190 MHz and 25 to 32 W, and 223 samples at boot or idle
carry reason 0x4). The seven boots ran
back to back on an otherwise idle unit.

Profiler cost: the client's wall per round is 0.8 to 1.8 ms (1 to 1.6%) longer under the
profiler than in `plain1`. Sampled less greedy on the client's clock is 11.3 / 13.2 / 12.3 ms
without it and 12.0 / 13.4 / 12.6 and 11.6 / 13.2 / 13.2 ms in the two profiled boots.

## Not determined

- The 34-token and 3.6k-token prompts were not profiled. The issue's table has the same gap
  there, and the cost is per burst, so the same split is expected, not measured.
- The RTX 5090 and the RTX PRO 6000. The mechanism is in shared code; the price of a T=1 pass
  differs per device.
- Concurrent requests and the pooled draft and verify paths.
- Seventeen pairs of a 2.26 MB memset and device copy, once per request in its first boundary
  (0.6 ms, in the after-tail phase), were not traced to their source line. Some after-tail and
  after-init units also hold one long 1 MB readback interval (1 to 4 ms); the cause was not
  traced. Both are inside the 0.6 to 0.7 ms per round the two "after" rows sum to.
- Why a `filter_stats_coop_f32` launch costs 10 times more with a filter on.
- Whether main without memra#1180 splits the same way (the binary is that change's head).
- The 75 to 92 ms between a sampled request's last chunk and its end was read from the client
  clock (`*/reqs.jsonl`), not from the trace.
- The closing greedy control lost its last four rounds from each trace (the profiler dropped
  the boot's last events at the stop, 20 s of idle before it notwithstanding). Its per-round
  figures use the 166 rounds the trace holds; the four classes are complete.
- The verify pass row moves by up to 0.5 ms between requests of one boot (the closing control
  reads +0.21 and +0.14 against the first greedy request). In-round differences under that are
  not resolved for the verify pass row; the draw rows are kernel counts and times and are.

## Files

- `nsys1/`, `nsys2/`: `summary.txt` (every table of `split.py`), `phases.tsv`, `delta.tsv`
  (sampled less greedy by component), `boundary.tsv` (the T=1 passes), `kernels.tsv` (device
  time by class and idle per phase), `hostcalls.tsv` (blocking host calls per phase),
  `reqs.jsonl` (the driver's rows with the texts), `server.log`, `guard.log` (binary sha256 and
  env), `run.log`, `telemetry.csv`. `nsys1/round-t1.txt` and `nsys1/boundary-t1.txt` are one
  round and one burst boundary of the temperature 1 request, every interval (`tools/fold.py`).
- `plain1/`, `phase1/`, `setup1/`, `burst128/`, `burst512/`: `reqs.jsonl`, `server.log`,
  `guard.log`, `run.log`, `telemetry.csv`; `phase1/doors.txt` and `setup1/doors.txt` are
  `tools/doors.py` over the two door cells.
- `cells.txt` (`tools/cells_tab.py`: every request of every cell on the client's clock),
  `thermal.txt` (`tools/thermal.py`).
- `tools/`: `req_driver.py`, `prof_cell.sh` and `gb10-build.sh` (run on the GB10 host under the
  caller's lease), `cells_driver.sh` (the one-lease-per-cell driver, machine names from the
  environment), `split.py`, `fold.py`, `doors.py`, `cells_tab.py`, `thermal.py`, `export.py`
  (wrote these directories), `check_shapes.py`.
- Profiles (`prof.nsys-rep`, `prof.sqlite`) stay on the machine.
