# DeepSeek-V4.1-Flash onboarding, step 7: the serving route, on one card (2026-11-09)

Verdict: memra-server serves DeepSeek-V4.1-Flash on one RTX PRO 6000 through its own deepseek-v4.1
route over the decode loop of step 6, and `memra model verify serve --against deepseek_v41` passes on
the pinned checkpoint. The route keeps every layer's own weights and the head on the card (9.9 GB)
and the 15,360 routed experts in 289 GB of pinned host memory behind a device LRU cache (4,736
slots, 89 GB, with the route idle); the cache moves bytes, never values: on the decode lane's 13 chat tapes teacher-forced and 5 greedy (the cache
at its floor, evicting on every layer), the cached runner writes the streamed runner's step files
byte for byte (95 of 95 files). CheckpointParity passes again with the native side on this
residency. Served greedy answers equal `dsv41-run --gen` on the seven chat and needle tapes
checked (render, text and token count), five requests served at once answer byte for byte as served
alone, a seeded sampled request repeats alone and beside four others, and stop strings, the token
budget, penalties and the numeric reasoning budget behave as specified. One request decodes at
7.5 tokens per second (median step 0.134 s, 298 steps) where the streamed runner took 11 to 31 s
per decode forward: a correctness route, not a fast one. The lane's shared benchmark client runs
against it unchanged. The pack is `NativeQualified` <!-- support: deepseek_v41-gpu-1x-pro6000 --> on this
topology (record `deepseek_v41-gpu-1x-pro6000`).

> **Re-pinned (2026-11-14, the row map).** The record `deepseek_v41-gpu-1x-pro6000` now pins `6371a9975b`, the
> head of memra#1315 on main `63c351a2f4`: the route's call flag with a row map, so a router row that names no
> expert fails its own request alone (`../row-faults/README.md`), the first route-code change after the pin
> below. On one RTX PRO 6000 Max-Q, `verify checkpoint` and `verify serve` pass at that head
> (`verified=Checkpoint`, `verified=Serve`; CheckpointParity with argmax 51119 on both sides and max abs 2.94):
> `receipts/verify-6371a9975b/` (the run, the binaries' sha256, the bundle with its large tables gzipped).
> `memra model verify tiny` and `ci:verify-tiny`'s test ran on the same tree, with every output hash equal to the
> earlier pins': `receipts/verify-6371a9975b-tiny/`. In the same window the row-fault gate over all 40 layers and
> the serve smoke with a poisoned row passed on this topology (`../row-faults/receipts/pro6000-6371a9975b/`). The
> two commits after the pin are receipts and docs, and the merge of main `0db086f87a` after them changes no code
> line of this route (`receipts/verify-6371a9975b-reach/`: one doc comment in `dsv41_gpu/experts.rs`). RewriteParity
> stays pending.
>
> **Earlier pin (2026-11-14, the CUTLASS lane).** The record pinned `a05173ee27`, the head of the
> CUTLASS lane (the route's FP8 and MXFP4 GEMMs on the block-scaled MMA and on vendored CUTLASS 4.8.1, with
> small-row decode bodies; `../perf/README.md`). On one RTX PRO 6000 Max-Q, `verify checkpoint` and `verify serve`
> pass at that head (`verified=Checkpoint`, `verified=Serve`; CheckpointParity with argmax 51119 on both sides and
> max abs 2.94): `../perf/receipts/cfix-pro6000/cfix2-m1/` (the run, the binaries' sha256, the bundle in
> `verify/`). `memra model verify tiny` and `ci:verify-tiny`'s test ran on the same tree:
> `receipts/verify-a05173ee27-tiny/`. The record's notes list the commits after the pin and what each can reach;
> the checkpoint and serve gates were not rerun on them. RewriteParity stays pending.
>
> **Earlier pin (2026-11-13).** The record pinned `0d074a7055`, the dsv41 landing
> head (the route with its two reviews' fixes, the peer copy call, the fp32 FMA family's streamed narrow outputs, the
> embedding table and the pool reserve at the load, main `d6ae4e525d` merged). At that head the route batches the
> running requests' one-token calls up to 8 rows a call (each row the bytes of its call alone) and takes a greedy
> pick as the card's argmax of the row. On one RTX PRO 6000 Max-Q, `verify checkpoint` and `verify serve` pass again
> (`verified=Checkpoint`, `verified=Serve`; Config, TokenizerTemplate, TensorCensus, CheckpointParity with argmax
> 51119 on both sides and max abs 2.91, and Serve passed), and `ci:verify-tiny`'s test and `memra model verify tiny`
> ran at the same tree: `receipts/verify-0d074a7055/` (the run, the binaries' sha256, the tiny gate). RewriteParity
> stays pending. The text below is the first run, at `87bbe3bfc8` (`receipts/verify-res/`, kept); its 7.5 tokens a
> second for one request is not remeasured at the new pin.

Hardware: one RTX PRO 6000 Blackwell Max-Q at a time (sm_120a, 96 GB, 250 W cap; the residency
gate ran on one card of the box and the final gates on its twin, while other work used the box's
other cards), CUDA 13.4, rustc 1.97.1, 1 TB host memory shared with other work, 16 host cores.
Every time below is a single run, not timed for publication.

Commits: the route is `83175e63f3`; `e44587d85c` makes its residency follow the cards;
`87bbe3bfc8` fixes a client stop string that completed across two tokens. Every receipt under
`receipts/` except the residency gate is from the binaries of `87bbe3bfc8`
(`memra-server` sha256 `b84f7cad...`); the residency gate ran `dsv41-run` (`918d22c2...`), which
the two later commits do not touch and which rebuilt to the same bytes.

## The route

`crates/memra-server/src/dsv41_serve.rs`, wired beside the DSv4 route in `worker.rs` (a V4.1
checkpoint directory, `model_type` `deepseek_v41`, loads here and never through `HybridModel`).

- **Program.** A request's prompt is one prefill call at position 0 (the vendor's prefill program),
  then every generated token is one decode call (the vendor's decode program), on the runner of step
  6 (`dsv41_gpu::decode::Runner`). A generated token is never prefilled again, and a later turn is a
  new request with its own prefill (no continuation, no prefix reuse), so no request crosses between
  the two programs, which below the 128-token window give different bits for the same row.
- **Residency, one card** (`dsv41_gpu::experts`). `Runner::experts_off_card` loads every trunk
  layer's own weights (attention, compressor, indexer, router, shared expert, mHC, Engram
  projections) and the head for good: 9,863,726,528 bytes. The routed experts of the 40 trunk layers
  are copied out of the checkpoint mmap into one anonymous region per layer and page-locked
  (`HostExperts`): 288,777,830,400 bytes, filled and locked in 14.2 to 16.3 s on 16 threads in
the final runs. A
  read-only file mapping cannot be page-locked on this driver
  (`CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED` is 0; `pin_probe.py`), so the store is a
  copy, not the page cache in place. Each rank keeps an `ExpertCache`: slabs of 32 slots of
  18,800,640 bytes (one expert's MXFP4 `w1`, `w3`, `w2` and their ue8m0 scales, packed as the
  checkpoint stores them). The serving route grows it into the free memory beside the resident
  weights less 1 GiB: 4,736 slots (89 GB) with the route idle; its floor is one slice's experts
  (384 slots). Before a slice's MoE, the slice's experts resolve: hits stamp their slot, misses take
  the least recently used slots outside the slice and copy on the cache's own stream behind a fence
  on the compute stream, the shared expert runs meanwhile, and the compute stream waits for the
  copies before the grouped GEMMs, which read the slots through the same pointer table as a streamed
  layer's experts. Pinned H2D on this card: 52 GB/s, 0.34 ms an expert (`pin_probe.out`).
- **Memory book.** Per rank: the ceiling is the effective free memory (driver free plus the pool's
  mapped, unused bytes) with the resident weights loaded and no cache; a request books its carried
  state at its capacity (prompt plus budget: 40 window rings of 128 KB, 1,280 bytes of compressed KV
  and index key per compressed row), the step's largest transient (its prompt's prefill, every buffer
  counted live: streams, keys side, the candidate pool `[s][s/8]`, per-slice heads, indexer scores,
  MoE pairs and Engram rows; or the largest decode call), and 1 GiB. A request that cannot fit beside
  the cache floor even with the route idle is refused as a context-length error naming the largest
  session that fits; one that fits alone waits in the FIFO queue until running requests finish; one
  that fits now makes the cache give back slabs (least recently used first, never below the floor)
  and runs. The cache grows back after the step. Every step logs the pool's high-water mark against
  the book: the measured transient stayed under the booked live need on every served step here,
  0.010 to 0.018 of it on the short prompts and 0.37 and 0.48 on the 3,761 and 18,343-token
  prompts (3.95 GB against 8.26 GB booked; `receipts/chat/serve-chat.log`, `rates.json`). After
  the 18,343-token prefill the cache's regrow failed twice, one slab short of full
  (`grow the expert cache: alloc 601620480: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of
  memory")`, both final runs), the route logged it and kept serving on 4,704 slots, and the cache
  was back at 4,736 by the end of the run: the pool's freed blocks count as free memory before
  the driver can hand them out as one slab.
- **Batching.** Up to `MEMRA_MAX_SESSIONS` requests (default 64) run at once; each step is one runner
  forward with at most one new prompt's prefill call and one decode call per running request. The
  runner runs every call alone through each block, so batching shares the layer walk and the
  resident experts, not arithmetic.
- **Sampling.** Greedy is the argmax, the lowest id among equal maxima (step 6's rule). Temperature,
  top-p and top-k draw through the DSv4 route's position-keyed host sampler (`dsv4_sample_row`:
  seeded and reproducible per absolute position), penalties through `dsv4_penalize_row` over the
  prompt and the generated tokens; `min_p` refuses by name. The vendor's own `sample` is Gumbel-max
  on the device; the distribution is the same, the random program is memra's.
- **Stops.** The end-of-sentence id, the request's stop ids, stop strings over the detokenized
  stream, and the token budget. A client's stop string ends the stream after the whole token
  that completes it, the worker's law, and the HTTP layer cuts the answer at the stop
  (`truncate_at_stop`, the stream scrubber). The DSv4 route's scan alone, which this route first
  copied, ends the stream at text already sent, so a stop that completed across two tokens left
  its first part in the answer and hid the stop from the HTTP cut: the chat gate's `stops` cell
  failed on it (stop `:"A` answered `{"iron":"Fe","gold":"`) and passes since `87bbe3bfc8`. The
  V4.1 DSML close `</｜DSML｜ calls>` stays in the stream so the parser closes the call.
- **Chat surface.** The HTTP layer renders with `encoding_dsv41` and parses V4.1's DSML calls and
  reasoning (step 2b); the route sets `ModelCaps { dsv4: true, dsv41: true }`, keyed on the
  tokenizer's census (the load refuses a tokenizer that is not `Dsv4Encoding::V41`).
  `reasoning_effort` now carries V4.1's numeric budget, `"1"` to `"100"` (canonical decimal), on
  chat completions and the OpenRouter `reasoning.effort` object, for V4.1 only; every other model
  still refuses a number, and `/v1/responses` keeps its level table. `none` is chat mode.
- **Policy contract** (`route_contract::RouteContract::dsv41_thread`): capacity, occupancy, progress
  (the runner stamps every layer pass through the thread's progress sink), fault ownership (a step
  that errors or panics fails its requests and the route keeps serving), shutdown, memory cost, the
  rewrite bundle (it must name this checkpoint's `config.json` and the plan this binary compiles;
  the route runs the plan's own eager decode loop and no rewrite) and service metrics implemented;
  prime fairness refused by name.
- **Not served, by name:** `response_format`, image parts, `min_p`, DSpark drafts.
- **N ranks, and the residency they get.** The route speaks `Runner::forward`/`Runner::logits` over
  `Seq`s, a `tp::Group` of ranks; `MEMRA_DSV41_DEVICES` names them. `bring_up` measures each rank's
  card before any weight loads (`experts_on_card`): where every rank's routed experts fit (four
  ranks on four 96 GB cards: 72 GB of experts a rank) the runner keeps the whole trunk and the head
  resident, no host store and no cache, and the book's ceiling is what the weights leave free;
  where they do not (one card: 289 GB) the residency above. No switch selects it. Only the
  one-card arm ran here; the resident arm ran on no hardware in this lane (see "Behind the same
  layer").

Engine changes under the route, all byte-neutral and gated below: `LayerW::load_part_with` (a
layer's share without its routed experts), `Ctx::experts` and `experts_grouped` reading cached slots,
the MoE queueing the copies before the shared expert, a resident layer borrowed instead of taken
(so an error part-way never loses it), sequences' compressed KV and index keys first sized for the
sequence's capacity instead of the runner's `max_pos`, and `dsv41-run --experts-host`.

## Gates

| gate | result |
|---|---|
| residency, a 3,904-slot cache (73 GB): the 13 chat tapes teacher-forced (517 calls, one pass) against the decode lane's streamed run `dec/native-final` | 65 / 65 files byte-identical (top-32 ids and values, lse, argmax, the full logits of 8 steps); 13,765 misses, 9,861 evictions; 106 s |
| residency, cache at its floor (384 slots: one layer's experts, evicting every layer): 5 chat tapes greedy against `dec/greedy` | 30 / 30 files byte-identical (tokens and every step file); 14,371 misses, 13,987 evictions |
| CheckpointParity, native side on the residency (`dsv41-run --experts-host` through `verify checkpoint`) | passed: argmax 51119 on both sides, max abs 2.14 inside the pack's band (`receipts/verify-res/checkpoint-parity.tsv`; its `native_runner_sha256` is the wrapper script's, the binary's is in `serve_gate_all.out`) |
| `memra model verify serve DeepSeek-V4.1-Flash-1017 --against deepseek_v41 --native-runner memra-server` | `family=deepseek_v41 verified=Serve`; Config, TokenizerTemplate, TensorCensus, CheckpointParity and Serve passed, TinyParity (CI's gate) and RewriteParity pending in this bundle (`receipts/verify-res/gates.txt`, `serve-gate.tsv`) |
| render: `/v1/tokenize` of the case messages against the tapes' prompt ids | 7 / 7 equal (16 to 18,343 tokens) |
| solo greedy against `dsv41-run --gen` | 7 / 7: content and completion token count equal, finish reason `stop` |
| batched: the five chat cases at once against solo | 5 / 5 byte-identical |
| streamed SSE against solo | equal (14 deltas of sys_json) |
| sampled, seed 1017, T 0.8, top-p 0.95, top-k 40 | the same answer twice solo and once beside four others |
| stop string | `:"A`, which completes across two tokens, cuts sys_json before its first occurrence, finish `stop` (failed before `87bbe3bfc8`) |
| `max_tokens` 5 | gsm cut at five tokens, the native run's first five, finish `length` |
| penalties (frequency 0.5, presence 0.3, greedy) | served, the same answer twice |
| numeric budget | `"63"` served (the prompt carries the budget line), `"101"` a 400 naming `[1, 100]` |
| lib suites on the final binaries, in a one-card lease | memra-server 1,120 passed (10 ignored), memra-cli 30, the engine's dsv41 tests 10; clippy `-D warnings` and fmt clean (`receipts/tests/`) |
| the shared benchmark client | every request served with its token counts, both stream shapes (below) |

Served greedy against the vendor: capital, multiturn, sys_json, hebrew, needle and needle1200 give
the vendor's tokens exactly (the decode lane's table, step 6); gsm parts from the vendor after
five tokens, exactly where memra's own greedy does (a vendor near-tie: the vendor's CPU run makes
memra's choice), and the served answer is memra's.

## Speed (a correctness route)

Decode steps from the chat gate's served log (`receipts/chat/rates.json`; single run, card capped
at 250 W, 71% cumulative cache hit rate, a median of 19 expert copies a decode token):

| calls in the step | steps | median | p90 | tokens/s |
|---|---|---|---|---|
| 1 | 298 | 0.134 s | 0.143 s | 7.5 |
| 2 | 22 | 0.281 s | 0.289 s | 7.1 aggregate |
| 3 | 1 | 0.431 s | | 7.0 aggregate |
| 4 | 1 | 0.569 s | | 7.0 aggregate |

Memory, measured: 9,863,726,528 bytes of resident weights on the card, 288,777,830,400 bytes of
routed experts pinned in host memory, and a device cache of 4,736 slots of 18,800,640 bytes
(89,039,831,040 bytes) with the route idle, 384 slots (7.2 GB) at its floor.

Batching adds no throughput here: a decode call costs about 0.13 s of per-call work (40 layers of
small launches, the MoE plan's host round trip, window index rows, the Engram gather), and every
call runs alone through each block. Short prefills (16 to 45 tokens) took 0.3 to 1.2 s, most of it
the experts' copies on a cold cache; the 3,761-token prompt took 5.4 s and the 18,343-token prompt
15.8 s (7,678 and 10,037 expert copies in the step). The streamed one-card runner of step 6 spent 11 to 31 s per
decode forward.

## The shared benchmark client

The lane measures every engine with one client (`bench.py`, stdlib only): token-id prompts on
`/v1/completions` (`prompt_ids` for memra), streamed with `stream_options.include_usage`, greedy,
generated tokens counted from the server's `usage`, cached tokens read from
`usage.prompt_tokens_details`. Run against this route as a wiring check, two samples a workload
after one warm-up, one card capped at 250 W (`scripts/bench_run.sh`; `receipts/bench-client/` for the
OpenAI stream shape, `MEMRA_COMPAT=openai`, which the lane's runs use, and
`receipts/bench-client-native/` for memra's own shape):

| stream shape | workload | prompt tokens | generated | TTFT | decode tokens/s | card power, hottest read |
|---|---|---|---|---|---|---|
| memra's own (`MEMRA_COMPAT` unset) | chat_c1 | 36, 41 | 384 | 1.16, 1.19 s | 7.40, 7.47 | 195 W, 89 C |
| | decode_c1 | 2,048 | 512 | 6.26, 6.51 s | 7.18, 7.27 | 191 W, 88 C |
| | prefill4k | 4,096 | 1 | 6.87, 8.72 s | | 181 W, 88 C |
| OpenAI (`MEMRA_COMPAT=openai`) | chat_c1 | 36, 41 | 384 | 1.24, 1.34 s | 7.16, 7.19 | 163 W, 76 C |
| | decode_c1 | 2,048 | 512 | 9.56, 10.62 s | 7.04, 7.42 | 162 W, 77 C |
| | prefill4k | 4,096 | 1 | 6.84, 13.59 s | | 153 W, 76 C |

The two shapes ran on different cards of the box at different times, with other work on the box's
cores and cards, so the rows are not a comparison of the shapes; time to first token varied more
in the second run and its cause was not isolated. Power and temperature are the mean of the two
samples' means and the hottest read.

The smoke request (the capital prompt as token ids, 8 tokens allowed) answered `Paris`.

Every request answered 200 with its token counts (the usage chunk `stream_options.include_usage`
asks for and finish reason `length` in the OpenAI shape, the closing `done` event in memra's own
shape, which the client also reads); one token a stream chunk; `cached_tokens` 0 (the route has
no prefix reuse). Two things the client cannot get from this route yet:

- `ignore_eos`: memra-server has no such field, so the client runs with `--no-ignore-eos` and a
  fixed-length decode workload ends early if the model emits end-of-sentence (it did not here:
  every decode sample reached its budget).
- The concurrent workloads (`conc8`, `conc32`) were not run: at this route's rate one round of
  eight 512-token answers is about ten minutes of card time, and the chat gate already covers
  concurrent requests byte for byte.

## Behind the same layer: what an N-rank resident engine implements

The session, sampler, stop and encoding layer (`serve_loop`, `Stream`, `choose`, `prepare`, the
memory book) touches the engine through these calls only, so an engine that keeps their meaning
serves through it unchanged:

| call | what the route needs from it |
|---|---|
| `Runner::new(stm, geo, engram, devices, max_pos, n_layers, keep, chunk)` | a `tp::Group` of ranks on `MEMRA_DSV41_DEVICES`, positions up to the served context |
| `set_keep(trunk)` then the warm-up forward (resident arm of `bring_up`) | every layer's share and the head on the ranks and kept; an engine with an explicit loader calls it here instead, and a load that does not fit must fail by name |
| `seq_with_capacity(cap)` | an empty sequence whose state is first sized for `cap` positions |
| `forward(&mut seqs, &calls)` | one step: at most one multi-token call (a new sequence's whole prompt at position 0, the prefill program) and one single-token call per running sequence (the decode program); a call's numbers must not depend on the other calls in the step |
| `logits(&out, row, 1)` | the full-vocabulary fp32 row on the host (the route samples and penalizes on the host) |
| `offloaded()`, `grow_caches`, `make_room(need)`, `cache_stats()` | `offloaded()` false makes the route skip the cache; `make_room` must still return each rank's effective free bytes; `cache_stats` may be empty |
| `grp.cx[r].ops` through `experts::device_memory`, `pool_peak`, `pool_used`, `reset_peak` | per-rank free memory for the book's ceiling and the logged transient |
| `memra_engine::progress` | a stamp per layer pass, or the route's stall watch sees no progress on a long prefill |

What changes the gates, not the code:

- **Batched equals solo.** The route puts up to `MEMRA_MAX_SESSIONS` calls in one forward and
  relies on each running alone through each block. An engine that batches decode GEMMs across
  calls must pass the chat gate's `batched` cell (five requests at once, byte for byte as solo)
  or serve with `MEMRA_MAX_SESSIONS=1`.
- **The memory book** (`MemModel`) prices a session's state and a step's transient for one rank
  holding every head; on N ranks it over-books, which is safe and wastes admission room. A split
  engine that wants the room gives the book its per-rank shapes.
- **Greedy without a host row.** `choose` reads the whole logits row; an engine with a device
  argmax can answer a greedy, penalty-free step without the read-back by giving `Runner` that
  call and using it where `sample` and `pen` are both `None`.
- **Its own evidence.** The record here is one card with the experts in host memory. Another
  topology needs CheckpointParity with its native runner, `verify serve`, and the chat gate on
  its own binary before it claims a state.

## What is left

- Speed: CUDA graphs, batched decode GEMMs across calls (needs its own batched-vs-solo gate), the
  per-call host round trips, prefetching the next layer's experts, kernel tuning (a separate lane).
- DSpark (`mtp.*`) drafts on the route.
- The 4-card full-residency route (76.5 GB per rank, step 5) behind the same session layer: the
  route now picks that residency from the cards, but it ran on no hardware here.
- `ignore_eos` on the completions surface, which the shared benchmark client wants for
  fixed-length decode workloads.
- The cache's regrow after a long prefill should stop quietly when a slab does not fit yet (two
  logged allocation failures a long prompt today; the route serves through them).
- The DSv4 route's stream has the stop-string shape this route had before `87bbe3bfc8` (it ends
  the stream at text already sent). Not served here to confirm; it needs its own two-card check.
- Port the vendor's `build_compressed_token_map`, so `MEMRA_DSV41_TOKEN_MAP` is not needed.
- Continuation and prefix reuse (they would key on the whole carried state), `response_format`,
  vision, logprobs, the numeric budget on `/v1/responses`.

## Commands

```bash
# the residency against the decode lane's receipts (13 chat tapes forced, 5 greedy at the cache floor)
scripts/res_gate.sh
# CheckpointParity on the residency, then the serve gate on the same bundle, then the chat gates
scripts/serve_gate.sh all       # ckpt, verify, chat; each step one one-card lease, every server stopped in it
# the suites the change touches (build, clippy -D warnings, fmt; then the lib suites in a lease)
scripts/test_serve.sh all
# the shared benchmark client against the route (COMPAT=native for memra's own stream shape)
scripts/serve_gate.sh bench
# everything in order, then the step rates and the receipts gathered
scripts/final_all.sh
```

Every script reads the run box's paths, card and lease wrapper from the file `LANE_ENV` names.

## Files

- `scripts/`: `res_gate.sh`, `serve_gate.sh`, `chat_run.sh`, `chat_gate.py`, `bench_run.sh`,
  `dsv41-run-experts-host` (the checkpoint gate's runner wrapper), `test_serve.sh`, `rates.py`,
  `bank.sh`, `final_all.sh`, `pin_probe.py`.
- `receipts/`: `pin_probe.out`; `residency/` (the gate's output, both runs' logs, sha256 of every
  compared file on both sides); `verify-res/` (the checkpoint and serve gates' bundle of `87bbe3bfc8`: lock, parity
  receipt, gates, serve receipt and response, logs); `verify-0d074a7055/` (the same gates at `0d074a7055`, with the
  run, the binaries' sha256, the plan, the census and the tiny gate, the large tables gzipped);
  `verify-6371a9975b/` (the same gates at `6371a9975b`, `raw.sha256` for the gzipped tables);
  `verify-6371a9975b-reach/` (what main's commits merged after that pin reach in this route: the script and its output);
  `verify-0d074a7055-tiny/`, `verify-a05173ee27-tiny/` and `verify-6371a9975b-tiny/` (`memra model verify tiny` at each pin's tree: gates,
  the tiny gate receipt, the oracles gzipped, the log; the last two also have the gate test's log); `serve_gate_all.out` (the binaries' sha256
  and the three gate steps); `chat/` (`chat-gate.json`, the client and server logs, `rates.json`);
  `bench-client/` and `bench-client-native/` (the client's samples, summary and logs, the server
  log, for the OpenAI and memra's own stream shape); `tests/` (suite summaries); `leases.tsv` (one line per lease of the lane, the failed
  chat gate of 22:16 included).
