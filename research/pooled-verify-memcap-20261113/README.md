# A pooled MTP verify fits the device; evictable memory funds the cohort's verify (memra#1266)

Verdict: fixed (memra#1266). On the RTX 5090 Laptop GPU (Qwen3.8-27B NVFP4 + Q5_K, masked MTP
head, the 29-turn agentic replay at c = 2) the pooled MTP verify held every member's GDN
checkpoint at once (about 1.5 GB for K=5 + K=5) with nothing charging or checking it, and ran
out at layer 61 of 64 after both streams had sent: main failed requests 19 and 21 in 9 of 9
boots. A member now joins a pooled verify forward only while the whole forward (its members'
checkpoint columns and logits, the FA partial pool's growth priced launch by launch, one per
same-rung run of rows) and a 256 MiB margin fit the allocatable memory; the rest verify after
it, bitwise the same. Before the cohorts step, evictable memory (prefix entries first, then
parked sessions, reaping the graveyard after each) funds the largest member's round plus every
member's burst snapshot, on a reading without the unbuilt sessions' booking. Every fix boot
answered 29 of 29, texts equal to main's and to the memra#1221 part 2 head's; the forced-cap arm
writes the same texts. With a test door that pads the need, the reclaim ran on both cards (15
reclaims on the RTX 5090; 48 reclaims and 240 short passes on GB10) with texts equal to main's.
The rider cap counted 307 capped rider rows on GB10 in the forced-cap arm with texts equal to
main's; on the RTX 5090 a rider steps plain inside each round, so that path does not run there.
GB10 never reaches the cap in serving shape. At c = 4 the RTX 5090 decodes every session plain (no cohort forms at 4 in flight), and F6, its forced-cap arm and main answer 29 of 29 with equal texts; the seeded-invariance and exact-resend gates pass on F6. Pending: the cost bench.

Decision record:
[`docs/decisions/POOLED-VERIFY-MEMORY-CAP.md`](../../docs/decisions/POOLED-VERIFY-MEMORY-CAP.md).

## Binaries

| label | tree | card | binary sha256 (first 16) | build id |
|---|---|---|---|---|
| main | `11b368f7d7` | RTX 5090 (sm_120a) | `3f94cfb8f871c2c9` | `memra-0.141.1-32737cebc309` |
| fix, round 0 | `6433a5a264` on `11b368f7d7` (the first cut) | RTX 5090 (sm_120a) | `8f1b48774396f5ac` | `memra-0.141.1-4e861bdcb961` |
| fix, round 1 (F2) | the review-round-1 code (`fb417c4643`) on `11b368f7d7` | RTX 5090 (sm_120a) | `d8101f1c793e106f` | |
| main (M3) | main `a17add18f3`'s crates (built as `7a43ac1e56` with memra#1256's `worker.rs`, byte for byte the crates of `7a32d9b10b` and `a17add18f3`) | RTX 5090 (sm_120a) | `2e04b66a07877895` | |
| fix (F3) | the round-1 code on main's crates (`7d7f56b46a`) | RTX 5090 (sm_120a) | `4a1abee9fb32937f` | |
| fix (F4) | `e8a306389a`: F3 plus the numeric-scope guard in `spec_round_bytes` (it launches nothing); built, superseded by F5 before a cell ran on it | RTX 5090 (sm_120a) | `8cb259d1f85008b5` | |
| fix (F5) | the review-round-2 code (`0551668a5d`, on `a17add18f3`) | RTX 5090 (sm_120a) | `6980707df19deedf` | |
| fix (F6) | the review-round-3 code (`deb6114414`, on `202e1411fe`); the RTX 5090's c = 4 cells and gates; the head (`3cf70f3009`) adds only the booking bound priced on the stacked cohort, which no cell reaches | RTX 5090 (sm_120a) | `4006b4aa92925f8e` | |
| diagnostic | main + `harness/rig/diag_probe.py` (log lines only) | RTX 5090 (sm_120a) | `8e7abde6637cc294` | `memra-0.141.1-c200db12045d` |
| main | `11b368f7d7` | GB10 (sm_121a) | `2e1211ff9466a203` | `memra-0.141.1-32737cebc309` |
| fix, round 0 | as above | GB10 (sm_121a) | `b9c35baaee789aca` | `memra-0.141.1-4e861bdcb961` |
| M3, F3 | as above (`spec.rs`, `worker.rs` and server `lib.rs` hashed in the build log against the trees) | GB10 (sm_121a) | `4e262794c7bfd775` / `7df050125df70bf1` | |
| F5 | as above (`spec.rs`, engine `lib.rs`, `worker.rs` and server `lib.rs` hashed against the tree); F6 differs only in the booking bound, which reaches no session in these cells (none builds its state in the tick), and a test | GB10 (sm_121a) | `065153e432ae8ac8` | |
| bench | `decode-batch-bench` from the F6 tree | RTX 5090 (sm_120a) | `c6d6f98e4f9ca48d` | |

The RTX 5090 binaries were built on the CPU build host (`harness/build-host/dd_build.sh`; M3 by
`dd_build_m3.sh`), the GB10 ones on the unit (`harness/gb10/build_gb10*.sh`). Card conditions:
RTX 5090 Laptop GPU, driver 615.71.09, on AC with `nv_tgp` 150 and `ppt_pl1_spl` 140; the
enforced power limit read 150 or 155 W at each cell's start (`guard.log`) and was not changed.
GB10: one DGX Spark, driver 580.178.04.

Platform profile. Until 06:41:49 (local, 2026-11-13) the rig ran its balanced profile (the
enforced limit read 140 to 160 W under load). From then the owner's leases set it: performance
(175 W) for a timed lease (`gpulock --perf`), balanced otherwise. Every cell's `guard.log` reads
the profile at its start and its end, and the tables carry it with the limit under load. Timings
are compared only between cells that read the same profile at both ends; OOM counts and texts do
not depend on it. A timed take on the RTX 5090 counts only when no arm shows software thermal
slowdown in more than 5% of its 1 Hz samples under load and none starts at 86 C or hotter
(`harness/rig/take_check.py`); a take that fails is kept as `DIR.takeN`, the card rests 5 minutes
under the held lease, and the step runs again, up to 3 times (`harness/rig/timed_take.sh`).

The thermal regime, measured: under `--perf` at 175 W a sustained load holds this card at its
87 C temperature target. The first timed take at c = 4 (`agentic-5090-c4-perf-take/01-F.take1`,
started at 73 C) reached 86 C within about two minutes and peaked at 89 C; software thermal
slowdown was active on 352 of its 360 samples under load (98%), hardware thermal slowdown on
none. No rest brings a sustained cell under the rule's 5%, so the sustained cells here (the
agentic replays, the normal cells, the gates) run as correctness cells on a balanced lease: OOM
counts, texts and capped-forward counts, no timing claim. The one timed step is the cost bench
(short, pooled and serial verify of the same shapes in one process), under `--perf` and the take
rule.

## The shape

Qwen3.8-27B NVFP4 + Q5_K with the masked 32,768-row MTP head, the MTP speculative route, ctx
32,768, 8 sessions, prefix cache 4,000 MB, no host tier, `MEMRA_PREFIX_STABLE_BOUNDARY=1`,
`MEMRA_TICK_TRACE=1`; the 2026-11-05 decision cell's 29 agentic turns (1.4k to 17k prompt
tokens) through its driver (`research/c8-workspace-20261108/harness/agent_turns.py`): a prime
pass at max_tokens 1, then the timed pass at c = 2 (or 4), thinking on, temperature 0, seed
1234, 768 tokens. The workload derives from private agent sessions and is not here; rows carry
lengths, ids and text hashes only. `ref-1221b-head-texts.jsonl` holds the memra#1221 part 2 head
boots' text hashes (k, ok, text_sha; that lane's `rtxag3/02-ST` and `03-ST`), the only boots that
answered k = 19 and 21 without this change.

## The mechanism (`diag-5090/`)

One boot of the diagnostic arm reproduced the failure at the same point (27 of 29) and printed
the memory around it (`diag-5090/tl-01-D.txt`, times relative to the step OOM):

| when | line | driver free | pool cached | allocatable |
|---|---|---:|---:|---:|
| -0.553 s | admission gate: cost 900 MiB + reserve 1,536 MiB = 2,436 MiB, sufficient | 515 MiB | 2,143 MiB | 2,658 MiB |
| -0.548 s | driver-headroom rung released 1,241 MiB; restore 3,488 of 3,905 tokens | 1,699 MiB | 516 MiB | 2,215 MiB |
| -0.107 s | after the 417-row suffix prime; the cohort enters | 1,539 MiB | 376 MiB | 1,915 MiB |
| -0.016 s | first round gate: graph replay suspended (two burst snapshots, +299 MiB used) | 99 MiB | 1,517 MiB | 1,616 MiB |
| 0.000 s | `alloc-fail bytes=15728640` at the ssm checkpoint slab of `pooled_linear_layer`, layer 61 of 64, rows 12 | 67 MiB | 128 MiB | 195 MiB |

The pooled forward (two K=5 rounds, 12 rows) had allocated 1,453 MiB of checkpoint slabs when
the second member's 15 MiB slab failed. The prefix cache held 2,678 MiB, 2,226 MiB of it
unleased, and two parked spec sessions were resident; the step-OOM teardown evicted them after
both requests had failed. Across the boot's two-member pooled rounds the allocatable reading was
median 2,135, 10th percentile 1,723 and minimum 1,294 MiB (578 rounds). Graph replay
suspension reads the driver figure alone at the round gate, before the chunk walk; how the
1,616 MiB splits between driver free and pool cached there depends on the run's history (the fix
boots below read the same sum without the suspension), so that column is not an effect of the
cap.

## Review round 2: the final code (F5)

Correctness cells on a balanced lease (OOM counts, texts and the memory cap's counters; no
timing claim), the final code (F5, then F6 for the RTX 5090's c = 4 cells and gates) against M3,
alternating; the one timed step is the cost bench. No prompt of
these cells is in memra#1274's k x 4,096 + 1 to 15 band (the agentic replays' 29 prompts run
1,604 to 17,101 tokens; the exact-resend gate's 11,000 and 11,150), so landing 17's prime
change, which the rebase brings, does not reach them.

### Agentic replay, RTX 5090, c = 4 (`agentic-5090-c4-r2/`)

F6, F6 with `MEMRA_SPEC_POOL_SERIAL=1` and M3, one balanced lease:

| boot | ok | failed | tokens | OOM lines | not parked | capped forwards | capped rider rows | pre-cohort reclaims | short passes | fence-funded passes | graph suspended | profile; limit under load W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-F | 29 | - | 9396 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | balanced; 155 (155 to 155) |
| 02-S | 29 | - | 9396 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | balanced; 155 (155 to 155) |
| 03-M | 29 | - | 9396 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 | balanced; 155 (155 to 155) |

- 01-F: 29 answered texts equal 01-F's; differing k: none
- 02-S: 29 answered texts equal 01-F's; differing k: none
- 03-M: 29 answered texts equal 01-F's; differing k: none
- 01-F against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 02-S against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 03-M against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 01-F against memra#1221 part 2 head rtxag3/03-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 02-S against memra#1221 part 2 head rtxag3/03-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 03-M against memra#1221 part 2 head rtxag3/03-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}

Every tick of these boots logged `spec=0`: with four requests in flight this card decodes every
session plain (the LOW/HIGH band's HIGH is 4 without a cost table; GB10's cost table also prices
n = 4 at K = 0), so no cohort forms and neither the cap nor the reclaim has anything to do. The
band past admission's floor (the record) is therefore unreachable at c = 4 under today's
policies; the short count reads 0 for that reason.

### The pre-cohort reclaim on hardware (`pad-5090/`, `pad-gb10/`)

`MEMRA_SPEC_COHORT_NEED_PAD_MB` adds to the need so the ladder runs where the real need is
funded. RTX 5090, F5 with a 1,024 MiB pad, the agentic replay at c = 2 (also this round's c = 2
boot of the final code on this card: the earlier rounds' c = 2 boots below hold the 29 of 29
across every fix boot):

| boot | ok | failed | tokens | OOM lines | not parked | capped forwards | capped rider rows | pre-cohort reclaims | short passes | fence-funded passes | graph suspended | profile; limit under load W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-F | 29 | - | 9396 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 4 | balanced; 155 (155 to 155) |

- 01-F: 29 answered texts equal 01-F's; differing k: none
- 01-F against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 01-F against memra#1221 part 2 head rtxag3/03-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- reference ag3/01-M: 27 answered texts equal 01-F's; differing k: none; k=19/21 equal: {}

The reclaims evicted prefix entries (15 passes, no parked session, no pass left short), and with
that headroom every pooled forward fit: no capped forward in this boot.

GB10, F5: unified memory reads 87 to 91 GiB here, so a 1,024 MiB pad funds every pass (01-F);
100,000 MiB keeps every pass short (02-F), and the ladder evicts every prefix entry and parked
session it can:

| boot | ok | failed | tokens | OOM lines | not parked | capped forwards | capped rider rows | pre-cohort reclaims | short passes | fence-funded passes | graph suspended | profile; limit under load W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-F | 29 | - | 9665 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| 02-F | 29 | - | 9665 | 0 | 0 | 0 | 0 | 48 | 240 | 0 | 0 | n/a |

- 01-F: 29 answered texts equal 01-F's; differing k: none
- 02-F: 29 answered texts equal 01-F's; differing k: none
- reference gag5/01-M: 29 answered texts equal 01-F's; differing k: none; k=19/21 equal: {19: True, 21: True}

### Riders (`riders-5090/`, `riders-gb10/`, `riders-gb10-n6/`)

memra#1244's riders cell (`harness/rig/riders_cell_1266.py`): greedy streams, then
greedy-with-penalty plain requests. On the RTX 5090 a rider steps plain inside each round (the
card has no cost table for the model), so no rider reaches the pooled forward and the cap's rider
path does not run there; the texts still compare (1 stream + 2 riders, 2 + 1):

| boot | cell | streams + riders | rider errors | capped forwards | capped rider rows | pre-cohort reclaims | short passes | OOM lines |
|---|---|---|---|---|---|---|---|---|
| 01-M | 0 | 1 + 2 pen | 0 | n/a | n/a | n/a | n/a | 0 |
| 01-M | 1 | 2 + 1 pen | 0 | n/a | n/a | n/a | n/a | 0 |
| 02-S | 0 | 1 + 2 pen | 0 | 230 | 0 | 0 | 0 | 0 |
| 02-S | 1 | 2 + 1 pen | 0 | 230 | 0 | 0 | 0 | 0 |
| 03-F | 0 | 1 + 2 pen | 0 | 0 | 0 | 0 | 0 | 0 |
| 03-F | 1 | 2 + 1 pen | 0 | 0 | 0 | 0 | 0 | 0 |

- rider and solo texts: 6 of 6 identical across the 3 boots; differing: none
- long streams: 3 of 3 prefix-equal across the boots

On GB10 the cost table runs everyone plain from 4 in flight (`riders-gb10-n6/`: 2 streams + 4
riders, `n=4` and `n=6` priced K = 0), so the cells that reach the forward are 1 + 1, 2 + 1 and
1 + 2:

| boot | cell | streams + riders | rider errors | capped forwards | capped rider rows | pre-cohort reclaims | short passes | OOM lines |
|---|---|---|---|---|---|---|---|---|
| 01-M | 0 | 1 + 1 pen | 0 | n/a | n/a | n/a | n/a | 0 |
| 01-M | 1 | 2 + 1 pen | 0 | n/a | n/a | n/a | n/a | 0 |
| 01-M | 2 | 1 + 2 pen | 0 | n/a | n/a | n/a | n/a | 0 |
| 02-S | 0 | 1 + 1 pen | 0 | 201 | 307 | 0 | 0 | 0 |
| 02-S | 1 | 2 + 1 pen | 0 | 201 | 307 | 0 | 0 | 0 |
| 02-S | 2 | 1 + 2 pen | 0 | 201 | 307 | 0 | 0 | 0 |
| 03-F | 0 | 1 + 1 pen | 0 | 0 | 0 | 0 | 0 | 0 |
| 03-F | 1 | 2 + 1 pen | 0 | 0 | 0 | 0 | 0 | 0 |
| 03-F | 2 | 1 + 2 pen | 0 | 0 | 0 | 0 | 0 | 0 |

- rider and solo texts: 8 of 8 identical across the 3 boots; differing: none
- long streams: 4 of 4 prefix-equal across the boots

### Agentic replay, GB10, c = 2 (`agentic-gb10-c2-r2/`)

| boot | ok | failed | tokens | OOM lines | not parked | capped forwards | capped rider rows | pre-cohort reclaims | short passes | fence-funded passes | graph suspended | profile; limit under load W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-M | 29 | - | 9665 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 | n/a |
| 02-F | 29 | - | 9665 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| 03-F | 29 | - | 9665 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| 04-M | 29 | - | 9665 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

- 01-M: 29 answered texts equal 02-F's; differing k: none
- 02-F: 29 answered texts equal 02-F's; differing k: none
- 03-F: 29 answered texts equal 02-F's; differing k: none
- 04-M: 29 answered texts equal 02-F's; differing k: none

### Normal cells, GB10 (`normal-gb10-r2/`)

M3 F5 then F5 M3 (the RTX 5090's normal cells ran in rounds 0 and below):

| boot | c | requests, errors | completion tokens |
|---|---|---|---|
| 01-M | 1 | 6, 0 | 960 |
| 01-M | 2 | 12, 0 | 1920 |
| 01-M | 4 | 24, 0 | 3840 |
| 01-M | 8 | 48, 0 | 7680 |
| 02-F | 1 | 6, 0 | 960 |
| 02-F | 2 | 12, 0 | 1920 |
| 02-F | 4 | 24, 0 | 3840 |
| 02-F | 8 | 48, 0 | 7680 |
| 03-F | 1 | 6, 0 | 960 |
| 03-F | 2 | 12, 0 | 1920 |
| 03-F | 4 | 24, 0 | 3840 |
| 03-F | 8 | 48, 0 | 7680 |
| 04-M | 1 | 6, 0 | 960 |
| 04-M | 2 | 12, 0 | 1920 |
| 04-M | 4 | 24, 0 | 3840 |
| 04-M | 8 | 48, 0 | 7680 |

- c=1: one set of (prompt, text sha) across the boots
- c=2: one set of (prompt, text sha) across the boots
- c=4: one set of (prompt, text sha) across the boots
- c=8: one set of (prompt, text sha) across the boots
- 01-M: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; spec-pool] pooled MTP cohort engaged=3; profile; limit under load W: n/a
- 02-F: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; spec-pool] pooled MTP cohort engaged=3; profile; limit under load W: n/a
- 03-F: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; spec-pool] pooled MTP cohort engaged=3; profile; limit under load W: n/a
- 04-M: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; spec-pool] pooled MTP cohort engaged=3; profile; limit under load W: n/a

## Agentic replay, RTX 5090, c = 2: earlier rounds

**Round 1 on main's crates (`agentic-5090-c2-f3/`):** M3 against F3, and S = F3 with
`MEMRA_SPEC_POOL_SERIAL=1`. 03-S changed profile in its last minute and 04-F ran at 87 C.

| boot | ok | failed | tokens | OOM lines | not parked | capped forwards | capped rider rows | pre-cohort reclaims | short passes | fence-funded passes | graph suspended | profile; limit under load W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-M | 27 | 19,21 | 8897 | 2 | 2 | n/a | n/a | n/a | n/a | n/a | 2 | balanced; 157 (140 to 160) |
| 02-F | 29 | - | 9396 | 0 | 0 | 23 | 0 | 0 | 0 | n/a | 0 | balanced; 156 (140 to 160) |
| 03-S | 29 | - | 9396 | 0 | 0 | 1457 | 0 | 0 | 0 | n/a | 0 | balanced, then performance from 06:41:49; 155 (155 to 175) |
| 04-F | 29 | - | 9396 | 0 | 0 | 23 | 0 | 0 | 0 | n/a | 0 | performance at start (end not read); 175 (149 to 175) |

- 01-M: 27 answered texts equal 02-F's; differing k: none
- 02-F: 29 answered texts equal 02-F's; differing k: none
- 03-S: 29 answered texts equal 02-F's; differing k: none
- 04-F: 29 answered texts equal 02-F's; differing k: none
- 01-M against memra#1221 part 2 head rtxag3/02-ST: 27 of 27 texts equal; k=19/21 equal: {}
- 02-F against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 03-S against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 04-F against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 01-M against memra#1221 part 2 head rtxag3/03-ST: 27 of 27 texts equal; k=19/21 equal: {}
- 02-F against memra#1221 part 2 head rtxag3/03-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 03-S against memra#1221 part 2 head rtxag3/03-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 04-F against memra#1221 part 2 head rtxag3/03-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}

Every main boot fails requests 19 and 21 with the step OOM; every fix boot answers all 29, with
23 capped forwards and no pre-cohort reclaim, and its texts equal main's 27 and the memra#1221
part 2 head boots' 29 (k = 19 and 21 included). The forced-cap arm (03-S: every member verifies
alone, 1,457 capped forwards) writes the same 29 texts as the default arm: a capped request
writes its uncapped text. These boots make no timing claim (01-M and 02-F ran balanced, 03-S
changed profile in its last minute, 04-F ran at 87 C).

**Earlier rounds.** Round 0 (`agentic-5090-c2/`, main against the first cut, M F F M M F) and
round 1 before memra#1256 landed (`agentic-5090-c2-r1/`, F2; paused for the landing after two
boots):

Round 0 (`agentic-5090-c2/`):

| boot | ok | failed | tokens | agg gen tok/s | per-user median tok/s | TTFT p50 s | OOM lines | not parked | capped forwards | pre-cohort reclaims | graph suspended |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-M | 27 | 19,21 | 8897 | 42.1 | 48.24 | 2.96 | 2 | 2 | n/a | n/a | 2 |
| 02-F | 29 | - | 9396 | 45.11 | 48.93 | 2.813 | 0 | 0 | 23 | 0 | 0 |
| 03-F | 29 | - | 9396 | 44.03 | 48.04 | 2.949 | 0 | 0 | 23 | 0 | 0 |
| 04-M | 27 | 19,21 | 8897 | 42.61 | 47.97 | 2.926 | 2 | 2 | n/a | n/a | 2 |
| 05-M | 27 | 19,21 | 8897 | 42.65 | 48.3 | 2.859 | 2 | 2 | n/a | n/a | 2 |
| 06-F | 29 | - | 9396 | 44.19 | 47.73 | 2.868 | 0 | 0 | 23 | 0 | 0 |

- 01-M against memra#1221 part 2 head rtxag3/02-ST: 27 of 27 texts equal; k=19/21 equal: {}
- 02-F against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 03-F against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 04-M against memra#1221 part 2 head rtxag3/02-ST: 27 of 27 texts equal; k=19/21 equal: {}
- 05-M against memra#1221 part 2 head rtxag3/02-ST: 27 of 27 texts equal; k=19/21 equal: {}
- 06-F against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}

Round 1 before the pause (`agentic-5090-c2-r1/`):

| boot | ok | failed | tokens | agg gen tok/s | per-user median tok/s | TTFT p50 s | OOM lines | not parked | capped forwards | pre-cohort reclaims | graph suspended |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-M | 27 | 19,21 | 8897 | 36.0 | 41.48 | 3.503 | 2 | 2 | n/a | n/a | 2 |
| 02-F | 29 | - | 9396 | 37.98 | 40.99 | 3.471 | 0 | 0 | 23 | 0 | 0 |

- 01-M against memra#1221 part 2 head rtxag3/02-ST: 27 of 27 texts equal; k=19/21 equal: {}
- 02-F against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 01-M against memra#1221 part 2 head rtxag3/03-ST: 27 of 27 texts equal; k=19/21 equal: {}
- 02-F against memra#1221 part 2 head rtxag3/03-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}

## Agentic replay, RTX 5090, c = 4: round 0 (`agentic-5090-c4/`)

| boot | ok | failed | tokens | agg gen tok/s | per-user median tok/s | TTFT p50 s | OOM lines | not parked | capped forwards | pre-cohort reclaims | graph suspended | profile; limit under load W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-M | 29 | - | 9396 | 47.3 | 14.6 | 3.38 | 0 | 0 | n/a | n/a | 0 | balanced; 155 (155 to 155) |

- 01-M against memra#1221 part 2 head rtxag3/02-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}
- 01-M against memra#1221 part 2 head rtxag3/03-ST: 29 of 29 texts equal; k=19/21 equal: {19: True, 21: True}

Round 0's main boot at c = 4 (`agentic-5090-c4/01-M`) answered all 29 with no OOM and texts equal
to the head boots'. The final code's c = 4 boots run as `agentic-5090-c4-r2/` (F6, F6 forced cap,
M3) on a balanced lease, under "Review round 2" above.

## Agentic replay, GB10, c = 2: earlier rounds (`agentic-gb10-c2/`, `agentic-gb10-c2-final/`)

Same shape on the GB10 launcher's model files (`harness/gb10/serve.sh`).

Round 0 (`agentic-gb10-c2/`):

| boot | ok | failed | tokens | agg gen tok/s | per-user median tok/s | TTFT p50 s | OOM lines | not parked | capped forwards | pre-cohort reclaims | graph suspended | profile; limit under load W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-M | 29 | - | 9665 | 33.34 | 23.59 | 2.403 | 0 | 0 | n/a | n/a | 0 | n/a |
| 02-F | 29 | - | 9665 | 33.32 | 23.64 | 2.404 | 0 | 0 | 0 | 0 | 0 | n/a |
| 03-F | 29 | - | 9665 | 33.33 | 23.62 | 2.409 | 0 | 0 | 0 | 0 | 0 | n/a |
| 04-M | 29 | - | 9665 | 33.29 | 23.61 | 2.404 | 0 | 0 | n/a | n/a | 0 | n/a |

- 01-M: 29 answered texts equal 02-F's; differing k: none
- 02-F: 29 answered texts equal 02-F's; differing k: none
- 03-F: 29 answered texts equal 02-F's; differing k: none
- 04-M: 29 answered texts equal 02-F's; differing k: none

Final code (`agentic-gb10-c2-final/`, M3 F3 F3 M3; the boots that have finished):

| boot | ok | failed | tokens | agg gen tok/s | per-user median tok/s | TTFT p50 s | OOM lines | not parked | capped forwards | pre-cohort reclaims | graph suspended | profile; limit under load W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01-M | 29 | - | 9665 | 33.51 | 23.86 | 2.402 | 0 | 0 | n/a | n/a | 0 | n/a |

GB10 never comes near the cap (unified memory, the allocatable reading counts `MemAvailable`):
0 capped forwards and 0 reclaims, 29 of 29 in every boot, one text per request across the
boots (gag3 01-M equals round 0's 02-F on all 29). (The GB10 texts differ from the RTX 5090's:
another build target.)

## Normal-concurrency cells, round 0 (`normal-5090/`, `normal-gb10/`)

The cell #1173 and #1255 used (`research/drafter-admission-20261112/harness/normal_cell.py`:
closed-loop clients at c = 1, 2, 4, 8, 6 requests per client, greedy chat, 160 new tokens,
streamed; ctx 8,192, 8 sessions), on the MTP route this change acts on (the masked MTP head, no
DFlash drafter). Two boots per lease, pairs in both orders.

**RTX 5090, round 0** (`normal-5090/`: main against the first cut, M F then F M, balanced
profile):

| boot | c | requests, errors | completion tokens | agg tok/s | TTFT p50 s | ITL p50 / p99 ms |
|---|---|---|---|---|---|---|
| 01-M | 1 | 6, 0 | 960 | 79.77 | 0.163 | 0.0 / 33.0 |
| 01-M | 2 | 12, 0 | 1920 | 94.77 | 0.223 | 0.0 / 87.1 |
| 01-M | 4 | 24, 0 | 3840 | 119.93 | 0.611 | 28.0 / 33.5 |
| 01-M | 8 | 48, 0 | 7680 | 152.18 | 1.294 | 41.5 / 52.5 |
| 02-F | 1 | 6, 0 | 960 | 79.83 | 0.163 | 0.0 / 33.2 |
| 02-F | 2 | 12, 0 | 1920 | 95.06 | 0.223 | 0.0 / 86.2 |
| 02-F | 4 | 24, 0 | 3840 | 119.46 | 0.618 | 28.0 / 33.9 |
| 02-F | 8 | 48, 0 | 7680 | 149.49 | 1.176 | 42.2 / 54.1 |
| 03-F | 1 | 6, 0 | 960 | 80.99 | 0.161 | 0.0 / 32.7 |
| 03-F | 2 | 12, 0 | 1920 | 95.59 | 0.22 | 0.0 / 85.1 |
| 03-F | 4 | 24, 0 | 3840 | 119.35 | 0.538 | 28.0 / 36.6 |
| 03-F | 8 | 48, 0 | 7680 | 146.5 | 1.412 | 42.8 / 54.7 |
| 04-M | 1 | 6, 0 | 960 | 79.86 | 0.163 | 0.0 / 33.1 |
| 04-M | 2 | 12, 0 | 1920 | 94.95 | 0.222 | 0.0 / 85.8 |
| 04-M | 4 | 24, 0 | 3840 | 119.78 | 0.521 | 28.0 / 33.9 |
| 04-M | 8 | 48, 0 | 7680 | 150.53 | 1.275 | 41.6 / 53.6 |

- c=1: one set of (prompt, text sha) across the boots
- c=2: one set of (prompt, text sha) across the boots
- c=4: one set of (prompt, text sha) across the boots
- c=8: one set of (prompt, text sha) across the boots
- 01-M: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; graph replay suspended=0; spec-pool] pooled MTP cohort engaged=1; profile; limit under load W: balanced; 155 (155 to 155)
- 02-F: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; graph replay suspended=0; spec-pool] pooled MTP cohort engaged=1; profile; limit under load W: balanced; 155 (155 to 155)
- 03-F: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; graph replay suspended=0; spec-pool] pooled MTP cohort engaged=1; profile; limit under load W: balanced; 155 (155 to 155)
- 04-M: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; graph replay suspended=0; spec-pool] pooled MTP cohort engaged=1; profile; limit under load W: balanced; 155 (155 to 155)

- c=1 agg tok/s: F 80.41 (79.83, 80.99), M 79.81 (79.77, 79.86)
- c=2 agg tok/s: F 95.33 (95.06, 95.59), M 94.86 (94.77, 94.95)
- c=4 agg tok/s: F 119.41 (119.46, 119.35), M 119.86 (119.93, 119.78)
- c=8 agg tok/s: F 148.00 (149.49, 146.5), M 151.36 (152.18, 150.53)

One text set per c across the four boots, no OOM, no capped forward and no reclaim in any boot.
c = 1 to 4 are level. At c = 8 the first cut read 149.49 and 146.50 tok/s against main's 152.18
and 150.53, 1.8% and 2.7% lower, one pair in each order (balanced profile). The plain c = 8 path
runs the same code in both binaries (no forward was capped in either boot), and the fix boots drew
about 140 W against 150 W at the same limit. Unresolved: the pairs that would settle it were
dropped, because a sustained cell on this card cannot meet the take rule (under `--perf` a
sustained load holds the card at its 87 C target with software thermal slowdown on 98% of the
samples), and balanced-profile timings on this card move with the draw. The cost bench below
prices a capped round directly.

**GB10, round 0** (`normal-gb10/`):

| boot | c | requests, errors | completion tokens | agg tok/s | TTFT p50 s | ITL p50 / p99 ms |
|---|---|---|---|---|---|---|
| 01-M | 1 | 6, 0 | 960 | 28.19 | 0.291 | 0.0 / 90.6 |
| 01-M | 2 | 12, 0 | 1920 | 43.27 | 0.38 | 0.0 / 269.5 |
| 01-M | 4 | 24, 0 | 3840 | 61.62 | 0.423 | 0.0 / 520.0 |
| 01-M | 8 | 48, 0 | 7680 | 79.98 | 1.406 | 87.3 / 102.3 |
| 02-F | 1 | 6, 0 | 960 | 28.19 | 0.291 | 0.0 / 91.1 |
| 02-F | 2 | 12, 0 | 1920 | 43.23 | 0.38 | 0.0 / 268.6 |
| 02-F | 4 | 24, 0 | 3840 | 61.73 | 0.423 | 0.0 / 521.1 |
| 02-F | 8 | 48, 0 | 7680 | 79.9 | 1.409 | 87.4 / 102.5 |

- c=1: one set of (prompt, text sha) across the boots
- c=2: one set of (prompt, text sha) across the boots
- c=4: one set of (prompt, text sha) across the boots
- c=8: one set of (prompt, text sha) across the boots
- 01-M: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; spec-pool] pooled MTP cohort engaged=3; profile; limit under load W: n/a
- 02-F: out of memory=0; NOT parked=0; pooled MTP verify capped=0; pre-cohort reclaim=0; spec-pool] pooled MTP cohort engaged=3; profile; limit under load W: n/a

- c=1 agg tok/s: F 28.19 (28.19), M 28.19 (28.19)
- c=2 agg tok/s: F 43.23 (43.23), M 43.27 (43.27)
- c=4 agg tok/s: F 61.73 (61.73), M 61.62 (61.62)
- c=8 agg tok/s: F 79.90 (79.9), M 79.98 (79.98)

Level at every c, one text set per c, 0 capped forwards. The final code's pairs on GB10 are under
"Review round 2" (`normal-gb10-r2/`).

## What a capped round costs

`decode-batch-bench --pooled` at the shapes the cap splits (6,6 / 6,3 / 3,3 / 3,3,3,3 / 4,4,4,4 /
4,4; depth 4,096, 20 reps; pooled and serial verifies bitwise equal first), from the F6 tree,
runs on the RTX 5090 under `--perf` and the take rule after the queued jobs; this section is
updated when it finishes.

## Gates

On F6 (`gates-5090/`, balanced lease, in block 1):

- seeded invariance (`tools/seeded-invariance-gate.py`, the device-penalised arm): `SEEDED INVARIANCE: PASS (54 rows, 10 prompts, cases ref,0,1,2,3,demote)`;
- exact resend (`tools/prefix-exact-resend-gate.py`): `PREFIX-EXACT-RESEND: p1=11000 p2=11150 grid=32 capture_p1=10976 resend_ok=2/2 growth_ok=2/2 identical=9/9 tails=4 covered_ok=4/4 V1=ok V2=ok V3=ok V4=ok V5=ok -> PASS`.

## CPU (`cpu/`)

The head (`cpu/full-chain-head.txt`, `cpu/green-head.txt`, `cpu/red-head/arms.txt`): `3cf70f3009`, the
fold-check code on main `202e1411fe`, on the CPU build host with the whole `research/` tree the suites read at
run time. `cargo fmt --check`, `cargo clippy --release --all-targets -D warnings`, the coalescer
contract, memra-server 1,243 passed / 0 failed, memra-kv 87 / 0, memra-engine lib 757 / 0,
memra-gguf lib 418 / 0, the portable suites 448 passed / 0 skipped / 0 failed;
`check-numeric-scope` passes on every entry site.

Red arms (`harness/build-host/red_arms.py`, `dd_red.sh`): each arm changes one thing and the named
test fails (cargo exit 101 with the test's own panic). The census arms use main's files
(`7a32d9b10b`, the crates of `202e1411fe` for these files).

| arm | what it changes | the test that fails |
|---|---|---|
| `accumulation-off` | `pooled_verify_chunk` forgets the members it took (`out.held += c.need` deleted) | `the_walk_joins_members_only_within_the_reading` (K=5 + K=5 at 1,616 MiB pools again), with the FA, rider and readiness tests |
| `fa-growth-off` | the walk leaves the FA partial pool's growth out | `the_fa_pool_growth_counts_in_the_forward` |
| `fa-deepest-rung` | `fa_rows_growth` prices one launch of every row at the deepest row's rung (round 1's price) | the straddle tests at 16,384 and 512 |
| `growth-max-not-sum` | `fa_rows_growth` keeps the largest grow, not the sum | `two_runs_that_both_grow_the_pool_add_up` |
| `rider-cap-off` | `pooled_rider_take` takes every rider | `riders_join_only_within_the_reading` |
| `rider-depth-off` | riders priced at the chunk's deepest row | `a_rider_is_priced_at_its_own_depth` |
| `margin-low` | `POOLED_VERIFY_MARGIN` at 128 MiB, under the measured 195 MiB | `the_margin_exceeds_what_the_pool_could_not_deliver` |
| `main-engine` | main's `spec.rs` with the lockstep census appended | `the_mtp_lockstep_chunk_walk_applies_the_memory_cap` |
| `snapshots-off` | `pre_cohort_need` leaves the burst snapshots out | `the_need_covers_the_round_and_every_burst_snapshot` |
| `parked-first` | the ladder runs the parked rung before the prefix rung | `the_ladder_evicts_prefix_first_then_parked_sessions` |
| `unreadable-acts` | the ladder reads an unreadable device as 0 | `an_unreadable_device_does_nothing` |
| `fence-funded-off` | the ladder does not mark a pass the fence funded | `a_reading_the_fence_funds_evicts_nothing` |
| `booking-off` | the reclaim's reading keeps the unbuilt booking | `the_reclaim_reads_without_the_unbuilt_booking` |
| `booking-unbounded` | the reclaim leaves the whole unbuilt booking out, not just the FA growth it overlaps | `only_the_fa_growth_overlaps_the_unbuilt_booking` |
| `cohort-rows-first-only` | `cohort_verify_rows` stacks only the first member (the booking bound priced on one round) | `a_cohort_stacked_grows_the_pool_where_no_round_alone_does` |
| `reap-off` | the parked rung does not reap after an eviction | `vmm_pending_releases_keep_the_idle_wait_polling_and_the_reclaims_reap` (a census: the reap does work only under on-demand KV planes, the RTX PRO 6000 default) |
| `main-worker` | main's `worker.rs`, server `lib.rs` and engine `spec.rs` with the Cohorts-pass census appended | `the_cohorts_pass_reclaims_before_it_forms_cohorts` |

Earlier heads: `cpu/full-chain-final.txt` and `cpu/red-final/` hold the review-round-2 code at
`0551668a5d` (14 arms, all exit 101; one memra-gguf failure in that chain, the per-process
temp-config race main fixed in landing 17, passed three reruns); `cpu/full-chain-rebased.txt` holds
it rebased onto `202e1411fe`. The review-round-3 code (`deb6114414`) ran the full chain and 16 arms
green and red as expected before the fold check added the 17th. `cpu/red/` and `cpu/red-r1/` hold
rounds 0 and 1, `cpu/full-chain.txt` round 0.

## What is here

| path | what |
|---|---|
| `diag-5090/01-D/`, `diag-5090/tl-01-D.txt` | the diagnostic boot (server log with the `[p1266]` lines, per-request rows, telemetry, `guard.log`) and its timeline |
| `agentic-5090-c2*/<boot>/`, `agentic-5090-c4*/<boot>/`, `agentic-gb10-c2*/<boot>/`, `pad-5090/<boot>/`, `pad-gb10/<boot>/` | `cells.jsonl`, per-request rows, `summary.txt` (line counts), `/metrics`, the server log, telemetry, `guard.log`, `run.log` (`take.json`: the take rule's verdict, `agentic-5090-c4-perf-take/`) |
| `riders-5090/<boot>/`, `riders-gb10/<boot>/`, `riders-gb10-n6/<boot>/` | the riders cell: `riders.jsonl` (per rider, solo re-run and stream: tokens, gaps, text sha), `texts/`, `metrics-<boot>.json`, the server log, `guard.log`, `summary.txt` |
| `normal-5090*/<boot>/`, `normal-gb10*/<boot>/` | `normal.jsonl` (per request: tokens, TTFT, inter-token gaps, text sha), the server log, telemetry, `guard.log`, `summary.txt` |
| `ref-1221b-head-texts.jsonl` | the memra#1221 part 2 head boots' text hashes |
| `gates-5090/` | the seeded-invariance and exact-resend gate runs on F6 (logs, `.guard` with the profile at start and end, each run's directory as a tarball) |
| `cpu/` | the red arms (`red/` round 0, `red-r1/` round 1, `red-final/` the review-round-2 code's 14, `red-head/` the head's), the new tests' green runs, the full CPU chains (`full-chain.txt` round 0, `full-chain-final.txt` the review-round-2 code, `full-chain-rebased.txt` it rebased onto `202e1411fe`, `full-chain-head.txt` the head) |
| `harness/rig/`, `harness/gb10/`, `harness/build-host/` | the scripts that ran all of it, with the step files; paths and hosts are written generically (`harness/rig/sanitize_receipts.py`) |
