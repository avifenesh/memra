# GU M1 plus packed half2, sampled plain model gate

2026-09-07. KEEP for controlled composition, not a serving default.

The conjunction of the existing process-local GU-M1 and GU-half2 setters
now selects `moe_kq_sktail_gu_kernel<108,true,true>` in the already eligible
single-row plain transaction. Both EP workspaces use the same dispatch.
No new environment flag is added. Batched/speculative/prefill transactions
remain outside this visitor. One combined enqueue advances the existing
GU-M1 and GU-half2 counters once each.

The full-model gate holds packed GU/down half2, grouped wo_a, radix index
top-k, whole-expert EP, and all other settings fixed. Only GU-M1 changes.
It restores a frozen prompt state for each row, samples at T=1, p=1, k=0,
seed=20260906, and generates 256 output tokens. Three ABBA cycles give six
scored rows per arm at each prompt length; warmups are excluded.

| Prompt | Baseline tok/s | Combined tok/s | Within-window gain |
| --- | ---: | ---: | ---: |
| 256 | 33.128051 | 33.537683 | +1.2365% |
| 8192 | 30.980860 | 31.329138 | +1.1242% |

All 28 rows pass full token, final-logit, committed-KV and state-position
identity, with no loops/EOS exclusions. Each 255-step row has 10,965 EP
transactions; combined GU-M1 enqueues are 21,930 in tuned rows and zero
in controls. GU-half2/down-half2 each have 21,930 enqueues in both arms.
Grouped wo_a has 10,965; radix top-k has 5,355 at 8K and zero at 256.
Graph captures/replays are zero in both arms. This is native sampled plain
decode, not HTTP throughput, speculative throughput, or a 120 tok/s result.

Binary SHA256:
`544cbc8cf2f0c98c36bc5533f024fb561f6e613b8f25a086044f23bf9a390986`.
Raw model-log SHA256:
`9fa6f0d27dce84ae425ac006905612cf97b191c2b3cbc84f5b3bef281ea811cb`.
Source corpus SHA256:
`f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded`.
Controller terminal: 2026-09-07T07:54:13Z, exit 0. The private ops receipt
namespace is `gu-m1-half2-model-20260907-r1`; it retains the model log,
telemetry, process ownership log, and validated schema-2 summary.

CPU validation: 413 engine-lib tests pass, 15 ignored; clippy for engine
lib and `dsv4_plain_perf_gate` passes with `-D warnings`. Ops reader tests:
29 pass. The paired R7 component H identity/memcheck receipt precedes this
model test. The candidate remains default OFF with the existing rollback
setters and 2026-09-21 decision date; no public serving claim is changed.
