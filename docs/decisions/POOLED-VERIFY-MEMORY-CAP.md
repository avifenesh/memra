# A pooled MTP verify forward fits the device; the largest member's round is funded before the cohorts step (2026-11-13)

Date: 2026-11-13. Issue memra#1266, branch `fix/1266-pooled-verify-oom`, receipts
`research/pooled-verify-memcap-20261113/`. Cards: one RTX 5090 Laptop GPU (24 GB; enforced power
limit 150 W, recorded per run and not changed) and one DGX Spark (GB10). Model: Qwen3.8-27B
NVFP4 + Q5_K with its masked 32,768-row MTP head.

## Problem

On the RTX 5090 Laptop the 29-turn agentic replay at c = 2 on the MTP route (ctx 32,768, 8
sessions, prefix cache 4,000 MB, no host tier, `MEMRA_PREFIX_STABLE_BOUNDARY=1`, thinking on,
temperature 0, seed 1234, 768 tokens) ended two requests with a CUDA out-of-memory in the
cohort's pooled verify, at the same point in every main boot: the warm admission of the
3,905-token prompt (k=19) beside the decoding k=21. Both sessions had streamed tokens, so
neither parked, and both requests failed while 2,226 MiB of unleased prefix entries and two
parked spec sessions stayed resident until the teardown.

A diagnostic arm (main plus logging only) named the failing allocation and the memory around it:

| step | driver free | pool cached | allocatable (sum) |
|---|---:|---:|---:|
| admission gate (charged cost 900 MiB + reserve 1,536 MiB = 2,436) | 515 MiB | 2,143 MiB | 2,658 MiB |
| after the driver-headroom rung (released 1,241 MiB of cached blocks) | 1,699 MiB | 516 MiB | 2,215 MiB |
| after the newcomer's 417-row suffix prime | 1,539 MiB | 376 MiB | 1,915 MiB |
| the cohort's first round, before its verify | 99 MiB | 1,517 MiB | 1,616 MiB |
| the failure | 67 MiB | 128 MiB | 195 MiB |

The failing allocation was 15,728,640 B, the second member's GDN ssm checkpoint slab (5 columns
of 3 MiB) at layer 61 of 64, in `pooled_linear_layer`. The forward pooled two K=5 rounds (12
rows) and had allocated 1,453 MiB of checkpoint slabs by then. Each member's verify keeps its
replay-free checkpoint, `rows - 1` columns of every linear layer's GDN state (156.9 MB a column
on this model), until its round finishes, so a pooled forward holds every member's checkpoint
at once: about 1.5 GB for K=5 + K=5.

Admission held its own arithmetic: after the newcomer's charge the allocatable reading stayed
above the 1,536 MiB transient floor. That floor funds one serial round. Nothing charged or
checked the extra members of a pooled forward. The DFlash2 cohort has such a check
(`pooled_verify_headroom`, below); the MTP lockstep (`generate_spec_sessions_unified`) had none.

Graph replay suspension was not the cause. On this dense model the MTP verify-graph pool is off
(`vgraph_family_default` needs linear layers and a routed MoE), so every cohort round is
poolable and runs the eager pooled verify whether or not replay is suspended; the suspension
(driver free 99 MiB while 1,517 MiB sat in the pool's cache) switched only the draft arms.

## Chosen

- **The memory cap.** The lockstep's verify chunk walk reads the allocatable memory once per
  forward (`pooled_verify_allocatable`: the driver's free bytes, or on unified memory the larger
  of them and `MemAvailable`, plus the async pool's cached blocks) and joins a member only while
  the whole forward fits it less the 256 MiB graph-launch floor (`pooled_verify_admits`). A
  member's need is its checkpoint columns plus two logits stacks (`pooled_verify_round_bytes`);
  plain riders join within the same cap. The first member always joins: alone it is the serial
  round. The rest verify in the next chunk, after this one's rounds have finished and freed
  their checkpoints, so the checkpoints live at once never exceed what the device holds. Pooled
  and serial verifies are bitwise equal (`decode-batch-bench --pooled`), so the cap moves no
  token. `[spec-pool] pooled MTP verify capped by memory: ...` (the first 16, then every 256th);
  `/metrics` `spec_pool_capped_forwards_total`.
- **The pre-cohort reclaim.** After the newcomers primed and before the cohorts form, when the
  allocatable reading cannot fund the largest cohort member's verify round
  (`spec_round_transient_bytes(k)`, about 790 MB at K=5) plus the floor, read again after a
  fence, unleased prefix entries go first, oldest first, until the shortfall is freed and
  settled (`prefix_yield_transient_reserve`), then parked sessions, globally oldest first.
  `[spec-pool] pre-cohort reclaim: ...`; `/metrics` `spec_cohort_reclaims_total`; the prefix
  evictions also count in `prefix_headroom_evictions`.
- **A mid-forward OOM still takes the honest error.** By then the forward has appended KV rows
  and advanced each member's GDN state in place for the layers it ran, so the round cannot be
  run again on that state, and a streamed session cannot replay. The fix keeps the forward from
  starting what it cannot finish.

## Rejected

- **DFlash2's rule for the MTP cohort** (`pooled_verify_headroom`: the extra members fit the
  reading less the 1,536 MiB transient floor, `sum - max <= free + cached - floor`). It keeps the
  floor unspent during the forward, but nothing else allocates during the forward: the newcomers
  primed before it and the prefix publications run after the tick's serving work. In this
  replay's two-member pooled rounds the allocatable reading was median 2,135, 10th percentile
  1,723 and minimum 1,294 MiB (578 rounds); at 1,723 MiB the DFlash2 rule would serialize a
  K=5 + K=2 round (extra about 320 MB against 187 MiB above the floor) that fits with about
  600 MiB to spare. The DFlash2 cohort keeps its rule; whether it should read the allocatable
  figure on unified memory and drop the floor is that route's call.
- **A per-session admission charge for the pooled transient.** The checkpoint of a pooled
  forward is a per-forward transient, not a per-session resident; charging each MTP session
  K columns (about 785 MB at K=5) at admission would cut c = 8 capacity on the 24 GB card to pay
  for an opportunistic speedup the cap already makes safe.
- **Evicting prefix entries so the cohort stays pooled.** It trades later cache hits for verify
  speed with no measurement behind the trade. The reclaim funds the serial round only.
- **Retrying the step after the OOM.** The state is already advanced (above).

## Measured

See `research/pooled-verify-memcap-20261113/README.md` for every table, binary and run.
