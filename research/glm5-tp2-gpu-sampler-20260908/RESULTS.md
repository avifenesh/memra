# GLM TP-2 device sampler qualification

Status: remote build, fmt, clippy, CPU suites and GPU sampler oracles PASS.
All five TP-2 p32k OFF/ON pairs, both p128k rows and the 160-token greedy twins
completed across an interrupted and resumed test run. No default promotion.
Base: `001c09e5d` (v0.135.0), including the existing GLM host prefix-radix sampler. The historical v0.132.0 comparison-sort measurements
are motivation, not this change's OFF baseline.

## Program and ownership

- `HybridModel::decode_step_hyper_output` shares the existing trunk and head
  arithmetic across the host and device arms. The device arm retains the full
  root-engine head row in `Cache::last_logits_dev`; the cache position advances
  exactly once. Eligibility requires loaded GLM TP-2 sidecars, HyperConnections,
  no pipeline stages, and a plain request.
- `Glm5TpDeviceSampler` reuses `Dsv4DeviceSampler::sample_ptr`, including its
  existing temperature/top-k/top-p CUDA kernels. There is no new CUDA kernel or
  external dependency. Sampling drains the root stream and reads one u32.
- `worker::sample_glm5_tp_plain` selects the token at the same worker boundary as
  host sampling. The existing accept, EOS, grammar, detokenization, stop, budget,
  context and event accounting remain in their original order.
- Prefill and restored boundary rows still arrive on the host. The ON arm uploads
  that row once and uses the same device sampler. Subsequent decode rows stay on
  device. Retire-time reuse-pool parking retains its existing single full-row D2H.

## RNG and greedy oracle

The ON sampled arm uses the existing DSV4 position-keyed SplitMix64 mapping from
the request seed and absolute cache position. Its numeric class is
`device-f64-exp-tree-cdf-v1`: logit/ID ordering, f64 exponentials, block prefix scan
and an unnormalized CDF. The host arm uses f32 scaling/softmax, its original tie
ordering and stateful SplitMix64 draws. Sampled token identity across arms is not
claimed; near-boundary nucleus membership can differ with arithmetic rounding.
The test checks ordinary finite rows away from a deliberately constructed cutoff.

Temperature <= 0 or top-k 1 selects `Engine::argmax_token_device_into`, the existing
two-pass raw-logit argmax with lowest-ID tie break. No temperature multiplication,
softmax or RNG enters that arm. The trunk, collapse, output norm and lm_head are
shared with the host path. For a finite row, this selects the same u32 as the
host raw argmax. All-invalid rows that return an invalid device ID are recovered
through the host path. The served 160-token greedy SHA twins also passed, as recorded below.

## Refusals and rollback

`MEMRA_GLM5_TP_DEVICE_SAMPLE` defaults to OFF. `=1` requests the new arm; unset or
`=0` followed by process restart restores host sampling. The FLAGS.md section 4
row has three columns and `decide-by: 2026-09-22`.

Wrong routes, any non-neutral repetition/frequency/presence coefficient (even
with a zero history window), min-p, constraints, full-row capture and invalid
sampler parameters log a refusal and retain the existing host path. Initialization
failures do likewise. Sampling errors recover the same logits row and disable
device sampling for that request; a failed CUDA readback still fails the request.

Current HTTP handlers reject requested logprobs/top_logprobs with 400 before
worker admission. This change preserves that existing API contract. The device
policy refuses requests requiring host logits, but this change does not add
logprobs response support or claim that an existing HTTP logprobs path works.

## Gates to run on the assigned non-production CUDA box

No cargo, tests, server or GPU work was run on the local rig. Direct rustfmt was
used only to format the source edits. Validation ran on an assigned non-production
2x B200 box with CUDA 13.1.115 and Rust 1.97.1. The initially assigned single card
was stopped by its provider before validation; no single-card results exist.

Build/check environment: `MEMRA_CUDA_ARCH=100a`, `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock`.
Use nice 19 and Cargo `-j 16`. Retain full build, clippy, test and server logs.

```sh
export MEMRA_CUDA_ARCH=100a MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
nice -n 19 cargo fmt --all -- --check
nice -n 19 cargo clippy --release --workspace --all-targets -j 16 -- -D warnings
nice -n 19 cargo test --release -p memra-sampling -p memra-engine --lib -j 16
nice -n 19 cargo test --release -p memra-server -j 16
flock /tmp/memra-gpu.lock nice -n 19 cargo test --release -p memra-engine --lib -j 16 \
  glm5_tp_device_sampler_gpu_oracles -- --ignored --nocapture
nice -n 19 cargo build --release -p memra-server --bin memra-server -j 16
tools/check-flags.sh
```

The ignored GPU test covers a random 154,880-logit row, tied maxima in different
blocks, both greedy selectors, top-p membership under three filter configurations,
non-degenerate sampled output, seed/position determinism and boundary-row adapter
identity. The CPU test covers flag defaults/parsing, sampling argument transfer,
route/full-row refusals and unsupported penalties/filters.

## Served A/B: five pairs complete

Ten accepted p32k rows complete the requested five OFF/ON pairs. The assigned
test container stopped during earlier attempts; completed receipts were retained
and the missing pairs resumed on the same physical B200 UUIDs with unchanged
source and binary hashes. One additional ON attempt was excluded when a compiler
appeared at the scored completion boundary. Interrupted attempts are not counted.
For the final two pairs, the shared GPU lock was acquired separately per pair and
released immediately afterwards. Every new row was written atomically and copied
to the rig before advancing to the next arm. Both pair completion markers and
all ten clean worker shutdowns were read back after measurement.

The same metered server binary ran both arms, SHA256
`c8bcbd80902d86f668733a790672d2d740bccec58c901fb5c60020b5b2076dc5`,
fingerprint `memra-0.135.0-eadd3136bc62`. The private deployment wrapper used
source `b5671b8f418d172a575cb729a6d8199100a5df2f` and this lane's unchanged
engine sources. A stock server rejects the launcher’s accounting/admin variables,
so the wrapper retained those surfaces with an isolated test keyring and ledger.
No serving process or real account was used.

The supplied launcher exec block was copied with its TP-2 expert split, symmetric
graphs, posture doors, `MEMRA_SERVE_SPEC=0`, context 1,048,576 and cache settings.
Only sandbox paths, credentials and loopback ports changed. The shared GPU lock
covered process restarts within each pair; the final pairs released it between
pairs so the other lane could run. Launcher source SHA256:
`8490641ceb7585d7bdfab4d99d5ad0e0fbf6ebd14055851842d971bc127a481b`.
Both arms added the existing `MEMRA_TICK_TRACE=1` diagnostic. The raw `[tick]`
records supply server decode time: mean of the 511 steady one-token ticks with no
prefill work, rounded to 0.1 ms individually by the server. This includes worker
sample/emit/decode work, and is not a GPU-kernel-only timer. Trace overhead is in
both arms. Compilation was absent from the accepted scored boundaries. The retained 250 ms
telemetry spans the original run, a later retry, and separate final-pair files,
with gaps across earlier attempt resets. Every accepted process logged TP graph
engagement and clean worker shutdown, recorded in `receipt-checks.json`. The
thermal regime is a fresh process after model load and a 32-token warmup.

Each row used a fresh process, a 32-token p32k warmup, then a vendor-default
512-token request with no sampling fields. The p32k file plus the fixed analysis
instruction rendered to 29,813 prompt tokens, with 29,792 cached tokens after
warmup. All measured rows completed 512 tokens with finish reason `length`.
The accepted order was OFF, ON, ON, OFF, OFF, ON, ON, OFF, OFF, ON,
with interruptions between completed blocks. Sampled outputs passed the repeated-text
screen (no 16-word span repeated four times); raw outputs remain available.

| Pair | OFF wall s | OFF tok/s | OFF server ms/token | ON wall s | ON tok/s | ON server ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.296 | 81.32 | 11.875 | 5.590 | 91.59 | 10.607 |
| 2 | 6.204 | 82.53 | 11.801 | 5.591 | 91.58 | 10.610 |
| 3 | 6.258 | 81.82 | 11.894 | 5.653 | 90.57 | 10.649 |
| 4 | 6.235 | 82.12 | 11.859 | 5.611 | 91.24 | 10.640 |
| 5 | 6.229 | 82.19 | 11.841 | 5.604 | 91.36 | 10.625 |

Five-observation arm medians: 82.116 -> 91.359 wall tok/s (+11.26%),
and 11.859 -> 10.625 server ms/token. Every row generated 512 tokens.
Rates use `completion_tokens / elapsed_wall_seconds`; prefill is included in wall
rate and excluded from the steady server tick summary.

The p128k requests were cold, each with 128,105 prompt tokens and zero cached
tokens. Wall rate therefore includes the roughly 29-second prefill.

| Arm | Completion tokens | Wall s | Wall tok/s | Server ms/token |
| --- | ---: | ---: | ---: | ---: |
| OFF | 512 | 35.458 | 14.440 | 12.423 |
| ON | 512 | 34.823 | 14.703 | 11.196 |

Greedy twins: all four requests produced exactly 160 tokens. Temperature zero and
top-k one, on both arms, had the same SHA256 of canonical generated content plus
reasoning, preserving the exact strings:
`b3bd73d02dfa61c28b11ccc51c5a56fd481e3962ea9c6555c24998ce0ca1ace9`.
This is output-byte identity, backed by the separate device/host token-ID argmax
oracle. Every ON scored row logged 512 device draws and no host fallback.

Raw rows, requests, SSE responses, generated strings, timing windows and checks:
`raw/served/`. Full deployment logs and private configuration remain outside the
public receipt. The requested five-pair campaign is complete. The flag remains
default OFF with its existing deadline; this receipt does not flip a serving default. Push uses
`MEMRA_SKIP_PERF_CI=1` because local-rig gates remain prohibited. The source
implementation did not change during measurement or this receipt update.
