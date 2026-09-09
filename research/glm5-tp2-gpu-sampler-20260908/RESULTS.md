# GLM TP-2 device sampler qualification

Status: remote build, fmt, clippy, CPU suites and GPU sampler oracles PASS.
Current receipt: five TP-2 p32k OFF/ON pairs, with pairs 1-3 from the original window and pairs 4-5 from the later owner-scheduled pair window. The original p128k controls and both windows' 160-token greedy twins are retained.
No default promotion.
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

## Served A/B: original three-pair window

The original window contains pairs 1 through 3. The later pair 4/5 window is recorded below. Additional earlier tune-box observations for pairs 4/5 remain in [SUPPLEMENTARY-TUNE-PAIRS.md](SUPPLEMENTARY-TUNE-PAIRS.md) and are excluded from the current five-pair selection.

Earlier container interruptions were followed by source/binary hash and physical
GPU UUID checks before resuming. One separate ON attempt was excluded for
compilation overlap. The final tune attempts acquired and released the shared
lock separately per pair, and mirrored each new row before proceeding. All ten
collected tune processes have clean worker shutdown receipts, including the
supplementary observations.

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
The original three-pair order was OFF, ON, ON, OFF, OFF, ON, with interruptions
between completed blocks. Sampled outputs passed the repeated-text
screen (no 16-word span repeated four times); raw outputs remain available.

| Pair | OFF wall s | OFF tok/s | OFF server ms/token | ON wall s | ON tok/s | ON server ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.296 | 81.32 | 11.875 | 5.590 | 91.59 | 10.607 |
| 2 | 6.204 | 82.53 | 11.801 | 5.591 | 91.58 | 10.610 |
| 3 | 6.258 | 81.82 | 11.894 | 5.653 | 90.57 | 10.649 |

Three-observation arm medians: 81.821 -> 91.577 wall tok/s (+11.92%),
and 11.875 -> 10.610 server ms/token. Every row generated 512 tokens.
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
public receipt. `current-receipt-rows.json` and `current-receipt-summary.json` select the original six rows plus the four later rows below. Earlier supplementary tune pairs 4/5 remain excluded. The flag stays default OFF with its existing deadline. Push uses `MEMRA_SKIP_PERF_CI=1`; no cargo or gates ran on the local rig.

## Owner-scheduled pairs 4 and 5

Both new pairs passed. Each process used the same lane A/B environment, p32k warmup, vendor-default 512-token request and tick diagnostic. The order was ON/OFF for pair 4, then OFF/ON for pair 5. Each measured row has 29,813 prompt tokens, 29,792 cached tokens, 512 output tokens and finish reason length. Both ON rows recorded 512 device draws and no host fallback. Repeated-text screens passed.

The rebuilt binary SHA256 is `8ed89bf30b552f7acf45e1d8b94f8aaea7b49ae631ab96d1b9c1537bc57c2298`, source fingerprint `memra-0.135.0-eadd3136bc62`. This is a separate binary from the original window's hash, with the same engine source fingerprint. Build on the target completed in 4m 37s; source manifest verification passed. No implementation source changed. One initial startup refused a mode 0644 sandbox ledger before GPU/model load; permissions were corrected to 0600 and that unsent attempt was excluded.

| Pair | OFF wall s | OFF tok/s | OFF server ms/token | ON wall s | ON tok/s | ON server ms/token |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 6.320590 | 81.005 | 11.990 | 5.610973 | 91.250 | 10.557 |
| 5 | 6.434862 | 79.567 | 12.113 | 5.631143 | 90.923 | 10.670 |

New two-pair medians: 80.285844 to 91.086349 wall tok/s; 12.051468 to 10.613112 server ms/token. Current five-pair medians: 81.316715 to 91.249779 wall tok/s (+12.215279%), 11.893542 to 10.609980 server ms/token. The five-pair summary combines two measurement windows; the new two-pair table is the direct comparison for the later pair.

All four new greedy/top-k-one requests produced 160 tokens and the same canonical generated-content-plus-reasoning SHA256 as the original oracle: `b3bd73d02dfa61c28b11ccc51c5a56fd481e3962ea9c6555c24998ce0ca1ace9`. This is output-byte identity, supported by the existing device/host token-ID argmax gate; the HTTP API does not return raw token IDs.

Long jobs ran under nohup/setsid, and each sampled row was copied to the rig and acknowledged before continuing. The stop driver signalled only the owned PID and waited for GPU quiet between processes. The driver required its hard-stop path after the graceful drain message; clean GPU-worker-shutdown markers are not claimed for these four processes. Request completion and usage receipts were retained before stopping.

Raw public requests by SHA, SSE, generated output, tick/sampler windows, paired rows, loop checks and telemetry: `raw/pair-box/`. Full private configuration, stop logs and the excluded startup remain in private custody. No default promotion.

The later binary was built from the source snapshot carried by `66b79c20ebc0422c87b6218e0e9719ff3b755416`; receipt-only commit `ad25ee7229c287c8115bad64266a16458f1ede25` preserves that implementation. Subsequent main integration is checked by hosted CI and does not constitute another performance measurement.
