# PRO device-penalty sampling

Status: qualified on the measured RTX PRO 6000 Server Edition topology. The engine default
remains off globally. Another board/deployment is not qualified until its own launcher smoke;
an owner-authorized reversible transfer canary may opt in earlier, but inherits no performance
claim and must retain an explicit rollback.

## Scope

- Target hardware: one RTX PRO 6000 Blackwell Server Edition GPU at a time across two
  same-class rented hosts (the first was reclaimed mid-campaign).
- Models: Qwen3.8-27B and Ornith-1.5-35B-A3B, using the owner's pinned Hugging Face cards.
- Serving shape: sampled plain/batched rows with repetition, frequency, or presence penalties.
- Explicitly out of scope: Step TP/EP, RTX 5090 performance, greedy-penalty promotion, and
  constrained penalty/filter composition.

Implementation base: `a2a810cdf2ef8caf3c6dd1cd06c811b7a2804df3`. Performance tree:
`3fab5c2599cd61cbfd917a2f38552692ad8fb4a6`; later integration rebases changed version/docs only.

## Mechanism

The host sampler used to be the only penalty-capable plain-decode path. Any non-neutral penalty
therefore disabled device sampling, forcing every row's full vocabulary logits through D2H and an
O(vocabulary) host sample. Qwen3.8's newly served non-thinking defaults make that path ordinary:
`temperature=.7`, `top_p=.8`, `top_k=20`, `presence_penalty=1.5`.

This lane keeps exact sparse counts for each sampler's active history window, uploads only unique positive-count
`(token,count)` entries, applies heterogeneous row coefficients in one Memra-owned CUDA launch,
then uses the existing device filter/Gumbel path. Raw logits are preserved before mutation and are
the only bytes parked for reuse.

## Qualification contract

1. `cargo test -p memra-sampling`: sliding-window and full-context sparse counts.
2. `sample-check`: heterogeneous rows, negative/positive coefficients, an untouched control row,
   and a 9,000-id window against the CPU house rule.
3. `decode-batch-gate`: mixed penalty/no-penalty rows; full and lean paths return/park identical
   raw logits and choose identical device tokens.
4. Model gates on both artifacts: kernel-check, run-gen, run-spec K=1..8, serve-smoke.
5. Same-binary interleaved N>=5 A/B: `MEMRA_SERVE_DEVPENALTY=1` vs a flag-absent,
   default-off baseline,
   isolated plain
   (`MEMRA_SERVE_SPEC=0`) and naked serving policy, c=1 through the first sustained decline.
6. Report TTFT/E2E/TPOT/ITL p50/p95/p99, request/output throughput, errors, engagement marker,
   binary/model hashes, and 250 ms telemetry. No default verdict without both model rows.

## Durability

The measurement host is disposable. Source custody stays in the local worktree and Git branch.
Every completed cell is pulled off-host immediately and sealed with SHA-256. Provider, machine,
storage, and private receipt coordinates live in the private deployment repository.

## Qualification result

Each paired comparison used one RTX PRO 6000 Server Edition GPU on one host while its peer GPU
stayed idle. Its arms used the same binary and alternated AB/BA across five fresh-process
repetitions. Isolated rows
disabled speculative decoding and every cache/reuse path; every request reported zero cached and
speculative tokens. Higher throughput and lower TPOT are better.

| model / shape | concurrency | baseline tok/s | device penalty tok/s | paired median | wins | baseline → device TPOT p50 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8 short | 1 | 46.49 | 68.66 | +47.57% | 5/5 | 19.85 → 12.83 ms |
| Qwen3.8 short | 16 (knee) | 94.95 | 300.93 | +216.75% | 5/5 | 150.54 → 34.39 ms |
| Qwen3.8 short | 32 | 96.10 | 302.12 | +214.37% | 5/5 | 308.01 → 74.42 ms |
| Qwen3.8 real long | 1 | 13.51 | 14.88 | +10.16% | 5/5 | 20.89 → 13.91 ms |
| Ornith-1.5 short | 1 | 79.46 | 176.78 | +121.70% | 5/5 | 11.68 → 4.61 ms |
| Ornith-1.5 short | 24 (knee) | 118.77 | 651.77 | +447.97% | 5/5 | 193.32 → 26.26 ms |
| Ornith-1.5 short | 32 | 118.89 | 642.07 | +439.16% | 5/5 | 257.94 → 35.59 ms |
| Ornith-1.5 real long | 1 | 36.91 | 49.69 | +34.30% | 5/5 | 11.47 → 4.54 ms |

The short isolated curves cover every rung c=1,2,4,8,12,16,24,32. Qwen plateaus at c16;
Ornith peaks at its served c24 ceiling and declines slightly at c32. Across 180 non-duplicated
isolated summary cells there were zero request errors, cache hits, or speculative tokens; every enabled
boot emitted the device-penalty marker and every rollback boot did not.

Scope: these are single-GPU closed-loop cells with a 241-token short prompt and 64 requested
output tokens; the real-long c1 prompt is about 12,074 tokens. The low-end/bracket campaigns use
different `MEMRA_MAX_SESSIONS` ceilings from the saturation campaigns, so every same-cell A/B is
paired and valid while the cross-c knee is an observed plateau, not one monolithic-config sweep.
Sampled arms are distribution-equal rather than sequence-identical; serving output hashes are
diagnostic only, while semantic parity comes from the CPU/GPU sampling and model exactness gates.

The separate naked-policy matrix retained product caching and speculative policy. Qwen c1 stayed
inside noise with active MTP drafting (`+0.24%` paired median); at c16/c32 the policy selected
plain decode and the device path won `+312.76%` / `+325.16%`, 5/5, with identical cached-token
totals. Ornith's published defaults carry no penalty, so the flag was unreachable: no marker in
either arm, identical cache totals, and c24 measured `+0.10%` (noise). These rows are controls,
not pooled with the isolated mechanism curve.

Correctness gates on both pinned artifacts passed: incremental host-count tests, heterogeneous
and 9,000-id sparse GPU parity, negative/duplicate/zero/wide metadata refusal, a causal
penalty-induced argmax flip, independent pristine raw logits plus mixed lean/full parking identity,
kernel-check, run-gen, and run-spec self-consistency K=1..8.
