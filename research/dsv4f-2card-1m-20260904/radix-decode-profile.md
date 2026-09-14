# Sampled plain decode profile after radix ordering

2026-09-06, two RTX PRO 6000 Max-Q cards. Explicit capture of sampled steps
32..64 after a real 8192-token prefix and warmup, active C4, matrix/EP, tiled
indexer/scorer, radix sampler. The complete 96-token output matches the frozen
pre-rewrite stream. Binary:
`f6277b88756a7da67729af28f9e8903ffcd2035c99c03ad569e72cff3b317c4e`.
Target completed 07:14:52 UTC; profile controller completed 07:14:53, status 0.

The captured phase wall is 1.485318 s. The observed kernel extent is
1481716051 ns, kernel sum 1354549378 ns, GPU-busy union 1145461623 ns,
and no-kernel interval time within the extent 336254428 ns. Both-device overlap
is 209087755 ns, 18.25% of the busy union. There are 122866 kernel launches
across 32 steps, about 3840 per generated/consumed token in this interval.
This is an instrumented mechanism profile, not a throughput measurement.

| Operation | Summed kernel time | Share of kernel sum |
| --- | ---: | ---: |
| selected-expert `moe_kq_sktail_kernel<108>` | 531546621 ns | 39.2% |
| dense FP8 `dsv4_gemv_fp8_m_kernel<1>` | 189189542 ns | 14.0% |
| active C4 host gather | 119545211 ns | 8.8% |
| f32-accumulated dots | 84561108 ns | 6.2% |
| mHC Sinkhorn | 53255063 ns | 3.9% |
| RMS norm | 51487027 ns | 3.8% |
| tiled sink scores | 48772537 ns | 3.6% |
| sink attention output | 47383976 ns | 3.5% |
| numeric index top-k | 40998380 ns | 3.0% |

The CUDA API table records 11140 D2H async calls (603192602 ns), 122866 kernel
launches (505114667 ns), and 8286 stream synchronizations. These API times
overlap device work and can include waits; they must not be added to GPU time
or treated as independent removable overhead. Next source checks are the
selected small-M expert form, dense projections, C4 gather reuse, and launch/
readback boundaries needed for reusable graphs. No power-limit cause is inferred.

Only the owned application PID appears in 3547 process samples, including eight
name gaps at teardown. Nsight retains generic CUDA/NVTX completeness warnings;
no explicit dropped-event count is reported. The report and interval audit are
banked with raw CUDA summaries and process/controller logs in the companion lane.
Report SHA256:
`a323c0c081b0ce606f8b3d83629278ea718cf968b10fa6eb61bd018e3563e138`;
SQLite SHA256:
`085b00a85e9c65526f61193892828e9e98925b8ca3e6956072954bfc74488246`.

The next capacity cell runs a real 1048448-token source prefix through direct
host C4 at chunk 512, followed by sampled plain/DSpark restoration checks.
It leaves room below the model's 1048576-token limit and is not an exact
1048576-token prompt. Its result, HTTP long-context behavior and concurrency
remain pending; no serving default is promoted.
