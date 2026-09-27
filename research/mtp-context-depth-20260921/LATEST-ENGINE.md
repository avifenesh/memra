# Latest Memra comparison

The latest engine fixes did not materially improve this contextual-depth study. Online-policy throughput changed by -0.021% on Qwen and +0.002% on Gemma. Every paired prompt and output token-ID tape matched exactly across versions, including the online policy.

This follow-up used 32 interleaved runs and 256 timed turns on the same RTX 5090 32GB. Each model had two fresh sampling seeds, four policies, and both runtime versions. Runtime order alternated between policies and reversed for the second seed. No group was excluded.

| Model / policy | Original tok/s | Updated tok/s | Updated change |
|---|---:|---:|---:|
| qwen / fixed K | 113.081 | 113.248 | +0.148% |
| qwen / native | 113.220 | 113.247 | +0.025% |
| qwen / frozen context | 113.430 | 113.468 | +0.033% |
| qwen / online | 113.184 | 113.161 | -0.021% |
| gemma / fixed K | 186.842 | 186.872 | +0.016% |
| gemma / native | 194.164 | 194.168 | +0.002% |
| gemma / frozen context | 198.852 | 198.860 | +0.004% |
| gemma / online | 193.012 | 193.015 | +0.002% |

The shared initial priors and workload pools came from the original study; sampling seeds were fresh. Settings remained temperature 0.7, top-k 20, top-p 0.95, a 2,048-token output cap, and a 49,152-token context reservation. The metric includes the complete native request, including suffix processing and policy work. Loading, warmup and receipt I/O are excluded.

These are two diagnostic source comparisons per model, not a retuned calibration study or a serving-default qualification. The small timing differences do not change the original conclusion: good fixed or calibrated contextual choices were stronger than this online update rule.

Updated source: `5450580fe2e5eb5c34cd17452f472ed368034f41`, based on upstream `dc192cd9543f9f4ba8bd45d489d12e2e871f7b54`. Original source: `cb2f1783a0818705c5528f5b11cae0b0bc176deb`.

The study policy and runner were unchanged. Relevant upstream changes included the GDN capture-grid correction and prefill cancellation checks. The Gemma MTP decoder and sampling crate were unchanged in this update.

The corrected updated runtime archive is `943d165b80f268ffacc63e78191c669ed4bda562b3151d45758876321e3f6dc8`. Its build includes the exact `docs/FLAGS.md` registry required by the newer boot audit. An earlier incomplete source export was stopped before scoring and retained privately.

The updated build passed Clippy, focused tests, eight ordinary native-runner checks, and 64 contextual correctness turns. Independent replay reconstructed every result and checked 128 paired turns for prompt/output identity. Raw commands, timings, token tapes, qualification and source bindings are in `latest-receipts/`; `publication/reproduce_latest.py` replays the report from verified archives.
