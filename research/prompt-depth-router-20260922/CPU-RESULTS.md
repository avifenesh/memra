# CPU classifier result

The bounded lexical classifier passed 95/95 fixed synthetic behavioral cases,
11 Rust regressions and four profile-fitting regressions. These cases establish
the documented behaviors; they are not a blind estimate of real-world accuracy.

Host: Intel(R) Xeon(R) 6973P-C, 4 logical CPUs available,
`x86_64`. Rust 1.97.1, release optimization. The function is
single-threaded and in-process. Timings include the measurement clock's overhead.

| Input group | Calls | Median us | p95 us | p99 us | Maximum us |
|---|---:|---:|---:|---:|---:|
| Classified fixture instructions | 100,000 | 0.179 | 0.268 | 0.298 | 13.457 |
| Mixed/unknown fixture instructions | 100,000 | 0.138 | 0.275 | 0.308 | 209.647 |
| 16 KiB quoted input | 100,000 | 3.983 | 4.008 | 5.297 | 42.370 |
| About 210 KB marked reference | 100,000 | 46.926 | 54.810 | 70.335 | 201.657 |
| 256 KiB quoted input | 100,000 | 62.632 | 69.402 | 75.685 | 265.212 |
| 512-word limit fallback | 100,000 | 9.204 | 9.348 | 13.121 | 32.507 |
| Oversized-input fallback | 100,000 | 0.017 | 0.019 | 0.019 | 7.446 |

Each group has five interleaved batches of 20,000 calls. Successful classification,
semantic fallback and input-limit fallback remain separate. Maximum observed
times include scheduling outliers; the implementation makes no real-time deadline claim.

The standalone process plus stdin/stdout and Python orchestration measured
1.268 ms median over 20 launches.
The native study calls the Rust function directly instead of launching this CLI per request.

This is CPU component evidence. It establishes no model-decoding speedup.

Source recipe: `fcdea815d59a4dcff75e62042796237ce5a465af`.
Measured executable SHA-256: `33734eecc5d6b255001647085b20400355b7c5c1dad82ddd32e37926320c2c81`.
CPU manifest SHA-256: `0d3c86bd71f53ca7801ab29f2657cd40bcc9136140e2f26a16d6b23b37fa8b32`.
Exact source, raw per-call samples, interface checks and compiler/command identity
are in `cpu-receipts/`. Exact executables and complete CI job output are retained privately.
