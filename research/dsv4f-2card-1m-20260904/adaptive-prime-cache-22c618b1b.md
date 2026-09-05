# Adaptive short-prime and sampled cache-transparency receipt

Date: 2026-09-04 UTC

Memra source: `22c618b1b1239d84a228e51ddc8e2a3ad4185875`

Server binary sha256:
`49290fbd67c7144edf4286f3f45ec17de9df713af32fbf979563569dab14da74`

Artifact: exact mixed NVFP4/MXFP4 Safetensors mint of
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

Runtime: PP2 on two RTX PRO 6000 Blackwell Workstation cards, DSpark device
path, `MEMRA_DSV4_PREFILL_CHUNK=32`, 32 GiB pinned parked-session budget. The
provider setting was 500 W per card. It is recorded only as metadata; this gate
did not profile power, clocks, bandwidth or compute saturation.

Prompts of at most 32 tokens now take the canonical monolithic prime path and
are not parked, keeping that numerical regime isolated. The public fixed
24-token prompt with 256 greedy, ignore-EOS output tokens measured 47.7626
tok/s in the post-change proof request. The preceding five-repetition selected
monolithic control had a 48.4773 tok/s median. This improves the selected short
path over chunked priming, but remains below the public comparison and earns no
performance claim.

The eight-turn fixed-seed sampled cache-transparency gate used generation
defaults plus seed `20260904` on `/v1/completions`:

| turn | prompt | restored | output | warm wall | cold wall | DSpark rounds/drafted/accepted | output identity |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 387 | 0 | 32 | 4.664689 s | 4.639271 s | 21 / 94 / 10 | exact |
| 2 | 433 | 418 | 32 | 1.788514 s | 5.168660 s | 25 / 115 / 6 | exact |
| 3 | 479 | 464 | 32 | 0.737533 s | 4.414285 s | 6 / 29 / 25 | exact |
| 4 | 524 | 510 | 32 | 0.929131 s | 4.910803 s | 10 / 42 / 21 | exact |
| 5 | 569 | 555 | 32 | 0.725973 s | 4.989839 s | 6 / 28 / 25 | exact |
| 6 | 614 | 600 | 32 | 0.842376 s | 5.416579 s | 8 / 37 / 23 | exact |
| 7 | 660 | 645 | 32 | 1.067758 s | 5.929405 s | 12 / 54 / 19 | exact |
| 8 | 706 | 691 | 32 | 0.797413 s | 5.966975 s | 7 / 33 / 24 | exact |

For every row, the warm and cold twins also had identical DSpark rounds,
drafted tokens and accepted tokens. Final verdict: `PASS`.
