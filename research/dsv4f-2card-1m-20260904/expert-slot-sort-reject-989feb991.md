# Rejected expert-major prefill slot order

Date: 2026-09-05 UTC

Implementation: `076237b0f`  
Revert: `989feb991`

The arm stably counting-sorted routed slots by expert on the device, supplied
each sorted slot with its original activation-row index, ran the existing exact
FP8-activation/FP4-weight kernels unchanged, and scattered w2 contributions
back to token-major slots before the unchanged ascending-expert combine.

On the exact DSV4 0731 Safetensors mint and two RTX PRO 6000 Blackwell
Workstation cards, the 160-token in-process gate passed bit-for-bit for final
logits and every live cache class. Three interleaved prefills measured:

| arm | median wall |
| --- | ---: |
| token-major reference | 1.768474 s |
| stable expert-major | 1.769360 s |

Speedup was 0.999x. The existing launch/cache behavior already captures the
available repeated-expert locality at the proven transaction width, while the
sort and scatter-back pay it back. The arm was fully reverted and earns no
performance claim.

Raw log: `/root/dsv4-prefill-expert-sort-gate.log` on dev instance 48400600.
The provider setting was 500 W/card; no power, clock, bandwidth or compute
attribution is made.
