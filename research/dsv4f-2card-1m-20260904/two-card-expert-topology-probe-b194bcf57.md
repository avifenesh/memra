# Two-card same-layer expert topology probe

Date: 2026-09-05 UTC

Probe implementation: `5ae389db2`  
Probe-code revert: `b194bcf57`

This was a compute/P2P feasibility cell, not a serving implementation. It
loaded the exact DSV4 0731 Safetensors mint, selected six real layer-3 NVFP4
experts, and compared two ways of using both RTX PRO 6000 cards against the
current single-card six-expert chain.

## Expert-ID partition

Experts 0/1/2 stayed on the layer owner; 128/129/130 were copied once into a
compact peer bank. The timed interval included a 16 KiB activation fan-out,
three exact selected experts per GPU, and a 48 KiB contribution return. The
one-time 42 MiB peer-bank setup was excluded, as it would be resident in a real
loader.

All 24,576 per-slot contribution values matched the serial baseline bit for
bit. Two 21-repetition runs measured 1.280x and 1.365x integrated speedup; the
latest row was 0.101703 ms serial versus 0.074511 ms parallel.

This topology preserves exact arithmetic but a batch-1 top-6 route does not
always split 3+3. It becomes more balanced across batched requests, making it
more promising for concurrency throughput than deterministic c1 latency.

## Intermediate-dimension tensor split

Every selected expert was split 1,024+1,024: w1/w3 output rows and w2 input
columns. Both cards therefore did equal work. The timed interval included a
16 KiB activation fan-out, 96 KiB partial return, event-ordered owner join and
f32 partial add.

The joined output's maximum absolute drift from the serial reduction was
`0.00000012`; maximum relative drift was `0.00000006`. Performance was flat:
0.095751 ms serial versus 0.095090 ms parallel, 1.007x. Halving the scalar
kernel shape plus P2P/join overhead consumed the theoretical balance benefit.

## Verdict

Do not build the full loader around the intermediate TP split with the current
kernel. Expert-ID EP is the viable native direction, especially for concurrent
requests, but its full-fleet value must include real-route ownership balance,
resident-memory placement and end-to-end scheduling. The probe code was removed
after banking the result.

Raw logs on dev instance 48400600:

- `/root/dsv4-ep2-expert-p2p-probe.log`
- `/root/dsv4-expert-tp-probe.log`

The provider setting was 500 W/card. No power, clock, bandwidth or compute-limit
attribution is made.
