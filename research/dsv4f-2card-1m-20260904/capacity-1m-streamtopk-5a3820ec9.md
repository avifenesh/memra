# Exact two-card 1M capacity and streaming-selection receipt

Date: 2026-09-04 UTC

Memra source: `5a3820ec9` for the gate implementation; the same gate output was
replayed from source `22c618b1b1239d84a228e51ddc8e2a3ad4185875` without a source
change to the measured path.

Artifact: `/root/models/dsv4-flash-0731-nvfp4`, exact Safetensors mint of
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

Hardware: two RTX PRO 6000 Blackwell Workstation GPUs, 94.970 GiB CUDA-visible
capacity each. The provider setting was 500 W per card. That value is metadata,
not a diagnosis of the measured path.

The DSpark-aware PP2 planner selected `split_at=23` and reserved 10.83 GiB for
the three DSpark tail blocks on device 1.

| point | device 0 | device 1 |
| --- | ---: | ---: |
| after weights and DSpark load | 83.487 GiB | 83.581 GiB |
| after full 1,048,576-token compact cache plus chunk-32 workspace | 90.769 GiB | 90.237 GiB |
| compact cache contribution | 7.044 GiB | 6.418 GiB |
| verify workspace contribution | 0.233 GiB | 0.232 GiB |
| unallocated headroom | 4.201 GiB | 4.733 GiB |

The exact hierarchical selector was compared with the host value-descending,
original-index-ascending oracle, including deliberate score ties:

| candidates | rows | selected | result |
| ---: | ---: | ---: | --- |
| 4,103 | 3 | 512 | exact |
| 16,397 | 3 | 512 | exact |
| 250,003 | 3 | 512 | exact |

`250,003` candidates covers the DSV4 4:1 compressed-index scale at one million
tokens without materializing the entire candidate matrix on the device. The gate
verdict was `PASS 1M compact state + DSpark + chunk32 workspace`.

This proves allocation and exact-selection reachability. It does not claim a
completed one-million-token prefill or a throughput result.
