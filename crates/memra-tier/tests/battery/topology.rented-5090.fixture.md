# Rented RTX 5090 topology fixture

Derived from the 2026-09-19 first-hour `TOPOLOGY.json` at source
`01e7b77f29c7b40fb29744e5321ca0e8b0c81f39` (receipt SHA-256
`cacf0fa44e749d776064d4bf275b94c41b4d54048d7238c9c5e98649c7fa4a79`). The original lives in
`research/spill-lead-20260919/rented-5090-20260919/receipts/` under the successful
`*run3` bootstrap. The fixture retains only topology/NUMA commands and the
`GPU Link Info` block; the GPU address is redacted. No provider/host identity.

One device. Reported generation: current **1**, effective max **4**, device max
**5**, host max **4**. Width current/max **16x**. This is a PCIe 4.0 x16 host
ceiling, not sustained bandwidth. An idle Gen1 sample is neither proof of a
broken link nor proof it trains to Gen4 under load. Fixture replay is CPU parser
coverage, not fresh GPU evidence or P2P qualification.
