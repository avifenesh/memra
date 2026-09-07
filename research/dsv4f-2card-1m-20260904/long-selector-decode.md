# Real decode beyond the 4096-candidate boundary

2026-09-05. The plain device path called `memra_dsv4_topk_idx`, whose bitonic
launcher returns 40008 above 4096 candidates. At CSA ratio four this blocked
decode beyond roughly 16K input positions despite successful 1M allocation.
The fix uses the existing exact hierarchical selector above that boundary,
with bounded per-stage scratch. Short-history bitonic behavior is unchanged.

Small-width verification was not capped: it copied the scores to the CPU and
sorted them. It now uses the same device selector above 4096, preserving the
short-history witness and the explicit host-math oracle.

On two RTX PRO 6000 Blackwell Server cards, the actual 16,416-token Rust-source
prompt passes plain/DSpark fixed-seed sampled identity for 16 output tokens.
The DSpark path engaged for five rounds. No input repetition or padding.

- gate binary: `ee63ab892e64b1f86b673912d5e6fa3c3b644cc7476fcc65b20161bd3fa5c889`
- source text: `a8ab08ed39dbc1ae891c1c4a52336bf6b27c98e788d2739ff5c0bae59b5c4dda`
- exact input tokens: `192d84faf04dbfd98056edf8eb8b2d37224bd418cdbc3915d1215c034d59340e`
- raw receipt: `long-decode-16416-pro.log`

The timed version of `dsv4_long_decode_gate` prints prefill wall separately;
plain/spec timings explicitly include restore and are not advertised decode TPS.
A real 262,144-token engine run is pending. HTTP end-to-end, actual 512K/native
1M, sampled chat behavior, active C4 offload and concurrency are separate gates.

The FP4 one-load gate also now permits exact negative-infinity masks only in
pending compressor/indexer score planes. It still rejects NaN/+Inf everywhere
and -Inf in other cache classes; every permitted mask bit participates in the
digest. The corrected block/warp gate passed on the same card class, including
both DSpark fused-MoE arms (`fp4-one-load-pro.log`). This is parity, not a default
or performance promotion.
