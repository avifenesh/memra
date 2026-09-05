# Native-limit allocation and exact selection re-gate

2026-09-05, two RTX PRO 6000 Blackwell Server Edition cards, driver 610.43.02.
Source commit `2436ce7403d10b30ea7ca94b3e44809f3ccfc1b5`; kernel/gate files
compiled before the commit and unchanged in it. Binary SHA256
`9b9ba955f18b5f54016449f9f2088b502d2f7ae3b61e4c784e4f596a02f7496e`.
Artifact `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`;
all 77 files verified, all LFS SHA256 values matched pinned Hub metadata.

Invocation (inside the isolated development model directory):

```sh
flock -n /tmp/memra-gpu.lock env MEMRA_DSV4_DRAFTER=dspark \
  MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_DECODE_PATH=device \
  MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_FP4_REDUCE=warp \
  ./dsv4_capacity_gate ./model ./dsv4_fixtures_ref.json 0,1
```

Fixture variant `ref` selects the 0731 FP8 activation contract; this gate uses
no fixture logits as a quality oracle. Raw log: `capacity-262147-pro.log`.

| stage | post-load GiB | compact cache GiB | verify workspace GiB | post-allocation / total GiB |
| --- | ---: | ---: | ---: | --- |
| 0 | 83.487 | 7.044 | 0.233 | 90.768 / 95.010 |
| 1 | 83.580 | 6.418 | 0.232 | 90.237 / 95.010 |

Split at layer 23; all three bundled DSpark blocks resident on the tail stage.
The exact hierarchical selector matched the independent host ordering on three
rows with deliberate ties at candidate counts 4,103; 16,397; 250,003; 262,144;
and 262,147. The latter two close the previous decimal-million coverage gap.

Verdict: `PASS 1M compact state + DSpark + chunk32 workspace`.
This is allocation plus component-selection proof, not full 1,048,576-token
prefill, latency, multiple concurrent 1M requests or serving qualification.
