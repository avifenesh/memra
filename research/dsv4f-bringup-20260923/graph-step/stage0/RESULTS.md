# DSv4 commit cleanup (memra #710 stage 0): one slot-row upload per stage, in-place scatter

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Server Edition, 2026-09-24. Served PP-2, serial route
(`MEMRA_DSV4_SESSIONS=1`). Lane `d888630a3` against its base `017f2bdae` (#699 with the drop
guard). One boot per row, order L B B L L B (N=3), for plain and for DSpark. Raw rows are in
`raw/stage0/` and the script is `raw/q-v3n.sh`.

What changed: the commit uploads the ring slots once per stage instead of once per layer, going
from 43 host-to-device copies per step to 2. It also scatters the transaction rows in place,
since they sit past every ring slot, so 43 bounce copies per step are gone. The indices and the
bytes written are the same.

| cell | lane | base | delta |
|---|---|---|---|
| plain greedy c1 | 52.26 | 51.76 | +1.0% |
| plain sampled c1 | 48.79 | 48.22 | +1.2% |
| DSpark greedy c1 | 60.87 | 60.70 | +0.3% |
| DSpark sampled c1 | 53.19 | 52.25 | +1.8% |

Correctness:
- `GPU DSPARK GATE [PASS]` on the lane binary.
- Every request's text hash is identical across all rows of both arms, on plain and DSpark,
  greedy and sampled.

Decision: this is the code, with no door.
