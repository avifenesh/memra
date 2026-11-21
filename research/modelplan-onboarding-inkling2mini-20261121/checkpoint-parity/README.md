# Inkling 2 Mini: declared-class checkpoint parity (lane H, 2026-11-21)

The oracle of the checkpoint's shipped class (`inkling2-declared-activation-quant`) without an
inference engine, and the route held against it. Decision and tables:
[docs/decisions/INKLING2-DECLARED-CHECKPOINT-PARITY.md](../../../docs/decisions/INKLING2-DECLARED-CHECKPOINT-PARITY.md).
Card: RTX 5090 Laptop (sm_120a, 24 GB); the harness's CPU runs on a 32-thread x86 host with no GPU.

## Files

| file | what |
|---|---|
| `p1-moe-ref.json`, `p2-moe-ref.json` | `ink2-gpu --score --force-moe-in --moe-ref N`: every slot's MoE output against lane D's host reference, bit for bit (`moe_reference`) |
| `p1-moe-ref-red.json` | the same with the reference's index-order accumulator (the check's red arm) |
| `p2-float-vs-f64.json`, `p3-float-vs-f64.json` | the float class against lane E's f64 tapes |
| `probe/` | `memra-checkpoint-oracle-v1` tapes on the probe: lane E's CPU oracle (`tape_oracle.py`), the harness on the GPU and on CPU (actq and actq_rev), the route (`ink2-gpu` under `MEMRA_ORACLE_OUT`); the harness's four are gzipped. `cmp_oracles.py` prints the pairwise spread (of plain files) |
| `kl-harness.json` | `parity_kl.py` on the GPU: per-token KL of the route and of the sum-order twin against the harness on the oracle prompts, rawdoc and chatwrap, and lane E's tapes against both (`tape_cross`) |
| `kl-peer.json` | the same with the harness's CPU run as the peer backend on every document |
| `cross-cpu.json` | `tape_cross` with the harness on CPU |
| `verify/` | `memra model verify checkpoint` and `verify serve` on the band |
| `logs/` | raw logs of every step |
| `run_gpu.sh`, `run_gpu_b.sh`, `run_gpu_c.sh`, `run_gpu_d.sh` | the GPU leases, in order |
| `summ_kl.py`, `cross_table.py` | the tables of the decision record |

The route's head rows (`ink2-gpu --dump-hidden`, 357 MB) and the harness's saved rows stay on the
machine; the post-merge run checks the dumps bit for bit (`../serve/s3/`).

Inputs: lane E's tapes (`reference/ink2_oracle.py`, the oracle lane's `tapes/` and `tapes-var/`),
lane F's harness and corpora (`klh.py`, `klrun.py`, `corpus/rawdoc.json`, `corpus/chatwrap.json`;
rawdoc and chatwrap are 15 documents of 1024 tokens each). `oracle-prompts.json` and
`oracle-p3.json` are the tapes' token ids.
