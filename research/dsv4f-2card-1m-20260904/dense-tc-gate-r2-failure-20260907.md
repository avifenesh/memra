# Dense-TC gate r2 failure receipt, 2026-09-07

The first GPU execution of r2 was stopped after the first numerical gate on
GPU0. No timing result was admitted and GPU1 did not run. Memcheck reported no
error.

```text
binary ca0f690c41074bd0531168b2a74a600f40284f769a32feb4895c71dbde1568c6
source 192ad714c1fd87d3fe8fc5f9c006289e332920a07834ed1295f65fb526c9ea9a
failure cycle=0 row=0 max_abs=1353.62524 max_rel=1.2759068
```

The r2 source used the in-tree `load_ldmatrix_A_trans` helper and selected its
`x0/x2` first-n=8 half, but the receipt does not establish that the standalone
output-neuron B staging has the same source orientation as the validated
attention V path. r3 must first pass a K=16/K=32 basis diagnostic against an
independent per-row oracle before any full-shape timing is considered. The
numeric thresholds must not be loosened to mask this operand/layout failure.
