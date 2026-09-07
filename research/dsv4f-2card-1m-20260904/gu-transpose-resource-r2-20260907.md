# GU transpose R2 resource receipt

This is a reporting-only correction after the R2 GPU rate/memcheck result was
already banked. No GPU row was rerun and no arithmetic or launch was changed.
The existing R2 timing receipt remains authoritative; this note preserves the
pre-correction source identity and the corrected resource identity.

## Source and binary identities

- R2 source before the reporting correction: `9c54167902fc554d05250d31b0103090066ed84e8024c47d0e34d6bc5ebc1d7a`
- Corrected source: `c946089d73f27c693e62c9baae88a247d06be1933b32e6784ffa8594fef59c9e`
- Corrected linked binary: `target/dsv4-gu-transpose-gate-r2`
- Corrected binary SHA256: `c314c7af24a7c25a919460f9e54b9c175b07521c57e33444061b619025c44789`

R2 timing rows are preserved in the root-owned rate receipt and were not
repeated here. This source-only metadata fix cannot change them.

## Actual compiled resources

Command, no GPU execution:

```text
cuobjdump --dump-resource-usage target/dsv4-gu-transpose-gate-r2
```

| kernel | registers | cuobjdump shared | declared accounting |
|---|---:|---:|---:|
| `moe_kq_sktail_gu_kernel<108,false,true>` current GU half2 control | 150 | 26,128 B | 25,092 B static/prefix + 1,024 B dynamic LUT = 26,116 B before 12 B alignment |
| `dsv4_gu_transpose_kernel<108,true>` candidate | 74 | 23,824 B | 22,788 B static/prefix + 1,024 B dynamic LUT = 23,812 B before 12 B alignment |

The previous printed control value incorrectly counted three staged B banks.
The production GU control declares `As[3][32][72]` plus one
`Bs[64][72]`; only the transpose candidate declares three B stages, each
`Bs[16][72]`. The source print now reports the static/prefix terms and the
dynamic 1,024-byte LUT separately.
