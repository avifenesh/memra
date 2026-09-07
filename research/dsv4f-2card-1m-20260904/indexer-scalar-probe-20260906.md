# Fine-indexer redirect scalar probe, 2026-09-06

The next graph seam is a component-only fine-indexer redirect builder. It adds
`memra_dsv4_build_idx_redirect_fine_pos`, whose query position and completed
compressed-block count are read from persistent device scalars. The explicit
`graph-indexer-scalar-probe` gate captures and replays only this array-builder
component. It does not enter a model layer, compressor state machine, score or
top-k, C4 gather, EP, or commit.

The CUDA component and FFI compile. Evidence:

```text
cargo build -p memra-engine --release --bin dsv4_decode_profile
Finished `release` profile [optimized] target(s) in 4m 20s

cargo check -p memra-engine --bin dsv4_graph_probe_gate
Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.62s
```

The release build also included the active parent small-M symbol repair that
was missing during the first check. That first failure was not caused by this
probe. The component has not been promoted or used by serving.

## Exact blocker to the next full-layer step

The fine ratio layer still cannot be captured as a layer. `cmp_decode_batch_dev`
advances pending KV/score state and append-only block counts in a host-ordered
state machine; T<=8 indexer score/top-k still has host metadata paths; active
C4 uses host gather/copy targets; and `commit_verify_dev` uploads slot rows and
replays compressor state. EP and PP boundary event dependencies remain outside
a single CUDA stream. The scalar redirect component therefore stays gate-only
and does not weaken the existing full-layer refusal.
