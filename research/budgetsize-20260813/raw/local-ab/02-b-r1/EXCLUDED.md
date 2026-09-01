# Excluded candidate diagnostic: MTP geometry overcount

This replay passed its request/counter/clock assertions on candidate source `772235e52` and binary
SHA-256 `8cf97fb0771caee87ac73b86186c7127ca91d3942c1dad3212f00d33d49e4840`, but it is excluded from
the final B arm because its boot receipt exposed an incorrect derived geometry:

- derived full-context entry: 415,367,168 B
- measured retained prefix entry at 8,192 tokens: 400,162,816 B
- excess: 15,204,352 B = 1,856 B/token x 8,192

The server reports 65 configured layers. Q27 has one `nextn_predict_layers` block, and the model
loader defines the retained trunk as `n_layer - nextn_predict_layers` (64). The generic whole-config
cache coefficient counted the MTP/NextN block as another full-attention cache layer, but the trunk
prefix snapshot never retains that block. The candidate therefore requested 830,734,336 B for two
entries instead of the exact 800,325,632 B.

No request from this directory contributes to the final N>=3 B reduction. Its raw logs are kept as
the receipt that found the bug before the campaign continued.
