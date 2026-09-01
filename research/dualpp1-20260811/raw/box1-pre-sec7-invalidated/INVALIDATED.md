# Invalidated pre-sec7 box1 battery

This directory preserves the complete battery run on source
`365e1eb71c6b635872447d4d1af1aeac4d7c087f`.

After the run completed, the orchestrator required the lane to rebase onto the sec7-fixed main tip
because the intervening constrained-decoding fail-closed latch did not rearm. The branch therefore
rebased onto local `main@1592253f`, which contains the named `2ddb9bd2` gate.

The captured raw data remains valid as a record of that earlier source, but it is not final-source
evidence and carries no merge or increment verdict. The replacement battery is stored separately
under `../box1/`.
