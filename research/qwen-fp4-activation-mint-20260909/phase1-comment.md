Phase 1 design is ready. Stopped for owner steering before minting, as requested.

The current Qwen loader imports ModelOpt NVFP4 weights but does not execute input_scale. The proposal keeps the existing weight payload and adds a typed prefill activation program for 400 large trunk projections. Decode and target verification keep W4A8. The design also fences generation-extended cache reuse, since that state cannot be treated as an all-prefill snapshot.

No artifact minted, engine changed, GPU job started or calibration box rented. Gates are unrun. Both requested worktrees remain active under this claim. Design and source evidence are in the private companion research/qwen-fp4-activation-mint-20260909/DESIGN.md; the engine worktree has the public technical design in the same receipt namespace.
