# Fast-gate goldens after the sm_120a GDN-MMA flip (2026-11-04)

Question (impact-map finding S22:F5): every fast-gate golden was minted 2026-08-02..05, and
`gdn_mma_default_on()` made the bf16 HMMA GDN chunk scan the sm_120a default on 2026-08-21.
Do the qwen35-class goldens now false-fail, or is the 20-token pin too short to see the scan
class?

Rig: RTX 5090 Laptop, sm_120a build, commit fe262e905b (top of the impact-map fix stack),
single runs through the 5090 lease. Models present locally: q9, q35 (q35, q35slru, q35spec),
g31, g26, e4b. k27, aw35, kat, o9, o35, g12 and g12d have no local model and did not run.

## Results

1. Tier 1, naked defaults (`raw/tier1-all-mma1/`): kernel-check FULL green (228 s); q9, q35,
   q35slru, q35spec, g31, g26 and e4b all PASS, gates green and golden token-identical.
2. Tier 1 with `MEMRA_GDN_MMA=0` (`raw/tier1-q9q35-mma0/`): q9 and q35 PASS, golden
   token-identical.
3. nsys on run-gen, both arms (`raw/nsys/`, `gdn-arms-nsys.sh`): the arms run different
   kernels. Only `MEMRA_GDN_MMA=1` launches `gdn_chunk_state_mma` and `gdn_chunk_output_mma`;
   only `=0` launches `gdn_chunk_state_f32` and `gdn_chunk_output_f32`. The 20 generated ids
   are identical across the arms on q9 and on q35.

## Verdict

The goldens do not false-fail after the flip. The token pin is blind to the GDN scan class:
two numeric programs that each run their own kernels give the same 20 greedy ids on both
qwen35-class probes. A GDN prefill change that keeps argmax over 20 tokens passes these probes,
so their teeth stay what docs/TESTING.md already says: structural breaks, not numeric class.

q9 runs without the chat template, and its 20 generated ids are the first 20 ids of its own
prompt (the model starts repeating the passage). A copy is a strong attractor that little
numeric drift can move. q35 runs with `MEMRA_CHAT=1` and generates fresh text.

The golden header's commit SHA already names the defaults generation it was minted under.
The refresh stays at the next full-battery green point (a 2x RTX PRO 6000 pair; the 5090
alone runs no tier 2). At that refresh, switch q9 to `MEMRA_CHAT=1` so it stops pinning a copy.

## Also found

The first tier-1 attempt (`raw/deadlock-run/console.log`) ran no probe: under `gpulock run`
the 5090 lock is held on FD 9, and each `lockrun` step opened a fresh descriptor on the same
file, so the first kernel-check waited until the one-hour lease ended. Runs 1 and 2 above set
`MEMRA_GPU_LOCK` to a private file inside the lease; `tools/fast-gate/fast-gate.sh` now runs its
steps under an inherited lock it owns.
