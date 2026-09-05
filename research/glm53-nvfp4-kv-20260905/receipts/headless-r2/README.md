# No-MTP allocation and prefix lifecycle gates

2026-09-05, local non-production SM120 device. Base HEAD72bbd416c with working
source bound by source-files.sha256; executable identities in binaries.sha256.
These are correctness gates, not model-scale inference or timing measurements.

Every GPU invocation used both existing local GPU locks:
`flock -n /tmp/memra-gpu.lock flock -n /tmp/memra-5090.lock timeout 90`.
Common explicit config for allocation/prefix tests:
`NVFP4_LATENT_TEST_CONFIG=/home/avifenesh/models/glm53-nvfp4-4o6-clamp448-v2/config.json`.

- native-gpu.log: `MEMRA_GLM53_NVFP4_LATENT=1`;
  `latent_nvfp4_gpu-e596f8749aaa9af7 --ignored --nocapture --test-threads=1`.
- tc-prefill.log: `memra_engine-a694dcb9cff670a7
  tensor_core_prefill_chain_uses_compressed_history_with_existing_prefix
  --ignored --nocapture --test-threads=1`.
- prefix-nvfp4.log / prefix-f32.log: `MEMRA_PREFIX_LATENT=1`,
  `MEMRA_GLM53_NVFP4_LATENT=1` / `=0` respectively;
  `memra_server-b2905d27d77daa7f planned_headless_prefix_roundtrips_across_capacities
  --ignored --nocapture --test-threads=1`.
- park-admission.log: same server executable,
  `park_compact_probe_and_park_site_wiring_is_real --nocapture` (CPU).
- latent-refusal.log: same engine unit-test executable,
  `checkpoint_restore_refuses_either_latent_plane_before_copy --nocapture` (CPU).

All executable paths are under target/debug/deps. Each raw log was captured with
pipefail and tee; a failed command stops the serial gate runner. Source changes
after the build were comments only. The first development prefix fixture refused
because it omitted resolved index-pool geometry; the corrected fixture explicitly
supplies pool4. It does not imply a model forward or loader test.

Production GLM measurement scope remains DFlash2 with a pinned D2T map, no native
MTP load or drafting. The included admitted-head control tests cache allocation
only, preserving generic allocator behavior for other users of the engine.
