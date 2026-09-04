# tools/ index

One line per tool, newest first. Started 2026-08-30 (lane/kv-host-spill-20260830); earlier
scripts are documented in their own headers and in docs/FLAGS.md / docs/SERVING.md rows.
When you add a tool, add its line.

- `peer-copy-direction-probe.cu`: byte-validating fabric diagnostic that contrasts Memra PP's
  producer-issued peer write with consumer-issued peer-read copies over the same retained primary
  contexts, peer grants, CUDA events, and 16 KiB to 64 MiB size ladder. It identifies whether a
  host's copy engine is sound in both directions; it is a correctness diagnostic, not a bandwidth
  benchmark.
- `box-health.sh`: the box-window boot receipt — *is this machine fit to be MEASURED on?* Run it
  FIRST on every box window. Nine documented degradation classes that report 100% utilisation
  with clean logs (persistent power cap, the false-600W ~600 MHz signature, a PCIe link silently
  negotiated down, 256 MB BAR1, out-of-range CPU affinity mask, IOMMU translated mode, ACS
  ReqRedir, P-state normalization) plus section 8's kernel peer-read probe. Idle P8 Gen1 is
  recorded, then classified from a required active-link recheck immediately after that ladder.
  `exit 0` = fit to measure, `exit 1` = do not open the window; OUTDIR is the receipt. Promoted from
  lane/glm5-tp-transport in lane/glm5-extract2; docs row in docs/TESTING.md ("Box health before
  measurement").
- `peer-read-probe.cu`: a self-contained `simpleP2P`-class KERNEL peer dereference — the only
  check that catches the driver staging SM-issued peer access through system memory while
  `nvidia-smi topo -p2p r` and `cudaMemcpy` both look healthy. Built and run by `box-health.sh`
  section 8; standalone for a fabric bring-up:
  `nvcc -O2 -arch=${MEMRA_CUDA_ARCH:-sm_120} -o peer-read-probe tools/peer-read-probe.cu`.
  Exit 0 = bytes validated both directions, 2 = WRONG BYTES, 3 = CUDA error, 4 = fewer than
  two devices, 5 = no peer-capable pair.

- `kv-host-spill-identity-gate.sh`: cached-vs-fresh byte-identity gate with a host-tier arm
  (demote -> promote -> restore must equal the tier-off cold bytes); `MEMRA_HOSTGATE_TEETH=1`
  is the forced-tiny `MEMRA_KV_HOST_MB=1` red arm whose verdict must invert.
- `kv-host-spill-failure-gate.sh`: executes the host tier's failure paths loudly (pool-full
  refusal, `MEMRA_KV_HOST_VERIFY` digest mismatch via the `MEMRA_KV_HOST_FAULT=flip-demote`
  door, pinned-alloc latch-off via `alloc-fail`) and pins byte-identical cold serving under
  each.
