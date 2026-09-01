# Excluded attempt 1 — unexpected GPU1 overlap

This attempt is excluded from scoring. It completed only Q27 repetition 1 / 1,024 MiB before a
separately locked grouped-regate job appeared on physical GPU1. `nvidia-smi topo -m` records the
two PRO 6000s on a shared `PIX` PCIe path, so continuing would violate the campaign's GPU1-idle
thermal and I/O regime even though memra itself remained pinned to physical GPU0.

The observed process, GPU topology, and owned-process shutdown receipts are retained in this
directory. The memra process tree was terminated cleanly, GPU0 returned to 0 MiB, the memra
ports cleared, and the campaign lock was released. No row from this attempt is used in the
scored reduction.
