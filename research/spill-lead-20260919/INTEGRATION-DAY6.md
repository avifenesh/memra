# Generic-spill pre-work — day-5/6 integration record (lead: agent-c07799, 2026-09-20)

Branch `lane/spill-integ3-20260919` from `main@b3487a03` (#519). Merged: A `2b2b90db` → B `31639532`
→ C `28d1f2b1` → D `16a5fda4` (one conflict: D had edited A-owned `research/spill-a-20260919/
storage_capture.py`; resolved by taking A's version — D's `tools/tier-battery.py` loads that file at
runtime for `--validate --schema storage-cell`, so D must not fork it) → E `3fd19501` → F `4e8b9e64`.
Lead fragments landed: engine `lib.rs` `mod ple_rows_tier; mod banked_residency;` (C) and
`[[bin]] kv-tier-gate` (B). CI on the PR is the first native compile with those two modules.

## What the rented box proved this round (all collector-locked, power cap 400 W / max 600 W recorded; overlay root, NVMe ancestry unproven; executed-not-qualified)
- **B**: `kv_tier_gate --case baseline` **8192 and 32768** same-program captures PASS (8064+128 / 32640+128
  tokens; full prefix/final state + logit + token hashes); 32k admitted vs measured 8k peak 15,008 MiB.
  `--case active` REFUSED exactly: "native CUDA materializer + scheduler binding with nonzero demote/reload
  engagement is missing" — the single named G1 seam. HostPrefix v2 compiles natively + 711 server tests;
  legacy identity/teeth/failure gates ALL GREEN (diagnostic setup recorded).
- **C**: `qwen4exp_gpu_gate --rows-via-tier` **BIT-IDENTICAL** PLE outputs + conv state (encodings=2, cases=16,
  values=6144, tier_calls=16, forced_read_chunks=144, budget_drained=true); existing gate still failures=0;
  Linux bank suite 46/46; native clippy -D warnings clean; HostExps bridge compiles natively (probe only).
- **A**: O_DIRECT accepted on overlayfs — 8/8 roundtrip/restore byte-exact (264/4097/1 MiB/4 MiB), no EINVAL,
  `backend_actual=linux-o-direct-read-write`; pinned-worker CUDA test 1/1; Linux storage 53/53, GC 6+1, catalog
  1/1; 200 GB-metadata catalog: 22.4 MB metadata, install 384 ms, touch 247 µs, zero payload reads (single run).
- **D**: collector fixes (literal `--` pass-through; explicit-refusal classification; per-cell power.limit/max
  with hash-bound CSV; storage-cell integrity validation from A's fragment); second topology fixture (Gen5 x16
  ceiling, idle Gen1); external-lock proof for legacy teeth (fragment, unapplied).
- **E**: v1.2 additive schedules (ReadyView owner/fence identity; per-trait framed-vs-logical bytes; directed
  grants bound to D's fake; BankSource fail-closed install; KvMaterializer over two record types); lead-owed
  docs: TESTING.md section, decision addendum, INDEX rows, ROUTER line (40/60); dry-run union 243 Rust + 52
  tool tests green.
- **F**: NVMe-proof options (recommendation: a writable NVMe/md0-backed bind is not obtainable inside the
  container; needs a VM/bare offer — M1 stays unproven); G2 envelope pre-registration; box hygiene.

Integrated CPU battery on this tip: **243 Rust tests pass, 0 fail**; Python battery tests OK; clippy -D warnings,
fmt, diff --check, flags census, publish census, Linux cross-check, docs census — all OK.

## Not run / not claimed
No G0–G7 gate is advanced to PASS. Active-KV demote/reload never engaged (seam missing). Storage numbers are
overlay development characterization. Runtime patches (HostPrefix v2, Hy3 guard) remain unapplied diffs.
Two host-side stops of the rented box today; receipts survived because lanes sync after every cell.
