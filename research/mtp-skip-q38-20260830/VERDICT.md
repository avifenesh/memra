# mtp-skip-q38 (2026-08-30): MEMRA_MTP_SKIP=1, all gates GREEN

Owner call 2026-08-30: "specifically for q38 skip loading the mtp head is good, we want the
extra space" (goal: session headroom on the serving card). The q38 production shape arms
dspark/DFlash2, which disables the MTP spec arm, yet the loader still uploaded the whole
embedded NextN block; its only live consumer was the FR-Spec trimmed rows, which on this
tied-head family gather from the trunk `output.weight`, not from `blk.N` tensors.

Binary pins: branch `eb5627b14`+`29c8a0f5b` (worktree mtp-skip-q38) vs pre-change baseline
`abc4014151d191b1e1d3afd7fc4853e9a51abd48` (origin/main at branch point). Model:
`/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (arch qwen35, n_layer 65,
nextn 1, n_trunk 64). Draft: `/data/ai-ml/models/q38-dflash2` (safetensors DFlash2 export;
NOTE the pinned-name dir `q38-dflash2-50307d4c` on this rig is EMPTY, the live export is the
unsuffixed dir, same as the production launcher's `$ROOT/models/q38-dflash2`). Ranks:
`q38-ranks-sxc32768.gguf.txt`. Rig: local RTX 5090 laptop, exactness-only, all boots behind
`/tmp/memra-5090.lock`. No timing claims anywhere in this lane.

## What shipped

- `MEMRA_MTP_SKIP=1` (strict 0/1, refuse-loud otherwise) skips the embedded MTP/NextN block
  at load; loud `[mtp-skip]` receipt quotes the exact on-disk bytes not loaded.
- Under `MEMRA_FRSPEC_TRIM`, the trimmed rows + d2t build anyway into a dedicated
  `DflashTrimHead` stub. NOT an `MtpHead`: `model.mtp` stays `None`, so `mtp_spec_capable`
  and every MTP forward path are off by construction. dflash.rs consumes exactly
  `shared_head_head` + `d2t` + `d2t_from_target_head` from the MTP struct and NOTHING else
  (both borrow sites audited; no norms, no eh_proj), so the stub carries exactly head + d2t.
- All refusals fire BEFORE any tensor upload (seconds, not after a 16 GB load).

## Gate results

1. **Default-OFF proof.** Flag unset: boot receipt lines (`[frspec-trim]`/`[dspark]`)
   byte-identical between branch and baseline binaries (`smoke/receipts-A-vs-C.diff` empty);
   run-gen greedy MATCH on 3 real prompts baseline-vs-branch (`rungen/`, hashes
   1994729ba085d15c / 610a57ee01165071 / 39bda8058c413238); flag-unset arm prints no
   `[mtp-skip]` line.
2. **Skip-ON, production shape (dspark + trim).** `smoke-matrix.out`: three full
   `tools/dspark-serve-smoke.sh` runs ALL GREEN (A=branch flag-unset, B=branch skip-on,
   C=baseline). Arm B boots with
   `[mtp-skip] ... skipping 1 embedded MTP/NextN block(s) blk.64..=blk.64 (~227 MiB of
   weights not loaded)` +
   `[mtp-skip] FR-Spec stub draft head built: 32768 rows of main output.weight (Q5_K)` and
   the SAME `[dspark] gate: DFlash2 draft head TRIMMED to 32768 rows` as arm A. Greedy
   completions byte-identical across A/B/C on all 3 prompts, spec-on AND plain
   (on==off==cross-arm, hashes in `smoke-matrix.out`); spec engagement receipts identical
   per prompt across arms: r1 rounds=27 drafted=101 accepted=70, r2 39/113/58, r3 34/108/62.
   Sampled engagement, LOW=2 concurrent pair, LOW=1 negative control, and `[dspark-acc]`
   log receipts green in every arm.
3. **VRAM.** Same production-shape boot pair, per-process `nvidia-smi` after ready:
   off=15522 MiB, on=15298 MiB, **reclaimed 224 MiB** (`vram/`), consistent with the ~227
   MiB on-disk skip figure. The owner's 0.3-0.5 GB estimate assumed a bigger head; this
   artifact's single NextN block is 227 MiB on disk (one block, tied head, quantized FFN).
   At the launcher's ~160 MB/session state cost that is ~+1 session of headroom, not +2.
4. **Refusal teeth, all EXECUTED with stderr quoted** (`refusals/`, `serve-policy/`):
   garbage value (`MEMRA_MTP_SKIP=2`), skip+`MEMRA_MTP_DRAFT` (contradictory),
   skip+trim with empty d2t, skip+trim on an own-per-block-head artifact (metadata-faithful
   fixture built by `make-ownhead-fixture.py`; refusal quotes
   `blk.64.nextn.shared_head_head.weight` and the 0/248 wrong-head receipt), and
   skip+explicit `MEMRA_SERVE_SPEC=1` with no dspark drafter (worker FATAL "cannot be
   honored"). Skip+default-spec+no-dspark boots with the loud
   `[mtp-skip] gate: serves PLAIN decode` line and a greedy request returns
   `.usage.spec == null` (the stub does NOT satisfy `mtp_spec_capable`).
5. **Suites.** cargo test --release -p memra-server: 407 passed (incl. the new
   `mtp_skip_verdict_tests`); dflash_parity + dspark_q38_parity bins: 11+11 passed;
   memra-gguf/tokenizer/validate/sampling libs: 162/16/48/3; cpu_experts: 4. fmt --check,
   `git diff --check`, flags census (`tools/check-flags.sh`), and
   `tools/test_flags_guard.sh` (14/14) all green.

## Stale gate found and fixed in-lane (29c8a0f5b)

`tools/dspark-serve-smoke.sh` tooth 5 still asserted the one-non-demotable-row block that
`62f48ac1c4` hardened on 2026-08-24 and `c4432f4a4` refuted and removed ON MEASUREMENT the
next day (LOW band now stacks dspark rows; seam `MEMRA_DSPARK_SAMPLED_WAVE=0`). The policy
commit updated worker.rs + FLAGS.md but not the smoke, so the gate had been red-on-main for
every binary since 2026-08-25; this lane's A/B/C matrix caught it because the PRE-CHANGE
baseline failed the tooth bit-for-bit alongside the branch. The tooth now asserts the
shipping default (observed in all six boots of the matrix).

## Surprises worth keeping

- dflash.rs's MtpHead consumption is exactly `{shared_head_head, d2t, d2t_from_target_head}`;
  nothing else. The worker boot receipt reads the same three.
- `MEMRA_MTP_HEADS=0` is silently ignored (`.filter(|&n| n > 0)`); FLAGS.md now carries the
  trap note pointing at `MEMRA_MTP_SKIP`.
- run-gen's GGUF path loads `without_mtp`, so it can never exercise the flag; server boots
  are the only executable surface for the refusals.
- bash 5.3 + `set -u`: `local a=$1 b="$a-x"` in ONE local statement errors ("unbound
  variable"); split the declarations (bit `vram-receipt.sh`).
