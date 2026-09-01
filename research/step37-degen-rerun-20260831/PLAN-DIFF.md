# Door-free re-run of the defect-7 deep-context degeneration cell: what changed

Pre-registered plan: `research/step37-deepctx-degen-20260829/PLAN.md`, followed as written.
That file is NOT restated here and NOT amended; read it first. This note exists so every
difference between the original cell and this re-run is auditable in one place, written
before any generation on this box.

## The one substantive change: the binary

| | original lane | this re-run |
|---|---|---|
| memra commit | `8695bdef4a` | `3999a92a6` (`3999a92a6e18a231ce8e18fb2b6f37997b00e882`) |
| bank-v2 / sel-down8 doors | **ARMED** (`[step37-defaults] serving doors armed ON for the SlidingGatedMoe program`, banked in the original `raw/server-d7boot.log` line 7) | **removed from the engine**; boot refuses if a recipe still sets them |
| binary md5 | `09fe2d670d82931248d4b0733898e6f4` | banked in `receipts/build-3999a92a6.receipt` and in every row (`bin_md5`) |

`MEMRA_NVFP4_BANK_V2` corrupts logits at prefill, margin-dependently, and it changed
GENERATED TEXT on the step37 serving path (bisect: memra
`research/step37-reasoning-effort-20260829`, landed `75bf4ce76`; incident narrative:
darklanes `research/step37-degen-incident-20260829/INCIDENT.md`). The original lane
therefore measured a corrupted decoder. Its INSTRUMENT findings (the collage trap, the
reasoning-replayed-as-history trap, the H1 byte-identical render result) do not depend on
the decoder and stand. Its MAGNITUDES do, which is what this re-run re-measures.

`3999a92a6` is the glsweep commit the production step37 stack currently serves (darklanes
`ops/serving/artifact-registry.tsv`, `memra-server-v0.121.0-dl3a03809-glsweep`), chosen so
the re-measured quirk magnitudes describe the engine program customers actually get.

Arm identity binds on the binary **md5**, never on a commit subject or on
`system_fingerprint`: a known `build.rs` fingerprint-staleness defect lets the baked sha
lie. `receipts/build-3999a92a6.receipt` banks md5, sha256, byte length, the baked
fingerprint, the post-checkout `git log -1`, and the toolchain.

## Unchanged, byte-verbatim

- `d7-drive.py`, md5 `2af48acc2415b3530f961b6d9f99509e`, identical to the original lane's
  file. Same prompts, same arms, same n, same 48-row structure, same
  vendor-default-sampled request shape (no `temperature`/`top_p` in any payload),
  same `reasoning_effort=low` pinned on exactly the two `*low` arms and omitted nowhere an
  arm defines it, same cold-per-row rule, same nonce practice, same clean-transcript
  accept rules including PLAN.md deviations 1, 2 and rule C, same mechanical counters.
- `d7-blind.py`, md5 `1c00f2f2eb5d0ef73185ba81d437fa9c`, identical. Same shuffle seed
  (`20260829`), same blind-id scheme, same mapping-written-at-shuffle-time protocol.
- `raw/transcript-contaminated.json`, md5 `ec7d8fb022e0797cdd3bd829269fe77c`, identical.
  This is the FIXED conversation both prior cells measured (U1 sha16 `d6cfca6cdb21edd5`,
  6266 chars, the collage prompt). Re-generating it would have changed the corpus axis.
- Turn-8 rubric: `research/step37-sampled-quality-20260828/RUBRIC.md`, reused verbatim,
  including its DQ classes (EMPTY / LOOP / TRUNC) and the thinking-model judge rule
  (content when non-empty, else reasoning).

## Mechanical rebase (paths, box, port), no measurement axis touched

- `d7-aggregate.py`: its hardcoded `LANE` constant now reads `$LANE`. Nothing else.
- `d7-run.sh`: rewritten for this box. Model `/data/models/step37-flash-nvfp4` (was
  `/root/models/...`), binary under `/home/ubuntu/degen-rerun/bin/`, lock
  `/tmp/memra-degen-rerun.lock` (was `/root/gemmprime.lock`, whose co-resident lane does
  not exist here), same port 18903, same one-boot block, same fault-counter gate
  (ILLEGAL / #87 / panics), same resume-safety.
- Serving env: the original took `ENVV` from that box's `/root/agentic8.sh` convention plus
  its own gate list. Reconstructed here as `ERA_BASE` + `GATES`, where `ERA_BASE` is the
  byte-verbatim 2026-08-29-era step37 serving env carried through
  `research/toolchain-ab-20260831` into `research/perf-chain-20260831/harness/launch.sh`
  (mode `era-nodoors`), and `GATES` is the original `d7-run.sh` gate string unchanged
  (`MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5
  MEMRA_SPEC_PMIN0=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1`), plus
  `MEMRA_STEP_GEMM_PRIME_SUFFIX=0` exactly as before. `era-nodoors` IS "the original
  launch env minus the doors": the doors were the launcher's
  `MEMRA_NVFP4_BANK_V2=1 MEMRA_SEL_DOWN8=1` pair and nothing else changed with them.
- Box: a reserved non-prod dev box, 2x RTX PRO 6000 Blackwell Server (96 GB each) - the same
  hardware class as the original lane. Provider, instance id and address are darklanes-side
  facts and are deliberately not recorded in this public repo.
  Serving is loopback-only with no keyring, ledger or admin listener: public memra carries
  no accounting at all, so this is the default and not a suppression.
- Model identity: all 22 registry-listed step37 files sha256-measured on this box and
  matched 22/22 against darklanes `ops/serving/artifact-registry.tsv`
  (`receipts/model-sha256.txt`); the registry pins them to
  `stepfun-ai/Step-3.7-Flash-NVFP4@4275532ffd9a9496ff36b7a2dc4a9db1048da438`. No R2/HF
  fetch was needed: the sha-verified artifact was already on the box's NVMe.

## Two additions, both recorded before any generation

### 1. `d7-doorfree-gate.py`, prove uncorrupted text before the cell runs

A door-free commit is not by itself proof that THIS boot on THIS box decodes correctly.
Two concrete reasons the re-run needs an output oracle up front: the 2026-08-29 corruption
was margin-dependent (first token already wrong at 25 prompt tokens, still right at 613),
so long-prompt gates cannot see it; and this box carries a 100 GB `.memra-repack`
expert-stack cache written 2026-08-28, i.e. in the door era, keyed by shape and length
rather than by content hash.

The gate is the incident's own post-fix battery: `What is 17*23? Reply with the number
only.`, 8 greedy + 8 vendor-default sampled, all 16 required to contain `391`. With the
door ON that battery scored 0/8 greedy and 1/8 sampled. Greedy appears here only as the
byte-deterministic instrument that was the incident's oracle; the cell itself is sampled
throughout, and the sampled arm also banks its spec-engagement receipt.

### 2. `d7-t2probe.py`, the stop-inside-think denominator, at n=8

The original quoted "turn-2 5/8 attempts" for stop-inside-think, but its clean-transcript
builder breaks on the first accepted attempt, so one build pass can never emit 8 turn-2
attempts; the 8 came from two separate build passes (PLAN.md deviations 1 and 2). To
reproduce the same n honestly, this probe fires the builder's exact turn-2 request shape
(same history, vendor-default sampled, maxtok schedule `[4096, 4096, 4096, 8192]` x2) for
exactly 8 attempts with no early break, and banks all of them. It imports `d7-drive.py`
rather than restating it, so the request path is literally the same code.

Also recorded: the clean transcript is REBUILT on the door-free binary. The original's
`raw/transcript-clean.json` was generated by the corrupted decoder, so reusing it would
have carried the door into the re-run's history axis. The contaminated transcript is
reused verbatim because it is fixed input corpus, not model output of this cell.

## One original number that is not reconstructible

The original FINDINGS quotes a 1024-budget turn-8 think-wall rate of "20/24 across
1024-budget arms with clean/contam history". Those arms are `ctrl`, `clean`, `cleanlow`,
`ctrllow` = 32 rows, not 24, and no natural statistic over its banked `rows.jsonl` yields
20/24: `finish=length` is 28/32 pooled (8/8, 8/8, 5/8, 7/8) and
`finish=length AND content==0` is 22/32 (7, 6, 4, 5). Three-arm subsets give 20/24 only
for `(ctrl, cleanlow, ctrllow)` and `(clean, cleanlow, ctrllow)`, neither of which is a
principled grouping. RESULTS.md therefore re-measures the think-wall with its denominator
stated explicitly, pooled over all four 1024-budget clean/contam arms (n=32) and per arm,
and reports the original's number as ambiguous rather than pretending to match it.

Same class of care on the t1 baseline: the original's footnote reads "6/6 finish=length
think-only at 1024", but its own rows show `t1-s3` emitting 4034 content chars. The
defensible original magnitude is 6/6 `finish=length`, 5/6 `finish=length AND content==0`.
Both are re-measured separately here.
