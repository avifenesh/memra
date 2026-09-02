# Releasing

Every board-moving or user-facing change gets a tagged release — that's the public change record.

## Version scheme

- **minor** (v0.X.0): new mechanism or board move — kernel defaults changed, model lane landed, published number moved.
- **patch** (v0.x.Y): fixes, docs, tooling.
- No retirement notes or migration prose — state current truth plainly.

## The gate (on the rig, before tagging)

GitHub CI is compile-only (no GPU). The release gate runs locally and must be green on the tagged commit:

```bash
cargo build --release --bins
tools/release-battery.sh          # exit 0 = PASS; prints a receipt block for the tag message
```

The models are `tools/release-roster.tsv`, not a judgment call. **An `own` model — one we
published and are the only endpoint for — is REQUIRED: the battery refuses the release when
its GGUF is absent from the rig rather than reporting it skipped.** A `vendor` model is
required when present and needs an explicit `--allow-missing-vendor` when not.

That rule is owner policy (2026-08-28: *"the main models that need to be tested are my
models"*) and it exists because the previous wording here — `run-gen <each affected model>` —
let v0.118.0 ship without ever running `ornith-1.5-35b-a3b`. The change touched no kernel and
the GGUF was not on the rig, and "affected" absorbed both facts. **A gate you satisfy by not
having the file is not a gate**, and the same shape bit twice in one hour: `argmax-margin-gate.sh`
answers `SKIP … exits 0` when its probe binary is missing, so the battery now requires
`target/release/argmax-margin-probe` up front and renders any `SKIP` as a refusal.

Note what the battery does NOT run: raw `run-gen`. Its prefill-vs-decode argmax assert is a
documented landmine (`tools/argmax-margin-gate.sh:12-47`, and
`research/q8-argmax-20260806/VERDICT.md`) — batched prefill and the tokenwise decode loop are two
legitimate arithmetics, so a near-tie position flips between them legitimately.
`tools/argmax-margin-gate.sh` judges each flip against the prompt's own distribution instead.
On 2026-08-28 the first roster run used raw `run-gen` and was about to file a defect against
our own model on a position whose prefill top-2 margin was 0.0256 against a config spread of
0.5540; the calibrated gate returns `flips=0 bad=0 PASS` on the same model.

If a published number moved: update `research/tune-data/current-board.json`, run `tools/update-perf-board.py`, and commit the regenerated `docs/MODELS.md` + `docs/PERFORMANCE.md` with the change (the pre-push hook enforces this).

## What "main is shippable" means

Every clause below is a claim, and every claim names the check that proves it. A clause with
no check is not in this list. Added 2026-08-23, after `main` spent a full day unreleasable
while CI was green and two tags died of the same defect — because **the thing that fails at
release time was not exercised at merge time.** That is the shape this section exists to close.

| Claim | Proven by | Where it runs | Cost |
|---|---|---|---|
| Rust and every CUDA fatbin compile | `cargo build --release --bins` at sm_120a | `ci.yml` `build` (its own job since 2026-09-02) | 542 s measured in the serial shape, run 33582547232 |
| The workspace is clippy-zero | `cargo clippy --release --all-targets -- -D warnings` | `ci.yml` `clippy` (parallel job) · `tools/local-ci.sh` | 380 s |
| memra-server's request contracts hold | `cargo test --release -p memra-server` | `ci.yml` `server-tests` (parallel job) · `tools/local-ci.sh` | 469 s |
| Every memra-engine lib test has a caller, unfiltered, through the skip census (memra#18) | `tools/skip-census.py run ... -- cargo test --release -p memra-engine --lib`, budget 0, floor 300 passed; the three `#[ignore]` GPU tests run in `tools/local-ci.sh` on the rig | `ci.yml` `engine-tests` (parallel job) · `tools/local-ci.sh` | 358 passed in 0.39 s once built (rig, 2026-09-02) |
| A docs-only change set never skips a compile it needed | `tools/ci-change-class.sh`: fail closed on every doubt; `ci.yml` gates the compile jobs on `code != 'false'` under `!cancelled()` | `ci.yml` `changes` job; teeth `tools/test_ci_change_class.sh` (14 arms) in `gates` | under 1 s |
| **Every arch LINKS, shipped or not** | same command at sm_90a and sm_89 — the arches where `build.rs` substitutes stub `.cu` files. CI covers a SUPERSET of the release matrix on purpose: sm_89 is compile-covered but no longer shipped | `ci.yml` `release-arch-mirror` (parallel) | 553 / 558 s measured, **0 s added wall time** |
| CI never silently stops covering a release arch | `tools/arch-matrix-census.sh` | pre-push · `ci.yml` · `release.yml` guard | 0.01 s |
| Every fail-closed stub still mirrors its real twin's ABI | `tools/stub-abi-census.py` | pre-push · `ci.yml` · both tag workflows | 0.08 s |
| crates.io can publish the whole workspace | `tools/workspace-publish-census.sh` (members vs `publish.yml` list, topological order, no `publish = false` dependency) | pre-push · `ci.yml` · both tag workflows | 0.06 s |
| Those three censuses can actually fail | `tools/test_releasability_census.sh` — 15 arms, every refusal forced, plus wiring and the advisory invariant | `ci.yml` | 0.7 s |
| **Every crate packages from its own tarball** (nothing reaching outside a crate root, `Cargo.lock` still resolves) | `cargo publish --workspace --exclude memra-probe --locked --dry-run` | `ci.yml` `publish-dryrun` (parallel) | 372 s, **0 s added wall time** |
| **Every kernel the Rust side looks up is IN the fatbins for that arch** | `tools/fatbin-lookup-census.py` — `cuobjdump --list-text` vs `Engine::func` literals | `ci.yml` `release-arch-mirror`, per arch | seconds, inside a cell already running |
| Those refusals can fail, in both directions of exceptions rot | `tools/test_fatbin_lookup_census.sh` — 16 arms | `ci.yml` | 0.74 s |
| An arch with known-missing kernels can never be shipped | `tools/arch-matrix-census.sh` refusal 3 (advisory ⇒ not in `release.yml`) | pre-push · `ci.yml` · `release.yml` guard | 0.01 s |
| `tools/install.sh` asks for an asset the release actually has | the glibc floor is read from the release's own `SHA256SUMS`, never restated | — (removed the duplication rather than gating it) | — |
| A tag's version matches its Cargo, its 9 internal pins match too, and the number was claimed | `tools/release-guard.sh` | both tag workflows | <1 s |
| The guard's three refusals can actually fail | `tools/test_release_guard.sh` — 8 arms | `ci.yml` | ~2 s |
| Every `MEMRA_*` flag has a `docs/FLAGS.md` row | `tools/check-flags.sh` | pre-push · `ci.yml` | 0.55 s |
| No new public-boundary violation | `tools/check-public-boundary.py check` | pre-push · `ci.yml` | 117 s |
| Every grandfathered boundary grant still describes a live finding | `tools/check-public-boundary.py verify-allowlist` | `ci.yml` | 45.9 s |
| Release notes measure from the last tag that actually shipped | `tools/changelog.sh` + `tools/changelog-skip-tags.txt` | `release.yml` | <1 s |

### CI wall time, and what a push actually pays (2026-09-02)

Until 2026-09-02 every check in the table lived in one serial `build` job: 42 min wall per push
(run 33582547232: CPU expert companion 767 s, release build 542 s, memra-server suite 469 s, clippy
380 s, boundary check 125 s, allowlist drift 124 s), while the three parallel jobs finished in under
15 min. The chain is now nine jobs. `gates` (every text census and fixture plus the CUDA-free unit
suites) and `boundary` always run; `build`, `clippy`, `server-tests`, `engine-tests`, the three
`release-arch-mirror` cells and `publish-dryrun` run when `tools/ci-change-class.sh` says the change
set touches something a compiler, linker or packager reads. A docs, research or corpus-only push
therefore finishes in the time of the text gates. Any doubt (zero or unreachable base, empty diff,
unknown event, a crate README, anything under `crates/`, `tools/`, `.github/`, `Cargo.*`) compiles.
Wall time for a code push is the slowest single job, the publish dry-run at about 15 min. Superseded
pull-request runs are cancelled; pushes to `main` never are.

`tools/local-ci.sh` got the same treatment on the rig: in the correctness mode the CPU chain
(clippy, the memra-server suite, the memra-engine lib suite) runs alongside the GPU gates and is
joined before the stage is called green; `MEMRA_CI_OVERLAP=0` restores the serial order. Perf modes
stay serial, because a compile sharing the box with a timing cell is the co-resident noise the perf
rows refuse.

The two publish checks are **complements, not alternatives**, and the difference is worth
knowing before anyone deletes one as redundant: `--dry-run` enumerates members from
`Cargo.toml` and never reads `publish.yml`'s crate list, so it packages `memra-reference`
happily while the real publish loop skips it — it would have passed on 2026-08-22 with the exact
stale list that then burned six crates.io version numbers. The census is what compares the list
to the members; the dry-run is what proves each crate builds from its own tarball.

What this list does **not** claim, stated so nobody reads more into it:

- **Numerical correctness.** No GPU runs in CI. The exactness battery below is still the only
  thing that proves kernels compute the right answers, and it runs on the rig.
- **That a shipped prebuilt runs.** Compiling and linking an arch is not the same as its
  fatbin containing every kernel the Rust side will look up at runtime. `Engine::func` resolves
  kernels lazily and panics on a miss, so an arch-scoped `#if` that drops a kernel with no
  `#else` is a **runtime** failure on a shipped binary that every compile gate here passes.
  That gap is now measured by `tools/fatbin-lookup-census.py` — which is how sm_89's 20 missing
  kernels were found and why sm_89 stopped shipping. See "Known unshippable arch" below.
- **sm_100a (B200): runtime-qualified from source and auto-detected; no prebuilt is released.**
  The 2026-08-23 state was "it does not compile" (two
  stub-gate polarity bugs, fixed then, plus a wrong `__CUDA_ARCH__ >= 1000` guard in
  `cu/mmq_q8_0_f32acc.cu` that admitted the one arch rejecting the f8f6f4 MMA — fixed by the
  b200-prep lane, `>= 1200`). Current state, measured by a 29-cell per-arch census (13 fatbin + 16 static-lib TUs)
  (`research/glm5-b200-prep-20260901/`): every cell of a `MEMRA_CUDA_ARCH=100a` build compiles clean,
  `ci.yml`'s `release-arch-mirror` carries a `100a` compile cell so it stays that way, and the
  fatbin census passes with two DECLARED exceptions (`qmatvec_gemm_nvfp4_fp4` — the `MEMRA_FP4`
  door asserts the 120a property on sm_100a builds, and kernel_check's Stage-C FP4 arm keys on
  the same property so 100a records a skip cell; portable builds keep their pre-existing
  early-return/refusal path; `qmatvec_gemm_q8_0_wgmma` — all call sites
  compiled out under `cfg!(memra_hopper_mma)`). The 2026-09-01 hardware closure at
  `69a2eb3684e1` passed the sealed synthetic battery on one NVIDIA B200, then pinned-checkpoint
  model, K=1..8, sampled serving, concurrency, admission, and rollback gates. `detect_arch()` now
  maps compute capability 10.0 to `100a`. Default NVFP4 W4A8 is `NativeQualified` on the pinned
  Qwen3.5-9B artifact. Raw-layout W4A4 is correct but remains explicit because it measured 0.521x
  raw W4A8 prefill. Block-FP8 is `NativeReference` only: its explicit B200 twin is correct and
  serves the pinned official Qwen3.8-27B-FP8 checkpoint, but measured 0.173x the established
  fallback with worse teacher-forced NLL. It does not default on. The release installer still
  refuses B200 before network access because the release manifest publishes no sm_100a binary.
  Owner and manifests: `research/b200-kernel-twins-dry-20260901/README.md` and `receipts/`.
- **The glibc/OS axis.** `release.yml` builds two glibc floors; CI mirrors only the arch axis.
  The OS changes the libc requirement of the shipped binary, not the source set or the symbols,
  so one OS cannot miss a compile or link failure the other would catch.

### sm_89: the panic claim was WRONG. What was real, and what got fixed.

**Corrected 2026-08-23, after it had been published.** An earlier version of this section — and
the v0.108.0 release notes — said the sm_89 asset panics on first batched GDN decode. It does
not. All three shipped arches are usable on default configuration.

**How the wrong claim was reached, because the method matters more than the conclusion.**
`cu/hybrid.cu:1575-2238` really does drop 17 GDN/varlen kernels with no `#else`, and
`tools/fatbin-lookup-census.py` really does flag them. A grep for `portable_mma_gated` inside
`src/hybrid_forward.rs` finds nothing, and that was reported as "the batched path is unguarded."
**The guard is one call deep.** `use_vl` (`:3937`) requires `e.gdn_mma_enabled(c)`, and
`gdn_mma_enabled` (`src/lib.rs:24097`) *begins* with `!portable_mma_gated()`. Each of the five
`*_vl8` wrappers has exactly one caller, all inside `use_vl` at `:4058-4099` — so on sm_89 the
block is skipped and `linear_attn_prime_core_pad_view` serves. The single-sequence twins are
guarded identically at `:23967` and `:24797`.

**A name-level census is not a reachability check.** It over-approximates on purpose (no false
negatives), which means every hit needs its call site traced before it is called a defect. That
tracing is what `tools/fatbin-lookup-exceptions.txt` records, per symbol.

**What WAS real: three env doors that ignored the arch.** `MEMRA_FA3=1` (`src/lib.rs:18527`,
`src/hybrid_forward.rs:3090`) and `MEMRA_FP4` (`src/lib.rs:15979`) had force arms — `Ok("1") =>
true` — that never consulted the build, so a forced path reached a kernel the arch omits and died
in `Engine::func` with `kernel <name> not in any fatbin`, several frames from the switch the
operator flipped. Default configuration never reached them.

**Fixed:** all three now call `refuse_portable_force()`, which fails at the switch with a named
message ("`MEMRA_FA3=1` forces a kernel path this build does not contain: it needs the sm_90a
fa3/bf16 kernels…"). Same idiom as `gemm_fatbin_bytes`'s existing refusal for
`MEMRA_GEMM_FATBIN` — one shape for "this switch cannot work on this build."

**sm_89 therefore ships.** Per owner ruling: fix the guard, do not drop the arch — dropping one
removes a whole GPU class, and "we stopped offering it" is a worse answer to a user than "it
works." It was briefly out of `release.yml`'s matrix while this was being traced; it is back.

#### The gate itself: `tools/fatbin-lookup-census.py`

Runs per arch inside `ci.yml`'s `release-arch-mirror` cells, after the build:
`cuobjdump --list-text` (a **host** tool — no GPU) over every fatbin `Engine::func` searches,
diffed against the `.func(…)` / `.func_g(…)` literals in the crate's Rust sources. Seconds, no new
job, no added wall time. Refusals: no fatbins, a missing module, a fatbin yielding zero kernels,
zero Rust lookups, an arch mismatch, a looked-up name in no fatbin and not declared, and
**exceptions-file rot in either direction**.

Measured on real builds of all three shipped arches: **sm_120a 1 declared, sm_90a 1 declared,
sm_89 20 declared, 0 unexcused anywhere.** Every declaration carries its reachability argument.

**It must never be landed without its exceptions file.** The sm_120a entry alone
(`qmatvec_gemm_q8_0_wgmma`, behind a compile-time `cfg!(memra_hopper_mma)` that is false there)
would otherwise red CI on the arch we actually serve, on a false positive.

**Residual gap, stated plainly: this proves the kernel is IN THE FATBIN, not that the path
EXECUTES.** Proving execution needs real Ada and Hopper silicon; the rig is sm_120a and prod boxes
are untouchable. That is the honest limit of everything in the table above.

#### Superseded: the original PROPOSED form of this gate

Owner ruling 2026-08-23: this class must be gated, and **the gate cannot be a compile.** Every
gate in the table above proves the tree builds and packages; none proves the built thing *runs*.
A tag that compiles six ways and then panics at kernel lookup is still unshippable, and it fails
in a customer's first request rather than in CI. Recorded here as an open gap with a named owner
so it is not rediscovered.

**Proposed gate — read the fatbin nvcc actually emitted, not the preprocessor's likely output.**
After each `release-arch-mirror` cell builds, run `cuobjdump --list-text` over the produced
fatbins, collect the `extern "C" __global__` names actually present, and diff against the set of
string literals passed to `Engine::func`. `cuobjdump` is a **host** tool — no GPU.

| | |
|---|---|
| **Cost** | seconds. It piggybacks on the `release-arch-mirror` cells that already run (553 / 558 s), so **no new job and no added wall time** |
| **Needs** | nothing CI does not already install (the CUDA toolkit is there for nvcc) |
| **Runs** | inside the existing arch cells, per arch — the arch dimension is the whole point: sm_120a was green at 109 `kernel-check` cells while the portable arches were the broken ones |

Why not the alternatives, stated so the choice is auditable:

- **A preprocessor-only static census** (evaluate the `#if`s ourselves, then diff) infers what
  nvcc *would* emit. Same cost, strictly less truth — it can disagree with the toolchain. Prefer
  reading the artifact.
- **A boot smoke that loads a model and issues one token** is the gate that would truly prove
  the path runs, and **we cannot run it**: it needs real sm_89 (Ada) and sm_90a (Hopper) silicon.
  The rig is sm_120a. Prod boxes are untouchable. This would need rented Ada/Hopper hardware, and
  it is the honest answer to "does it run" — the proposed census proves only that *the kernel is
  in the fatbin*. That residual gap should be stated whenever sm_89/sm_90a are called supported.
- **On the rig:** a lookup panic is an exactness question, not a timing one, so the rig's
  throttle does not disqualify it — but the rig cannot build or run sm_89/sm_90a SASS, so it
  cannot answer this for the arches that are actually broken.

Two design constraints the gate must satisfy, both learned the hard way this week:

1. **The empty case must REFUSE, not pass.** If no fatbin is found, or no `Engine::func` literals
   are extracted, the census must fail loudly — the same rule `tools/check-flags.sh --list`
   follows. A census that can silently answer "nothing to compare" is green precisely on the
   machine where it matters least.
2. **It needs a declared-exceptions file, or it will be a false-positive machine.** Some lookups
   are *correctly* arch-conditional — `src/lib.rs:23967` guards `gdn_chunk_attn_f32` behind
   `!portable_mma_gated()` and is right to. A naive name diff flags those too, and a gate that
   cries wolf gets bypassed. The `docs/FLAGS.md` pattern applies: a census plus a file of
   declared arch-conditional lookups, so a *new* unguarded one is what shows up.

Until it exists, sm_89 and sm_90a assets are compile-proven only.

### Open releasability findings, with owners

Recorded rather than swept: an honest open list is worth more than a rushed one, and every entry
here is a measured finding with a location. None of them fails for a user today except where
noted.

| # | Finding | Fails at | Owner |
|---|---|---|---|
| 1 | **`tools/check-flags.sh`'s grandfather list is inert but still non-blocking.** `research/docsync3-20260811/flags-drift.txt` holds 75 names; **all 75 are now documented in `docs/FLAGS.md`**, so every exemption is dead. Verified by probe in a throwaway tree: with a baselined name present in the list, deleting its `docs/FLAGS.md` row still **exits 0** (the name prints under "uncovered runtime names" as a non-fatal line); removing it from the baseline makes the same tree exit 1. **For those 75 names, deleting documentation keeps the gate green.** Changes gate semantics, so it needs an owner ruling, not a lane. The fix shape is the one `tools/fatbin-lookup-exceptions.txt` already uses: give the list a drift check that refuses a dead entry. | merge (silently not at all) | owner ruling |
| 2 | **`MEMRA_FP4=1` panics on sm_89/sm_90a.** The Stage-C FP4 gate (`src/lib.rs:15979`) is an ENV door, not an arch predicate, so it can reach `qmatvec_gemm_nvfp4_fp4`, which portable arches do not compile. Declared in `tools/fatbin-lookup-exceptions.txt` because no default path reaches it. Fix: also test `portable_mma_gated()` and refuse by name. *(Since resolved twice over: the portable refusal landed 2026-08-23 (`refuse_portable_force`), and lane/glm5-b200-prep-20260901 replaced the enumeration with the 120a PROPERTY after finding the same door — plus kernel_check's Stage-C arm — reachable on sm_100a builds.)* | first use, opt-in only | engine lane |
| 3 | **sm_89's 20 missing kernels** — `cu/hybrid.cu:1575-2238` has no `#else`, and the batched GDN call site `src/hybrid_forward.rs:3912` is unguarded where its single-sequence twin (`src/lib.rs:23967`) is guarded. sm_89 no longer ships because of it. | first batched GDN decode | engine lane |
| 4 | **The stub-gate polarity bug is a CLASS, not a one-off — and this is the entry that matters most.** `build.rs`'s substitution chain guards each sm_120a-only MMA translation unit; two of the three tested `portable` (an ENUMERATION of `89\|90a`) where the property is `!= 120a`. Both were fixed 2026-08-23, and the second was found only by fixing the first and watching the ptxas failure MOVE (`mmq_nvfp4_w4a8.cu` → `mmq_fp8_blk.cu`, ~40 then ~400 sites). **The cost was never sm_100a specifically: an enumeration means every future non-120a arch silently inherits the breakage.** `cu/mmq_fp4.cu` in the same chain always had the correct test — the inconsistency was the tell. The durable fix is a census asserting every stub-substituted TU is stubbed on every arch lacking its instruction class, rather than fixing them one at a time. **sm_100a still does not compile after both fixes**: a THIRD unit, `cu/mmq_q8_0_f32acc.cu` (~256 sites), fails for a different reason — its `build.rs` entry says it "needs no portable stub" because an in-TU `__CUDA_ARCH__ >= 1000` guard covers its f8f6f4 arm, and that threshold is wrong (1000 *is* sm_100a, so the guard admits the arch that rejects the instruction). Polarity was separable and cheap; sm_100a is not. *(Since resolved: lane/glm5-b200-prep-20260901 fixed the threshold to `>= 1200`, sm_100a compiles 29/29 census cells and ci.yml carries a 100a compile cell — see the sm_100a bullet above for current state.)* | explicit opt-in build today; the NEXT new arch, silently | engine lane |
| 5 | **Published-crate `cargo test` escape.** `crates/memra-server/src/main.rs` (9 sites) and `crates/memra-gguf/src/source.rs:2233` read `{CARGO_MANIFEST_DIR}/../../research/...`. `cargo publish --verify` builds but does not test, so this is not a publish blocker; it breaks a contributor building from the published tarball. | consumer, from a crates.io tarball | release lane |
| 6 | **Three orphan fixtures with no automated caller** — `tools/test_check_hardware_gate.py` (the teeth for `tools/check_hardware_gate.py`, which `tools/hooks/pre-push` *does* invoke in `step-pro` mode), `tools/test_cpu_expert_shm.sh`, `tools/check-batch-exact.py`. A live push gate whose teeth nobody runs. | nothing runs it | release lane |
| 7 | **`release.yml`'s stable-name upload is silently conditional.** It is produced only by the `ubuntu-22.04 && 120a` coordinate under `if: env.STABLE_NAME != ''`, so if that coordinate ever changes the upload skips silently and `if-no-files-found: error` never fires. That asset is the `cargo-binstall` / `install.sh` target. | release, silently | release lane |

Three rules for anyone adding to that table:

0. **A new pre-push arm carries its stub in the fixture's `stage()` in the SAME commit.**
   House law, adopted 2026-08-23. `tools/test_flags_guard.sh` drives a real `git push` through
   the real hook into a throwaway repo and stubs every non-census arm green, so a refusal there
   can only be the flags census. An arm that lands without its stub reds that fixture for every
   lane behind it — which is exactly what happened when the releasability arm landed. The stub
   block in `stage()` is a contiguous section of `printf` lines plus one `chmod`; a fourth is
   mechanical. A `.py` census needs a `#!/usr/bin/env python3` shebang and the executable bit,
   because the hook invokes the script directly. `EXPECTED_ASSERTIONS` at the bottom of that
   fixture is a floor: stubs do not change it, new *arms* do — bump it in the same commit or the
   fixture exits 3 as BROKEN. Do not ping another lane to do this; land it yourself.
1. **Run it against the real thing, not a fixture.** On 2026-08-23 `tools/test_release_guard.sh`
   had five green arms while `main` could not be released, because all five inspected fixtures.
   Fixtures belong in the *teeth*, which force refusals; the census itself must read the real
   manifests and the real workflow files.
2. **Latency is a real cost.** A gate that lengthens every merge gets bypassed — that is how
   `MEMRA_SKIP_PERF_CI` became a habit. Anything cheap (<1 s) goes in pre-push so the defect
   cannot even be pushed; anything expensive goes to CI, where nobody waits on it; and anything
   that needs a matrix goes in a **parallel** job, which costs runner-minutes instead of wall
   time. The releasability censuses total 0.15 s and are in both.

## Claiming a version number (parallel sessions)

Several sessions release from this repo in the same day. The week of 2026-08-22 that
produced three tag races: v0.102.0, v0.104.0 and v0.104.1 all went out while `Cargo.toml`
still said an older version — binaries self-reporting the wrong version under a "Latest"
badge, crates refused, three renumbers, one near-bad deploy. All three are annotated
`[SKIPPED — version mismatch]` on GitHub; v0.103.0 is the nearest release where tag,
binaries and crates agree.

**v0.105.0 and v0.106.0 are a different failure and must not be filed with those three.**
Neither has a GitHub release, and neither is a version-discipline story:

- **v0.105.0** — release run `32581590491` died on `rust-lld: error: undefined symbol` in
  the sm_89 matrix cell. `cu/mmq_fp8_blk_stub.cu` had lost ABI parity with
  `cu/mmq_fp8_blk.cu` at `58ce746ad3`. It also had a Cargo mismatch, which is what it was
  originally blamed on — wrongly. A correctly bumped v0.105.0 would have failed identically.
- **v0.106.0** — proved that: Cargo and tag agreed, the guard passed, and release run
  `32608092743` died on the *same* link error. Its publish run `32608092737` also
  half-finished, leaving **six crates permanently live at 0.106.0** (gguf, kv, runtime,
  sampling, tokenizer, validate) with engine, lanes and server missing, because
  `publish.yml`'s crate list had gone stale against the workspace.

`main` was therefore **unreleasable from 2026-08-22 morning until 2026-08-23**, with CI
green throughout. Both defects now have gates — see "What main is shippable means" above.

### `release/claim-v0.106.0` stays on origin. Do not tidy it away.

Two separate facts, because they are often collapsed and only one of them is enforced by
anything:

1. **The number 0.106.0 is permanently consumed.** Six crates went live on crates.io at
   0.106.0 before that publish run died (`memra-gguf`, `memra-kv`, `memra-runtime`,
   `memra-sampling`, `memra-tokenizer`, `memra-validate`). crates.io versions are immutable, so
   nothing can ever republish that version. This is a property of the registry, not of this repo.
2. **`release/claim-v0.106.0` is what enforces it here.** The `release` job sweeps a claim
   branch once its release lands; v0.106.0's never did, so the claim remains — and
   `tools/release-guard.sh` refuses any tag without a claim but happily accepts a tag whose
   claim already exists. That branch is therefore the only thing in this repository that will
   stop a second `v0.106.0` tag being cut. **It is load-bearing, not residue.**

So it reads as a false "in-flight release" in the `release/claim-*` namespace. That is the
lesser harm, and it is deliberate. **A lane cleaning up dead `release/*` branches must skip
this one.** If it is ever deleted, a different permanent block has to replace it first.

### The six orphaned 0.106.0 crate versions are deliberately NOT yanked

**Owner ruling 2026-08-23: do not yank them.** The exposure is nil rather than merely
tolerated, which is what makes this the right call and not a convenient one:

- **Nothing reaches them by accident.** Cargo resolves to the newest compatible version, so
  anything depending on `memra-*` gets 0.107.0.
- **Nothing reaches them quietly.** An explicit `=0.106.0` pin fails LOUDLY at resolve time —
  `no matching package named memra-reference` — rather than silently building a half-published
  set, because the three crates that complete the workspace were never published at that
  version.

A partial publish that cannot be reached by accident and cannot fail silently is not a hazard
worth a destructive, irreversible operation. **The `cargo yank` commands for these six are
deliberately not to be run.** Do not resurrect them as a to-do.

The fix is a claim that is atomic on origin, so two sessions can never hold the same
number:

```bash
# 1. CLAIM the number first — before bumping, before tagging:
git push --force-with-lease=refs/heads/release/claim-vX.Y.Z: \
    origin HEAD:refs/heads/release/claim-vX.Y.Z
```

The `--force-with-lease=<ref>:` form (empty expectation) tells origin "create this ref
only if it does not exist" — a server-side compare-and-swap. If the push is refused,
the number is taken: pick the next one and claim again. Do not delete another session's
claim branch, and do not tag a number you did not claim — `tools/release-guard.sh`
fails both tag workflows when the claim branch is missing. The `release` workflow
sweeps the claim branch after the release lands, so `release/claim-*` on origin always
reads as the list of in-flight releases.

## Cutting the release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Before tagging, in this order:

1. Claim the number (section above) — the guard refuses an unclaimed tag.
2. Bump `[workspace.package].version` AND the pinned `[workspace.dependencies]`
   versions in the root `Cargo.toml` to `X.Y.Z` (one sed pass), commit, and make sure
   that commit is what the tag points at. `tools/release-guard.sh` runs first in BOTH
   tag workflows (`release.yml` gates the build matrix on it; `publish.yml` runs it
   before the CUDA install) and refuses a tag whose checkout says a different version —
   the refusal is the guard doing its job, not an obstacle: a mismatched tag ships
   old-version binaries under a new-version name.

That's it. Two workflows fire on the tag:

- `release` — builds the prebuilt binary matrix (glibc 2.35/2.39 x sm_120a/sm_90a/sm_89,
  fatbins embedded — self-contained), drafts the changelog from conventional commits since
  the previous tag (`tools/changelog.sh` — `perf:`/`feat:`/`fix:`/`config:`/`docs:` grouped;
  `data:`/`chore:` dropped as research-log noise), attaches tarballs + `SHA256SUMS` + the
  stable-name `memra-server-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` (the cargo-binstall /
  `tools/install.sh` target), and publishes the GitHub release. Edit notes on GitHub
  afterwards if the draft needs headline or context — draft is floor, not ceiling.
- `publish` — per-crate `cargo publish -p <crate> --locked` to crates.io in dependency
  order, skipping versions already live (registry API check) and waiting out crates.io's
  new-crate burst limit on 429 (~1 new crate/10 min; learned on the v0.69.0 first publish,
  which shipped 5/9 before a 429 and could not resume under the old all-or-nothing
  `--workspace` form). The step is idempotent — rerunning a partly-published tag finishes
  the remainder. Dry-run first without tagging: Actions → publish → Run workflow
  (`workflow_dispatch` runs the full package+verify with no upload and no token); the
  dispatch `publish=true` input is the recovery door that runs the REAL publish on a ref
  when a tag's run needs finishing.

## Publishing to crates.io — one-time setup (owner)

- crates.io account (GitHub login), email verified.
- Generate an API token at <https://crates.io/settings/tokens> with `publish-new` +
  `publish-update` scopes.
- Add it as the repo secret `CARGO_REGISTRY_TOKEN` (Settings → Secrets and variables →
  Actions). The publish workflow fails with a pointed error if it's missing.
- Nine `memra-*` crate names are publishable (verified available 2026-08-04,
  `research/crates-release-20260804/`); `memra-probe` stays unpublished (`publish = false`
  — dev spike). Note the v0.69.0 first publish landed **5 of 9** before crates.io's
  new-crate burst limit returned 429 — which is why the workflow is now per-crate and
  resumable (skip-if-live + backoff + the `publish=true` dispatch door). Do not state "all
  nine crates are live" from this doc; check the registry.
- Publishing is irreversible (yank ≠ delete): the tag must already have passed the on-rig
  gate battery like any release.

Preview the draft locally before tagging:

```bash
bash tools/changelog.sh            # previous tag -> HEAD
bash tools/changelog.sh v0.1.0     # explicit range
```

## A commit subject is a PUBLISHING SURFACE

`tools/changelog.sh` drafts release notes from commit subjects **verbatim**, and `release.yml`
publishes that draft. So a subject line is not an internal note — it is copy on a public release
page, and whatever number it contains becomes the number readers quote.

This is not hypothetical. v0.108.0's notes shipped with **"+19.7% on a current-generation host"**
from `3b49a524fb`'s subject, plus **"~20% for ornith"** from the release commit's own subject —
mine. Both were the friendliest window of a multi-window study whose own body concludes
**+3–7%** and explicitly warns against quoting its best window. The notes had to be edited after
publish to name both figures and the real range.

Rules that follow:

1. **A subject line states WHAT changed. A number belongs in the body, next to its receipt.**
   `perf(spec): verify-graph default ON for the GDN+MoE family` is a good subject. Appending the
   best window is not.
2. **If a subject must carry a number, it carries the conclusion, not the best cell.** The range,
   or the figure the study itself concludes with.
3. **Read the drafted notes before the release is announced**, and edit them if a subject leaked a
   number — `gh release edit <tag> --notes-file …`. The docs already say the draft is a floor, not
   a ceiling; this is the main thing that makes editing it necessary.
4. **A correction goes where the reader looks.** Do not rewrite pushed history to hide it: edit
   the release page, and say what the earlier text claimed. v0.108.0's asset guidance was
   corrected in place that way after I published a false claim about `sm_89`.

## Commit prefixes that feed the changelog

`perf:` kernel/throughput wins · `feat:` new capability · `fix:` correctness/bugs · `config:` defaults/flags · `docs:` documentation · `data:` tune-data rows (excluded) · `chore:` plumbing (excluded).
