# Releasing

How to cut a memra release, from claiming the version number to published binaries, crates and
notes. For maintainers. The evidence a tag needs, and how to produce it, is defined in
[Release qualification](RELEASE-QUALIFICATION.md); this page is the procedure around it.

## Release in one screen

1. [Claim the version number](#claiming-a-version-number) on origin.
2. [Prepare the release commit](#prepare-the-release-commit): version, the 11 internal pins,
   `Cargo.lock` and its boundary grant. CI green on the release PR, publish dry run green.
3. [Qualify the release commit](#qualify-the-release-commit): byte check, one sealed record per
   OS profile, the push-gate cell records and a qualification note, each banked in a data commit.
4. [Tag](#tag) the last banked commit and push the tag.
5. [The tag workflows](#what-the-tag-workflows-do) build, sign and publish the GitHub release and
   the crates. Read the drafted notes and edit them.

Every board-moving or user-facing change gets a tagged release; the tag is the public change
record.

## Version scheme

- **Minor** (`v0.X.0`): a new mechanism or a board move: a kernel default changed, a model lane
  landed, a published number moved.
- **Patch** (`v0.x.Y`): fixes, docs, tooling.
- State the current truth plainly; no retirement notes or migration prose.

## Claiming a version number

Several sessions release from this repo, sometimes on the same day. Claim the number on origin
before bumping or tagging:

```bash
git push --force-with-lease=refs/heads/release/claim-vX.Y.Z: \
    origin HEAD:refs/heads/release/claim-vX.Y.Z
```

The `--force-with-lease=<ref>:` form (empty expectation) asks origin to create the ref only if it
does not exist: a server-side compare-and-swap. If origin refuses, the number is taken; pick the
next one and claim again. `tools/release-guard.sh` fails both tag workflows when the claim branch
is missing.

- Do not tag a number you did not claim.
- Never delete a `release/claim-*` branch, yours or another session's. A claim stays after the
  release lands: it reserves the number and lets later publish retries pass the same guard. Lane
  cleanup must preserve every claim ref; check the tag and the publication state to tell a
  finished release from one in flight.
- **Broken tags.** Never renumber over a broken tag, never delete a tag and never recut a release
  under the same number. Mark a release whose binaries disagree with its tag as skipped for
  version mismatch and as a prerelease, then release the next number.
- crates.io versions are immutable. The six crate versions a half-finished publish left live at
  0.106.0 are deliberately not yanked (owner, 2026-08-23); do not reopen that as a to-do. The
  2026-08 tag races and the reasoning are in the [history](history/RELEASING.md#claiming-a-version-number-parallel-sessions).

## Prepare the release commit

The release commit is the commit the tag's record qualifies (its *tested commit*). It holds only
the version change:

1. Set `[workspace.package].version` and every pinned memra-* `version = "=X.Y.Z"` in the root
   `Cargo.toml`'s `[workspace.dependencies]` (11 today) to the new version in one pass.
   `tools/release-guard.sh` refuses a tag whose version or any pin differs, and refuses an empty
   pin set.
2. Refresh `Cargo.lock` (`cargo update --workspace` updates only the workspace entries).
3. Re-pin the `Cargo.lock` grant in `tools/public-boundary-allowlist.jsonl`: set its `sha256` to
   the new file's (`sha256sum Cargo.lock`). The grant is pinned to the file's bytes, which the bump
   changes; `python3 tools/check-public-boundary.py verify-allowlist` reports a stale pin.
4. Commit it as `release: vX.Y.Z` and open a PR to main. CI must be green on that commit.
5. Dry-run the crates publish on the release branch: Actions, `publish`, Run workflow, with
   `publish` left false (`gh workflow run publish.yml --ref <branch>`). It packages and verifies
   the workspace and reads the registry state; it uploads nothing and needs no token.

## Qualify the release commit

<a id="the-gate-on-the-rig-before-tagging"></a>

The tag workflows refuse a tag without a sealed, content-bound record of the release commit. A
bare `tools/release-battery.sh` run on a development card qualifies nothing. Steps, each detailed
in [Produce a record](RELEASE-QUALIFICATION.md#produce-a-record):

1. **Byte check.** Dispatch `gpu-ci.yml` with `candidate=<40-character release commit>` and
   `build_reuse_only=true`, and build the `ubuntu-24.04` profile on the capture host. Capture only
   when the two `build.json` files name the same six digests; `qualify-release.py capture
   --reference-build` refuses otherwise.
2. **Records.** Build both OS profiles (`ubuntu-22.04`, `ubuntu-24.04`) and capture each on one
   leased RTX PRO 6000 Blackwell card with no other work on it. Seal each, bank the first and
   `--append` the second, each in its own `data(release):` commit.
3. **Cell records.** Main needs the push gate's evidence for the release commit like any push: a
   version bump is a `build` cell on every push-gated target, so bank an `sm_120a` and an
   `sm_121a` record ([Change impact](RELEASE-QUALIFICATION.md#change-impact-and-per-cell-evidence)).
4. **Qualification note.** Write `research/release-vX.Y.Z-<yyyymmdd>/README.md`: the tested
   commit, the records and their cards, the byte check, box conditions, the cell records and the
   other checks on the tested commit
   ([v0.141.1's note](../research/release-v0.141.1-20261111/README.md) shows the form).

What a record covers, and what it does not (other families, other formats, the served path), is
in [What a record covers](RELEASE-QUALIFICATION.md#what-a-record-covers).

Between the release commit and the tag, commit only qualification data: the records under
`research/release-qualification/` and new research files such as the note. The record pins every
other tracked file, docs included, so a change to any existing file voids it
([verifier contract](RELEASE-QUALIFICATION.md#verifier-contract)).

## Tag

Tag the last banked data commit, not the release commit. The record names the tested commit, and
`release.yml` checks that commit out and rebuilds it.

```bash
git tag -a vX.Y.Z -F tag-message.txt <last data commit>
git push origin vX.Y.Z
```

The tag message summarizes the release: the headline changes, the release path (records, cards,
byte check) and the path of the qualification note.

## What the tag workflows do

Both workflows trigger on a `v*` tag, and both first call `ci.yml` as `cpu-suite`.

`release.yml`:

1. **guard**: `release_qualification.py verify --fetch-source`, then `tools/release-guard.sh`
   (version, the 11 pins, the claim branch), then three releasability censuses
   (`workspace-publish-census.sh`, `stub-abi-census.py`, `arch-matrix-census.sh`).
2. **build**, on `ubuntu-22.04` and `ubuntu-24.04` at `sm_120a`: reads the tested commit from the
   record and checks it out, installs CUDA 13.1 and the release sandbox, builds the six battery
   binaries with `qualify-release.py build` on Rust 1.97.1, refuses packaging unless their ELF
   digests equal the record's for that profile, and packages.
3. **release**: drafts the notes with `tools/changelog.sh`, writes `SHA256SUMS`, signs it with
   cosign keyless (the workflow's GitHub OIDC identity) and publishes the GitHub release.

What ships:

| Asset | Contents |
|---|---|
| `memra-vX.Y.Z-linux-x86_64-glibc2.35-sm120a.tar.gz` | `run-gen`, `run-spec`, `memra-server`, `kernel-check`, `README.txt`; built on `ubuntu-22.04` |
| `memra-vX.Y.Z-linux-x86_64-glibc2.39-sm120a.tar.gz` | the same, built on `ubuntu-24.04` |
| `memra-server-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` | the `ubuntu-22.04` package under a stable name, the `cargo-binstall` target (`tools/install.sh` takes the highest glibc floor the host satisfies instead) |
| `SHA256SUMS` | the SHA-256 of every tarball |
| `SHA256SUMS.sigstore.json` | the cosign bundle for `SHA256SUMS` |

The binaries embed their fatbins. They need an NVIDIA driver with CUDA 13 runtime support (580 or
newer) and the CUDA 13 runtime libraries (cudart, cuBLAS, cuBLASLt); the driver floor is stated,
not checked at run time. `tools/install.sh` verifies the bundle before it trusts the sums. By hand:

```bash
cosign verify-blob --bundle SHA256SUMS.sigstore.json \
  --certificate-identity "https://github.com/avifenesh/memra/.github/workflows/release.yml@refs/tags/vX.Y.Z" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS
```

`publish.yml` runs in the `crates-io` environment: it verifies the record, refuses a non-tag ref
for a real publish, runs `tools/release-guard.sh` and the publish and stub-ABI censuses, installs
CUDA 13.1, then publishes 13 crates one at a time through `tools/crates_publish.py`: memra-tier,
-gguf, -tokenizer, -reference, -kv, -runtime, -sampling, -validate, -net-guard, -engine, -lanes,
-server, -cli. memra-probe is `publish = false`. Per crate it skips a version that is already
live, waits out a 429 from crates.io's new-crate limit (620 s, up to 6 attempts), and polls the
sparse index until that exact version is listed (up to 1,200 s, then a named failure), because
cargo itself waits only 60 s and exits 0. A dependent that still misses a listed crate retries
with a growing back-off inside the same bound.

A `workflow_dispatch` with `publish` false is the dry run. The two publish checks complement each
other: `cargo publish --dry-run` proves each crate builds from its own tarball, and the census
proves `publish.yml`'s list matches the workspace members (the dry run never reads that list).

## Re-running the publish on a tag

The publish is per crate and skip-if-live, so a re-run uploads nothing twice and finishes the
remainder. Two refusals are the guard doing its job:

- A dispatch must name the tag ref in full, or the runner sees a branch ref and the guard refuses
  ("real publishing requires dispatching the immutable vX.Y.Z tag ref"):

  ```bash
  gh workflow run publish.yml --ref refs/tags/vX.Y.Z -f publish=true
  ```
- "no claim branch release/claim-vX.Y.Z on origin": the claim was lost. Re-create it at the
  tagged commit through the refs API (a stale checkout's pre-push hook refuses a plain push):

  ```bash
  gh api repos/avifenesh/memra/git/refs -f ref=refs/heads/release/claim-vX.Y.Z -f sha=<tagged commit>
  ```

## Publishing to crates.io: one-time setup

- A crates.io account (GitHub login) with a verified email.
- An API token from <https://crates.io/settings/tokens> with the `publish-new` and
  `publish-update` scopes, stored as the Actions secret `CARGO_REGISTRY_TOKEN` (Settings, Secrets
  and variables, Actions). The publish job runs in the `crates-io` environment and fails with a
  named error when the secret is missing.
- Thirteen crates publish. Check the registry for what is live; do not infer it from this page.
- Publishing is irreversible (a yank is not a delete), so a crate version ships only from a
  qualified tag.

## Release notes

`release.yml` publishes the draft from `tools/changelog.sh`, which reads commit subjects since
the last tag that became a release (`tools/changelog-skip-tags.txt` names tags that never
shipped). It drops merge commits and groups the rest by prefix (an optional `(scope)` and `!` are
accepted):

| Prefix | Section |
|---|---|
| `perf:` | Performance |
| `feat:` | Features |
| `fix:` | Fixes |
| `config:` | Configuration |
| `docs:` | Documentation |
| `data:`, `chore:`, `wip:`, `probe:` | dropped as research-log noise |
| anything else, `release:` included | Other |

The footer links the bench repository on Hugging Face. A draft over 120,000 bytes is cut to fit
GitHub's 125,000-character limit: whole sections in priority order (configuration, performance,
features, fixes, documentation, other), the omitted entries counted, a compare link to the full
range. Preview before tagging:

```bash
bash tools/changelog.sh            # previous release tag -> HEAD
bash tools/changelog.sh v0.141.0   # explicit range start
```

A commit subject is a publishing surface: the draft copies it verbatim to a public page.

1. A subject states what changed. A number belongs in the body, next to its receipt.
2. If a subject must carry a number, it carries the study's conclusion, not its best cell.
3. Read the drafted notes before the release is announced and edit them on GitHub
   (`gh release edit vX.Y.Z --notes-file notes.md`): the draft is a floor, not a ceiling.
4. Correct a published error where the reader looks: edit the release page and say what the
   earlier text claimed. Do not rewrite pushed history to hide it.

## The 2026-09-01 history swap

The repository was rebuilt from a zero-commit content snapshot on 2026-09-01. The last release on
the old history is `v0.123.0`; its GitHub release and assets did not survive the swap, and GitHub
releases here start at `v0.124.0`. Three mechanisms keep the record:

- `tools/changelog-baseline.txt` names what the history root stands for
  (`<root sha> <predecessor tag> <old commit> <archive notes path>`). When `tools/changelog.sh`
  finds no reachable release tag it falls back to the root, and with the baseline it heads the
  draft "Changes since v0.123.0" and appends
  [the pre-snapshot notes](archive/CHANGELOG-preswap-v0.123.0-to-469c8898e.md) under "Before the
  snapshot". A shallow clone says the baseline does not apply; a missing archive file prints
  RECORD MISSING.
- `tools/changelog-skip-tags.txt` keeps its old-history entries as the record.
- crates.io continuity is unaffected: versions are global to the crate name.

The swap map is in the [history](history/RELEASING.md#the-2026-09-01-history-swap-and-the-first-release-on-this-history).

## Prebuilt targets and limits

- Only `sm_120a` ships prebuilt (RTX 50-series and RTX PRO 6000 Blackwell). `sm_121a` (GB10),
  `sm_100a` (B200), `sm_90a` (H100) and `sm_89` (Ada) build from source with `MEMRA_CUDA_ARCH`.
  `tools/install.sh` refuses `100a` before any download and fails for every other non-`120a`
  target. Per-card status: [rigs](rigs/README.md).
- CI compiles `100a` and `121a` beside `120a` and runs the fatbin lookup census on those two; it
  does not compile `89` or `90a`. The sm_89 and sm_90a prebuilt lanes and their CI cells were
  retired together on 2026-09-02 (owner ruling, an economics decision). `build.rs` still accepts
  both, `tools/stub-abi-census.py` keeps their stub units in step with the real ones, and a
  fatbin census on such a build is the only further check.
- `tools/fatbin-lookup-census.py` proves each kernel the Rust side looks up is in that arch's
  fatbin, not that the path executes. Proving execution needs that silicon. This is the limit of
  every merge-time check; no GPU runs in CI. The check inventory is in [CI](CI.md).
- The glibc axis: `release.yml` builds two floors, 2.35 and 2.39. The OS changes the libc
  requirement of the binary, not the source set or the symbols.

Known gaps:

- The stable-name asset is produced only by the `ubuntu-22.04` and `120a` cell under
  `if: env.STABLE_NAME != ''`. If that coordinate changes, the upload is skipped silently and
  `if-no-files-found: error` never fires.
- `tools/test_cpu_expert_shm.sh` and `tools/check-batch-exact.py` have no automated caller.
- Unit tests in memra-gguf and memra-engine read files under `research/` through paths relative
  to the crate (for example `crates/memra-gguf/src/config.rs` and
  `crates/memra-engine/src/ep_map.rs`). A published crate's tarball does not carry those files, so
  `cargo test` on a downloaded crate fails. `cargo publish --verify` builds without tests, so the
  publish is not blocked.

## History

The previous text of this page is in [history/RELEASING.md](history/RELEASING.md): the per-release
notes for v0.124.1 to v0.141.1, the 2026-08 tag races and the 0.106.0 crates, the sm_89 and
fatbin-census record, the merge-time check table as of 2026-09-02, the open releasability findings
and the rules for adding a gate. Each release's qualification note is under
`research/release-vX.Y.Z-*/`, and its edited notes are on its GitHub release page.
