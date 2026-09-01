# PUBLIC-BOUNDARY-DETECTION.md — what the boundary gate matches, and what it refuses to (2026-08-18)

Two independent read-only reviews reported, on the same night, that rented-host network details
and cloud inventory were committed in this public repo and that
`tools/check-public-boundary.py` did not match them. Both were **confirmed**. This record is the
decision behind the fix: which rules were adopted, which candidate rules were measured and
**rejected**, and why a whole new scan mode had to exist rather than another pattern.

Companion surfaces: `tools/public-boundary-policy.toml` (the rules, each with its why-comment
and measured cost), `tools/test_public_boundary.py` (the shapes, asserted through the combined
union the gate actually evaluates), `.github/workflows/boundary-refs.yml` (the scheduled ref
scan).

## What the gate missed, and the two distinct causes

**Cause 1 — rule shape.** Three families were invisible:

- `rented_ipv4` only fires when one of its keywords sits within 60 characters *before* the
  address. A bench box written as a bare transfer target — `git push <user>@<addr>:/path`, `scp`,
  `rsync` — has no keyword on the line to anchor on, so the address passed.
- `cloud_instance_id` and `security_group_id` were the only two members of a large family of
  prefix-plus-hex account resource ids. An image, a subnet, a VPC, a volume, an interface, a
  snapshot or a capacity block names the same account and the same placement as an instance id
  does, and a provisioning script that quoted those instead matched nothing.
- `provider_machine_id` requires the literal word *machine*. A rented box is just as
  identifiable through the contract it was rented on, which is the number the provider's own
  console shows, and a lane receipt that opened with the contract id matched nothing.

**Cause 2 — corpus, not shape.** The two affected lanes were pushed 2026-08-15; the
account-identity rules landed 2026-08-16. The pre-push hook judges only the blob versions a push
introduces, and CI judges only the checkout. **Nothing re-reads a branch that is already
published.** So every rule added after a push is retroactively blind on that branch — a
structural hole no additional pattern can close.

## Adopted

Three rules (`ssh_destination`, `aws_resource_id`, `provider_contract_id`) and one new scan mode
(`check --refs`), plus `--summary-only` for public logs. Each rule's measured cost on the tracked
tree is recorded in its policy comment: 3 files, 5 files, and 0 files respectively. The first
three were already `rented_ipv4` hits through a keyword, so `ssh_destination` widens detection
without widening the allowlist.

`--refs [GLOB]` re-scans every blob version carried by the matching published refs under the
**current** policy, deduplicated by `(path, sha256)` so shared history costs one violation with a
carrier list rather than one per branch. It reuses `evaluate_content`, so a ref finding and a
checkout finding are the same judgement, and it deliberately skips the stale-allowlist invariant:
published refs are a different corpus from the checkout, and only the full-tree check owns that
invariant.

## Rejected, with the measurement that rejected it

Both candidates below would have caught real material. Both were dropped because the noise they
add is the failure mode this policy already learned once: a broad `serving|fleet` pattern fired on
305 files, nearly all dated research prose, and **a gate that noisy is a gate nobody reads** —
the same dynamic that let real leaks hide inside an eleven-hundred-entry allowlist.

| candidate | what it would catch | measured new hits | verdict |
|---|---|---|---|
| hourly-rate shape (a currency amount followed by a per-hour unit) | rented-rig cost, a deployment fact that belongs in darklanes | 11 tracked files, including a wiki-corpus prompt where the amount is ordinary prose | rejected |
| unanchored bare IPv4 (drop `rented_ipv4`'s keyword requirement entirely) | the transfer-target case, and every other keyword-free address | 15 tracked files, mostly vendor HTML, pip resolver logs and environment freezes | rejected |

`ssh_destination` is the narrow form of the second candidate that survived: the `@` is itself the
assertion that the number is a host someone logs into, so it needs no keyword and it does not fire
on a vendor log. The same private-range exclusions are kept, which is also why profiler run
labels of the form `<label>@<private-addr>` stay silent — by the range lookahead, not by luck.

## `--summary-only`: the findings list is itself the leak

Actions logs on a public repo are public. A job that prints `path:line` for each unremediated
finding publishes a map to every one of them. `--summary-only` prints per-rule counts and no
paths: enough to fail the job, useless to a stranger reading the log. A maintainer re-runs without
the flag, locally, to get paths.

For the same reason the per-path output of the first full ref scan is **not committed anywhere in
this repo**, and neither is a remediation checklist keyed by path. That is a deliberate exception
to the evidence-discipline rule that raw output ships next to the summary: here the raw output is
the exposure. The counts below are the committed form.

## The scheduled ref scan is not a merge gate

`.github/workflows/boundary-refs.yml` runs `check --refs --summary-only` daily and on demand, and
is deliberately absent from `push`/`pull_request`. The first run found a backlog on history that
the owner remediates branch by branch; a required check that stays red for the length of that
backlog is, again, a gate nobody reads. The daily cadence is what makes a *new* leak on a pushed
lane surface within a day; the manual dispatch is what to run right after a policy change.

## First full ref scan, 2026-08-18

64 published refs, one multi-tree `git grep` prefilter plus batched blob reads, 3m32s wall on the
local dev rig under a CPU cap. 647 matches, 499 grandfathered, **148 new** across 127 distinct
paths, 113 of which were never allowlisted in any version.

Read by hand, the 148 split into two classes:

- **Real exposure that the checkout no longer carries but published branches still do** — hits
  from `payout_wallet`, `aws_account_id`, `security_group_id` and `provider_contract_id` in
  research lanes. This class is precisely what no existing mode could see, and it is the reason
  the mode exists.
- **Noise with zero marginal exposure** — the largest single cluster (48 of 56 `personal_email`
  hits) is captured `git log` output, where the address appears on the `Author:` line. The policy
  comment on that rule already anticipates this: it exists to catch pasted logs and captures
  rather than authorship, and the commit trailers carry the same address regardless of file
  content.

Remediation of already-public content — history rewrite, branch deletion, receipt edits, key or
host rotation — is an owner decision and is not recorded here.

## 2026-08-19 — the delivery failure, and the two reporting defects behind it

A full read-only audit of every pushed ref (207 refs, 5,107 commits, 39,991 blobs — banked in
darklanes `research/security/public-ref-audit-20260819.md`) established that **none of the work
above had ever been delivered.** Measured against the remote:

| artifact | on the rig | on any pushed ref |
|---|---|---|
| `.github/workflows/boundary-refs.yml` | yes | **zero of 66 heads** |
| `--refs` mode | yes | **absent** |
| `aws_resource_id` — the only rule matching `cr-`/`ami-`/`subnet-`/`vpc-`/`vol-`/`snap-`/`eni-` | yes | **absent** |
| `ssh_destination`, `provider_contract_id` | yes | **absent** |

Two consequences, and neither is subtle. GitHub schedules `on: schedule` only from the **default
branch**, so the nightly ref audit designed for exactly this class of leak **had never run, not
once**. And the pushed policy had no rule for `cr-`, so the capacity-block id that motivated the
audit was **undetectable by the gate as deployed**, on any branch, at any time.

The lesson is not about patterns. A detection improvement that is designed, justified in
comments, tested and committed **locally** is worth nothing; the gate is the version that runs in
the place it is supposed to run. Treat "landed" as meaning pushed to the default branch.

The same audit proved two reporting defects on real bytes, both fixed here.

### Defect 1 — the allowlist had no rule scoping

Keyed on `(path, sha256)` alone, a blob grandfathered for one rule was exempt from **every**
rule, including rules added years later. Verified byte-for-byte:
`research/model-selection-20260801/recon-journal-v2.jsonl` was allowlisted as
`production_endpoint`, its sha256 matched `main` exactly — and the same bytes carried a provider
capacity-block id at offset 9,616. That id was permanently exempt from `aws_resource_id`, a rule
written specifically to catch it, and appeared in **no report**.

Entries are now keyed on `(path, sha256, rules)` and `rules` is required — an entry that names no
rule is rejected rather than read as "everything", because a silent breadth default is the defect
itself. The invariant sharpens in both directions: the grant now also **expires with the finding
it was granted for**, so remediating the rule an entry was granted for no longer leaves the
exemption in place covering the rest.

Migration was mechanical and lossless: the `reason` field has always led with the rule names that
matched at seed time, so all **578** existing entries were narrowed to their recorded rules and
**zero** needed to keep their old breadth. Verified by running the pre-push range check over
current `main` (`v0.92.0..main`, 86 commits): 63 matches, all still grandfathered, so the
narrowing broke nothing.

Narrowing then surfaced **61** blobs whose allowlist entry covered a *different* rule than one
they also match — against the 4 the audit had found by hand. Two clusters dominate:
`serve_prefix` inside files pinned for `serve_data_root` (the same serve-box path written two
ways), and `ssh_destination` inside files pinned for `rented_ipv4`. Plus **5** genuinely new
findings from the three previously-unpushed rules, including `tools/provision-aug2.sh`. All 66 are
now pinned as explicit rule-scoped entries whose `reason` says `UNREMEDIATED, owner decision
pending`: greppable, diffable, and automatically stale the moment the file changes. They are
pinned so the gate is deployable today, **not** because the findings are accepted — remediation is
an owner call and is out of scope here.

### Defect 2 — first-match-only reporting buried the worst finding

`scan_secret_bytes` reported one hit per file. That was correct for the enforce/don't-enforce
decision — one hit makes a blob a violation — and it is how the worst file in the repo hid:

| file | rules that match | was reported as |
|---|---|---|
| `research/gemma4-bringup/corpus-prompts/wiki-055.txt` | `personal_email` @367, **`aws_account_id` @556** | `personal_email` |
| `research/per-expert-quant/finalize_hy3_smart100_research.sh` | `cloud_instance_id` @264, **`aws_account_id` @294** | `cloud_instance_id` |

`wiki-055.txt` carries a cloud account id, an `AdministratorAccess` assertion and a named
identity principal. Reported as `personal_email`, it was one line inside **56** `personal_email` hits — a
class a reviewer correctly triages as known authorship noise — and the account id was named
nowhere.

Now every matching rule is reported, and the report is ordered by **severity**, not scan order.
Both halves are needed: a complete list still hides its own worst line if it is sorted by offset.

The scan stays two-phase for cost. The union matcher remains the cheap gate that keeps a 1.8 GB
tree scannable, and only a blob that already hit gets the per-rule pass — 24 searches on ~580
blobs of ~40,000. The per-rule pass is not merely a nicety: **alternation reports only the branch
that wins at the earliest position**, so two rules matching the same span, or a narrower rule
inside a span a wider rule already consumed, are invisible to the union no matter how it is
iterated.

Severity lives in a `[severity]` table in the policy, 1..5, and `load_policy` **refuses a policy
that leaves any rule unranked**. An unranked rule would sort by an implicit default, and a default
is exactly how a severity-5 finding ends up buried in a class of noise. The bands: 5 names the
account or a principal inside it (permanent, and it is the target-selection fact that turns a
future key leak from "a key" into "a key, and we know what it opens"); 4 is reachability and
persistent account objects; 3 is private product surface; 2 is owner identity in file content; 1
is build provenance — which is 451 of the 578 entries, and the reason it must not sort alongside
anything above it.

### The nightly job, made decisive

The workflow as written covered heads only: it fetched `--no-tags` against a default glob of
`refs/remotes/origin/**`, which cannot match `refs/tags/*`. Tags carried **more** violations than
heads (151 vs 131 under the pushed policy) including a rule class heads did not have, and they are
the worst namespace for a leak — immutable release markers are what forks and package consumers
pull, and rewriting them is worse than the disclosure, so **detection is the only lever that works
on them at all.** `refs/pull/*` were also out of scope; 20 of 25 carried a confirmed identifier at
tip, and they are not owner-writable, so an unscanned PR ref is how someone concludes a head
deletion was sufficient.

Now: heads, tags and PR refs; a ref inventory checked against `git ls-remote` so a fetch that
quietly returned one ref **fails** instead of scanning one ref and reporting green; a printed blob
count so "clean" and "never looked" are different outputs; and a per-ref count in the summary, on
the ground that ref names are already public on GitHub while paths and lines are the map that must
not be. No `if:`, no `continue-on-error:`, and the scanner reads no environment variables — the
real bypasses are `--no-verify` and `core.hooksPath`, both local git config, absent from any fork
and any CI checkout, which is precisely why the server-side job matters.

### Second full ref scan, 2026-08-19 — heads AND tags, under the fixed scanner

187 refs (68 heads, 119 tags; the local mirror carries no `refs/pull/*`, so that namespace is
covered by the workflow's fetch and not by this run), 45,237 candidate blob versions read, ~25
minutes wall on 4 nice'd cores. **739 matches, 565 grandfathered, 174 new**, on 137 of the 187
refs. For comparison, the same corpus under the *pushed* policy scanned one namespace at a time
gave 131 new on heads and 151 new on tags, and the 2026-08-18 heads-only run gave 148 new on 64
refs.

The new-violation profile, worst-first, is the argument for the severity ranks in one table:

| sev | count | rules |
|---|---|---|
| 5 | 4 | `aws_account_id` |
| 4 | 65 | `rented_ipv4` 17, `payout_wallet` 13, `cloud_instance_id` 10, `ssh_destination` 10, `aws_resource_id` 9, `security_group_id` 3, `ssh_source_cidr` 3 |
| 3 | 49 | `serve_home` 32, `production_endpoint` 29, `serve_prefix` 17, `serve_data_root` 16, `provider_machine_id` 8, `onlist` 6, `openmodels` 3, `vast_serve_host` 3 |
| 2 | 56 | `personal_email` |

`personal_email` is the largest single class and now sorts **last**. That is the whole point: it
was first before, and the four `aws_account_id` findings were inside it.

Both defect fixes reproduce the audit's headline findings on the real bytes:

- `research/gemma4-bringup/corpus-prompts/wiki-055.txt` now reports
  `aws_account_id,personal_email` at **sev5 and first in the report**, where the old scanner
  reported `personal_email` and sorted it into the middle of 56 like hits.
- `research/per-expert-quant/finalize_hy3_smart100_research.sh` reports
  `aws_account_id,cloud_instance_id`; `research/moe-levers-20260801/recon-journal.jsonl` and
  `synthesis-raw.txt` likewise. All three match the audit's measured table exactly.
- The `cr-`/`subnet-`/`ami-` class is detected: **9 blobs** under `aws_resource_id`, a rule that
  did not exist on any pushed ref. All six capacity-block, subnet and AMI ids the audit named by
  hand were checked against both policies — every one matches `aws_resource_id` under this
  policy and matches **nothing** under the 20 rules on `main`.
- One blob reports **five** rules where first-match-only reported one:
  `tools/provision-aug2.sh` on a lane ref matches `aws_resource_id`, `security_group_id`,
  `cloud_instance_id`, `rented_ipv4` and `ssh_source_cidr` together.

An incidental confirmation of the root cause, worth recording because it is the mechanism rather
than an anecdote: the v0.93.0 release scrub removed the instance ids and host addresses from
`main`'s copy of `tools/provision-aug2.sh` (11,121 bytes, matches `aws_resource_id` only) while
the lane copy (11,137 bytes) still carries all five classes. The AMI, subnet and capacity-block
ids survived the scrub **in both** — because the gate that guided it had no rule for them.

### Still gets through, after this lane

1. **History below the ref tips.** `--refs` reads tip trees; a blob added and later removed is
   invisible. The audit's residential-IP finding and its two nsys profiles sit in pushed history
   on 46 heads and 36-63 tags. The fix is the object-DB sweep the audit itself used
   (`git cat-file --batch-all-objects`, attributed with `git rev-list --all --objects`), which is
   *cheaper* than 66 tree walks and provably cannot skip a commit. Not done here.
2. **Commit messages and author/committer identity.** Never scanned, in any mode. Two `cr-` ids
   and a box public IP are in commit messages on `main` and 37-47 tags; an
   `ubuntu@ip-*.compute.internal` trailer is on 4 commits. One `git log --all --format=…` pass
   over 5,107 commits costs under a second. Message and trailer hits are **unfixable without a
   full history rewrite**, which is the whole argument for catching them before the push.
3. **A fresh clone or a fork has no gate at all.** `core.hooksPath=tools/hooks` is local config,
   not in the repo. `ci.yml` runs the full-tree check only on `main` push and PRs to `main`, so a
   lane branch pushed straight to the remote is judged by nothing server-side. Either widen
   `ci.yml` to all branches or enable GitHub push protection with a custom pattern set.
4. **Anything already pushed.** No detection change un-discloses a byte. GitHub keeps unreachable
   objects fetchable by SHA and the repo has 36 forks.

## `scrump` is complementary, not a substitute

The owner's credential scrubber (`scrump scan`, trufflehog-derived ruleset) was run read-only
across the tracked tree and both lane worktrees. It found **no credentials**; its hits on the
files in question are generic-shape false positives. It does not model deployment identity —
account resource ids, provider contracts, rented-host addresses — so it neither confirms nor
refutes a boundary finding. Both tools stay: `scrump` for credential material, this gate for the
public/private repo split. `scrub` mutates files in place and was not run.
