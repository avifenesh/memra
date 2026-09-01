# Where the pre-squash narrative of the step37 08-25/29 lanes lives

**Short version: nothing is missing from this repo.** The receipts, verdicts and
raw logs of the step37 work from 2026-08-25 to 08-29 are all here, landed
2026-08-29 as two lane-squash commits:

- `068cbc425` step37 serving readiness: the six-fix arc, qualified spec serving,
  and the measurement corpus
- `46f700291` step37 lane landing 2: the dcw draft-graph arc, the
  sampled-quality verdict bank, and the 30k cell

This file exists because those squashes make the work look unlanded to anyone
searching by commit subject: the lane's own 134 commit messages appear nowhere on
`main`, which reads exactly like a lane whose receipts never got pushed. It was
mistaken for one on 2026-09-01, and the check that settled it is worth writing
down: compare FILES, not subjects (`git diff --name-only <lane-base> <lane-tip>`,
then diff each path against `main` - 276 of 280 were byte-identical and the other
4 were ones where `main` was NEWER).

The pre-squash narrative - the 134 individual commits - is not in this repo's
history. It is preserved in:

| copy | location |
|---|---|
| off-machine, verified | R2 bucket `tiyuvta-capture`, key `repo-bundles/20260901/step37-mtp-masked-vocab-134commits.bundle` (184,782,290 B, sha256 `6bf5a96d024ad87a150d…`, round-trip verified) |
| rig | `~/repo-backups-20260901/step37-mtp-masked-vocab-134commits.bundle` (`git bundle verify`: records a complete history) |
| local ref | tag `backup/mtp-mask-134-preflatten` (134 commits ahead of the squash base; also what keeps the objects reachable against a gc) |

`git bundle unbundle` either bundle to read it.

**Why the lane branch never landed as a branch:** the pre-push public-boundary
hook refused the range, because an intermediate revision of
`research/step37-sampled-quality-20260828/PLAN.md` carried a dev-box identity
(provider name plus IP) even though the lane's last commit had scrubbed the tip.
The hook judges every commit being pushed, not the tip, and a blob published to a
public repo cannot be unpublished by deleting a ref - so the guard was right and
the squashes were the correct way for the content to land.

**SHA caveat (2026-09-01):** the repos were rebuilt from a content snapshot with
zero-commit history, so the SHAs above - and SHAs cited in receipts generally -
resolve only against the archive bundles, not against this repo's history.
