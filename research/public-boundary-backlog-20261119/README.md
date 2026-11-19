# The ref scan's backlog (memra#1357)

Once PR #1370 lands, the daily `boundary-refs` run finishes, and it reports the backlog that main's
timed-out scan never reached: 615 findings across the published refs, on a mirror of origin's refs
(heads, tags and PR heads, as `boundary-refs.yml` fetches them). This PR takes the part that needs
no branch deletion.

## Commit 1: three rules stop matching shapes that name nothing private

- `serve_prefix` no longer reads `prod` on into `producer` (a receipt schema name); every other
  continuation of a serve unit's name still hits.
- `live_fingerprint` skips cargo's `deps/memra-<hash>` test binaries in build logs, whose hash is
  always 16 hex; any other length after `deps/` still hits.
- `provider_name_aws` skips the `aws-lc-rs`, `aws-lc-sys` and `aws-lc-fips-sys` crate names
  (`Cargo.lock`, build logs).

Each tightening has a red and a green arm (a real-shaped positive is still caught, the false
positive is gone), and `test_backlog_false_positive_shapes_are_not_hits` pins both through the
union, the per-rule pass, the raw-bytes prefilter and `git grep -P`. The 10 grandfathered entries
these rules no longer match are removed. `runpod_pod_id_bare` is not tightened: a narrower form
misses real ids.

## Commit 2: `refs_only` entries for versions only on refs nobody rewrites

`gen_refs_only.py` writes them, in two reviewed classes (its docstring has the rules). A version is
listed only when EVERY ref that carries it is frozen, a merged or closed PR's head or a release
tag: a `refs_only` entry exempts its bytes at every ref, so a version that an open PR's head or a
live branch also carries stays a finding until a commit or a branch deletion removes it there.

- `released_box_id`: versions whose only rule is `provider_bare_id`. They predate the scrub of that
  form from main's live docs. A box id is no credential. 183 entries.
- `auth_test_fixture`: `auth.rs` versions whose only `minted_api_key` matches sit after the
  opening of the `#[cfg(test)]` `mod tests`, each key's random part one repeated character. None
  today: the one such version is also on live branches. The credential-grant count stays at 24.

A `provider_bare_id` version in a file main still keeps the form in, as a dated receipt under
its own grant, is refused and left for review, since the class's reason would be false for it;
none is today.

On the mirror, `verify-allowlist`'s ref check judges all `refs_only` entries live (no drift, none
unnamed), and `check --refs` reports 123 new findings, each on at least one open PR head or live
branch: the 90 versions the generator now leaves out for that reason, and the 33 it left out
before. Those are for a commit there or the owner's branch cleanup, not for this allowlist.

Run, from a clone that mirrors origin's refs:

    gh pr list -R avifenesh/memra --state all --limit 5000 --json number,state > states.json
    python3 research/public-boundary-backlog-20261119/gen_refs_only.py states.json --write
