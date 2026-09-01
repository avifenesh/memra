# Orchestrator note on this lane's import — it ran on the UNREPAIRED tree

Imported EVIDENCE-ONLY at this commit. The lane branch `lane/cx-cachesize` was NOT merged: it based on
`18885ec47` ("chore: workspace 0.81.2"), and `main` has since advanced 195 commits, so a merge would have
reverted ~347k lines. Same handling as `cx-splitiso` (`0b0ffa13c`). The lane's only non-research changes
were 8 lines of docs (`docs/FLAGS.md`, `docs/SERVING.md`) and `main` already carries the equivalent
FLAGS correction at :530, so nothing behavioural was dropped.

## Why the campaign failed closed — and why that is NOT a cachesize defect

RESULTS.md reports the runner failing closed at Q27 repetition 2 / 16,384 MiB after **three restored hits
stopped at 11/60 tokens**, and correctly refuses to emit a capacity recommendation.

**That is the early-EOS numeric class, not a cache-budget defect.** Verified: none of `904a5d5f3`,
`6aba8b2e5`, or `96361c531` is an ancestor of this lane's base, so the whole campaign ran on a tree
WITHOUT the eosclass repair and WITHOUT the zero-insert prefix-capture fix.

The signature corroborates precisely. Independent sightings of the same class:
- `cx-gscost` on box1: Q27 EAGER at c=16 selected early EOS **at token 11** within eleven requests.
- this lane: Q27 restored hits stopped at **11/60 tokens**.

Same model, same token index, different lane, different harness, different intent. This is the **seventh**
independent trigger of the class and the second to land on exactly token 11. The lane's fail-closed was
the correct call: an arm truncating at token 11 cannot produce a capacity number.

## What IS valid from this campaign

The snapshot-byte geometry is a measurement of cache entry SIZE, not of decode output, so the defect does
not touch it. These are authoritative sold-shape sizing inputs:

| model | fixed bytes | bytes/token | 4,860-tok sold entry |
|---|--:|--:|--:|
| Q27 | 156,893,184 | 29,696 | **301,215,744** |
| Q35 | 65,863,680 | 9,280 | **110,964,480** |

Exactly linear, and independently reconciled three ways: the isolated per-entry probes, the mixed
3-full/3-partial serial gate average, and the older 14-entry full-prompt requalification (which divides
exactly to 301,215,744 for Q27).

Also confirmed: `prefix_cache_bytes` is the cache's `total_bytes` (prefix KV snapshot plus recurrent
state per entry), and each entry probe reconciled one miss, one insert, one retained entry, zero
evictions/admission-defers/OOM-parks.

## Required follow-up

**The capacity campaign must be RE-RUN on the repaired tip.** This is the owner's #1 engine priority
(cache-hit concurrency is where the money is), and the repair plausibly unblocks the exact cells that
failed. Do not carry the partial capacity table forward as a recommendation — carry only the byte
geometry above. Re-run inherits the protocol in `protocol.lock.json` and the attempt-1 discard lesson
(`raw/attempt1-gpu1-overlap/ATTEMPT-DIAGNOSIS.md`: a separately-locked second-card job invalidated a full
scored attempt because both PRO 6000s share a PIX path).
