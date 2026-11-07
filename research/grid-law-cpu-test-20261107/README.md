# The device restore-grid-law probes run in the default test set (memra#1088, 2026-11-07)

Verdict: `the_device_prefix_probes_apply_the_restore_grid_law` replays the 2026-11-07 receipt's shape (an off-grid 2,992-token recurrent entry, a 3,109-token prompt) through the device prefix cache's probes on CPU, in the default test set; applied to 9805016bb1, the predecessor of the restore grid law, it fails at its first assertion (the lookup returns the off-grid entry), and it passes on the landing head.

## Why a marker

memra#1083's law broke at the device default, and its device-side probes (`PrefixCache::lookup`,
`lookup_where`, `deepest_covering`, `admission_restore_rows`) had a behavioural test only in the
`#[ignore]` GPU set, which no default run executes. A CPU test cannot allocate the GDN conv/ssm planes
`PrefixEntry::carries_recurrent` reads, so the entry gets a test-only marker (`recurrent_in_test`,
`#[cfg(test)]`, compiled out of production) that `carries_recurrent` honours.

## The test

One cache, the receipt's entries, the 3,109-token turn-2 prompt:

- the off-grid recurrent 2,992-token entry: refused by `lookup`, `lookup_where`, `deepest_covering`;
  still a whole cover of its own prompt; pinned by that whole cover, `admission_restore_rows` refuses it
  for the longer prompt;
- an on-grid 2,976-token entry (133 suffix tokens): served by every probe and the retained-prefix plan;
- an on-grid 3,104-token entry (5 suffix tokens, under `PRIME_MIN_T`): refused; the filter never falls
  back to it;
- a recurrent whole cover (3,109): served;
- attention-only 2,992 and 3,104 entries: served (split-invariant).

## Fail before, pass after

- Red: 9805016bb1 plus `red-9805.patch` (the marker on `PrefixEntry`, its literals, and the test; the
  predecessor has no `carries_recurrent`, so its probes ignore the marker as they ignore recurrence):
  `red-test.log`, `assertion left == right failed: lookup: off-grid strict prefix`, `left: Some(2992)`,
  `right: None`.
- Green: the landing head, memra-server lib, the test included (`green-lib.txt`).
