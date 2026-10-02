# Source-binding amendment

This corrects metadata in `lifecycle/identity.json` and
`lifecycle-retry/identity.json`. It records no new execution and changes no measured
result or executed driver bytes.

The recorded drivers read the source head from `review-fixes/source-head.txt`,
but mistakenly read the patch digest from the earlier `combined-v2/source.diff`.
Both runs measured server binary
`6ca4ea6a08606c459605ac65f9b40f458d928a63c9fc9af3afae5a347c6a4514`,
which exactly matches the retained review-fixes build. Its correct source binding is:

- Source head: `bef76711297eaf86fe8aef51c8fb2d8e8d968b6b`.
- Patch: `research/cache-metrics-20261002/review-fixes/source.diff`.
- Patch SHA256: `dfb654e642962b5c1c5455c6bc00ea6403402daabdf57343a90fad914f3f23bb`.
- Compiled input tree: `4e6290633581ae265e7c3c7fe61d680af863d5e6`.

The public identity fields now point to that build. Byte-preserved original
identities remain in the private execution archive, and the executed drivers and
their original hashes remain unchanged. `provenance-amendment.json` lists each
original field value, corrected value, original identity hash and amended hash.
Treat this amendment as authoritative for the two binary-source patch fields.
The short-prime identity already used the correct patch digest.
