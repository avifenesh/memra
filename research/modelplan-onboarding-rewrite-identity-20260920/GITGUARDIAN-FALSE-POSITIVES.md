# Verified checksum false positives

GitGuardian incidents37463655,37463656,37463657 (workspace145584) report six
occurrences in attempts002/003 `files-sha256.json`, lines11-13. Each value is the
SHA256 of the corresponding committed pre-install-tokenwise F32 output file, verified
by recomputing the digest from those bytes. They are not authentication material.

| Prompt | Output SHA256 |
|---|---|
| 0 | a1a60e5a6d7587b2338fcbce3a980ed27efa56a8606d62d7b08226237fd363ad |
| 1 | 3a197dfe63d6c4faf70fdf3ad1cefea19df31340b65c6a9071fe9b131d4bc83c |
| 2 | f8e8a9f8b129dd4de07579c48ea8c653671e511f2fcf2b8805ee07f14c736cdd |

The path keys contain "tokenwise", which appears to trigger the generic high-entropy
heuristic. Raw evidence remains byte-for-byte intact. The appropriate resolution is
incident-specific false-positive triage, not disabling secret scanning, broad path
exclusions, or rewriting historical evidence.
