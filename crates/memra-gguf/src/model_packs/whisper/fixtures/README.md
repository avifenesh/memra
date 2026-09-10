# Whisper metadata fixture

Verbatim config, frontend, generation policy, token additions and index from
`ivrit-ai/whisper-large-v3@766847c9795b3b5cc0d42f8476199c711d5cee21`.
The `.header` files contain only the 8-byte little-endian length and JSON header,
obtained with HTTP Range. No model payload is included. Provenance is in `source.lock.json`.
Source metadata accompanies the Apache-2.0 checkpoint; Memra code remains FSL-1.1-ALv2.

Tests bind the real census and exercise the mmap loader with sparse, zero-filled payload holes.
Those holes are test fixtures, never actual weights or numerical correctness evidence.
