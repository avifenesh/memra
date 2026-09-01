# Rejected pilot: default-effort prompt

This run is not a scored matrix cell. It is retained because the evidence protocol preserves
failed attempts.

Both `ctx131072-policy-d2048` repetitions reached `MaxNew` after 2048 generated tokens without
emitting `</think>` or an HTML start. The native `/v1/completions` request used the legacy
single-turn `chat:true` switch, which cannot carry Step35's `reasoning_effort` render input. The
detector therefore reported `missing_html_start` at completion token 0. That is a harness
invalidation, not evidence of long-depth token corruption.

The scored matrix uses a byte-frozen prompt rendered through the artifact chat template with
`Reasoning: low`, submitted as a raw native completion so exact response token ids remain
available.
