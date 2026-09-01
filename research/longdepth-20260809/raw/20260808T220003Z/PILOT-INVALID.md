# Rejected pilot: low-effort reasoning did not reach code

This run is not a scored matrix cell. Both `ctx131072-policy-d2048` repetitions used the
artifact-rendered `Reasoning: low` prompt, reached `MaxNew` after 2048 generated tokens, and emitted
neither `</think>` nor an HTML start. The raw output repeats planning about response-token limits;
the detector's `missing_html_start` at token 0 therefore means the task never reached code, not
that code corrupted at token 0.

The scored matrix adds a frozen assistant continuation prefix containing `</think>` and
`<!doctype html>` only. It holds the requested task, temperature, artifacts, runtime settings, and
generated-token accounting constant while forcing the measured completion to be HTML code.
