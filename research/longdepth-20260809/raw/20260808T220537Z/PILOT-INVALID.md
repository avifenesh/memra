# Rejected pilot: detector omitted the frozen prefix

Both `ctx131072-policy-d2048` repetitions generated deterministic HTML immediately and contained
no non-ASCII or non-Latin code points. They are not scored because the detector parsed only the
generated continuation, while the HTML5 doctype deliberately lives in the frozen assistant
prefix. It consequently reported `missing_html5_doctype` at completion token 0 even though the
combined assistant continuation begins `<!doctype html><html ...`.

The corrected detector takes an explicit `--doctype-prefilled` receipt, records that fact in its
JSON, and still parses and token-maps generated HTML from completion token 0. No runtime or prompt
bytes change in this correction; the scored matrix nevertheless gets a new commit and run id.
