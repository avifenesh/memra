# OpenRouter submission receipt

The owner-authorized two-model provider application was submitted through the live OpenRouter
form on 2026-08-12 at approximately 15:53 UTC (18:53 Asia/Jerusalem). The form returned the exact
visible confirmation `Thanks for submitting the form.` and no confirmation id. `confirmation.jpg`
is a viewport capture of that result after the completed fields had been replaced by the success
message; it contains no submitted field values or credentials.

Immediately before submission, the public gate was rerun for both exact model ids. Each result
contains 21 checks, zero failures, and an exact expected/actual usage delta of 13 admitted and
completed requests, 34,716 prompt tokens, 10,013 cached tokens, and 528 output tokens:

- `gates/qwen3.6-27b/summary.json` — manifest
  `bb7c32d16eb0bcea96074a66287a94e326a0566690402ccd4a2e9cac7e9ad99c`
- `gates/qwen3.6-35b-a3b/summary.json` — manifest
  `3b34e85cea549d9555cfd03031a1bd2e3c389cacf9311545f2f2806f1c0df69d`

The live OpenRouter `shape=v7` weighted-pricing recheck returned Q27
`$0.3035585338997156/$2.892072976187206` and Q35-A3B
`$0.1301003999538507/$1.0851909290239605` per million input/output tokens. The submitted list
prices therefore remained 5.1–7.8% below that same current metric.

The screenshot SHA-256 is
`b51fd3d786b50614c680f2c498e67e1037a7e99a92f417dab2c56c53c894253f`. Exact-value scans found
zero API-key and zero metrics-token occurrences in both gate trees; a local pattern scan also
found no bearer header, private-key block, or likely memra-key value.
