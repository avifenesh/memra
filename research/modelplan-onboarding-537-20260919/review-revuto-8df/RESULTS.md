# Source RoPE refusal review repair

Two review findings at `8df7ca7787decc409616ae6f34358abf03bd78a2` exposed implicit
GGUF factor declarations that metadata-only checks missed:

- Step's official header has no scaling-type key. A missing factor tensor was accepted
  although the compiled plan required checkpoint factors.
- Llama-style GGUFs can declare factors solely through `rope_freqs.weight`. A plan with
  no factor consumer accepted that source and ignored its declared scaling.

Both regressions fail before the repair ([before.log](before.log): 12 tests passed,
2 failed). Step refusal now follows the compiled factor width. Shared source preflight
rejects a factor tensor whenever the plan has no consumer. It checks the plan first,
so Step's validated factor buffer is still read only once during preflight.

The Step trunk/MTP tests use their unchanged official-header fixtures before testing the
metadata-spelled form; no scaling key is inserted or cleared to reach the new refusal.
The Llama/Qwen GGUF tests cover factor-bearing and factor-free sources with no scaling
metadata. Compact and full-head storage/binding tests remain in the suite.

The focused GPU test now requires actual eager-load refusal for missing factors, including
the unchanged official-header fixture. Its numerical diagnostic uses a separately cloned
synthetic source containing explicit identity factors. Pinned checkpoint files are not
changed or replaced. The valid HF/tensor and compact/full-head equality checks remain;
the qualification runner requires the new refusal/identity-fixture marker.

## CPU evidence

| Gate | Result |
|---|---|
| Focused semantic suite | [14 passed](final-focused.log), 0 failed, 0 ignored |
| Full GGUF/CLI/tokenizer suites | [Passed](final-cpu-suites.log): GGUF lib 295 reported passes / 2 ignored; CLI 11 unit + 1 process integration; tokenizer 72; dequant bin 7. Existing artifact-dependent GGUF skips remain part of libtest's reported passes. |
| Strict clippy, all targets for those crates | [Passed](final-clippy.log) |
| Qualification runner | [9 passed](final-runner-tests.log) |
| Formatting and diff check | Passed |

The initial test-only clippy refusal is retained in `clippy-initial.log`; the final tests
use explicit result matching without a lint allowance. [manifest.json](manifest.json)
binds the source files and all logs. Public logs redact only generated test executable
basenames and trim final blank lines; original hashes and original private logs are kept.

No native result is claimed for this repair yet. Earlier native runs remain bound to their
original source and executables, including their earlier diagnostic behavior. A fresh
reviewed build must rerun the scoped model/factor gates and required full release battery.
