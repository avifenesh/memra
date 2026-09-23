# Qualification compiler pin — issue 547 subfinding

The frozen `cbde7231a3cc8d36bbf1d4537e99d20c897b24c8` preparation selected CUDA 13.1 through
`MEMRA_NVCC`. The public qualification runner removed all `MEMRA_*` variables, including that
build input. Its stock build consequently selected an installed CUDA 13.2 automatically.
The raw requested-tool probe and actual stock-build warning are retained below. That mixed
attempt remains unqualified; none of its binaries or compiler identities is relabeled, and
no GPU phase ran. Global CUDA/Cargo configuration was not changed.

The corrected build phase requires an explicit `--nvcc` or `MEMRA_NVCC`; the CLI path wins.
It resolves and checks the executable, restores only that build input after runtime controls
are filtered, hides GPUs, records the executable SHA256 and actual `--version` output, and
checks the compiler hash again before writing the binary manifest. Missing pins refuse rather
than permitting an ambient toolkit to decide a qualification build.

The CPU red control selects a 13.1 executable while an ambient 13.2 executable is available.
The old runner loses the override and the simulated build selects the ambient path. The fixed
runner preserves the explicit path in all three Cargo commands and records the selected tool.
Thirteen CPU controls pass, including CLI precedence, missing-pin refusal, compiler replacement
refusal, and the existing lease/artifact/receipt checks. Compiler stubs execute on CPU; Cargo
and GPU work are mocked in these orchestration controls. No native result is inferred.

Logs are losslessly compressed in `raw/`; `requested-build-inputs.json` identifies the complete
original source-input manifest by hash. Runtime crates are unaffected by this runner repair.
The changed upstream runtime from main must still receive its own composed-source review and
fresh source/compiler/binary-bound native qualification.
