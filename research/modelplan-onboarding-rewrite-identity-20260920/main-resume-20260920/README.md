# Current-main integration CPU evidence

Source: `e0bfca3154fb4e1aaa906b8689ad9c6caa6becc6`; main parent:
`8a1559b48f38cd67977d3631e735e61ccdc5fcdc`. No GPU qualification.

`integration-source.json` records the parents and changed source hashes.
`manifest.json` hashes the captured raw records. Python commands and outcomes are
in `python-controls.json`; Linux commands, source hashes and outcomes are in
`linux/summary.json`. The Linux controller command uses `--require-linux` and
passes without skips. Both environments run nine actual SIGTERM scenarios through
three test methods.

Other commands, from the candidate checkout:

```sh
cargo test --locked -p memra-gguf -p memra-cli --lib
python3 -B research/modelplan-onboarding-rewrite-identity-20260920/run-host-tests.py --offline
python3 -B crates/memra-engine/src/model/repack/run-host-tests.py
sh tools/test-model-device-memory.sh
sh tools/test-model-memory-fixture.sh
DOCS_RS=1 MEMRA_MMQ_ARCHIVE_HASH=host-check-placeholder cargo clippy \
  --target x86_64-unknown-linux-gnu --target-dir target/resume-crosscheck \
  -p memra-engine --lib --bins --tests -p memra-server -- -D warnings
cargo fmt --all -- --check
git diff --check
python3 tools/update-perf-board.py --check
bash tools/check-flags.sh
git push origin HEAD:refs/heads/codex/542-trusted-rewrite-identity
```

Cargo used three build jobs for compiler/CLI and clippy, two for isolated host
harnesses. The clippy run uses placeholder fatbins and never executes CUDA.
`flags.log` retains a failed `sh tools/check-flags.sh` invocation; `flags-bash.log`
and `push.log` show the corrected invocation passed. The hook reports no local
model directory, so its performance check is not evidence here. The GitHub snapshot
records MERGEABLE with CI pending, not a complete CI verdict.
