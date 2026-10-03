# Shared cache-helper contract inputs

The actual serving collector and its sampled-MTP consumer both changed their CPU
cache expectation from 224 to 241 when only a copied cache helper changed. The
baseline planner selected only the background collector. The repaired registry
selects all three present consumers and preserves their CPU commands and native
obligations. No unrelated compilation is selected for that helper edit.

Question: can the declared Python contract graph omit a real shared-helper consumer?
Comparison: main 75bac45b8a against the repaired input registry, using the actual
source imports in fresh CPU subprocesses. The temporary helper mutation is a
dependency witness, not a change to production cache accounting.
Budget: manager lane, local CPU tooling only, cash ceiling $0. No GPU time,
model downloads, remote workloads or production access.
Decision: fix the demonstrated omission; reject incomplete selected contracts and
retain absent-consumer controls. No serving, model, native or release qualification.

Validation: 124 composed planner/coverage controls, registry census for 14 packages,
eight serving, seven sampled-MTP and nine background collector controls passed.
The actual local NumPy version was 2.3.5, matching the sampled requirement.
The dependency replay pins six baseline source/manifest hashes and the candidate
registry hash. It executes both actual imports, compares original/repaired planner
selection and verifies unchanged CPU commands and native requirements.

Reproduce with `python3 research/shared-cache-contract-20261003/replay.py` using
the repository's pinned NumPy dependency. Results are in `PROOF.json`.
An initial replay omitted the baseline planner's validation_inputs.json and
correctly expanded rather than producing the intended scoped proof. The replay
now copies and hashes that required manifest too. Temporary directories are owned
by the replay and removed on exit.

Publicity: skipped: maintenance release.
