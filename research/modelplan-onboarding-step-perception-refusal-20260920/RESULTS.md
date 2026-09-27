# Refuse a fabricated Step vision program

The generic `vision_config` parser filled Gemma-style defaults when the declared type was
`perception_encoder`. Step uses `width`/`layers`, not Gemma's `hidden_size`/`num_hidden_layers`;
the old result was an invented 768-wide, 16-layer executable vision plan for the actual
1,536-wide, 47-layer perception encoder.

Correction to the initial wording: Step already has a native perception encoder in
`crates/memra-engine/src/vision_step.rs`, with historical qualification under
`research/step37-vision-20260830/`. The gap is its canonical ModelPlan representation, not the
existence of execution support. This isolated correction removes the unrelated invented plan. It does not alter or filter the original config, index, shard
headers or tensors. #542 captures raw config, the source census retains all physical names, and
#541's complete inventory/selected-execution contract handles the declared unsupported component.
Text-only qualification is not vision qualification.

The new CPU regression fails before this correction. Afterward it verifies that the executable
Step text plan is identical with and without the not-yet-represented vision declaration, and that no
factored vision program or multimodal operation is manufactured. The ModelPlan suite also passes.
Raw failure/pass logs are preserved.

Dependencies: isolated Step A/B (`944516e44`, `ef131c82f`) and existing #537 compiler helpers.
The test reuses B's small contract fixture. No #541 bound-source/runtime API is required by this
patch. Serialized plans change again; final #537/#541 composition must be reviewed and rebuilt,
and historical receipts must not be relabeled. No native, vision or merge approval is claimed.

The existing Step tower route (`worker.rs`, `MEMRA_STEP_VISION_DIR` and `StepVisionTower::load`)
does not depend on `cfg.vision`; it remains intact. This patch is not a vision support downgrade.
The complete canonical binding migration must represent and preserve that existing path.
