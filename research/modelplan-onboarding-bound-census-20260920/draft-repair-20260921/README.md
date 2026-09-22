# External draft consumer-contract repair

This bounded repair follows immutable `80483aaf2f66563c29612c529f2e465e3517fc01`.
Carver closed ROOT01/02 at804, then completed the separate1254 review with mandatory
`ARCH541-DRAFT-01` and `ARCH541-DRAFT-02`. Both current reports are preserved in `reviewed_input`.
Earlier1254/3a refs and evidence remain unchanged. Independent closure of this repair is pending.

## Activation alignment

Draft preparation now compares declared FFN activation with the effective target-driven
consumer, including SwiGLU-OAI alpha/limit. The pure `FfnActivation` selector is shared with
the actual native FFN dispatcher, so validation does not merely compare model-family names.
The native call signatures, branch order, scale arguments and unscaled-SiLU fast branch are
preserved. Step's dense MTP clamp remains resolved from the source block's plan, using the
same helper as `Step35MtpGeom`; it is not replaced by the target's final-layer clamp.
Routed/shared draft FFNs check the target's existing external-block dispatch index separately.

The `MtpHead::load_draft`, `load_prepared_draft`, `load_t`, and `GpuTensor::load_from_source`
function bodies are byte-identical to804. The audited chain is:

1. `MtpHead::load_draft` prepares the standalone source before loading tensors.
2. Head and student output-up loads call `load_t`, then `GpuTensor::load_from_source_inner`.
3. The standalone GGUF source exposes no HF-native operand. An NVFP4 view reaches the resident
   NVFP4 branch and the checked `model::nvfp4_scale::load` reader.
4. Draft FFN execution in unchanged `spec.rs` passes target `self.cfg` and the resolved Step
   limit into `ffn_act_lim`, which delegates to the production `ffn_activation::apply` dispatcher.

## Scale alignment

Preparation requires selected NVFP4 macro-scale rows to be F32 `[1]`, exactly four bytes.
F16/BF16 selected scales refuse before allocation. This includes private/file heads and student
output-up projections. Unselected scale rows retain their original encoding and remain in
the complete opened identity; they cannot be read through the selected runtime scope.

The actual resident NVFP4 reader validates one F32 value and exact byte width rather than
slicing unchecked bytes. It preserves absent-scale=1 and existing scalar HF views. No scale
conversion, format fallback, metadata rewrite or raw-source bypass is added. Raw opened bytes
and existing standalone artifact-identity implementations remain unchanged.

## Evidence

Carver's preserved real-file probe fails before the repair (exit101, three bad admissions) and
passes afterward (exit0). The SiLU-target and F32-scale controls remain accepted; the OAI-target
mismatch and F16/BF16 scales refuse. Probe source, lockfile and all four GGUF fixtures match
Carver's originals byte-for-byte. Only its dependency path points to the owner checkout.

Fourteen draft tests pass, including two new controls for activation refusal before payload
reads and distinct source-owned Step limits. Six consumer-harness tests compile the actual
production FFN and scalar-reader modules by path. Only terminal device calls are replaced by
a recorder: it verifies kernel selection and all alpha/limit/macro arguments for SiLU, OAI,
post-clamp and pre-clamp. Real GGUF tests verify accepted/rejected target programs and head/student
scale encodings through the actual resident reader. OAI parameter controls use valid compiled
target metadata; they do not claim a new OAI standalone-GGUF format or native inference result.

Strict GGUF, consumer-harness and Linux DOCS_RS engine lib/test Clippy pass, along with both
formatting checks and whitespace. The harness lock uses the existing production dependency
versions. Initial compiler diagnostics and the initial unlocked harness run are preserved as
development evidence; the final harness is locked. Existing targets were reused and no broad
root suite was rerun. `evidence.json` pins source, commands, raw receipts and unique CPU executables.

The preservation receipt checks916 other tracked crate files, including #542 backend/snapshots,
worker/LRU, config/ModelPlan, spec execution, placement, Gemma root repair and canonical cache
consumers. No composite-draft/identity acceptance work, main intake/merge, default change,
remote/native job, rental or topup occurred. Parent coordination owns original Carver closure.
