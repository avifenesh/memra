# Step centered-norm declaration

The existing Step HF materializer implements `Step3p7RMSNorm` by folding `weight + 1`.
The serialized ModelPlan omitted Step from that weight-transform declaration. This change
records the existing numerical program in the plan; it does not add a new execution path.

The regression checks output, trunk and MTP norms, then checks that the GGUF tensor contract
still applies identity transforms to its already-folded weights. Removing only the new Step
clause reproduces the intended `Identity` versus `AddOne` failure. Restoring it passes.
Raw CPU logs and source hashes are retained alongside this record.

Dependencies: the Step plan/compiler and metadata-only fixture helpers already present in
#537's `f2c6fdcfc3ec4b61ebdaab67f1e5bc76d1b79322`. This patch does not require #541's bound-source
foundation, runtime adapter, identity composition or universal loader activation. HF q/k and
private-head schema corrections follow separately.

Step serialized plan hashes change. Old qualification receipts remain historical and cannot
qualify the new plan/binary. This is CPU/source evidence only; fresh reviewed native qualification
belongs to the coordinated #537/#541 integration.
