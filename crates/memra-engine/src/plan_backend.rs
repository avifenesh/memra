//! Compatibility re-export for build-time execution manifests owned by the ModelPlan compiler.

pub use memra_gguf::execution_manifest::*;

pub fn bind_rewrite_artifact(
    receipt: RewriteParityReceipt,
) -> Result<RewriteParityReceipt, Box<dyn std::error::Error>> {
    let path = std::env::var_os("MEMRA_ARTIFACT_LOCK")
        .ok_or("rewrite receipt requires MEMRA_ARTIFACT_LOCK")?;
    Ok(receipt.bind_artifact_lock(&std::fs::read(path)?))
}
