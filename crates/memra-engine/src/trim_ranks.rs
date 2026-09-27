//! One opened rank input per model load. Paths label inputs; they are never reopened for identity.
use memra_gguf::bound_source::ranks::RankArtifact;

#[derive(Default)]
pub(crate) struct RankInput {
    captured: Option<CapturedRanks>,
}
pub(crate) struct CapturedRanks {
    spec: String,
    path: String,
    artifact: RankArtifact,
}
impl CapturedRanks {
    pub(crate) fn path(&self) -> &str {
        &self.path
    }
    pub(crate) fn artifact(&self) -> &RankArtifact {
        &self.artifact
    }
    pub(crate) fn sha16(&self) -> &str {
        &self.artifact.identity().source_sha256()[..16]
    }
}
impl RankInput {
    pub(crate) fn captured(&self) -> Option<&CapturedRanks> {
        self.captured.as_ref()
    }
    pub(crate) fn capture(&mut self, spec: &str) -> Result<&CapturedRanks, String> {
        if self.captured.is_none() {
            let path = memra_gguf::hf::resolve_arg(spec)
                .map_err(|e| format!("MEMRA_FRSPEC_TRIM={spec:?}: {e}"))?;
            let artifact = RankArtifact::open(std::path::Path::new(&path))?;
            self.captured = Some(CapturedRanks {
                spec: spec.into(),
                path,
                artifact,
            });
        }
        let captured = self.captured.as_ref().unwrap();
        if captured.spec != spec {
            return Err("MEMRA_FRSPEC_TRIM changed during this model load".into());
        }
        Ok(captured)
    }
}

// The existing byte gather, shared with hybrid::frspec_gather_rows and the CPU consumer tests.
pub(crate) fn gather_rows(rows: &[u8], row_bytes: usize, ids: &[u32]) -> Vec<u8> {
    let mut gathered = Vec::with_capacity(ids.len() * row_bytes);
    for &id in ids {
        let offset = id as usize * row_bytes;
        gathered.extend_from_slice(&rows[offset..offset + row_bytes]);
    }
    gathered
}
