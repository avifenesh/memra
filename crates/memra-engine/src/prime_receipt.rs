//! Read-only GDN prime diagnostics, enabled only by the existing allocation trace.
//! Hash actual capture storage, not just the logits that caused the capture.

use crate::{Engine, spec::SpecBoundaryCapture};
use sha2::{Digest, Sha256};

fn floats(hash: &mut Sha256, values: &[f32]) {
    hash.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
}

pub(crate) fn logits_digest(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    floats(&mut hash, values);
    format!("{:x}", hash.finalize())
}

pub(crate) fn capture_digest(
    e: &Engine,
    captures: &[SpecBoundaryCapture],
) -> Result<String, Box<dyn std::error::Error>> {
    let mut hash = Sha256::new();
    hash.update((captures.len() as u64).to_le_bytes());
    for cap in captures {
        if cap.latent_tails.iter().any(Option::is_some) {
            return Err("GDN prime capture oracle does not cover latent tail storage".into());
        }
        hash.update((cap.pos as u64).to_le_bytes());
        hash.update((cap.snap.pos as u64).to_le_bytes());
        floats(&mut hash, &cap.logits);
        floats(&mut hash, &cap.last_h);
        for lengths in [&cap.snap.kv_len, &cap.snap.tp_kv_len] {
            hash.update((lengths.len() as u64).to_le_bytes());
            for len in lengths {
                hash.update([u8::from(len.is_some())]);
                hash.update((len.unwrap_or(0) as u64).to_le_bytes());
            }
        }
        for (ssm, planes) in [(false, &cap.snap.conv), (true, &cap.snap.ssm)] {
            hash.update((planes.len() as u64).to_le_bytes());
            for (il, plane) in planes.iter().enumerate() {
                hash.update([u8::from(plane.is_some())]);
                if let Some(plane) = plane {
                    let mut words = e.dtoh(plane)?;
                    // a GDN plane holds its state in the class the snapshot recorded
                    // (crate::gdn_state): hash only the words that hold it (the hashed length
                    // tells the classes apart; an fp32 plane hashes whole, as before)
                    if ssm {
                        let bf16 = cap.snap.ssm_class(il);
                        words.truncate(crate::gdn_state::gdn_state_words(words.len(), bf16));
                    }
                    floats(&mut hash, &words);
                }
            }
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}
