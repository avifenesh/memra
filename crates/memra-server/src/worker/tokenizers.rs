//! Tokenizers are one immutable startup snapshot shared by worker and HTTP consumers.
use memra_tokenizer::Tokenizer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(crate) type SharedTokenizers = Arc<HashMap<String, Arc<Tokenizer>>>;

struct LoadedTokenizer {
    source: String,
    tokenizer: Arc<Tokenizer>,
}

/// Owned by the worker supervisor, so a GPU-worker respawn keeps the same tokenizer
/// objects as the already-running HTTP listener. A process restart selects new metadata.
#[derive(Default)]
pub struct TokenizerSnapshots {
    loaded: Mutex<HashMap<String, LoadedTokenizer>>,
}

impl TokenizerSnapshots {
    pub fn load(
        &self,
        name: &str,
        source: &str,
        open: impl FnOnce() -> Result<Tokenizer, String>,
    ) -> Result<Arc<Tokenizer>, String> {
        let mut loaded = self
            .loaded
            .lock()
            .map_err(|_| "tokenizer snapshot poisoned")?;
        if let Some(existing) = loaded.get(name) {
            if existing.source != source {
                return Err(format!(
                    "model {name:?}: tokenizer source changed within one server"
                ));
            }
            return Ok(existing.tokenizer.clone());
        }
        let tokenizer = Arc::new(open()?);
        loaded.insert(
            name.into(),
            LoadedTokenizer {
                source: source.into(),
                tokenizer: tokenizer.clone(),
            },
        );
        Ok(tokenizer)
    }

    /// Called only after every model has loaded successfully. No filesystem reads occur.
    pub fn snapshot(&self, names: &[String]) -> Result<SharedTokenizers, String> {
        let loaded = self
            .loaded
            .lock()
            .map_err(|_| "tokenizer snapshot poisoned")?;
        let tokenizers = names
            .iter()
            .map(|name| {
                loaded
                    .get(name)
                    .map(|entry| (name.clone(), entry.tokenizer.clone()))
                    .ok_or_else(|| format!("model {name:?}: loaded tokenizer snapshot is missing"))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        Ok(Arc::new(tokenizers))
    }
}

#[cfg(test)]
#[path = "tokenizers/tests.rs"]
mod tests;
