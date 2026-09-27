//! Root preparation chooses a validated source bundle, never a legacy-load fallback.
use super::*;
use crate::source::Hy3RepackSource;

pub(crate) struct CompositeInput<'a> {
    source: &'a Hy3RepackSource,
}
impl<'a> CompositeInput<'a> {
    pub(crate) fn new(source: &'a Hy3RepackSource) -> Self {
        Self { source }
    }
}

enum Prepared<'a> {
    Single(Box<BoundTensorSource<'a>>),
    Composite(Box<composite::BoundCompositeSource<'a>>),
    Existing(&'a dyn TensorSource),
}

/// Root-owned input bundle. The callback cannot outlive its runtime adapter, while returned
/// device allocations and opaque disk views own the backing they need after loading finishes.
/// External adapters cannot substitute another composite artifact at this boundary:
/// ```compile_fail
/// use memra_gguf::{bound_source::CompositeInput, config::ModelConfig,
///     source::{TensorSource, TensorView, Hy3RepackSource}};
/// struct Swap<'a>(&'a Hy3RepackSource);
/// impl TensorSource for Swap<'_> {
///     fn config(&self)->ModelConfig { self.0.config() }
///     fn find(&self,_:&str)->Option<TensorView<'_>> { None }
///     fn composite_input(&self)->Option<CompositeInput<'_>> { self.0.composite_input() }
/// }
/// ```
/// A temporary runtime cannot lend a tensor beyond the loading callback:
/// ```compile_fail
/// use memra_gguf::{bound_source::PreparedModelSource, source::TensorView};
/// fn leak<'a>(prepared: &'a PreparedModelSource<'a>) -> TensorView<'a> {
///     prepared.with_runtime(|source| source.try_find("token_embd.weight").unwrap().unwrap()).unwrap()
/// }
/// ```
pub struct PreparedModelSource<'a> {
    inner: Prepared<'a>,
}
impl<'a> PreparedModelSource<'a> {
    pub fn text(source: &'a dyn TensorSource) -> Result<Self, String> {
        let inner = if let Some(program) = source.bound_program() {
            if matches!(program, BoundProgramRef::ExternalDraft(_)) {
                return Err("external draft bundle cannot be used at a text model root".into());
            }
            let (_, plan) = program.cloned_pair();
            if plan.vision.is_some() || plan.multimodal.is_some() {
                return Err("text root requires a text-scoped bound program; prepare the source with LoadScope::Text".into());
            }
            Prepared::Existing(source)
        } else if let Some(input) = source.composite_input() {
            Prepared::Composite(Box::new(composite::BoundCompositeSource::compile(
                input.source,
                LoadScope::Text,
            )?))
        } else {
            Prepared::Single(Box::new(
                BoundTensorSource::compile_for_scope(source, LoadScope::Text)
                    .map_err(|e| e.to_string())?,
            ))
        };
        Ok(Self { inner })
    }

    pub fn with_runtime<T>(&self, load: impl FnOnce(&dyn TensorSource) -> T) -> Result<T, String> {
        match &self.inner {
            Prepared::Single(bound) => Ok(load(&bound.runtime()?)),
            Prepared::Composite(bound) => Ok(load(&bound.runtime()?)),
            Prepared::Existing(source) => Ok(load(*source)),
        }
    }
}
