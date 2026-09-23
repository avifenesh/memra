use memra_gguf::{bound_source::{BoundProgramRef, BoundRuntimeSource},
    config::ModelConfig, source::{TensorSource, TensorView}};
struct Swapped<'a,'b,'c> { config:ModelConfig, other:&'a BoundRuntimeSource<'b,'c> }
impl TensorSource for Swapped<'_, '_, '_> {
 fn config(&self)->ModelConfig {self.config.clone()}
 fn find(&self,_:&str)->Option<TensorView<'_>> {None}
 fn bound_program(&self)->Option<BoundProgramRef<'_>> {self.other.bound_program()}
}
fn main() {}
