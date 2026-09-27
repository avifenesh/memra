//! Day 47 (`research/spill-c-20260919/DAY47.md`): host buffers a bank lease can own instead of a
//! heap `Vec<u8>`. A native owner installs a `HostBufferSource` (the MoE slot cache door's pool of
//! cached pinned buffers); the bank then reads each record straight into a buffer from it, verifies
//! it there, and the lease owns it. Dropping a buffer returns it to its pool, and a lease's
//! backing is dropped only when the lease is released, after every ticket on it has retired.

/// Fixed host bytes a lease can own. `Send` because the host fill fills buffers on its worker
/// threads and hands them to the owner.
pub trait HostBuffer: Send {
    fn as_slice(&self) -> &[u8];
    fn as_mut_slice(&mut self) -> &mut [u8];
}

/// Where the bank takes a record's buffer from. `None` is an exhausted pool: the read refuses
/// (`Capacity`) and nothing is allocated in its place.
pub trait HostBufferSource {
    fn take(&mut self, len: usize) -> Option<Box<dyn HostBuffer>>;
}

/// A record's bytes: a heap `Vec` (no source installed) or a pooled buffer.
pub enum HostBytes {
    Heap(Vec<u8>),
    Pooled(Box<dyn HostBuffer>),
}
impl HostBytes {
    pub fn bytes(&self) -> &[u8] {
        match self {
            HostBytes::Heap(v) => v,
            HostBytes::Pooled(b) => b.as_slice(),
        }
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            HostBytes::Heap(v) => v,
            HostBytes::Pooled(b) => b.as_mut_slice(),
        }
    }
    pub fn len(&self) -> usize {
        self.bytes().len()
    }
    pub fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }
    /// The lease backing: `Vec<u8>` for heap bytes (today's type, so every existing
    /// `resource::<Vec<u8>>()` reader is unchanged), `Box<dyn HostBuffer>` for a pooled buffer.
    pub fn into_backing(self) -> Box<dyn std::any::Any> {
        match self {
            HostBytes::Heap(v) => Box::new(v),
            HostBytes::Pooled(b) => Box::new(b),
        }
    }
}
impl std::fmt::Debug for HostBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HostBytes::Heap(v) => write!(f, "HostBytes::Heap({} bytes)", v.len()),
            HostBytes::Pooled(b) => write!(f, "HostBytes::Pooled({} bytes)", b.as_slice().len()),
        }
    }
}
