//! Frozen metadata lives only in memra-tier. No local wire/identity compatibility shim.
pub use memra_tier::contracts::{
    ByteSegment, Digest, EncodingId, Epochs, GroupRequirement, KvBlockId, OwnerAlias,
    PageRequirement, ProgramIdentity, RecordLayout, Role, StateBundle, StateKind, TierError,
    WIRE_VERSION, Wire, checksum, digest,
};
