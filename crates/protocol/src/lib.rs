//! kilop-protocol — the frozen v7.5.6 wire contract.
//!
//! Golden tests lock request/response/SSE/JSON-field-presence/null-behavior/
//! error-code behavior against the permanent fixtures in
//! `compat/kilo-v756/`. Changing wire behavior requires updating fixtures.

pub mod error;
pub mod fixtures;
pub mod sse;
pub mod v756;

pub use error::ApiError;
pub use v756::*;
