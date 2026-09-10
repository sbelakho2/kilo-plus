//! Faktor-native evidence/CCR layer (audit 34/36/37/38/39).
//!
//! Raw tool/process/search output is normalized into typed compact
//! representations with full backing in CAS, provenance that can never
//! gain instruction authority, compression policies hardcoded by kind,
//! and retrieval by typed selector — never whole-blob auto-dumps.

pub mod compress;
pub mod normalize;
pub mod provenance;
pub mod render;
pub mod retrieve;
pub mod store;
pub mod types;
