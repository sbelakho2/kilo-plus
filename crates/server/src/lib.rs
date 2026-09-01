//! kilop-server — the HTTP/SSE surface of the daemon, speaking the frozen
//! v7.5.6 protocol. The UI connection is disposable: turns run detached from
//! any SSE connection and resume from the journal.
//!
//! Auth: the frontend generates `KILO_SERVER_PASSWORD` and passes it via env;
//! every endpoint except `/global/health` requires it, in either the
//! `Authorization: Bearer` or the `x-kilo-server-password` header form.

pub mod api;
pub mod auth;
pub mod coalesce;
pub mod global;
pub mod permission;

pub use api::{serve, ServerDeps, ServerHandle};
pub use auth::{check_bearer, check_password, AuthToken, ServerPassword};
pub use coalesce::DeltaCoalescer;
pub use global::GlobalEventBus;
pub use permission::{ChannelPermissionRequester, PendingPermission};
