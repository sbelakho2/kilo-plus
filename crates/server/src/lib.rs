//! kilop-server — the HTTP/SSE surface of the daemon, speaking the frozen
//! v7.5.6 protocol. The UI connection is disposable: turns run detached from
//! any SSE connection and resume from the journal.
//!
//! Auth: a random per-start token; every endpoint except `/api/hello`
//! requires `Authorization: Bearer <token>`.

pub mod api;
pub mod auth;
pub mod permission;

pub use api::{serve, ServerDeps, ServerHandle};
pub use auth::{AuthToken, check_bearer};
pub use permission::ChannelPermissionRequester;
