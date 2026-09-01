//! Permission requester for the HTTP world: the agent waits (with a timeout)
//! until the frozen UI resolves the permission through
//! `POST /api/perm/{id}/resolve`.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kilop_agent::PermissionRequester;
use kilop_core::capability::PermissionDecision;
use kilop_core::id::SessionId;
use kilop_session::ops::PermissionRequest;
use tokio::sync::Notify;

#[derive(Clone, Debug)]
pub struct PendingPermission {
    pub id: i64,
    pub session_id: SessionId,
    pub capability: String,
    pub detail: serde_json::Value,
}

#[derive(Clone)]
pub struct ChannelPermissionRequester {
    waiters: Arc<Mutex<HashMap<i64, Arc<Notify>>>>,
    decisions: Arc<Mutex<HashMap<i64, PermissionDecision>>>,
    pending: Arc<Mutex<HashMap<i64, PendingPermission>>>,
    timeout: Duration,
}

impl ChannelPermissionRequester {
    pub fn new(timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            waiters: Arc::new(Mutex::new(HashMap::new())),
            decisions: Arc::new(Mutex::new(HashMap::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            timeout,
        })
    }

    /// Called by the HTTP resolver. Returns false when the permission is
    /// unknown or already resolved (never double-resolves).
    pub fn resolve(&self, permission_id: i64, decision: PermissionDecision) -> bool {
        {
            let mut decisions = self.decisions.lock().unwrap();
            if decisions.contains_key(&permission_id) {
                return false;
            }
            decisions.insert(permission_id, decision);
        }
        self.pending.lock().unwrap().remove(&permission_id);
        if let Some(notify) = self.waiters.lock().unwrap().remove(&permission_id) {
            notify.notify_one();
        }
        true
    }

    pub fn pending_count(&self) -> usize {
        self.waiters.lock().unwrap().len()
    }

    /// The ids currently waiting for resolution (sorted, stable).
    pub fn pending_ids(&self) -> Vec<i64> {
        let mut v: Vec<i64> = self.waiters.lock().unwrap().keys().copied().collect();
        v.sort_unstable();
        v
    }

    /// Snapshot of pending permission requests (id, session, capability,
    /// detail) for `GET /permission/list`.
    pub fn pending_views(&self) -> Vec<PendingPermission> {
        let mut v: Vec<PendingPermission> =
            self.pending.lock().unwrap().values().cloned().collect();
        v.sort_by_key(|p| p.id);
        v
    }
}

impl PermissionRequester for ChannelPermissionRequester {
    fn request(
        &self,
        session: SessionId,
        permission: &PermissionRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = kilop_core::Result<PermissionDecision>> + Send>>
    {
        let id = permission.id;
        let notify = Arc::new(Notify::new());
        let detail =
            serde_json::to_value(&permission.capability).unwrap_or(serde_json::Value::Null);
        self.waiters.lock().unwrap().insert(id, notify.clone());
        self.pending.lock().unwrap().insert(
            id,
            PendingPermission {
                id,
                session_id: session,
                capability: detail
                    .get("capability")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                detail,
            },
        );
        // A decision may already exist (the resolver raced ahead of us).
        if let Some(d) = self.decisions.lock().unwrap().get(&id).copied() {
            self.waiters.lock().unwrap().remove(&id);
            self.pending.lock().unwrap().remove(&id);
            return Box::pin(async move { Ok(d) });
        }
        let me = self.clone();
        Box::pin(async move {
            tokio::select! {
                _ = notify.notified() => {
                    match me.decisions.lock().unwrap().get(&id).copied() {
                        Some(decision) => Ok(decision),
                        None => Err(kilop_core::error::Error::permission(
                            format!("permission {id} resolved without a decision"),
                        )),
                    }
                }
                _ = tokio::time::sleep(me.timeout) => {
                    me.waiters.lock().unwrap().remove(&id);
                    me.pending.lock().unwrap().remove(&id);
                    Err(kilop_core::error::Error::timeout(format!(
                        "permission {id} not resolved within {}ms",
                        me.timeout.as_millis()
                    )))
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::capability::Capability;
    use kilop_core::id::OpId;

    fn permission(id: i64) -> PermissionRequest {
        PermissionRequest {
            id,
            op_id: OpId::new(1),
            capability: Capability::ExecuteShell {
                command: "ls".into(),
            },
            event_seq: kilop_core::id::EventSeq::new(1),
        }
    }

    #[tokio::test]
    async fn resolver_wakes_waiter() {
        let r = ChannelPermissionRequester::new(Duration::from_secs(5));
        let r2 = r.clone();
        let handle =
            tokio::spawn(
                async move { r2.request(SessionId::new(1), &permission(7)).await.unwrap() },
            );
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(r.resolve(7, PermissionDecision::Allow));
        let decision = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(decision, PermissionDecision::Allow);
        assert_eq!(r.pending_count(), 0);
    }

    #[tokio::test]
    async fn double_resolve_loses_second() {
        let r = ChannelPermissionRequester::new(Duration::from_secs(5));
        assert!(r.resolve(1, PermissionDecision::Allow));
        assert!(
            !r.resolve(1, PermissionDecision::Deny),
            "first decision wins"
        );
    }

    #[tokio::test]
    async fn timeout_returns_permission_error() {
        let r = ChannelPermissionRequester::new(Duration::from_millis(30));
        let result = r.request(SessionId::new(1), &permission(2)).await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind,
            kilop_core::error::ErrorKind::Timeout
        );
        assert_eq!(r.pending_count(), 0, "waiter cleaned up after timeout");
    }

    #[tokio::test]
    async fn pre_resolved_decision_is_returned_immediately() {
        let r = ChannelPermissionRequester::new(Duration::from_secs(5));
        assert!(r.resolve(3, PermissionDecision::Deny));
        let d = r.request(SessionId::new(1), &permission(3)).await.unwrap();
        assert_eq!(d, PermissionDecision::Deny);
    }

    #[tokio::test]
    async fn many_concurrent_waiters_resolve_independently() {
        let r = ChannelPermissionRequester::new(Duration::from_secs(5));
        let mut handles = Vec::new();
        for id in 100..110 {
            let r = r.clone();
            handles.push(tokio::spawn(async move {
                r.request(SessionId::new(1), &permission(id)).await
            }));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        for id in 100..110 {
            assert!(r.resolve(id, PermissionDecision::Allow));
        }
        for h in handles {
            assert_eq!(h.await.unwrap().unwrap(), PermissionDecision::Allow);
        }
    }

    #[tokio::test]
    async fn pending_views_reflect_live_requests_and_cleanup() {
        let r = ChannelPermissionRequester::new(Duration::from_millis(30));
        let mut handles = Vec::new();
        for (id, session) in [(1, SessionId::new(10)), (2, SessionId::new(20))] {
            let r = r.clone();
            handles.push(tokio::spawn(async move {
                r.request(session, &permission(id)).await
            }));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        let views = r.pending_views();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].id, 1);
        assert_eq!(views[0].session_id, SessionId::new(10));
        assert_eq!(views[0].capability, "execute_shell");
        assert_eq!(views[0].detail["detail"]["command"], "ls");
        // Resolving removes the view.
        assert!(r.resolve(1, PermissionDecision::Allow));
        assert_eq!(r.pending_views().len(), 1);
        assert_eq!(r.pending_views()[0].id, 2);
        // Timeout cleans the remaining view.
        for h in handles {
            let _ = h.await;
        }
        assert!(
            r.pending_views().is_empty(),
            "timed-out request must not linger"
        );
        // Double resolve: no second view, decision wins.
        assert!(!r.resolve(1, PermissionDecision::Deny));
        assert!(r.pending_views().is_empty());
    }
}
