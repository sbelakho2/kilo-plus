//! Cancellation tokens. Std-only (no tokio) so core stays testable in plain
//! threads. Parent cancellation cascades to children; a cancel races with
//! `wait` and never loses (wait observes cancellation even if it started
//! before `cancel`). [`CancellationToken::cancelled`] is a wake-driven async
//! wait built on std waker registration — `cancel()` wakes every registered
//! waker, so async waiters surface cancellation immediately without polling
//! (audit round 14: the guarded-line transport used to poll on a timer).

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

#[derive(Default)]
struct Inner {
    cancelled: AtomicBool,
    waiters: Mutex<Vec<Arc<Waiter>>>,
    /// Registrations of live [`CancellationToken::cancelled`] futures. One
    /// entry per awaiting future, removed when the future drops; `cancel()`
    /// wakes every entry.
    async_waiters: Mutex<Vec<Waker>>,
}

/// A registration on a parent token. `on_cancel` (if set) is the token that
/// must be cancelled when the parent is — this is how `child()`/`attach()`
/// cascade. `wait()` registers with `on_cancel = None`.
struct Waiter {
    cond: Condvar,
    notified: Mutex<bool>,
    on_cancel: Mutex<Option<CancellationToken>>,
}

impl Waiter {
    fn for_wait() -> Arc<Self> {
        Arc::new(Self {
            cond: Condvar::new(),
            notified: Mutex::new(false),
            on_cancel: Mutex::new(None),
        })
    }

    fn for_cascade(other: CancellationToken) -> Arc<Self> {
        Arc::new(Self {
            cond: Condvar::new(),
            notified: Mutex::new(false),
            on_cancel: Mutex::new(Some(other)),
        })
    }
}

#[derive(Clone, Default)]
pub struct CancellationToken {
    inner: Arc<Inner>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// Resolve when this token is cancelled. Wake-driven and
    /// executor-agnostic (std waker registration — core stays
    /// dependency-free): the first poll registers the caller's waker with
    /// the token and `cancel()` wakes every registered waker, so a waiter
    /// surfaces cancellation immediately instead of polling. The cancel /
    /// register race never loses: the flag is checked before AND under the
    /// registration lock. Already-cancelled tokens resolve on the first
    /// poll. A dropped future unregisters itself (see
    /// [`CancelledAwait::drop`]).
    pub fn cancelled(&self) -> CancelledAwait<'_> {
        CancelledAwait {
            token: self,
            registered: None,
        }
    }

    /// Cancel. Returns true if this call performed the cancellation
    /// (first caller wins; subsequent calls return false).
    pub fn cancel(&self) -> bool {
        if self.inner.cancelled.swap(true, Ordering::AcqRel) {
            return false;
        }
        let waiters = self.inner.waiters.lock().unwrap();
        for w in waiters.iter() {
            {
                let mut n = w.notified.lock().unwrap();
                *n = true;
                w.cond.notify_all();
            }
            // Propagate to registered children. Reentrancy is bounded: a
            // child's cancel() sees the parent's flag already set and stops.
            let to_cancel = w.on_cancel.lock().unwrap().clone();
            if let Some(t) = to_cancel {
                t.cancel();
            }
        }
        drop(waiters);
        // Async waiters: drain under the lock, wake outside it — a wake may
        // re-poll the waiting task inline, and the poll would take the same
        // lock (std Mutex is not reentrant).
        let to_wake = {
            let mut async_waiters = self.inner.async_waiters.lock().unwrap();
            std::mem::take(&mut *async_waiters)
        };
        for w in to_wake {
            w.wake();
        }
        true
    }

    /// A child token: cancelling the parent cancels the child. Cancelling the
    /// child does not cancel the parent.
    pub fn child(&self) -> CancellationToken {
        let child = CancellationToken::new();
        self.attach(child.clone());
        child
    }

    /// Register a token that must be cancelled when this one is. Used by
    /// `child()` and by structured concurrency to fan cancellation out.
    pub fn attach(&self, other: CancellationToken) {
        let waiter = Waiter::for_cascade(other.clone());
        let mut waiters = self.inner.waiters.lock().unwrap();
        if self.inner.cancelled.load(Ordering::Acquire) {
            other.cancel();
            return;
        }
        waiters.push(waiter);
        // Re-check under the same lock so cancel() cannot interleave between
        // registration and this check: if the parent was cancelled before we
        // registered, cancel() already iterated waiters and never saw us.
        if self.inner.cancelled.load(Ordering::Acquire) {
            other.cancel();
        }
    }

    /// Block until cancelled (or immediately if already cancelled).
    pub fn wait(&self) {
        if self.inner.cancelled.load(Ordering::Acquire) {
            return;
        }
        let waiter = Waiter::for_wait();
        {
            let mut waiters = self.inner.waiters.lock().unwrap();
            if self.inner.cancelled.load(Ordering::Acquire) {
                return;
            }
            waiters.push(waiter.clone());
        }
        if self.inner.cancelled.load(Ordering::Acquire) {
            return;
        }
        let mut notified = waiter.notified.lock().unwrap();
        while !*notified && !self.inner.cancelled.load(Ordering::Acquire) {
            notified = waiter.cond.wait(notified).unwrap();
        }
    }
}

/// Future returned by [`CancellationToken::cancelled`]. Registers the task
/// waker with the token on the first poll and holds it so the registration
/// can be released when this future is dropped.
pub struct CancelledAwait<'a> {
    token: &'a CancellationToken,
    /// The waker this future registered (cleared on completion so drop
    /// never removes someone else's entry after `cancel()` drained the list).
    registered: Option<Waker>,
}

impl Future for CancelledAwait<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.token.is_cancelled() {
            this.registered = None;
            return Poll::Ready(());
        }
        let mut async_waiters = this.token.inner.async_waiters.lock().unwrap();
        // Re-check under the lock: cancel() cannot interleave between this
        // check and the registration below, so no wake can be missed.
        if this.token.is_cancelled() {
            this.registered = None;
            return Poll::Ready(());
        }
        match this.registered.as_ref() {
            // Still registered with the executor's current waker: nothing to
            // do (a spurious re-poll must not duplicate the registration).
            Some(existing) if existing.will_wake(cx.waker()) => {}
            Some(existing) => {
                // The executor switched wakers: replace our slot.
                if let Some(pos) = async_waiters.iter().position(|w| w.will_wake(existing)) {
                    async_waiters.remove(pos);
                }
                async_waiters.push(cx.waker().clone());
                this.registered = Some(cx.waker().clone());
            }
            None => {
                async_waiters.push(cx.waker().clone());
                this.registered = Some(cx.waker().clone());
            }
        }
        Poll::Pending
    }
}

impl Drop for CancelledAwait<'_> {
    fn drop(&mut self) {
        let Some(registered) = self.registered.take() else {
            return;
        };
        let mut async_waiters = self.token.inner.async_waiters.lock().unwrap();
        // Entries are per-future clones of the same waker; removing any one
        // matching entry is safe — at most one registration per live future
        // exists and wake semantics only need one survivor per task.
        if let Some(pos) = async_waiters.iter().position(|w| w.will_wake(&registered)) {
            async_waiters.remove(pos);
        }
    }
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::task::RawWakerVTable;
    use std::thread;
    use std::time::Duration;

    // ------------------------------------------------------------ async wait

    /// Minimal wake-driven executor (no tokio in core): polls the future;
    /// a wake sends `()` on the channel and the executor re-polls. Proves
    /// the async wait is woken by `cancel()` from another thread.
    fn wake_channel_vtable() -> &'static RawWakerVTable {
        use std::task::{RawWaker, RawWakerVTable};
        unsafe fn clone_raw(ptr: *const ()) -> RawWaker {
            let arc = Arc::from_raw(ptr as *const mpsc::Sender<()>);
            let cloned = arc.clone();
            std::mem::forget(arc);
            RawWaker::new(Arc::into_raw(cloned) as *const (), wake_channel_vtable())
        }
        unsafe fn wake_raw(ptr: *const ()) {
            let arc = Arc::from_raw(ptr as *const mpsc::Sender<()>);
            let _ = arc.send(());
        }
        unsafe fn wake_by_ref_raw(ptr: *const ()) {
            let arc = Arc::from_raw(ptr as *const mpsc::Sender<()>);
            let _ = arc.send(());
            std::mem::forget(arc);
        }
        unsafe fn drop_raw(ptr: *const ()) {
            drop(Arc::from_raw(ptr as *const mpsc::Sender<()>));
        }
        static VTABLE: RawWakerVTable =
            RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);
        &VTABLE
    }

    fn thread_waker(tx: mpsc::Sender<()>) -> Waker {
        use std::task::{RawWaker, Waker};
        let ptr = Arc::into_raw(Arc::new(tx)) as *const ();
        unsafe { Waker::from_raw(RawWaker::new(ptr, wake_channel_vtable())) }
    }

    /// Block until `fut` resolves or 5s pass without a wake (returns the
    /// outcome). Requires at least one wake per re-poll.
    fn block_on_wake_driven(fut: CancelledAwait<'_>) -> bool {
        use std::task::{Context, Poll};
        let mut fut = Box::pin(fut);
        let (tx, rx) = mpsc::channel::<()>();
        let waker = thread_waker(tx);
        let mut cx = Context::from_waker(&waker);
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(()) => return true,
                Poll::Pending => match rx.recv_timeout(Duration::from_secs(5)) {
                    Ok(()) => continue,
                    Err(_) => return false,
                },
            }
        }
    }

    /// Poll once; true when the future resolved immediately.
    fn poll_once(fut: &mut Pin<Box<CancelledAwait<'_>>>) -> bool {
        let (tx, _rx) = mpsc::channel::<()>();
        let waker = thread_waker(tx);
        let mut cx = std::task::Context::from_waker(&waker);
        fut.as_mut().poll(&mut cx).is_ready()
    }

    #[test]
    fn cancelled_async_resolves_immediately_when_already_cancelled() {
        let t = CancellationToken::new();
        let mut fut = Box::pin(t.cancelled());
        assert!(!poll_once(&mut fut), "uncancelled token must park");
        drop(fut);
        t.cancel();
        let mut fut = Box::pin(t.cancelled());
        assert!(poll_once(&mut fut));
    }

    #[test]
    fn cancel_wakes_registered_async_waiter() {
        let t = Arc::new(CancellationToken::new());
        let t2 = t.clone();
        let h = thread::spawn(move || block_on_wake_driven(t2.cancelled()));
        thread::sleep(Duration::from_millis(30));
        assert!(
            !h.is_finished(),
            "waiter must still be parked before cancel"
        );
        t.cancel();
        assert!(h.join().unwrap(), "cancel must wake the registered waiter");
    }

    #[test]
    fn many_async_waiters_all_wake() {
        let t = Arc::new(CancellationToken::new());
        let mut handles = vec![];
        for _ in 0..16 {
            let t = t.clone();
            handles.push(thread::spawn(move || block_on_wake_driven(t.cancelled())));
        }
        thread::sleep(Duration::from_millis(20));
        t.cancel();
        for h in handles {
            assert!(h.join().unwrap());
        }
    }

    #[test]
    fn dropped_async_waiter_unregisters() {
        let t = CancellationToken::new();
        let fut = t.cancelled();
        let mut fut = Box::pin(fut);
        {
            let (tx, _rx) = mpsc::channel::<()>();
            let waker = thread_waker(tx);
            let mut cx = std::task::Context::from_waker(&waker);
            assert!(fut.as_mut().poll(&mut cx).is_pending());
        }
        assert_eq!(t.inner.async_waiters.lock().unwrap().len(), 1);
        drop(fut); // no cancel ever: the registration must be released
        assert!(
            t.inner.async_waiters.lock().unwrap().is_empty(),
            "a dropped waiter must not leave a stale registration behind"
        );
        // Re-polling registers again and cancel still wakes it.
        let t2 = t.clone();
        let h = thread::spawn(move || block_on_wake_driven(t2.cancelled()));
        thread::sleep(Duration::from_millis(10));
        t.cancel();
        assert!(h.join().unwrap());
    }

    #[test]
    fn parent_cancel_wakes_child_async_waiter() {
        let parent = CancellationToken::new();
        let child = parent.child();
        let h = thread::spawn(move || block_on_wake_driven(child.cancelled()));
        thread::sleep(Duration::from_millis(20));
        parent.cancel();
        assert!(
            h.join().unwrap(),
            "cancelling the parent must cascade to the child's async waiters"
        );
    }

    #[test]
    fn async_cancel_race_never_loses() {
        // Hammer: async waiters race a cancel from another thread. Every
        // waiter must complete.
        let t = Arc::new(CancellationToken::new());
        let mut handles = vec![];
        for _ in 0..8 {
            let t = t.clone();
            handles.push(thread::spawn(move || {
                thread::sleep(Duration::from_micros(50));
                block_on_wake_driven(t.cancelled())
            }));
        }
        thread::sleep(Duration::from_millis(5));
        t.cancel();
        for h in handles {
            assert!(h.join().unwrap(), "a racing waiter must never hang");
        }
    }

    // -------------------------------------------------------------- sync API

    #[test]
    fn cancel_before_wait_returns_immediately() {
        let t = CancellationToken::new();
        assert!(!t.is_cancelled());
        assert!(t.cancel());
        assert!(!t.cancel(), "second cancel loses");
        assert!(t.is_cancelled());
        t.wait(); // must return instantly
    }

    #[test]
    fn cancel_wakes_blocked_waiters() {
        let t = Arc::new(CancellationToken::new());
        let t2 = t.clone();
        let h = thread::spawn(move || {
            t2.wait();
            true
        });
        thread::sleep(Duration::from_millis(30));
        t.cancel();
        assert!(h.join().unwrap());
    }

    #[test]
    fn many_waiters_all_wake() {
        let t = Arc::new(CancellationToken::new());
        let mut handles = vec![];
        for _ in 0..32 {
            let t = t.clone();
            handles.push(thread::spawn(move || {
                t.wait();
            }));
        }
        thread::sleep(Duration::from_millis(20));
        t.cancel();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn child_cancel_does_not_cancel_parent() {
        let parent = CancellationToken::new();
        let child = parent.child();
        assert!(child.cancel());
        assert!(!parent.is_cancelled());
        assert!(child.is_cancelled());
    }

    #[test]
    fn parent_cancel_cascades_to_children() {
        let parent = CancellationToken::new();
        let c1 = parent.child();
        let c2 = parent.child();
        let c3 = c1.child(); // grandchild
        parent.cancel();
        assert!(c1.is_cancelled());
        assert!(c2.is_cancelled());
        assert!(c3.is_cancelled());
    }

    #[test]
    fn attach_before_cancel_propagates() {
        let a = CancellationToken::new();
        let b = CancellationToken::new();
        a.attach(b.clone());
        a.cancel();
        assert!(b.is_cancelled());
    }

    #[test]
    fn attach_after_cancel_immediately_cancels() {
        let a = CancellationToken::new();
        a.cancel();
        let b = CancellationToken::new();
        a.attach(b.clone());
        assert!(b.is_cancelled(), "late attach must not leak a live token");
    }

    #[test]
    fn cancel_while_attaching_is_race_safe() {
        // Hammer: spawn threads that attach while another cancels; afterwards
        // every attached token must be cancelled.
        let parent = Arc::new(CancellationToken::new());
        let children: Vec<Arc<CancellationToken>> = (0..64)
            .map(|_| Arc::new(CancellationToken::new()))
            .collect();
        let mut handles = vec![];
        for (i, c) in children.iter().enumerate() {
            let parent = parent.clone();
            let c = c.clone();
            handles.push(thread::spawn(move || {
                if i % 2 == 0 {
                    thread::sleep(Duration::from_micros(5));
                }
                parent.attach((*c).clone());
            }));
        }
        thread::sleep(Duration::from_millis(3));
        parent.cancel();
        for h in handles {
            h.join().unwrap();
        }
        for c in &children {
            assert!(c.is_cancelled());
        }
    }

    #[test]
    fn wait_never_hangs_after_cancel_under_stress() {
        let parent = Arc::new(CancellationToken::new());
        let mut waiters = vec![];
        for _ in 0..16 {
            let p = parent.clone();
            waiters.push(thread::spawn(move || p.wait()));
        }
        parent.cancel();
        for w in waiters {
            w.join().unwrap();
        }
    }
}
