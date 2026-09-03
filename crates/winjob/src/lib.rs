//! Windows Job Objects (audit: taskkill /T is not equivalent to OS-enforced
//! kill-on-close). One job per supervisor with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`: when the daemon process dies, the
//! OS itself terminates every assigned child tree — the guarantee does not
//! depend on Kilo+ staying alive long enough to call taskkill.
//!
//! Certification status: code is `cargo check`-verified against
//! `x86_64-pc-windows-msvc`; runtime certification requires a Windows
//! runner (declared platform blocker on this host).

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    pub struct JobGuard {
        handle: HANDLE,
    }

    // SAFETY: assignment is thread-safe; Drop closes the handle.
    unsafe impl Send for JobGuard {}
    unsafe impl Sync for JobGuard {}

    impl JobGuard {
        pub fn create() -> Option<Self> {
            unsafe {
                let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if handle.is_null() {
                    return None;
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    handle,
                    9, // JobObjectExtendedLimitInformation
                    &info as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    CloseHandle(handle);
                    return None;
                }
                Some(Self { handle })
            }
        }

        /// No-op guard for when job creation failed at startup.
        pub fn null() -> Self {
            Self {
                handle: std::ptr::null_mut(),
            }
        }

        /// Assign a child pid (best-effort; a failed assignment is logged —
        /// the supervisor's own kill paths still apply).
        pub fn assign(&self, pid: u32) {
            unsafe {
                let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
                if process.is_null() {
                    return;
                }
                let r = AssignProcessToJobObject(self.handle, process);
                if r == 0 {
                    tracing::warn!("AssignProcessToJobObject({pid}) failed");
                }
                CloseHandle(process);
            }
        }
    }

    impl Drop for JobGuard {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    CloseHandle(self.handle);
                }
            }
        }
    }
}

#[cfg(windows)]
pub use imp::JobGuard;

#[cfg(not(windows))]
pub struct JobGuard;

#[cfg(not(windows))]
impl JobGuard {
    pub fn create() -> Option<Self> {
        None
    }
    pub fn null() -> Self {
        Self
    }
    pub fn assign(&self, _pid: u32) {}
}
