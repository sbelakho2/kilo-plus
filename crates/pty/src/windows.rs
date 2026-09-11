//! Windows ConPTY backend for `faktor-pty`.
//!
//! `Pty::spawn` builds the canonical Microsoft pseudoconsole session
//! (docs/creating-a-pseudoconsole-session):
//!
//! 1. two anonymous pipes — the ConPTY input channel's write end carries
//!    child stdin, the ConPTY output channel's read end delivers child
//!    stdout/stderr (merged through the console);
//! 2. `CreatePseudoConsole` binds the input pipe's READ end and the output
//!    pipe's WRITE end;
//! 3. the child is spawned with `CreateProcessW` carrying
//!    `PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE` and
//!    `EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT` (never
//!    `CREATE_NEW_CONSOLE`/`DETACHED_PROCESS`/`STARTF_USESTDHANDLES` — the
//!    pseudoconsole owns the console), attaching its console I/O to the
//!    pseudoconsole;
//! 4. the conpty-side pipe ends are released after spawn (per the docs) so
//!    a closed session surfaces as a broken pipe instead of a deadlock.
//!
//! Output is drained by a dedicated reader thread into the shared bounded
//! ring (identical semantics to the unix backend). The reader never blocks
//! in `ReadFile`: it peeks first and polls the child's process handle, so a
//! dead child or a closed session ends the thread on its own and `Drop`
//! never races an outstanding read.
//!
//! Job-object enforcement is deliberately NOT duplicated here: `faktor-pty`
//! does not depend on `faktor-winjob` (scope is ConPTY only), so killing a
//! pty terminates the direct child, while grandchildren-tree guarantees
//! stay with winjob's own call sites.
//!
//! Certification status: code is `cargo check`/`clippy`-verified against
//! `x86_64-pc-windows-msvc`; runtime certification requires a Windows
//! runner (declared platform blocker on this host — same posture as
//! `crates/winjob`).

use std::ffi::c_void;
use std::fmt;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use faktor_core::error::Error;

use windows_sys::Win32::Foundation::{
    CloseHandle, DuplicateHandle, GetLastError, DUPLICATE_SAME_ACCESS, HANDLE, WAIT_OBJECT_0,
    WAIT_TIMEOUT,
};
use windows_sys::Win32::Storage::FileSystem::{ReadFile, SearchPathW, WriteFile};
use windows_sys::Win32::System::Console::{
    ClosePseudoConsole, CreatePseudoConsole, ResizePseudoConsole, COORD, HPCON,
};
use windows_sys::Win32::System::Pipes::{CreatePipe, PeekNamedPipe};
use windows_sys::Win32::System::Threading::{
    CreateProcessW, DeleteProcThreadAttributeList, GetCurrentProcess,
    InitializeProcThreadAttributeList, TerminateProcess, UpdateProcThreadAttribute,
    WaitForSingleObject, CREATE_UNICODE_ENVIRONMENT, EXTENDED_STARTUPINFO_PRESENT,
    PROCESS_INFORMATION, PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE, STARTUPINFOEXW,
};

use crate::ring::Ring;
use crate::validation::validate_spawn_config;
use crate::win_common;
use crate::PtyConfig;

/// `ERROR_FILE_NOT_FOUND` (winerror.h, frozen ABI value): SearchPathW's
/// "module not found" result.
const ERROR_FILE_NOT_FOUND: u32 = 2;

/// CreateProcessW appends `.exe` to an extension-less application name;
/// SearchPathW does so only when handed an explicit extension, so pass the
/// documented one to keep the resolution semantics identical.
const EXE_EXTENSION: [u16; 5] = [b'.' as u16, b'e' as u16, b'x' as u16, b'e' as u16, 0];

/// Grace period after ClosePseudoConsole before the TerminateProcess
/// fallback, and the bounded wait after TerminateProcess (drop must never
/// hang).
const KILL_GRACE_MS: u32 = 500;
const TERMINATE_WAIT_MS: u32 = 1000;

/// Reader idle poll interval: keeps output latency low without a busy loop
/// (the thread never blocks on ReadFile, so 5 ms of sleep is worst-case
/// wakeup latency).
const READER_IDLE: std::time::Duration = std::time::Duration::from_millis(5);

fn last_error() -> u32 {
    unsafe { GetLastError() }
}

fn win_err(operation: &str, code: u32) -> Error {
    Error::new(
        win_common::classify_win32(code),
        format!("{operation} failed (win32 error {code})"),
    )
}

fn hresult_err(operation: &str, hr: u32) -> Error {
    Error::new(
        win_common::classify_hresult(hr),
        format!("{operation} failed (hr 0x{hr:08X})"),
    )
}

fn closed_error() -> Error {
    Error::internal("pty is closed (conpty session ended)")
}

/// Loud INVALID_HANDLE_VALUE detection (CreatePipe/CreatePseudoConsole/
/// CreateProcessW successes must hand back usable handles).
fn valid_handle(h: HANDLE) -> bool {
    !h.is_null() && (h as isize) != -1
}

/// Raw Win32 HANDLE for crossing thread boundaries: HANDLE is `*mut
/// c_void`, which is not `Send`, but the kernel objects behind the handles
/// this crate passes to its reader thread (a pipe + a process) tolerate the
/// operations we perform from any thread (PeekNamedPipe/ReadFile/
/// WaitForSingleObject/CloseHandle). Copy so the spawn error path can keep
/// using the original handle after the closure captured its copy.
#[derive(Clone, Copy)]
struct SendHandle(HANDLE);

// SAFETY: see the struct docs — the wrapped kernel objects are thread-safe
// for the wait/peek/read/close operations the reader thread performs.
unsafe impl Send for SendHandle {}

fn pack_size(rows: u16, cols: u16) -> u32 {
    ((rows as u32) << 16) | cols as u32
}

fn unpack_size(v: u32) -> (u16, u16) {
    ((v >> 16) as u16, v as u16)
}

/// Zeroed storage for the opaque `PPROC_THREAD_ATTRIBUTE_LIST`, aligned to
/// `MEMORY_ALLOCATION_ALIGNMENT` (16 on x64; the canonical MSDN sample gets
/// that from HeapAlloc). A `Vec<u64>` alone guarantees only 8-byte
/// alignment, so one extra word lets the pointer advance to the next
/// 16-byte boundary. Returns the owning Vec (it MUST outlive every use of
/// the pointer) and the aligned pointer.
fn aligned_attr_storage(bytes: usize) -> (Vec<u64>, *mut c_void) {
    let words = bytes.div_ceil(size_of::<u64>()).max(1);
    let mut storage = vec![0u64; words + 1];
    let base = storage.as_mut_ptr();
    let aligned = if (base as usize).is_multiple_of(16) {
        base
    } else {
        // u64 storage is 8-aligned, so one word lands on the next 16.
        unsafe { base.add(1) }
    };
    (storage, aligned.cast::<c_void>())
}

/// Windows-only pre-spawn validation that needs no Win32 call. The shared
/// `validate_spawn_config` already rejects an empty program, NUL bytes, and
/// oversized fields; a whitespace-only program slips through it but can
/// never produce a valid `lpApplicationName`/command-line token, so it is
/// Malformed here before any handle exists.
fn validate_program(command: &str) -> Result<(), Error> {
    if command.trim().is_empty() {
        return Err(Error::malformed(
            "pty program must not be empty or whitespace-only",
        ));
    }
    Ok(())
}

/// True when `command` already names a path (separator or drive designator)
/// rather than a bare module name. Explicit paths are passed to
/// CreateProcessW verbatim so Windows reports its own not-found/bad-image
/// error instead of this crate inventing a PATH-search miss.
fn has_path_component(command: &str) -> bool {
    command.contains('\\') || command.contains('/') || command.as_bytes().get(1) == Some(&b':')
}

/// One `SearchPathW` invocation with a growable buffer. `Ok(None)` means the
/// file was not found; any other failure is returned with its win32 code.
fn search_path(dir: *const u16, name: &[u16]) -> Result<Option<Vec<u16>>, u32> {
    let mut buf = vec![0u16; 260];
    loop {
        let len = unsafe {
            SearchPathW(
                dir,
                name.as_ptr(),
                EXE_EXTENSION.as_ptr(),
                buf.len() as u32,
                buf.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if len == 0 {
            let code = last_error();
            return if code == ERROR_FILE_NOT_FOUND {
                Ok(None)
            } else {
                Err(code)
            };
        }
        if len as usize > buf.len() {
            // Too small: SearchPathW returned the required size INCLUDING
            // the terminating NUL; retry with exactly that much room.
            buf.resize(len as usize, 0);
            continue;
        }
        buf.truncate(len as usize); // logical length WITHOUT the NUL
        while buf.last() == Some(&0) {
            buf.pop();
        }
        return Ok(Some(buf));
    }
}

/// Resolve the program to the module path handed to `CreateProcessW` as
/// `lpApplicationName`:
///
/// - explicit paths (separator/drive) pass through verbatim;
/// - a bare name is resolved with `SearchPathW` (documented search order
///   with the `.exe` append), never left to CreateProcessW's own module
///   guessing, so `powershell.exe` deterministically becomes
///   `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`;
/// - PowerShell's install directory is a PATH entry, not the system
///   directory itself, so it is probed explicitly before failing, keeping
///   the lifecycle tests runnable even with a stripped child environment.
fn resolve_application(command: &str) -> Result<Vec<u16>, Error> {
    validate_program(command)?;
    if has_path_component(command) {
        let mut w = to_wide(command);
        while w.last() == Some(&0) {
            w.pop();
        }
        return Ok(w);
    }
    let name = to_wide(command);
    let mut code = ERROR_FILE_NOT_FOUND;
    match search_path(std::ptr::null(), &name) {
        Ok(Some(path)) => return Ok(path),
        Err(c) => code = c,
        Ok(None) => {}
    }
    if let Some(root) = std::env::var_os("SystemRoot") {
        let dir = format!(
            "{}\\System32\\WindowsPowerShell\\v1.0",
            root.to_string_lossy()
        );
        let dir_wide = to_wide(&dir);
        match search_path(dir_wide.as_ptr(), &name) {
            Ok(Some(path)) => return Ok(path),
            Err(c) => code = c,
            Ok(None) => {}
        }
    }
    Err(win_err(&format!("SearchPathW({command})"), code))
}

/// One live ConPTY pty. Sync API with identical semantics to the unix
/// backend: writes block until accepted, reads are non-blocking snapshots
/// of a ring drained by a background thread, `kill`/`Drop` close the
/// pseudoconsole and terminate the child with a bounded fallback.
pub struct Pty {
    /// Pseudoconsole handle; 0 once closed (kill/Drop).
    pc: HPCON,
    /// Write end of the ConPTY input pipe — WriteFile here feeds child
    /// stdin.
    input: HANDLE,
    /// Read end of the ConPTY output pipe (closed on Drop; the reader
    /// thread works on its own duplicate).
    output: HANDLE,
    /// Child process handle (wait/terminate; never inherited).
    child: HANDLE,
    pid: u32,
    /// Last size applied through ResizePseudoConsole (ConPTY exposes no
    /// size query, so this is the honest source of truth).
    last_size: AtomicU32,
    shared: Arc<(Mutex<Ring>, Condvar)>,
    reader_stop: Arc<AtomicBool>,
    /// Set once the session has been closed (kill/Drop/observed broken
    /// channel): further writes return a typed error instead of touching a
    /// dead session.
    closed: AtomicBool,
}

// SAFETY: Pty performs only thread-safe Win32 calls on its handles. The
// output pipe is duplicated before the reader thread starts, so the thread
// never touches a handle this struct also closes; handle lifetime is
// enforced by Drop/ownership on the main side.
unsafe impl Send for Pty {}
// SAFETY: all &self methods (write/resize/size/read/kill-adjacent) operate
// on Win32 objects that tolerate concurrent use from separate threads (the
// process handle is waitable by both the reader thread and is_alive).
unsafe impl Sync for Pty {}

impl fmt::Debug for Pty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pty")
            .field("pid", &self.pid)
            .field("pc", &self.pc)
            .finish_non_exhaustive()
    }
}

impl Pty {
    /// Create the pseudoconsole and spawn the child attached to it.
    pub fn spawn(cfg: &PtyConfig) -> Result<Self, Error> {
        validate_spawn_config(cfg)?;
        // Geometry is validated pre-spawn: ConPTY COORD dims are i16, so a
        // config that cannot be honored errors loudly instead of silently
        // truncating.
        win_common::validate_geometry(cfg.rows, cfg.cols)?;
        // Resolve the module BEFORE any handle exists: a blank program is
        // Malformed here and a PATH miss is a typed SearchPathW error, so no
        // pipe or pseudoconsole ever has to be reaped on the failure path.
        let app_wide = resolve_application(&cfg.command)?;

        let mut pc: HPCON = 0;
        // ConPTY input pipe: input_read is consumed by the pseudoconsole,
        // input_write feeds child stdin.
        let mut input_read: HANDLE = std::ptr::null_mut();
        let mut input_write: HANDLE = std::ptr::null_mut();
        // ConPTY output pipe: output_write is consumed by the
        // pseudoconsole, output_read delivers child stdout.
        let mut output_read: HANDLE = std::ptr::null_mut();
        let mut output_write: HANDLE = std::ptr::null_mut();

        // (1) pipes. NULL security attributes = non-inheritable handles;
        // the child is attached through the pseudoconsole attribute, so no
        // automatic inheritance is needed or wanted.
        if unsafe { CreatePipe(&mut input_read, &mut input_write, std::ptr::null(), 0) } == 0 {
            return Err(win_err("CreatePipe(conpty input)", last_error()));
        }
        if !valid_handle(input_read) || !valid_handle(input_write) {
            unsafe {
                CloseHandle(input_read);
                CloseHandle(input_write);
            }
            return Err(Error::internal("CreatePipe returned an invalid handle"));
        }
        if unsafe { CreatePipe(&mut output_read, &mut output_write, std::ptr::null(), 0) } == 0 {
            let code = last_error();
            unsafe {
                CloseHandle(input_read);
                CloseHandle(input_write);
            }
            return Err(win_err("CreatePipe(conpty output)", code));
        }
        if !valid_handle(output_read) || !valid_handle(output_write) {
            unsafe {
                CloseHandle(input_read);
                CloseHandle(input_write);
                CloseHandle(output_read);
                CloseHandle(output_write);
            }
            return Err(Error::internal("CreatePipe returned an invalid handle"));
        }

        // (2) the pseudoconsole itself (geometry already validated).
        let coord = COORD {
            X: cfg.cols as i16,
            Y: cfg.rows as i16,
        };
        let hr = unsafe { CreatePseudoConsole(coord, input_read, output_write, 0, &mut pc) };
        if hr < 0 || !valid_handle(pc as HANDLE) {
            let code = hr as u32;
            unsafe {
                CloseHandle(input_read);
                CloseHandle(input_write);
                CloseHandle(output_read);
                CloseHandle(output_write);
            }
            return Err(hresult_err("CreatePseudoConsole", code));
        }

        // (3) STARTUPINFOEXW carrying PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE.
        // `cb` MUST be sizeof(STARTUPINFOEXW) with
        // EXTENDED_STARTUPINFO_PRESENT (not sizeof(STARTUPINFOW)).
        let mut si: STARTUPINFOEXW = unsafe { std::mem::zeroed() };
        si.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
        let mut attr_bytes: usize = 0;
        // First call with NULL only sizes the buffer (documented to fail
        // with ERROR_INSUFFICIENT_BUFFER); the second call initializes it in
        // caller-owned storage.
        unsafe { InitializeProcThreadAttributeList(std::ptr::null_mut(), 1, 0, &mut attr_bytes) };
        if attr_bytes == 0 {
            let code = last_error();
            unsafe {
                CloseHandle(input_read);
                CloseHandle(input_write);
                CloseHandle(output_read);
                CloseHandle(output_write);
                ClosePseudoConsole(pc);
            }
            return Err(win_err(
                "InitializeProcThreadAttributeList(size query)",
                code,
            ));
        }
        // The opaque list must outlive CreateProcessW; the underscore
        // binding (not a bare `_`) keeps the owning Vec alive past the call.
        let (_attr_storage, attr_list) = aligned_attr_storage(attr_bytes);
        if unsafe { InitializeProcThreadAttributeList(attr_list, 1, 0, &mut attr_bytes) } == 0 {
            let code = last_error();
            unsafe {
                CloseHandle(input_read);
                CloseHandle(input_write);
                CloseHandle(output_read);
                CloseHandle(output_write);
                ClosePseudoConsole(pc);
            }
            return Err(win_err("InitializeProcThreadAttributeList", code));
        }
        // Canonical MSDN ConPTY shape: for
        // PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE, `lpValue` IS the HPCON value
        // (not the address of the local variable) and `cbSize` is
        // sizeof(HPCON). Passing `&pc` armed the attribute with a stack
        // address instead of the console handle, which CreateProcessW
        // rejects (ERROR_INVALID_PARAMETER, 87).
        let ok = unsafe {
            UpdateProcThreadAttribute(
                attr_list,
                0,
                PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE as usize,
                pc as *const c_void,
                size_of::<HPCON>(),
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        if ok == 0 {
            let code = last_error();
            unsafe {
                DeleteProcThreadAttributeList(attr_list);
                CloseHandle(input_read);
                CloseHandle(input_write);
                CloseHandle(output_read);
                CloseHandle(output_write);
                ClosePseudoConsole(pc);
            }
            return Err(win_err("UpdateProcThreadAttribute(PSEUDOCONSOLE)", code));
        }
        si.lpAttributeList = attr_list;

        // (4) environment: THE identical authority — the resolved EnvSpec
        // (env-clear semantics, deny-set filtered) encoded as the
        // double-NUL block. The block is ALWAYS passed: the child never
        // inherits the daemon environment implicitly.
        let env_entries: Vec<(String, String)> = cfg
            .env
            .resolve()
            .into_iter()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.to_string_lossy().into_owned(),
                )
            })
            .collect();
        let mut env_block: Vec<u16> = win_common::build_env_block(&env_entries);
        // A Unicode environment block MUST end in two NUL UTF-16 units.
        // `resolve()` always emits the GIT_TERMINAL_PROMPT default, but an
        // empty block must still be well formed rather than a one-NUL
        // buffer CreateProcessW would read past.
        if env_block == [0] {
            env_block.push(0);
        }
        let env_ptr: *const c_void = env_block.as_ptr().cast();
        let cwd_wide = cfg.cwd.as_deref().map(to_wide).unwrap_or_default();
        let cwd_ptr: *const u16 = if cfg.cwd.is_some() {
            cwd_wide.as_ptr()
        } else {
            std::ptr::null()
        };

        // (5) spawn with the canonical MSDN ConPTY parameter shape:
        // - lpApplicationName = the resolved absolute module path (no
        //   CreateProcessW module guessing on PATH);
        // - lpCommandLine = a MUTABLE UTF-16 buffer containing
        //   `"<exe>" <quoted args...>`, which the API may rewrite;
        // - process/thread attributes NULL, bInheritHandles FALSE (the
        //   pseudoconsole attribute, not handle inheritance, wires I/O);
        // - flags = EXTENDED_STARTUPINFO_PRESENT (lpStartupInfo is the
        //   STARTUPINFOEXW above) | CREATE_UNICODE_ENVIRONMENT (the env block
        //   is UTF-16). Deliberately NOT CREATE_NEW_CONSOLE/DETACHED_PROCESS
        //   or STARTF_USESTDHANDLES: each conflicts with the pseudoconsole
        //   and arms ERROR_INVALID_PARAMETER (87).
        let app_display = String::from_utf16_lossy(&app_wide[..app_wide.len() - 1]);
        let mut cmdline_wide = to_wide(&win_common::build_command_line(&app_display, &cfg.args));
        let mut pi: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
        let creation_flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT;
        // lpApplicationName must be NUL-terminated: resolve_application
        // returns the logical (unterminated) path, so terminate here.
        let mut app_wide_z = app_wide.clone();
        app_wide_z.push(0);
        let created = unsafe {
            CreateProcessW(
                app_wide_z.as_ptr(),
                cmdline_wide.as_mut_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0, // no automatic handle inheritance
                creation_flags,
                env_ptr,
                cwd_ptr,
                &si.StartupInfo,
                &mut pi,
            )
        };
        unsafe {
            DeleteProcThreadAttributeList(attr_list);
        }
        if created == 0 {
            let code = last_error();
            unsafe {
                CloseHandle(input_read);
                CloseHandle(input_write);
                CloseHandle(output_read);
                CloseHandle(output_write);
                ClosePseudoConsole(pc);
            }
            // The step string carries the exact module so a CI failure is
            // root-causable from the message alone.
            return Err(win_err(&format!("CreateProcessW({app_display})"), code));
        }
        if !valid_handle(pi.hProcess) || !valid_handle(pi.hThread) || pi.dwProcessId == 0 {
            unsafe {
                CloseHandle(pi.hThread);
                CloseHandle(pi.hProcess);
                CloseHandle(input_read);
                CloseHandle(input_write);
                CloseHandle(output_read);
                CloseHandle(output_write);
                ClosePseudoConsole(pc);
            }
            return Err(Error::internal(
                "CreateProcessW returned an invalid process",
            ));
        }
        unsafe {
            CloseHandle(pi.hThread);
        }
        // (6) release the conpty-side pipe ends. The pseudoconsole holds
        // its own references; dropping ours lets I/O detect a broken
        // channel when the session closes (per the ConPTY docs) instead of
        // deadlocking on a full pipe.
        unsafe {
            CloseHandle(input_read);
            CloseHandle(output_write);
        }

        let shared = Arc::new((Mutex::new(Ring::new()), Condvar::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let closed = AtomicBool::new(false);
        let last_size = AtomicU32::new(pack_size(cfg.rows, cfg.cols));

        // (7) reader thread on its OWN duplicate of the output pipe, so
        // Drop closing the original never races an outstanding read.
        let mut reader_dup: HANDLE = std::ptr::null_mut();
        let duplicated = unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                output_read,
                GetCurrentProcess(),
                &mut reader_dup,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        };
        if duplicated == 0 || !valid_handle(reader_dup) {
            let code = last_error();
            unsafe {
                TerminateProcess(pi.hProcess, 1);
                CloseHandle(pi.hProcess);
                CloseHandle(input_write);
                CloseHandle(output_read);
                ClosePseudoConsole(pc);
            }
            return Err(win_err("DuplicateHandle(conpty output)", code));
        }

        let t_shared = shared.clone();
        let t_stop = stop.clone();
        let child_handle = SendHandle(pi.hProcess);
        let dup_copy = SendHandle(reader_dup);
        match std::thread::Builder::new()
            .name("faktor-pty-conpty-reader".to_string())
            .spawn(move || conpty_reader(child_handle, dup_copy, t_shared, t_stop))
        {
            Ok(_handle) => {} // detached: exits on its own when the channel dies
            Err(e) => {
                unsafe {
                    CloseHandle(reader_dup);
                    TerminateProcess(pi.hProcess, 1);
                    CloseHandle(pi.hProcess);
                    CloseHandle(input_write);
                    CloseHandle(output_read);
                    ClosePseudoConsole(pc);
                }
                return Err(Error::internal(format!(
                    "pty reader thread failed to start: {e}"
                )));
            }
        }

        Ok(Self {
            pc,
            input: input_write,
            output: output_read,
            child: pi.hProcess,
            pid: pi.dwProcessId,
            last_size,
            shared,
            reader_stop: stop,
            closed,
        })
    }

    /// The child pid.
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// Write raw bytes to the child's stdin through the ConPTY input pipe.
    /// A closed session returns a typed error, never a panic.
    pub fn write_all(&self, bytes: &[u8]) -> Result<(), Error> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(closed_error());
        }
        let mut written = 0usize;
        while written < bytes.len() {
            // A dead child will never drain the ConPTY input queue; bail
            // out with a typed error instead of blocking forever.
            if !self.is_alive() {
                self.closed.store(true, Ordering::SeqCst);
                return Err(Error::internal("pty child exited; write aborted"));
            }
            let chunk = (bytes.len() - written).min(u32::MAX as usize);
            let mut n: u32 = 0;
            let ok = unsafe {
                WriteFile(
                    self.input,
                    bytes[written..written + chunk].as_ptr(),
                    chunk as u32,
                    &mut n,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                let code = last_error();
                if win_common::channel_closed(code) {
                    self.closed.store(true, Ordering::SeqCst);
                    return Err(closed_error());
                }
                return Err(Error::internal(format!(
                    "pty write failed (win32 error {code})"
                )));
            }
            if n == 0 {
                return Err(Error::internal("pty write made no progress"));
            }
            written += n as usize;
        }
        Ok(())
    }

    /// Write a line (the pseudoconsole's line discipline handles CR/echo).
    pub fn write_line(&self, line: &str) -> Result<(), Error> {
        let mut b = line.as_bytes().to_vec();
        b.push(b'\n');
        self.write_all(&b)
    }

    /// Resize the pseudoconsole surface. Zero and >32767 are rejected
    /// before any call (COORD is i16).
    pub fn resize(&self, rows: u16, cols: u16) -> Result<(), Error> {
        win_common::validate_geometry(rows, cols)?;
        if self.pc == 0 {
            return Err(closed_error());
        }
        let coord = COORD {
            X: cols as i16,
            Y: rows as i16,
        };
        let hr = unsafe { ResizePseudoConsole(self.pc, coord) };
        if hr < 0 {
            return Err(hresult_err("ResizePseudoConsole", hr as u32));
        }
        self.last_size
            .store(pack_size(rows, cols), Ordering::SeqCst);
        Ok(())
    }

    /// Last size applied through ResizePseudoConsole. ConPTY exposes no
    /// size query, so this is the honest best answer (unlike the unix
    /// backend's kernel TIOCGWINSZ).
    pub fn size(&self) -> (u16, u16) {
        unpack_size(self.last_size.load(Ordering::SeqCst))
    }

    /// Drain all currently available output.
    pub fn read_available(&self) -> Vec<u8> {
        self.shared.0.lock().unwrap().drain()
    }

    /// Snapshot the current output WITHOUT draining.
    pub fn snapshot(&self) -> Vec<u8> {
        self.shared.0.lock().unwrap().snapshot()
    }

    /// Total bytes ever read from the ConPTY output pipe.
    pub fn total_bytes(&self) -> u64 {
        self.shared.0.lock().unwrap().total()
    }

    /// Block until `needle` appears in the accumulated output or `timeout`
    /// elapses (test/consumer helper).
    pub fn wait_for_contains(&self, needle: &str, timeout: std::time::Duration) -> bool {
        let (ring, cv) = &*self.shared;
        let mut guard = ring.lock().unwrap();
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let snap = guard.snapshot();
            let text = String::from_utf8_lossy(&snap);
            if text.contains(needle) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            let (g, _t) = cv
                .wait_timeout(guard, std::time::Duration::from_millis(50))
                .unwrap();
            guard = g;
        }
    }

    /// Is the child process still running?
    pub fn is_alive(&self) -> bool {
        !self.child.is_null() && unsafe { WaitForSingleObject(self.child, 0) } == WAIT_TIMEOUT
    }

    /// Terminate the child: ClosePseudoConsole (the attached client is
    /// terminated by the OS when the session closes), then a bounded
    /// TerminateProcess fallback if it does not exit within the grace
    /// period. Idempotent; Drop runs the same path.
    pub fn kill(&mut self) {
        self.closed.store(true, Ordering::SeqCst);
        self.reader_stop.store(true, Ordering::SeqCst);
        self.close_pc();
        if self.child.is_null() {
            return;
        }
        if unsafe { WaitForSingleObject(self.child, KILL_GRACE_MS) } == WAIT_TIMEOUT {
            unsafe {
                TerminateProcess(self.child, 1);
            }
            // Bounded: never hang a caller (or Drop) on a stuck process.
            unsafe {
                WaitForSingleObject(self.child, TERMINATE_WAIT_MS);
            }
        }
    }

    fn close_pc(&mut self) {
        if self.pc != 0 {
            unsafe {
                ClosePseudoConsole(self.pc);
            }
            self.pc = 0;
        }
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        self.kill();
        // The reader thread works on its own duplicate and exits on its own
        // once the pipe breaks; only the original handles are closed here.
        if !self.input.is_null() {
            unsafe {
                CloseHandle(self.input);
            }
            self.input = std::ptr::null_mut();
        }
        if !self.output.is_null() {
            unsafe {
                CloseHandle(self.output);
            }
            self.output = std::ptr::null_mut();
        }
        if !self.child.is_null() {
            unsafe {
                CloseHandle(self.child);
            }
            self.child = std::ptr::null_mut();
        }
    }
}

fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Reader thread: drains the ConPTY output pipe into the bounded ring.
/// Never blocks in ReadFile — it peeks for available bytes and polls the
/// child's process handle, so it terminates on its own when the session
/// ends, the child exits, or `stop` is set.
fn conpty_reader(
    child: SendHandle,
    output_dup: SendHandle,
    shared: Arc<(Mutex<Ring>, Condvar)>,
    stop: Arc<AtomicBool>,
) {
    let child = child.0;
    let output_dup = output_dup.0;
    let mut buf = [0u8; 8192];
    let mut child_exited = false;
    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        if !child_exited && unsafe { WaitForSingleObject(child, 0) } == WAIT_OBJECT_0 {
            child_exited = true;
        }
        let mut available: u32 = 0;
        let peeked = unsafe {
            PeekNamedPipe(
                output_dup,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                &mut available,
                std::ptr::null_mut(),
            )
        };
        if peeked == 0 {
            let code = last_error();
            if win_common::channel_closed(code) {
                break;
            }
            std::thread::sleep(READER_IDLE);
            continue;
        }
        if available == 0 {
            if child_exited {
                // Drained everything the session emitted; the child is
                // gone and no new output can arrive.
                break;
            }
            std::thread::sleep(READER_IDLE);
            continue;
        }
        let mut n: u32 = 0;
        let read_ok = unsafe {
            ReadFile(
                output_dup,
                buf.as_mut_ptr(),
                available.min(buf.len() as u32),
                &mut n,
                std::ptr::null_mut(),
            )
        };
        if read_ok == 0 {
            let code = last_error();
            if win_common::channel_closed(code) {
                break;
            }
            std::thread::sleep(READER_IDLE);
            continue;
        }
        if n == 0 {
            break; // EOF: the pseudoconsole closed its write end
        }
        let (ring, cv) = &*shared;
        ring.lock().unwrap().push(&buf[..n as usize]);
        cv.notify_all();
    }
    unsafe {
        CloseHandle(output_dup);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::error::ErrorKind;

    // Windows-only tests: they compile and RUN on a Windows host; on unix
    // they simply do not exist (cfg(windows), not #[ignore]).

    #[test]
    fn spawn_rejects_invalid_configs_before_any_win32_call() {
        let err = Pty::spawn(&PtyConfig::default()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
        let mut cfg = PtyConfig {
            command: "cmd.exe".into(),
            ..Default::default()
        };
        cfg.command = "cmd.exe\0owned".into();
        let err = Pty::spawn(&cfg).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
    }

    #[test]
    fn spawn_rejects_geometry_conpty_cannot_honor() {
        let mut cfg = PtyConfig {
            command: "cmd.exe".into(),
            ..Default::default()
        };
        cfg.rows = 0;
        assert_eq!(Pty::spawn(&cfg).unwrap_err().kind, ErrorKind::Malformed);
        cfg.rows = 24;
        cfg.cols = 40_000; // > i16::MAX: would silently truncate in COORD
        assert_eq!(Pty::spawn(&cfg).unwrap_err().kind, ErrorKind::Oversized);
    }

    #[test]
    fn pre_spawn_validation_rejects_blank_program_before_any_win32_call() {
        // Empty program: the shared validator rejects it.
        assert_eq!(
            Pty::spawn(&PtyConfig::default()).unwrap_err().kind,
            ErrorKind::Malformed
        );
        // Whitespace-only program: rejected by the Windows resolver BEFORE
        // SearchPathW/CreatePipe, so no Win32 resource exists to leak.
        let mut cfg = PtyConfig {
            command: "cmd.exe".into(),
            ..Default::default()
        };
        cfg.command = " \t\n ".into();
        let err = Pty::spawn(&cfg).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
        for blank in ["", " ", "\t", "  \r\n "] {
            assert_eq!(
                validate_program(blank).unwrap_err().kind,
                ErrorKind::Malformed,
                "blank program {blank:?} must be Malformed"
            );
        }
    }

    #[test]
    fn pre_spawn_validation_accepts_empty_args_and_empty_argument() {
        // Empty argv is legal: only the program is required.
        let cfg = PtyConfig {
            command: "cmd.exe".into(),
            args: Vec::new(),
            ..Default::default()
        };
        assert!(validate_program(&cfg.command).is_ok());
        assert!(validate_spawn_config(&cfg).is_ok());
        // An empty argument is a legal argv element (argv[i] == "").
        let cfg = PtyConfig {
            command: "cmd.exe".into(),
            args: vec![String::new()],
            ..Default::default()
        };
        assert!(validate_spawn_config(&cfg).is_ok());
    }

    #[test]
    fn bare_program_resolves_to_a_module_path_and_explicit_paths_pass_through() {
        // `cmd.exe` lives in the system directory and on PATH: the resolved
        // path is absolute and handed to CreateProcessW as lpApplicationName.
        let resolved =
            resolve_application("cmd.exe").expect("cmd.exe must resolve via PATH/system dir");
        let text = String::from_utf16(&resolved).expect("resolved path is valid UTF-16");
        assert!(text.to_ascii_lowercase().ends_with("cmd.exe"), "{text}");
        assert!(
            text.contains(':') && text.contains('\\'),
            "not absolute: {text}"
        );

        // Explicit paths are not PATH-searched: Windows owns the image error.
        let explicit = resolve_application(r"C:\Windows\System32\cmd.exe").unwrap();
        assert_eq!(explicit, to_wide(r"C:\Windows\System32\cmd.exe"));

        // Resolution always exposes a NUL-terminated buffer.
        assert_eq!(*resolved.last().unwrap(), 0);
    }

    #[test]
    fn size_round_trips_through_the_stored_last_size() {
        let pty = Pty {
            pc: 0,
            input: std::ptr::null_mut(),
            output: std::ptr::null_mut(),
            child: std::ptr::null_mut(),
            pid: 0,
            last_size: AtomicU32::new(pack_size(24, 80)),
            shared: Arc::new((Mutex::new(Ring::new()), Condvar::new())),
            reader_stop: Arc::new(AtomicBool::new(false)),
            closed: AtomicBool::new(false),
        };
        assert_eq!(pty.size(), (24, 80));
        pty.resize(30, 100).unwrap_err(); // pc == 0 -> typed closed error
        assert_eq!(pty.size(), (24, 80), "failed resize must not lie");
    }

    #[test]
    fn closed_session_writes_are_typed_errors_not_panics() {
        let pty = Pty {
            pc: 0,
            input: std::ptr::null_mut(),
            output: std::ptr::null_mut(),
            child: std::ptr::null_mut(),
            pid: 0,
            last_size: AtomicU32::new(pack_size(24, 80)),
            shared: Arc::new((Mutex::new(Ring::new()), Condvar::new())),
            reader_stop: Arc::new(AtomicBool::new(false)),
            closed: AtomicBool::new(true),
        };
        let err = pty.write_all(b"hello").unwrap_err();
        assert_eq!(err.kind, ErrorKind::Internal);
        let err = pty.write_line("hello").unwrap_err();
        assert_eq!(err.kind, ErrorKind::Internal);
        assert!(!pty.is_alive());
        assert_eq!(pty.read_available(), Vec::<u8>::new());
    }

    #[test]
    fn kill_without_a_child_is_idempotent() {
        let mut pty = Pty {
            pc: 0,
            input: std::ptr::null_mut(),
            output: std::ptr::null_mut(),
            child: std::ptr::null_mut(),
            pid: 0,
            last_size: AtomicU32::new(pack_size(24, 80)),
            shared: Arc::new((Mutex::new(Ring::new()), Condvar::new())),
            reader_stop: Arc::new(AtomicBool::new(false)),
            closed: AtomicBool::new(false),
        };
        pty.kill();
        pty.kill(); // double kill must not panic or double-close
    }
}
