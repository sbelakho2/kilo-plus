//! Adversarial tests of the shadow mutation-root service (P0-48).
//!
//! The tests break the invariants the feature exists for: user-checkout
//! isolation until a clean commit, conflict-aware integration (external
//! drift during the drive), crash replay of a partially applied commit,
//! daemon-shutdown removal, symlink escapes, oversize refusals, `.git`
//! plumbing never copied, and deterministic reopen recovery. The "shadowed
//! drive writes" of the end-to-end semantics are staged as DIRECT writes
//! into the shadow root — the exact operation a shadow-aware tool context
//! performs once the next wave re-points the session file consumers (the
//! wiring tests in `task_executor_tests.rs` drive the real executor).

use std::fs;
use std::path::{Path, PathBuf};

use faktor_core::id::SessionId;
use faktor_session::{SessionManager, ShadowRowState};

use super::*;
use crate::runtime::shadow::{
    ShadowCopyLimits, ShadowRoots, SHADOW_MAX_BASE_ENTRIES, SHADOW_MAX_COPY_BYTES,
};

struct Fix {
    _dir: tempfile::TempDir,
    manager: Arc<SessionManager>,
    shadows: Arc<ShadowRoots>,
    user: PathBuf,
    session: SessionId,
}

fn seed_user(root: &Path) {
    fs::create_dir_all(root.join("sub")).unwrap();
    fs::write(root.join("a.txt"), b"alpha").unwrap();
    fs::write(root.join("sub/b.txt"), b"beta").unwrap();
    fs::write(root.join("c.txt"), b"gamma").unwrap();
}

fn open_fix(limits: ShadowCopyLimits) -> Fix {
    let dir = tempfile::tempdir().unwrap();
    let user = dir.path().join("user");
    fs::create_dir_all(&user).unwrap();
    seed_user(&user);
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = manager.create_workspace(user.to_str().unwrap()).unwrap();
    let session = manager
        .create_session(ws, "shadow-service", "fake", "m")
        .unwrap()
        .id();
    let shadows = ShadowRoots::new_with_limits(manager.clone(), dir.path().join("shadows"), limits);
    Fix {
        _dir: dir,
        manager,
        shadows,
        user,
        session,
    }
}

fn default_limits() -> ShadowCopyLimits {
    ShadowCopyLimits {
        max_entries: SHADOW_MAX_BASE_ENTRIES,
        max_total_bytes: SHADOW_MAX_COPY_BYTES,
    }
}

fn user_bytes(fix: &Fix, rel: &str) -> Vec<u8> {
    fs::read(fix.user.join(rel)).unwrap()
}

fn shadow_row_of(fix: &Fix) -> ShadowRow {
    fix.manager
        .shadow_row(fix.session)
        .unwrap()
        .expect("a shadow row exists")
}

/// The "shadowed drive" write: stage `bytes` at `rel` inside the shadow
/// root (the next-wave consumers resolve this root via
/// `SessionManager::active_root`).
fn drive_write(fix: &Fix, rel: &str, bytes: &[u8]) {
    let row = shadow_row_of(fix);
    let dst = PathBuf::from(&row.root).join(rel);
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(dst, bytes).unwrap();
}

fn assert_no_shadow(fix: &Fix) {
    assert!(fix.manager.shadow_row(fix.session).unwrap().is_none());
}

fn arm_seam(fix: &Fix, after: usize) {
    fix.shadows.arm_apply_seam(after);
}

// ---------------------------------------------------------------- isolation

#[test]
fn user_checkout_untouched_until_commit_then_integrated() {
    // (a)+(b) at service level: while the shadow is live, staged writes
    // never leak into the user checkout; a clean auto-approve commit lands
    // the content, removes the shadow and retires the durable row.
    let fix = open_fix(default_limits());
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    let row = shadow_row_of(&fix);
    assert_eq!(row.state, ShadowRowState::Active);
    let dir = PathBuf::from(&row.root);
    assert!(dir.is_dir());
    // Mid-drive: the shadow holds the new world, the user checkout still
    // holds the base bytes.
    drive_write(&fix, "a.txt", b"agent alpha v2");
    drive_write(&fix, "new.txt", b"agent new file");
    assert_eq!(user_bytes(&fix, "a.txt"), b"alpha");
    assert!(!fix.user.join("new.txt").exists());
    // The staged candidate presents exactly the two changed files.
    let cs = fix.shadows.present_change_set(fix.session).unwrap();
    let paths: Vec<String> = cs
        .files
        .iter()
        .map(|f| f.path.to_string_lossy().into_owned())
        .collect();
    assert_eq!(paths, vec!["a.txt".to_string(), "new.txt".to_string()]);
    let (approved, rejected) = super::auto_approve(&cs);
    assert_eq!(approved.len(), 2);
    assert!(rejected.is_empty());
    // Clean commit: content lands, shadow removed, row retired.
    let out = fix
        .shadows
        .commit_back(fix.session, &approved, &rejected)
        .unwrap();
    assert!(out.clean());
    assert_eq!(out.merged.len(), 2);
    assert_eq!(user_bytes(&fix, "a.txt"), b"agent alpha v2");
    assert_eq!(user_bytes(&fix, "new.txt"), b"agent new file");
    assert!(
        !dir.exists(),
        "shadow dir removed after a clean integration"
    );
    let row = shadow_row_of(&fix);
    assert_eq!(row.state, ShadowRowState::Integrated);
    // A later run on the same session begins a FRESH shadow (retired rows
    // never block).
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    let row2 = shadow_row_of(&fix);
    assert_eq!(row2.state, ShadowRowState::Active);
    assert_ne!(row2.shadow_id, row.shadow_id, "generations are distinct");
}

#[test]
fn no_op_drive_integrates_nothing_and_discards() {
    // An empty diff (the drive changed nothing) is a clean no-op: the user
    // checkout is byte-identical and the shadow is retired.
    let fix = open_fix(default_limits());
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    let dir = PathBuf::from(&shadow_row_of(&fix).root);
    let cs = fix.shadows.present_change_set(fix.session).unwrap();
    assert!(cs.files.is_empty(), "no writes -> no change set");
    let out = fix.shadows.commit_all(fix.session).unwrap();
    assert!(out.clean());
    assert!(out.merged.is_empty());
    assert_eq!(user_bytes(&fix, "a.txt"), b"alpha");
    assert!(!dir.exists());
    assert_eq!(shadow_row_of(&fix).state, ShadowRowState::Integrated);
}

#[test]
fn deletions_commit_cas_and_retain_user_drift() {
    // A shadow that deleted a base file commits the deletion through the
    // CAS anchor; a user file that moved on meanwhile is a conflict and is
    // never removed.
    let fix = open_fix(default_limits());
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    // Shadowed drive deletes sub/b.txt and adds d.txt.
    let dir = PathBuf::from(&shadow_row_of(&fix).root);
    fs::remove_file(dir.join("sub/b.txt")).unwrap();
    drive_write(&fix, "d.txt", b"deleted b, added d");
    // External drift on c.txt (untouched by the shadow) must NOT surface:
    // only changed files are decided.
    let out = fix.shadows.commit_all(fix.session).unwrap();
    assert!(out.clean());
    assert!(!fix.user.join("sub/b.txt").exists(), "deletion landed");
    assert_eq!(user_bytes(&fix, "d.txt"), b"deleted b, added d");
    assert_eq!(user_bytes(&fix, "a.txt"), b"alpha", "untouched file intact");
    // Second run: the drive deletes a.txt again, but the user changed it
    // meanwhile -> per-file conflict, deletion refused.
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    let dir = PathBuf::from(&shadow_row_of(&fix).root);
    fs::remove_file(dir.join("a.txt")).unwrap();
    fs::write(fix.user.join("a.txt"), b"user drift on a").unwrap();
    let out = fix.shadows.commit_all(fix.session).unwrap();
    assert!(!out.clean(), "{out:?}");
    assert_eq!(out.conflicts.len(), 1);
    assert!(
        out.conflicts[0].1.contains("changed since the base"),
        "{out:?}"
    );
    assert_eq!(
        user_bytes(&fix, "a.txt"),
        b"user drift on a",
        "a drifted user file is never removed"
    );
    assert_eq!(
        shadow_row_of(&fix).state,
        ShadowRowState::IntegrationBlocked
    );
}

#[test]
fn external_user_drift_conflicts_then_resolves() {
    // (c): the user edits a file while the drive is staged; the integration
    // conflicts, the user checkout is untouched, the shadow is RETAINED
    // with the conflict list durably recorded, and a second attempt after
    // the user reverts integrates cleanly.
    let fix = open_fix(default_limits());
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    let shadow_dir = PathBuf::from(&shadow_row_of(&fix).root);
    drive_write(&fix, "a.txt", b"agent new alpha");
    fs::write(fix.user.join("a.txt"), b"user edit while drive").unwrap();
    let out = fix.shadows.commit_all(fix.session).unwrap();
    assert!(!out.clean());
    assert_eq!(out.conflicts.len(), 1);
    assert!(out.conflicts[0]
        .1
        .contains("changed since the base snapshot"));
    assert_eq!(
        user_bytes(&fix, "a.txt"),
        b"user edit while drive",
        "a conflicted user file is never overwritten"
    );
    assert!(shadow_dir.is_dir(), "shadow retained on conflict");
    let row = shadow_row_of(&fix);
    assert_eq!(row.state, ShadowRowState::IntegrationBlocked);
    // A second commit while the drift persists surfaces the SAME conflict.
    let again = fix.shadows.commit_all(fix.session).unwrap();
    assert_eq!(again.conflicts.len(), 1);
    assert_eq!(
        user_bytes(&fix, "a.txt"),
        b"user edit while drive",
        "still untouched"
    );
    // The user resolves the drift (reverts to the base content); the same
    // auto decision now integrates.
    fs::write(fix.user.join("a.txt"), b"alpha").unwrap();
    let out = fix.shadows.commit_all(fix.session).unwrap();
    assert!(out.clean(), "{out:?}");
    assert_eq!(user_bytes(&fix, "a.txt"), b"agent new alpha");
    assert!(!shadow_dir.exists());
    assert_eq!(shadow_row_of(&fix).state, ShadowRowState::Integrated);
}

#[test]
fn exclusive_create_conflicts_with_user_creation() {
    // A file the shadow created can never overwrite a user file created
    // during the drive (exclusive-create semantics).
    let fix = open_fix(default_limits());
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    drive_write(&fix, "e.txt", b"shadow-owned");
    fs::write(fix.user.join("e.txt"), b"user-owned, created during drive").unwrap();
    let out = fix.shadows.commit_all(fix.session).unwrap();
    assert!(!out.clean());
    assert_eq!(out.conflicts.len(), 1);
    assert!(
        out.conflicts[0]
            .1
            .contains("exists although the base snapshot had no such file"),
        "{out:?}"
    );
    assert_eq!(
        user_bytes(&fix, "e.txt"),
        b"user-owned, created during drive"
    );
}

#[test]
fn hostile_shadow_paths_and_reuse_refused() {
    // begin_shadow refuses: a base root inside the shadow root, a session
    // with a live shadow, and unknown sessions.
    let fix = open_fix(default_limits());
    let inner = fix._dir.path().join("shadows").join("sneaky");
    fs::create_dir_all(&inner).unwrap();
    let err = fix
        .shadows
        .begin_shadow(fix.session, &inner)
        .expect_err("a shadow can never shadow a shadow");
    assert!(
        err.to_string().contains("inside the daemon shadow root"),
        "{err}"
    );
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    let err = fix
        .shadows
        .begin_shadow(fix.session, &fix.user)
        .expect_err("a live shadow refuses a second begin");
    assert!(err.to_string().contains("live shadow"), "{err}");
    let err = fix
        .shadows
        .begin_shadow(SessionId::new(424_242), &fix.user)
        .expect_err("unknown sessions refuse");
    assert!(
        matches!(err, crate::runtime::ExecError::NotFound(_)),
        "{err}"
    );
    // stage/commit/discard on a shadow-less session are typed refusals.
    assert!(fix
        .shadows
        .stage_change_set(SessionId::new(424_242))
        .is_err());
}

// ---------------------------------------------------------- hostile inputs

#[test]
fn symlink_escape_from_shadow_copy_rejected() {
    // (f): a checkout whose symlink leaves the tree must refuse the shadow
    // begin loudly — nothing is copied, no row is written.
    let dir = tempfile::tempdir().unwrap();
    let outside = dir.path().join("outside");
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join("secret.txt"), b"outside secret").unwrap();
    let user = dir.path().join("user");
    fs::create_dir_all(&user).unwrap();
    fs::write(user.join("a.txt"), b"alpha").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(outside.join("secret.txt"), user.join("leak.txt")).unwrap();
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = manager.create_workspace(user.to_str().unwrap()).unwrap();
    let session = manager
        .create_session(ws, "escape", "fake", "m")
        .unwrap()
        .id();
    let shadows = ShadowRoots::new_with_limits(
        manager.clone(),
        dir.path().join("shadows"),
        default_limits(),
    );
    let err = shadows
        .begin_shadow(session, &user)
        .expect_err("escape refused");
    assert!(
        err.to_string().contains("symlink") || err.to_string().contains("escape"),
        "{err}"
    );
    assert!(manager.shadow_row(session).unwrap().is_none());
    // Nothing of the copy survives: the daemon-owned shadow area is empty
    // of shadow directories (the session-level dir may exist empty).
    let shadows_area = dir.path().join("shadows");
    if shadows_area.exists() {
        for session_dir in fs::read_dir(&shadows_area).unwrap().flatten() {
            let leftovers: Vec<_> = fs::read_dir(session_dir.path())
                .unwrap()
                .flatten()
                .collect();
            assert!(leftovers.is_empty(), "{:?}", leftovers);
        }
    }
}

#[test]
fn oversize_base_refused_before_any_mutation() {
    // (g): a base tree beyond the copy caps is a typed Oversized refusal;
    // nothing is copied, no row exists, the user checkout is untouched.
    let fix = open_fix(ShadowCopyLimits {
        max_entries: 2, // the seed has three files
        max_total_bytes: SHADOW_MAX_COPY_BYTES,
    });
    let err = fix
        .shadows
        .begin_shadow(fix.session, &fix.user)
        .expect_err("entry cap exceeded");
    assert!(
        matches!(err, crate::runtime::ExecError::Oversized(_)),
        "{err}"
    );
    assert_no_shadow(&fix);
    assert_eq!(user_bytes(&fix, "a.txt"), b"alpha");
    // Byte cap too.
    let fix2 = open_fix(ShadowCopyLimits {
        max_entries: SHADOW_MAX_BASE_ENTRIES,
        max_total_bytes: 4, // every seed file is 5 bytes
    });
    let err = fix2
        .shadows
        .begin_shadow(fix2.session, &fix2.user)
        .expect_err("byte cap exceeded");
    assert!(
        matches!(err, crate::runtime::ExecError::Oversized(_)),
        "{err}"
    );
    assert_no_shadow(&fix2);
}

#[test]
#[should_panic(expected = "shadow copy caps must be >= 1")]
fn zero_caps_refused_at_construction() {
    let dir = tempfile::tempdir().unwrap();
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let _ = ShadowRoots::new_with_limits(
        manager,
        dir.path().join("s"),
        ShadowCopyLimits {
            max_entries: 0,
            max_total_bytes: 1,
        },
    );
}

#[test]
fn git_plumbing_never_copied_git_or_plain_root() {
    // (h): no git API materializes an at-revision tree outside the repo
    // (faktor-git's only creation path adds a worktree INSIDE the user
    // checkout — forbidden), so every root uses the bounded fs copy; the
    // `.git` plumbing of a git root is detected and skipped by the copy.
    let fix = open_fix(default_limits());
    fs::create_dir_all(fix.user.join(".git/objects/aa")).unwrap();
    fs::write(fix.user.join(".git/HEAD"), b"ref: refs/heads/main").unwrap();
    fs::write(fix.user.join(".git/objects/aa/bb"), vec![0x7f; 1024 * 1024]).unwrap();
    let shadow = fix
        .shadows
        .begin_shadow(fix.session, &fix.user)
        .expect("the copy skips .git, so the byte cap is not hit");
    let row = shadow_row_of(&fix);
    assert_eq!(row.base_entries, 3, "only the three content files copied");
    assert!(
        !shadow.root.join(".git").exists(),
        "no plumbing in the shadow"
    );
    // The copied content is intact and committable.
    assert!(fix.user.join(".git/HEAD").exists(), "user .git untouched");
    drive_write(&fix, "a.txt", b"alpha via git-root shadow");
    // Staging after the write yields only content entries (never .git).
    let manifest = fix.shadows.present_change_set(fix.session).unwrap();
    assert!(
        manifest.files.iter().all(|f| !f.path.starts_with(".git")),
        "no .git entry can ever be staged: {:?}",
        manifest.files
    );
    let out = fix.shadows.commit_all(fix.session).unwrap();
    assert!(out.clean());
    assert_eq!(user_bytes(&fix, "a.txt"), b"alpha via git-root shadow");
    assert!(fix.user.join(".git/HEAD").exists());
    // A non-git root (no .git at all) uses the same fs-copy path.
    let fix2 = open_fix(default_limits());
    fix2.shadows.begin_shadow(fix2.session, &fix2.user).unwrap();
    assert!(PathBuf::from(&shadow_row_of(&fix2).root).is_dir());
}

// ------------------------------------------------------- crash and teardown

#[test]
fn crash_mid_apply_replays_identically() {
    // A crash inside the apply loop (after 1 of 2 files) leaves the
    // in-flight durable envelope; the identical auto decision replays each
    // apply CAS-idempotently (AlreadyCurrent) and finalizes Applied.
    let fix = open_fix(default_limits());
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    drive_write(&fix, "a.txt", b"first change");
    drive_write(&fix, "b.txt", b"second change"); // b.txt under sub/? no —
    arm_seam(&fix, 1);
    let err = fix
        .shadows
        .commit_all(fix.session)
        .expect_err("seam fires after one apply");
    assert!(matches!(
        err,
        crate::runtime::ExecError::InjectedCrashSeam(_)
    ));
    // Deterministic residue: a.txt already applied (its CAS merge is
    // idempotent on replay), b.txt pending; the row is still live.
    assert_eq!(user_bytes(&fix, "a.txt"), b"first change");
    let row = shadow_row_of(&fix);
    assert_eq!(row.state, ShadowRowState::Active);
    let out = fix.shadows.commit_all(fix.session).unwrap();
    assert!(out.clean(), "{out:?}");
    assert_eq!(out.merged.len(), 2, "replay merges both files");
    assert_eq!(user_bytes(&fix, "b.txt"), b"second change");
    assert_eq!(shadow_row_of(&fix).state, ShadowRowState::Integrated);
}

#[test]
fn discard_removes_dir_marks_row_and_retires_cleanly() {
    let fix = open_fix(default_limits());
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    let dir = PathBuf::from(&shadow_row_of(&fix).root);
    fix.shadows.discard(fix.session).unwrap();
    assert!(!dir.exists());
    assert_eq!(shadow_row_of(&fix).state, ShadowRowState::Discarded);
    assert_eq!(user_bytes(&fix, "a.txt"), b"alpha", "discard never writes");
    // stage/commit after discard are typed refusals (the tombstone row
    // exists; only live shadows may stage).
    let err = fix.shadows.stage_change_set(fix.session).unwrap_err();
    assert!(
        err.to_string().contains("only Active/IntegrationBlocked"),
        "{err}"
    );
    // Double discard is idempotent on the directory; the tombstone stays.
    fix.shadows.discard(fix.session).unwrap();
    assert_eq!(shadow_row_of(&fix).state, ShadowRowState::Discarded);
    // A shadow-less session has nothing to discard.
    let err = fix
        .shadows
        .discard(SessionId::new(999_999))
        .expect_err("unknown session");
    assert!(
        matches!(err, crate::runtime::ExecError::NotFound(_)),
        "{err}"
    );
}

#[test]
fn drop_removes_every_live_shadow() {
    // (e): daemon shutdown (the service's Drop) removes every shadow dir
    // and retires the durable rows; the user checkouts are untouched.
    let dir = tempfile::tempdir().unwrap();
    let manager =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let user_a = dir.path().join("user-a");
    let user_b = dir.path().join("user-b");
    for u in [&user_a, &user_b] {
        fs::create_dir_all(u).unwrap();
        fs::write(u.join("a.txt"), b"alpha").unwrap();
    }
    let shadows = ShadowRoots::new(manager.clone(), dir.path().join("shadows"));
    let ws_a = manager.create_workspace(user_a.to_str().unwrap()).unwrap();
    let ws_b = manager.create_workspace(user_b.to_str().unwrap()).unwrap();
    let s_a = manager.create_session(ws_a, "a", "fake", "m").unwrap().id();
    let s_b = manager.create_session(ws_b, "b", "fake", "m").unwrap().id();
    let row_a = manager.shadow_row(s_a).unwrap();
    assert!(row_a.is_none());
    shadows.begin_shadow(s_a, &user_a).unwrap();
    shadows.begin_shadow(s_b, &user_b).unwrap();
    let dir_a = PathBuf::from(&manager.shadow_row(s_a).unwrap().unwrap().root);
    let dir_b = PathBuf::from(&manager.shadow_row(s_b).unwrap().unwrap().root);
    assert!(dir_a.is_dir() && dir_b.is_dir());
    drop(shadows);
    assert!(
        !dir_a.exists() && !dir_b.exists(),
        "Drop removed both shadows"
    );
    let manager2 =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    assert_eq!(
        manager2.shadow_row(s_a).unwrap().unwrap().state,
        ShadowRowState::Discarded,
        "rows retired durably"
    );
    assert_eq!(
        manager2.shadow_row(s_b).unwrap().unwrap().state,
        ShadowRowState::Discarded
    );
    assert_eq!(fs::read(user_a.join("a.txt")).unwrap(), b"alpha");
}

#[test]
fn reconcile_deterministically_settles_crash_residue() {
    let fix = open_fix(default_limits());
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    // Crash residue 1: the shadow dir vanished without a discard.
    let dir = PathBuf::from(&shadow_row_of(&fix).root);
    fs::remove_dir_all(&dir).unwrap();
    let actions = fix.shadows.reconcile().unwrap();
    assert!(
        actions.iter().any(|a| a.contains("directory gone")),
        "{actions:?}"
    );
    assert_eq!(shadow_row_of(&fix).state, ShadowRowState::Discarded);
    // Crash residue 2: a row-less shadow dir (crash between the copy and
    // the row write) is removed.
    fix.shadows.begin_shadow(fix.session, &fix.user).unwrap();
    let session_dir = dir.parent().unwrap();
    let stray = session_dir.join("sh-00000000deadbeef");
    fs::create_dir_all(&stray).unwrap();
    fs::write(stray.join("partial"), b"x").unwrap();
    let actions = fix.shadows.reconcile().unwrap();
    assert!(
        actions.iter().any(|a| a.contains("row-less")),
        "{actions:?}"
    );
    assert!(!stray.exists());
    // A live shadow that survived reconcile stays live (its dir is the
    // CURRENT row's dir — a fresh generation was begun above).
    let row = shadow_row_of(&fix);
    assert_eq!(row.state, ShadowRowState::Active);
    assert!(PathBuf::from(&row.root).is_dir());
}

#[test]
fn shadow_survives_reopen_and_commits_from_durable_rows() {
    // (d): a crashed daemon's shadow (row + dir + base manifest + staged
    // change set are all durable) is fully resolvable after a manager
    // reopen, and commit_back works from the durable rows alone.
    let dir = tempfile::tempdir().unwrap();
    let user = dir.path().join("user");
    fs::create_dir_all(&user).unwrap();
    fs::write(user.join("a.txt"), b"alpha").unwrap();
    let (session, shadow_id, shadow_dir) = {
        let manager =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let ws = manager.create_workspace(user.to_str().unwrap()).unwrap();
        let session = manager
            .create_session(ws, "reopen", "fake", "m")
            .unwrap()
            .id();
        let shadows = ShadowRoots::new(manager.clone(), dir.path().join("shadows"));
        let shadow = shadows.begin_shadow(session, &user).unwrap();
        // The shadowed drive wrote before the crash...
        fs::write(shadow.root.join("a.txt"), b"agent post-crash state").unwrap();
        let shadow_dir = shadow.root.clone();
        // Simulate a CRASH (no Drop): the service is leaked, exactly like a
        // killed daemon — the durable row + dir survive for reopen.
        std::mem::forget(shadows);
        (session, shadow.shadow_id.clone(), shadow_dir)
    };
    // The daemon restarts: reopen reads the durable shadow row + dir.
    let manager2 =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let shadows2 = ShadowRoots::new(manager2.clone(), dir.path().join("shadows"));
    let row = manager2.shadow_row(session).unwrap().expect("row survives");
    assert_eq!(row.shadow_id, shadow_id);
    assert_eq!(row.state, ShadowRowState::Active);
    assert!(shadow_dir.is_dir(), "shadow dir survives the crash");
    assert_eq!(
        fs::read(shadow_dir.join("a.txt")).unwrap(),
        b"agent post-crash state"
    );
    // Deterministic continuation: the staged change set + commit work off
    // the durable base manifest and the surviving shadow tree.
    let cs = shadows2.present_change_set(session).unwrap();
    assert_eq!(cs.files.len(), 1);
    assert_eq!(user_bytes_reopen(&user), b"alpha");
    let out = shadows2.commit_all(session).unwrap();
    assert!(out.clean());
    assert_eq!(
        fs::read(user.join("a.txt")).unwrap(),
        b"agent post-crash state"
    );
    assert!(!shadow_dir.exists());
    assert_eq!(
        manager2.shadow_row(session).unwrap().unwrap().state,
        ShadowRowState::Integrated
    );
}

fn user_bytes_reopen(user: &Path) -> Vec<u8> {
    fs::read(user.join("a.txt")).unwrap()
}
