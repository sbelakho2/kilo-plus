// Canonical workspace -> session binding for the VS Code window.
//
// Defect: the extension bound `sessions[0]` from `/session/list` — an
// arbitrary session, often one created for a DIFFERENT workspace. This
// module resolves the exact canonical workspace identity of the current
// window (the folder URI string, never a basename or index) and binds the
// session stored for that identity only; an unknown or stale binding
// creates a new session instead of guessing. The binding map is bounded
// and dependency-free so scripts/selftest.mjs can drive it directly.

export type SessionBindings = Readonly<Record<string, string>>;

/** Bound on remembered workspace -> session bindings (LRU-ish by insert). */
export const MAX_SESSION_BINDINGS = 64;

export interface BindableSession {
  readonly id: string;
}

/**
 * The canonical key of one workspace URI: trimmed, trailing slashes
 * stripped (except a lone `/`). `null` for missing/blank input — never a
 * guessed identity, never a filesystem-path case fold.
 */
export function canonicalWorkspaceKey(uri: string | null | undefined): string | null {
  if (typeof uri !== 'string') {
    return null;
  }
  let key = uri.trim();
  if (key.length === 0) {
    return null;
  }
  while (key.length > 1 && key.endsWith('/')) {
    key = key.slice(0, -1);
  }
  return key;
}

/** The canonical key of a folder URI list (first folder wins, explicitly). */
export function windowWorkspaceKey(folderUris: readonly string[]): string | null {
  return folderUris.length > 0 ? canonicalWorkspaceKey(folderUris[0]) : null;
}

/**
 * The session bound to `key`, ONLY when that exact id is present in the
 * current session list (a stale binding resolves to null, never to
 * `sessions[0]`).
 */
export function boundSessionFor(
  key: string | null,
  sessions: readonly BindableSession[],
  bindings: SessionBindings,
): string | null {
  if (key === null) {
    return null;
  }
  const bound = bindings[key];
  if (typeof bound !== 'string' || bound.length === 0) {
    return null;
  }
  return sessions.some((session) => session.id === bound) ? bound : null;
}

/** Record a binding; the map stays bounded (oldest key evicted first). */
export function withBinding(
  bindings: SessionBindings,
  key: string | null,
  sessionId: string,
): SessionBindings {
  if (key === null || sessionId.length === 0) {
    return bindings;
  }
  const next: Record<string, string> = { ...bindings, [key]: sessionId };
  const keys = Object.keys(next);
  if (keys.length > MAX_SESSION_BINDINGS) {
    const drop = keys.length - MAX_SESSION_BINDINGS;
    for (const stale of keys.slice(0, drop)) {
      delete next[stale];
    }
  }
  return next;
}

/** Drop bindings whose session no longer exists (bounded cleanup). */
export function pruneBindings(
  bindings: SessionBindings,
  sessions: readonly BindableSession[],
): SessionBindings {
  const alive = new Set(sessions.map((session) => session.id));
  const next: Record<string, string> = {};
  for (const [key, id] of Object.entries(bindings)) {
    if (alive.has(id)) {
      next[key] = id;
    }
  }
  return next;
}
