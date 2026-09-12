// Deterministic pixel-avatar presence for child agents.
//
// Every ChildId maps to a STABLE avatar (hash -> color + symmetric pixel
// sprite) and a typed presence state with its own animation class. The
// mapping is a pure function of the id, so the same child keeps the same
// avatar across frames, reconnects and daemon restarts — no randomness, no
// external assets, no canvas dependency in the extension host. The webview
// renders the sprite as inline SVG with the CSS animation named here.
//
// Dependency-free so scripts/selftest.mjs drives the whole state machine
// over native mock frames.

export type PixelState =
  | 'running'
  | 'paused'
  | 'waiting'
  | 'blocked'
  | 'done'
  | 'failed'
  | 'cancelled';

export interface PixelAvatar {
  readonly childId: string;
  readonly hash: number;
  /** Primary sprite color (deterministic hue from the hash). */
  readonly color: string;
  /** Highlight color used by the webview for the sprite's eyes. */
  readonly accent: string;
  /** 5x5 row-major sprite bits (1 = filled) — symmetric by construction. */
  readonly pixels: readonly number[];
  /** Avatar version so a future sprite change can be detected. */
  readonly version: number;
}

export interface PixelPresence {
  readonly childId: string;
  readonly state: PixelState;
  readonly avatar: PixelAvatar;
  /** CSS animation class name the webview applies to the sprite. */
  readonly animation: string;
}

export const PIXEL_AVATAR_VERSION = 1;

/** FNV-1a 32-bit over the UTF-16 code units — stable across platforms. */
export function pixelHash(childId: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < childId.length; i += 1) {
    hash ^= childId.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** Deterministic 5x5 symmetric sprite + colors for one ChildId. */
export function pixelAvatar(childId: string): PixelAvatar {
  const hash = pixelHash(childId);
  const pixels: number[] = new Array(25).fill(0);
  // Left half (3 columns) from 15 hash bits; mirror onto the right half.
  for (let y = 0; y < 5; y += 1) {
    for (let x = 0; x < 3; x += 1) {
      const bit = (hash >>> ((y * 3 + x) % 15)) & 1;
      pixels[y * 5 + x] = bit;
      pixels[y * 5 + (4 - x)] = bit;
    }
  }
  const hue = hash % 360;
  return {
    childId,
    hash,
    color: `hsl(${hue} 72% 55%)`,
    accent: `hsl(${(hue + 42) % 360} 85% 72%)`,
    pixels,
    version: PIXEL_AVATAR_VERSION,
  };
}

/**
 * Native state tag -> presence state. Terminal states are Done / Failed /
 * Cancelled; Waiting/Blocked/Paused keep their own animations; unknown or
 * idle tags read as `waiting` (never as running) so the pixel never lies
 * about activity.
 */
export function pixelStateOf(state: string): PixelState {
  const tag = state.trim().toLowerCase();
  if (tag === 'done' || tag === 'completed' || tag === 'verifiedcomplete') {
    return 'done';
  }
  if (tag === 'failed' || tag === 'failedrecoverable' || tag === 'failedpermanent') {
    return 'failed';
  }
  if (tag === 'cancelled' || tag === 'canceled') {
    return 'cancelled';
  }
  if (tag === 'blocked' || tag === 'needsuserinput') {
    return 'blocked';
  }
  if (tag === 'paused' || tag === 'suspended') {
    return 'paused';
  }
  if (
    tag === 'running' ||
    tag === 'preparing' ||
    tag === 'buildingcontext' ||
    tag === 'waitingformodel' ||
    tag === 'streaming' ||
    tag === 'toolrequested' ||
    tag === 'executingtool' ||
    tag === 'validating'
  ) {
    return 'running';
  }
  return 'waiting';
}

/** The CSS animation class for one presence state. */
export function pixelAnimation(state: PixelState): string {
  return `pixel-${state}`;
}

export function pixelPresence(childId: string, state: string): PixelPresence {
  const pixelState = pixelStateOf(state);
  return {
    childId,
    state: pixelState,
    avatar: pixelAvatar(childId),
    animation: pixelAnimation(pixelState),
  };
}

/**
 * Fold one native agent frame (the `GET /native/agents` payload, snake_case
 * ids or the camelCase summaries) into a presence map. `previous` is
 * consulted only to keep a child's last presence when a frame omits it
 * (transient page gaps) — the avatar itself is always recomputed from the
 * id, never carried as mutable state.
 */
export function foldPixelPresence(
  previous: ReadonlyMap<string, PixelPresence>,
  frame: readonly { agentId?: string; agent_id?: string; kind?: string; state: string }[],
): Map<string, PixelPresence> {
  const next = new Map<string, PixelPresence>();
  const idOf = (agent: { agentId?: string; agent_id?: string }): string | null =>
    typeof agent.agentId === 'string' && agent.agentId.length > 0
      ? agent.agentId
      : typeof agent.agent_id === 'string' && agent.agent_id.length > 0
        ? agent.agent_id
        : null;
  // Existing children keep insertion order first (stable rendering).
  for (const [childId, presence] of previous) {
    const updated = frame.find((agent) => idOf(agent) === childId);
    if (updated === undefined) {
      next.set(childId, presence);
    }
  }
  for (const agent of frame) {
    const childId = idOf(agent);
    if (childId !== null) {
      next.set(childId, pixelPresence(childId, agent.state));
    }
  }
  return next;
}
