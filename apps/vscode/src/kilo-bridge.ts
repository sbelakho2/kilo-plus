// Message-ABI bridge between the frozen Kilo v7.5.6 webview and the Faktor
// native runtime.
//
// The boundary is deliberately one-way and strict:
//
//   - inbound (frozen UI -> bridge): every message is validated against the
//     small command set Faktor can actually honor. Unknown kinds, malformed
//     envelopes, oversized payloads and unsupported structured fields are
//     DROPPED with an explicit reason (`BridgeDrop`) so the caller can log
//     them loudly — nothing is forwarded unvalidated and nothing is silently
//     coerced.
//   - outbound (native snapshot -> frozen UI): Faktor state is translated
//     into the subset of upstream `ExtensionMessage` shapes the vendored UI
//     consumes (ready, connectionState, sessionsLoaded, sessionStatus,
//     messagesLoaded, messageCreated, partUpdated, todoUpdated, error).
//     Payloads are bounded by entry count and serialized bytes.
//
// Additive Faktor extension messages (`faktorTaskState`, `faktorAgents`,
// `faktorCockpit`, `faktorTournament`, `faktorEvidence`, `faktorBoardState`)
// are emitted AFTER the frozen Kilo messages and are consumed by the
// Faktor companion panel that the overlay build step mounts next to the
// vendored chat. The frozen upstream message semantics are never mutated:
// the additive tail is a separate namespace, bounded like everything else.
//
// The module is dependency-free (no vscode, no vendored imports) so
// scripts/selftest.mjs can drive every accept/reject path, and it contains no
// Faktor state of its own: it is a pure translation layer.

import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type {
  FaktorSnapshot,
  SessionSummary,
  TaskSummary,
  TranscriptEntry,
  TranscriptTool,
} from './state';

/** Upstream UI message envelope (structure only; the frozen UI owns the types). */
export interface WebviewBoundMessage {
  readonly type: string;
  readonly [key: string]: unknown;
}

/** The host command vocabulary the bridge maps accepted UI messages onto. */
export interface HostChatMessage {
  readonly type: string;
  readonly [key: string]: unknown;
}

export const BRIDGE_PROTOCOL = 'faktor-kilo-bridge/1';

export const BRIDGE_LIMITS = {
  /** Inbound envelope cap (serialized JSON bytes). */
  maxInboundBytes: 256 * 1024,
  /** Outbound `messagesLoaded` cap (serialized JSON bytes). */
  maxOutboundBytes: 1024 * 1024,
  /** Prompt text cap. */
  maxTextChars: 64 * 1024,
  /** Identifier/URL cap. */
  maxStringChars: 4096,
  /** Newest transcript entries per `messagesLoaded` page. */
  maxPageEntries: 200,
  /** Largest honored `loadMessages.limit`. */
  maxMessagesLimit: 500,
  defaultMessagesLimit: 100,
  /** Tool parts rendered per message. */
  maxToolPartsPerMessage: 200,
  /** Faktor panel: agent cards per `faktorAgents` frame. */
  maxFaktorAgents: 64,
  /** Faktor panel: evidence refs per `faktorEvidence` frame. */
  maxFaktorEvidenceRefs: 64,
  /** Faktor panel: characters of one expanded evidence artifact. */
  maxFaktorEvidenceChars: 256 * 1024,
  /** Faktor panel: cockpit lines rendered per section. */
  maxFaktorCockpitLines: 64,
  /** Faktor panel: serialized-byte budget of one additive frame. */
  maxFaktorFrameBytes: 128 * 1024,
  /** Faktor panel: one inline attachment (decoded bytes). */
  maxAttachmentBytes: 128 * 1024,
  /** Faktor panel: binary attachment references per message. */
  maxAttachments: 64,
  /** Mirrors faktor_session::MAX_FILES_PER_PROMPT. */
  maxFilesPerPrompt: 64,
  /** Mirrors faktor_session::MAX_FILE_PATH_BYTES. */
  maxFilePathBytes: 4096,
  /** Mirrors the runtime's steering note bound (chars). */
  maxSteerChars: 500,
  /** Model ids are bounded identifiers, never free text. */
  maxModelChars: 256,
  /** Mirrors faktor_server::native::task::MAX_ABORT_REASON_BYTES. */
  maxAbortReasonBytes: 512,
  /** Mirrors faktor_session board bounds. */
  maxBoardSubjectBytes: 512,
  maxBoardBodyBytes: 16 * 1024,
  maxBoardRefs: 32,
  maxBoardRefBytes: 1024,
  /** Mirrors faktor_session::MAX_BOARD_SCAN_PAGE. */
  maxBoardPage: 500,
} as const;

const SUPPORTED_INBOUND: ReadonlySet<string> = new Set([
  'webviewReady',
  'sendMessage',
  'abort',
  'createSession',
  'loadSessions',
  'loadMessages',
  'openExternal',
  // Additive Faktor companion-panel actions. Never part of the frozen Kilo
  // vocabulary; unknown `faktor*` kinds are dropped with a reason too.
  'faktorAgentAction',
  'faktorTournamentAction',
  'faktorEvidenceExpand',
  'faktorBoardAction',
  // The pre-rename alias for `faktorTournamentAction`, kept accepted so the
  // additive surface is backward compatible within the same release.
  'tournamentControl',
]);

const AGENT_ACTIONS: ReadonlySet<string> = new Set([
  'retry',
  'pause',
  'resume',
  'cancel',
  'steer',
  'model',
  'budget',
  'presentation',
]);

const BOARD_ACTIONS: ReadonlySet<string> = new Set(['read', 'post']);

const PRESENTATION_STATES: ReadonlySet<string> = new Set(['foreground', 'background']);


/** Mirrors the daemon's bounded abort reason (native tournament abort DTO). */
export const MAX_TOURNAMENT_ABORT_BYTES = BRIDGE_LIMITS.maxAbortReasonBytes;

const MESSAGE_LOAD_MODES: ReadonlySet<string> = new Set([
  'replace',
  'prepend',
  'focus',
  'reconcile',
]);

/** One validated binary/image attachment: a REFERENCE, never inline bytes. */
export interface BridgeAttachmentRef {
  /** `faktor-attachment:sha256:<hex>` — stable, content-addressed, opaque. */
  readonly ref: string;
  readonly mime: string;
  readonly filename: string | null;
  readonly bytes: number;
}

/** One refused entry of a Kilo file payload (kept, never a whole drop). */
export interface BridgeAttachmentRefusal {
  readonly index: number;
  readonly reason: string;
}

export interface BridgeFilesMapping {
  /** Workspace-relative paths handed to the daemon (bounded per its rules). */
  readonly files: readonly string[];
  /** Binary/image references (never bytes, never in the prompt). */
  readonly attachments: readonly BridgeAttachmentRef[];
  /** Per-entry refusals; the message itself is always kept. */
  readonly refused: readonly BridgeAttachmentRefusal[];
}

/** The Task-mode completion contract accepted from the UI (wire snake_case). */
export interface BridgeCompletionContract {
  readonly include_commit: boolean;
  readonly include_push: boolean;
  readonly include_pr: boolean;
}

/**
 * Strict parse of one `sendMessage.completionContract`. Returns `null` for
 * an absent value or the all-false default (no contract: today's path),
 * `{ contract }` for a valid non-default contract, or `{ reason }` for a
 * malformed value — a malformed contract is a LOUD per-message drop, never
 * a silently contract-free task start (that would claim a workflow the run
 * never recorded).
 */
export function completionContractOf(
  raw: unknown,
): { readonly contract: BridgeCompletionContract | null } | { readonly reason: string } {
  if (raw === undefined || raw === null) {
    return { contract: null };
  }
  if (!isRecord(raw)) {
    return { reason: 'completionContract must be an object' };
  }
  const keys = Object.keys(raw);
  for (const key of keys) {
    if (key !== 'include_commit' && key !== 'include_push' && key !== 'include_pr') {
      return { reason: `completionContract.${key} is not a known member` };
    }
  }
  for (const key of ['include_commit', 'include_push', 'include_pr'] as const) {
    if (
      !Object.prototype.hasOwnProperty.call(raw, key) ||
      typeof raw[key] !== 'boolean'
    ) {
      return { reason: `completionContract.${key} must be a boolean` };
    }
  }
  const contract: BridgeCompletionContract = {
    include_commit: raw.include_commit as boolean,
    include_push: raw.include_push as boolean,
    include_pr: raw.include_pr as boolean,
  };
  if (!contract.include_commit && !contract.include_push && !contract.include_pr) {
    return { contract: null };
  }
  return { contract };
}

export type BridgeCommand =
  | { readonly kind: 'ready' }
  | {
      readonly kind: 'sendMessage';
      readonly text: string;
      readonly sessionId: string | null;
      readonly files: readonly string[];
      readonly attachments: readonly BridgeAttachmentRef[];
      readonly refusedAttachments: readonly BridgeAttachmentRefusal[];
      /** Task-mode completion contract (null = default path). */
      readonly completionContract: BridgeCompletionContract | null;
    }
  | { readonly kind: 'abort'; readonly sessionId: string }
  | { readonly kind: 'createSession' }
  | { readonly kind: 'loadSessions' }
  | {
      readonly kind: 'loadMessages';
      readonly sessionId: string;
      readonly before: string | null;
      readonly limit: number;
      readonly mode: string;
    }
  | { readonly kind: 'tournamentDecide'; readonly tournamentId: string }
  | {
      readonly kind: 'tournamentAbort';
      readonly tournamentId: string;
      readonly reason: string | null;
    }
  // Additive Faktor companion-panel actions (the `faktor*` namespace).
  | {
      readonly kind: 'faktorAgentAction';
      readonly agentId: string;
      readonly action: string;
      readonly state: string | null;
    }
  | { readonly kind: 'faktorEvidenceExpand'; readonly evidenceId: number }
  | {
      readonly kind: 'faktorBoardAction';
      readonly action: 'read' | 'post';
      readonly since: number | null;
      readonly limit: number | null;
      readonly subject: string | null;
      readonly body: string | null;
      readonly refs: readonly string[];
    }
  | { readonly kind: 'openExternal'; readonly url: string };

export interface BridgeDrop {
  readonly dropped: true;
  readonly type: string | null;
  readonly reason: string;
  readonly bytes: number;
}

export type BridgeIngest = BridgeCommand | BridgeDrop;

// ------------------------------------------------------------------ helpers

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Copy only own enumerable properties: inherited state must not smuggle values. */
function ownProperties(record: Record<string, unknown>): Record<string, unknown> {
  const copy: Record<string, unknown> = {};
  for (const key of Object.keys(record)) {
    copy[key] = record[key];
  }
  return copy;
}

function serializedBytes(value: unknown): number {
  try {
    const text = JSON.stringify(value);
    return typeof text === 'string' ? Buffer.byteLength(text, 'utf8') : 0;
  } catch {
    return Number.POSITIVE_INFINITY;
  }
}

function drop(type: string | null, reason: string, bytes: number): BridgeDrop {
  return { dropped: true, type, reason, bytes };
}

function boundedString(value: unknown, max: number): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim();
  if (trimmed.length === 0 || trimmed.length > max) {
    return null;
  }
  return trimmed;
}

function isNonEmptyStructured(value: unknown): boolean {
  if (value === undefined || value === null) {
    return false;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === 'object') {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
}

// ------------------------------------------------------------- attachments

/** RFC-ish mime type shape; anything else falls back to the declared mime. */
const MIME_RE = /^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}\/[a-z0-9][a-z0-9!#$&^_.+-]{0,63}$/i;
const DATA_URL_RE = /^data:([^,]*),([\s\S]*)$/;
const ATTACHMENT_REF_PREFIX = 'faktor-attachment:sha256:';
const DRIVE_PREFIX_RE = /^[a-zA-Z]:[\\/]/;

function utf8Bytes(value: string): number {
  return Buffer.byteLength(value, 'utf8');
}

function hasControlChars(value: string): boolean {
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code < 0x20 || code === 0x7f) {
      return true;
    }
  }
  return false;
}

function sanitizeFilename(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const base = value.replace(/\\/g, '/').split('/').pop() ?? '';
  const trimmed = base.trim();
  if (
    trimmed.length === 0 ||
    trimmed.length > 255 ||
    trimmed === '.' ||
    trimmed === '..' ||
    hasControlChars(trimmed)
  ) {
    return null;
  }
  return trimmed;
}

function attachmentMime(value: unknown, fallback: string): string {
  if (typeof value === 'string') {
    const candidate = value.trim().toLowerCase().split(';')[0]!.trim();
    if (MIME_RE.test(candidate)) {
      return candidate;
    }
  }
  return fallback;
}

/**
 * Map one absolute or workspace-relative candidate onto the daemon's
 * attachment rule: workspace-relative, no control characters, no `..`
 * segment, bounded by `MAX_FILE_PATH_BYTES`. Absolute paths are accepted
 * only under the current workspace root; everything else is refused with a
 * reason (the containing message is never dropped for one bad entry).
 */
function workspaceRelativePath(
  candidate: string,
  workspaceDirectory: string | null,
): { readonly path: string } | { readonly reason: string } {
  const trimmed = candidate.trim();
  if (trimmed.length === 0) {
    return { reason: 'attachment path is empty' };
  }
  if (utf8Bytes(trimmed) > BRIDGE_LIMITS.maxFilePathBytes) {
    return { reason: `attachment path exceeds ${BRIDGE_LIMITS.maxFilePathBytes} bytes` };
  }
  if (hasControlChars(trimmed)) {
    return { reason: 'attachment path carries control characters' };
  }
  if (DRIVE_PREFIX_RE.test(trimmed) || isAbsolute(trimmed)) {
    if (workspaceDirectory === null || workspaceDirectory.trim().length === 0) {
      return { reason: 'absolute attachment path without a workspace root' };
    }
    const rel = relative(resolve(workspaceDirectory), resolve(trimmed));
    const posix = rel.split('\\').join('/');
    if (posix.length === 0 || posix.startsWith('..') || isAbsolute(posix)) {
      return { reason: 'attachment path is outside the workspace' };
    }
    if (posix.split('/').some((segment) => segment === '..')) {
      return { reason: 'attachment path traverses outside the workspace' };
    }
    return { path: posix };
  }
  const segments = trimmed.split(/[\\/]/);
  if (segments.some((segment) => segment === '..')) {
    return { reason: 'attachment path traverses outside the workspace' };
  }
  return { path: segments.filter((segment) => segment.length > 0).join('/') };
}

/**
 * Map a Kilo `FileAttachment` entry to a content-addressed binary/image
 * reference. The decoded bytes exist only long enough to hash and size them:
 * they are NEVER placed into the prompt or into `files` — only the opaque
 * `faktor-attachment:sha256:<hex>` reference is, per the defined attachment
 * mechanism.
 */
function binaryAttachmentRef(
  url: string,
  rawMime: unknown,
  filename: unknown,
  index: number,
): BridgeAttachmentRef | BridgeAttachmentRefusal {
  const match = DATA_URL_RE.exec(url);
  if (!match) {
    return { index, reason: 'malformed data URL attachment' };
  }
  const meta = match[1] ?? '';
  const payload = match[2] ?? '';
  const mime = attachmentMime(meta.split(';')[0], attachmentMime(rawMime, 'application/octet-stream'));
  let bytes: Buffer;
  if (/;base64$/i.test(meta)) {
    const compact = payload.replace(/\s+/g, '');
    if (
      compact.length > BRIDGE_LIMITS.maxAttachmentBytes * 2 ||
      !/^[A-Za-z0-9+/]*={0,2}$/.test(compact)
    ) {
      return { index, reason: 'malformed or oversized base64 attachment' };
    }
    bytes = Buffer.from(compact, 'base64');
  } else {
    let decoded: string;
    try {
      decoded = decodeURIComponent(payload);
    } catch {
      return { index, reason: 'malformed percent-encoded data URL attachment' };
    }
    bytes = Buffer.from(decoded, 'utf8');
  }
  if (bytes.byteLength > BRIDGE_LIMITS.maxAttachmentBytes) {
    return {
      index,
      reason: `attachment exceeds ${BRIDGE_LIMITS.maxAttachmentBytes} bytes`,
    };
  }
  return {
    ref: `${ATTACHMENT_REF_PREFIX}${createHash('sha256').update(bytes).digest('hex')}`,
    mime,
    filename: sanitizeFilename(filename),
    bytes: bytes.byteLength,
  };
}

/**
 * Validate one Kilo `files` payload against the daemon's attachment rules
 * and split it into workspace-relative paths (for `sendGoal.files`) and
 * binary/image references. Malformed entries are refused individually with
 * a reason: a message carrying files is NEVER dropped wholesale because of
 * them, and bytes never enter the prompt.
 */
export function mapKiloFiles(
  raw: unknown,
  workspaceDirectory: string | null,
): BridgeFilesMapping {
  const files: string[] = [];
  const attachments: BridgeAttachmentRef[] = [];
  const refused: BridgeAttachmentRefusal[] = [];
  if (raw === undefined || raw === null) {
    return { files, attachments, refused };
  }
  if (!Array.isArray(raw)) {
    refused.push({ index: 0, reason: 'files must be an array of attachments' });
    return { files, attachments, refused };
  }
  // Refusals are themselves bounded: one reason per malformed entry while
  // the list is under the file cap; the message is never dropped wholesale.
  const refuse = (index: number, reason: string): void => {
    if (refused.length < BRIDGE_LIMITS.maxFilesPerPrompt) {
      refused.push({ index, reason });
    }
  };
  for (let index = 0; index < raw.length; index += 1) {
    const entry = raw[index];
    if (!isRecord(entry)) {
      refuse(index, 'attachment must be an object');
      continue;
    }
    const url = typeof entry.url === 'string' ? entry.url : null;
    if (url !== null && url.startsWith('data:')) {
      const ref = binaryAttachmentRef(url, entry.mime, entry.filename, index);
      if ('reason' in ref) {
        refuse(ref.index, ref.reason);
      } else if (attachments.length >= BRIDGE_LIMITS.maxAttachments) {
        refuse(index, `more than ${BRIDGE_LIMITS.maxAttachments} binary attachments`);
      } else {
        attachments.push(ref);
      }
      continue;
    }
    const source = isRecord(entry.source) ? entry.source : null;
    const sourcePath = source !== null && typeof source.path === 'string' ? source.path : null;
    let candidate = sourcePath ?? url;
    if (candidate === null || candidate.trim().length === 0) {
      refuse(index, 'attachment has no url or source path');
      continue;
    }
    if (candidate.startsWith('data:')) {
      const ref = binaryAttachmentRef(candidate, entry.mime, entry.filename, index);
      if ('reason' in ref) {
        refuse(ref.index, ref.reason);
      } else if (attachments.length >= BRIDGE_LIMITS.maxAttachments) {
        refuse(index, `more than ${BRIDGE_LIMITS.maxAttachments} binary attachments`);
      } else {
        attachments.push(ref);
      }
      continue;
    }
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(candidate)) {
      if (!candidate.startsWith('file:')) {
        refuse(index, `unsupported attachment url scheme in ${candidate.slice(0, 32)}`);
        continue;
      }
      try {
        candidate = fileURLToPath(candidate);
      } catch {
        refuse(index, 'malformed file URL attachment');
        continue;
      }
    }
    const mapped = workspaceRelativePath(candidate, workspaceDirectory);
    if ('reason' in mapped) {
      refuse(index, mapped.reason);
      continue;
    }
    if (files.length >= BRIDGE_LIMITS.maxFilesPerPrompt) {
      refuse(index, `more than ${BRIDGE_LIMITS.maxFilesPerPrompt} file attachments`);
      continue;
    }
    files.push(mapped.path);
  }
  return { files, attachments, refused };
}

// --------------------------------------------------------------- inbound ABI

/** Context the attachment mapping needs (the current workspace identity). */
export interface BridgeIngestOptions {
  /** Absolute workspace root used to relativize absolute `file:` attachments. */
  readonly workspaceDirectory?: string | null;
}

/**
 * Validate one message from the frozen UI. Returns a typed command, or a
 * `BridgeDrop` that the caller must log (never forward).
 */
export function ingestWebviewMessage(
  raw: unknown,
  options: BridgeIngestOptions = {},
): BridgeIngest {
  const bytes = serializedBytes(raw);
  if (bytes > BRIDGE_LIMITS.maxInboundBytes) {
    return drop(
      null,
      `payload ${bytes} bytes exceeds ${BRIDGE_LIMITS.maxInboundBytes} byte bound`,
      bytes,
    );
  }
  if (!isRecord(raw)) {
    return drop(null, 'message must be a JSON object', bytes);
  }
  // Work on own properties only: prototype-inherited fields must not count.
  const message = ownProperties(raw);
  const rawType = message.type;
  if (typeof rawType !== 'string' || rawType.length === 0 || rawType.length > 64) {
    return drop(null, 'message type must be a non-empty string', bytes);
  }
  if (!SUPPORTED_INBOUND.has(rawType)) {
    return drop(rawType, `unsupported message kind "${rawType}"`, bytes);
  }

  if (rawType === 'webviewReady') {
    return { kind: 'ready' };
  }
  if (rawType === 'createSession') {
    return { kind: 'createSession' };
  }
  if (rawType === 'loadSessions') {
    return { kind: 'loadSessions' };
  }
  if (rawType === 'sendMessage') {
    if (message.review !== undefined && isNonEmptyStructured(message.review)) {
      return drop(rawType, 'review comments are not supported by the native bridge', bytes);
    }
    if (isNonEmptyStructured(message.agentManagerContext)) {
      return drop(rawType, 'agent-manager context is not supported by the native bridge', bytes);
    }
    if (message.files !== undefined && message.files !== null && !Array.isArray(message.files)) {
      return drop(rawType, 'sendMessage.files must be an array of attachments', bytes);
    }
    if (typeof message.text !== 'string' || message.text.trim().length === 0) {
      return drop(rawType, 'sendMessage.text must be a non-empty string', bytes);
    }
    if (message.text.length > BRIDGE_LIMITS.maxTextChars) {
      return drop(
        rawType,
        `sendMessage.text exceeds ${BRIDGE_LIMITS.maxTextChars} character bound`,
        bytes,
      );
    }
    let sessionId: string | null = null;
    if (message.sessionID !== undefined && message.sessionID !== null) {
      sessionId = boundedString(message.sessionID, BRIDGE_LIMITS.maxStringChars);
      if (sessionId === null) {
        return drop(rawType, 'sendMessage.sessionID must be a bounded non-empty string', bytes);
      }
    }
    // Attachments are mapped, never refused wholesale: valid entries become
    // workspace-relative daemon paths and/or content-addressed binary refs;
    // malformed entries are refused individually with a reason.
    const mapping = mapKiloFiles(message.files, options.workspaceDirectory ?? null);
    // The Task-mode completion contract is strict: a malformed contract is
    // a loud drop (starting the task contract-free would lie about the
    // requested workflow); absent/all-false is the default path.
    const completion = completionContractOf(message.completionContract);
    if ('reason' in completion) {
      return drop(rawType, completion.reason, bytes);
    }
    return {
      kind: 'sendMessage',
      text: message.text,
      sessionId,
      files: mapping.files,
      attachments: mapping.attachments,
      refusedAttachments: mapping.refused,
      completionContract: completion.contract,
    };
  }
  if (rawType === 'abort') {
    const sessionId = boundedString(message.sessionID, BRIDGE_LIMITS.maxStringChars);
    if (sessionId === null) {
      return drop(rawType, 'abort.sessionID must be a bounded non-empty string', bytes);
    }
    return { kind: 'abort', sessionId };
  }
  if (rawType === 'loadMessages') {
    const sessionId = boundedString(message.sessionID, BRIDGE_LIMITS.maxStringChars);
    if (sessionId === null) {
      return drop(rawType, 'loadMessages.sessionID must be a bounded non-empty string', bytes);
    }
    let before: string | null = null;
    if (message.before !== undefined && message.before !== null) {
      before = boundedString(message.before, BRIDGE_LIMITS.maxStringChars);
      if (before === null) {
        return drop(rawType, 'loadMessages.before must be a bounded non-empty string', bytes);
      }
    }
    let limit: number = BRIDGE_LIMITS.defaultMessagesLimit;
    if (message.limit !== undefined) {
      if (
        typeof message.limit !== 'number' ||
        !Number.isInteger(message.limit) ||
        message.limit <= 0 ||
        message.limit > BRIDGE_LIMITS.maxMessagesLimit
      ) {
        return drop(
          rawType,
          `loadMessages.limit must be an integer in 1..${BRIDGE_LIMITS.maxMessagesLimit}`,
          bytes,
        );
      }
      limit = message.limit;
    }
    let mode = 'replace';
    if (message.mode !== undefined) {
      if (typeof message.mode !== 'string' || !MESSAGE_LOAD_MODES.has(message.mode)) {
        return drop(rawType, `loadMessages.mode "${String(message.mode)}" is not a known mode`, bytes);
      }
      mode = message.mode;
    }
    return { kind: 'loadMessages', sessionId, before, limit, mode };
  }

  // Tournament decide/abort (Faktor companion vocabulary): the durable
  // tournament id must be bounded; the decision takes NO operator input (the
  // engine compares candidates) while abort may carry a reason bounded by the
  // daemon's own limit. Nothing is coerced: an unknown action, a structured
  // reason on decide, or an oversized reason is dropped loudly.
  if (rawType === 'faktorTournamentAction' || rawType === 'tournamentControl') {
    const tournamentId = boundedString(message.tournamentId, BRIDGE_LIMITS.maxStringChars);
    if (tournamentId === null) {
      return drop(rawType, `${rawType}.tournamentId must be a bounded non-empty string`, bytes);
    }
    if (message.action === 'decide') {
      if (isNonEmptyStructured(message.reason)) {
        return drop(rawType, `${rawType} decide takes no reason`, bytes);
      }
      return { kind: 'tournamentDecide', tournamentId };
    }
    if (message.action === 'abort') {
      let reason: string | null = null;
      if (message.reason !== undefined && message.reason !== null) {
        if (typeof message.reason !== 'string') {
          return drop(rawType, `${rawType}.reason must be a string`, bytes);
        }
        if (Buffer.byteLength(message.reason, 'utf8') > MAX_TOURNAMENT_ABORT_BYTES) {
          return drop(
            rawType,
            `${rawType}.reason exceeds ${MAX_TOURNAMENT_ABORT_BYTES} bytes`,
            bytes,
          );
        }
        const trimmed = message.reason.trim();
        reason = trimmed.length === 0 ? null : trimmed;
      }
      return { kind: 'tournamentAbort', tournamentId, reason };
    }
    return drop(rawType, `${rawType}.action must be "decide" or "abort"`, bytes);
  }

  // Agent control (retry/pause/resume/cancel/steer/model/budget/presentation).
  // The runtime is the guard for every state transition; the bridge only
  // enforces shapes and bounds, never coerces.
  if (rawType === 'faktorAgentAction') {
    const agentId = boundedString(message.agentId, BRIDGE_LIMITS.maxStringChars);
    if (agentId === null) {
      return drop(rawType, 'faktorAgentAction.agentId must be a bounded non-empty string', bytes);
    }
    if (typeof message.action !== 'string' || !AGENT_ACTIONS.has(message.action)) {
      return drop(
        rawType,
        'faktorAgentAction.action must be one of retry|pause|resume|cancel|steer|model|budget|presentation',
        bytes,
      );
    }
    let state: string | null = null;
    if (message.action === 'presentation') {
      if (typeof message.state !== 'string' || !PRESENTATION_STATES.has(message.state)) {
        return drop(
          rawType,
          'faktorAgentAction.state must be "foreground" or "background"',
          bytes,
        );
      }
      state = message.state;
    } else if (message.state !== undefined && message.state !== null) {
      return drop(rawType, `faktorAgentAction.state is not valid for action "${message.action}"`, bytes);
    }
    if (message.action === 'steer') {
      if (typeof message.text !== 'string' || message.text.trim().length === 0) {
        return drop(rawType, 'faktorAgentAction.text must be a non-empty string for steer', bytes);
      }
      if (message.text.length > BRIDGE_LIMITS.maxSteerChars) {
        return drop(
          rawType,
          `faktorAgentAction.text exceeds ${BRIDGE_LIMITS.maxSteerChars} character bound`,
          bytes,
        );
      }
    }
    if (message.action === 'model') {
      // The inline model is optional: the host prompts when the panel sends
      // the bare control action; a provided value must still be bounded.
      if (
        message.model !== undefined &&
        message.model !== null &&
        boundedString(message.model, BRIDGE_LIMITS.maxModelChars) === null
      ) {
        return drop(rawType, 'faktorAgentAction.model must be a bounded non-empty string', bytes);
      }
    }
    if (message.action === 'budget') {
      if (
        message.maxTokens !== undefined &&
        message.maxTokens !== null &&
        (typeof message.maxTokens !== 'number' ||
          !Number.isInteger(message.maxTokens) ||
          message.maxTokens <= 0)
      ) {
        return drop(rawType, 'faktorAgentAction.maxTokens must be a positive integer', bytes);
      }
    }
    return { kind: 'faktorAgentAction', agentId, action: message.action, state };
  }

  // Evidence expansion: a positive durable evidence id only (the host
  // retrieves it through the session-scoped native route).
  if (rawType === 'faktorEvidenceExpand') {
    const evidenceId = message.evidenceId;
    if (
      typeof evidenceId !== 'number' ||
      !Number.isInteger(evidenceId) ||
      evidenceId <= 0 ||
      !Number.isSafeInteger(evidenceId)
    ) {
      return drop(rawType, 'faktorEvidenceExpand.evidenceId must be a positive integer', bytes);
    }
    return { kind: 'faktorEvidenceExpand', evidenceId };
  }

  // Coordination board read/post. Bounds mirror the durable ledger rules;
  // the daemon remains the only authority on scope and terminal children.
  if (rawType === 'faktorBoardAction') {
    if (typeof message.action !== 'string' || !BOARD_ACTIONS.has(message.action)) {
      return drop(rawType, 'faktorBoardAction.action must be "read" or "post"', bytes);
    }
    if (message.action === 'read') {
      let since: number | null = null;
      if (message.since !== undefined && message.since !== null) {
        if (
          typeof message.since !== 'number' ||
          !Number.isInteger(message.since) ||
          message.since < 0
        ) {
          return drop(rawType, 'faktorBoardAction.since must be a non-negative integer', bytes);
        }
        since = message.since;
      }
      let limit: number | null = null;
      if (message.limit !== undefined && message.limit !== null) {
        if (
          typeof message.limit !== 'number' ||
          !Number.isInteger(message.limit) ||
          message.limit <= 0 ||
          message.limit > BRIDGE_LIMITS.maxBoardPage
        ) {
          return drop(
            rawType,
            `faktorBoardAction.limit must be an integer in 1..${BRIDGE_LIMITS.maxBoardPage}`,
            bytes,
          );
        }
        limit = message.limit;
      }
      return { kind: 'faktorBoardAction', action: 'read', since, limit, subject: null, body: null, refs: [] };
    }
    const subject = message.subject;
    if (typeof subject !== 'string' || subject.trim().length === 0) {
      return drop(rawType, 'faktorBoardAction.subject must be a non-empty string', bytes);
    }
    if (utf8Bytes(subject) > BRIDGE_LIMITS.maxBoardSubjectBytes) {
      return drop(
        rawType,
        `faktorBoardAction.subject exceeds ${BRIDGE_LIMITS.maxBoardSubjectBytes} bytes`,
        bytes,
      );
    }
    let body: string | null = null;
    if (message.body !== undefined && message.body !== null) {
      if (typeof message.body !== 'string') {
        return drop(rawType, 'faktorBoardAction.body must be a string', bytes);
      }
      if (utf8Bytes(message.body) > BRIDGE_LIMITS.maxBoardBodyBytes) {
        return drop(
          rawType,
          `faktorBoardAction.body exceeds ${BRIDGE_LIMITS.maxBoardBodyBytes} bytes`,
          bytes,
        );
      }
      body = message.body;
    }
    const refs: string[] = [];
    if (message.refs !== undefined && message.refs !== null) {
      if (!Array.isArray(message.refs)) {
        return drop(rawType, 'faktorBoardAction.refs must be an array of strings', bytes);
      }
      if (message.refs.length > BRIDGE_LIMITS.maxBoardRefs) {
        return drop(rawType, `faktorBoardAction.refs exceeds ${BRIDGE_LIMITS.maxBoardRefs} entries`, bytes);
      }
      for (let index = 0; index < message.refs.length; index += 1) {
        const ref = message.refs[index];
        if (typeof ref !== 'string' || ref.trim().length === 0) {
          return drop(rawType, `faktorBoardAction.refs[${index}] must be a non-empty string`, bytes);
        }
        if (utf8Bytes(ref) > BRIDGE_LIMITS.maxBoardRefBytes) {
          return drop(
            rawType,
            `faktorBoardAction.refs[${index}] exceeds ${BRIDGE_LIMITS.maxBoardRefBytes} bytes`,
            bytes,
          );
        }
        refs.push(ref.trim());
      }
    }
    return {
      kind: 'faktorBoardAction',
      action: 'post',
      since: null,
      limit: null,
      subject: subject.trim(),
      body,
      refs,
    };
  }

  // openExternal: https only; anything else is refused, never opened.
  const rawUrl = message.url;
  if (typeof rawUrl !== 'string' || rawUrl.length > BRIDGE_LIMITS.maxStringChars) {
    return drop(rawType, 'openExternal.url must be a bounded string', bytes);
  }
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return drop(rawType, 'openExternal.url is not a valid URL', bytes);
  }
  if (parsed.protocol !== 'https:') {
    return drop(rawType, `openExternal refuses non-https protocol "${parsed.protocol}"`, bytes);
  }
  return { kind: 'openExternal', url: parsed.toString() };
}

/**
 * Map an accepted inbound command onto the existing extension-host message
 * vocabulary (`extension.ts`), or null when the webview layer handles it.
 */
export function bridgeCommandToHostMessage(command: BridgeCommand): HostChatMessage | null {
  switch (command.kind) {
    case 'ready':
      return { type: 'ready' };
    case 'sendMessage':
      return {
        type: 'sendGoal',
        goal: command.text,
        ...(command.files.length > 0 ? { files: command.files } : {}),
        ...(command.attachments.length > 0 ? { attachments: command.attachments } : {}),
        ...(command.refusedAttachments.length > 0
          ? { refusedAttachments: command.refusedAttachments }
          : {}),
        ...(command.completionContract !== null
          ? { completionContract: command.completionContract }
          : {}),
      };
    case 'abort':
      return { type: 'cancelRun' };
    case 'createSession':
      return { type: 'refresh' };
    case 'loadSessions':
      return { type: 'refresh' };
    case 'loadMessages':
      return { type: 'refresh' };
    case 'tournamentDecide':
      return {
        type: 'tournamentControl',
        tournamentId: command.tournamentId,
        action: 'decide',
      };
    case 'tournamentAbort':
      return {
        type: 'tournamentControl',
        tournamentId: command.tournamentId,
        action: 'abort',
        reason: command.reason ?? undefined,
      };
    case 'faktorAgentAction':
      return {
        type: 'agentControl',
        agentId: command.agentId,
        action: command.action,
        ...(command.state !== null ? { state: command.state } : {}),
      };
    case 'faktorEvidenceExpand':
      return { type: 'retrieveEvidence', evidenceId: command.evidenceId };
    case 'faktorBoardAction':
      if (command.action === 'read') {
        return {
          type: 'boardRead',
          ...(command.since !== null ? { since: command.since } : {}),
          ...(command.limit !== null ? { limit: command.limit } : {}),
        };
      }
      return {
        type: 'boardPost',
        subject: command.subject,
        ...(command.body !== null ? { body: command.body } : {}),
        ...(command.refs.length > 0 ? { refs: command.refs } : {}),
      };
    case 'openExternal':
      return null;
    default:
      return null;
  }
}

// -------------------------------------------------------------- outbound ABI

export interface BridgeContext {
  readonly extensionVersion: string;
  readonly workspaceDirectory: string;
  readonly daemonVersion?: string | null;
  readonly port?: number | null;
}

const EPOCH_ISO = new Date(0).toISOString();

function isoFromMs(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) {
    return EPOCH_ISO;
  }
  try {
    return new Date(ms).toISOString();
  } catch {
    return EPOCH_ISO;
  }
}

/**
 * The native session listing carries no timestamps (strict DTO), so the
 * frozen `SessionInfo.createdAt/updatedAt` required fields are filled with
 * the epoch. Deterministic and documented, never a fabricated "now".
 */
export function sessionToUpstream(session: SessionSummary): Record<string, unknown> {
  return {
    id: session.id,
    title: session.title,
    createdAt: EPOCH_ISO,
    updatedAt: EPOCH_ISO,
  };
}

export function readyMessage(ctx: BridgeContext): WebviewBoundMessage {
  return {
    type: 'ready',
    extensionVersion: ctx.extensionVersion,
    workspaceDirectory: ctx.workspaceDirectory,
    serverInfo:
      typeof ctx.port === 'number' && Number.isInteger(ctx.port) && ctx.port > 0
        ? { port: ctx.port, version: ctx.daemonVersion ?? undefined }
        : undefined,
  };
}

export function connectionStateMessage(
  daemon: FaktorSnapshot['daemon'],
  detail?: string | null,
): WebviewBoundMessage {
  if (daemon === 'running') {
    return { type: 'connectionState', state: 'connected' };
  }
  if (daemon === 'starting') {
    return { type: 'connectionState', state: 'connecting' };
  }
  if (daemon === 'error') {
    return {
      type: 'connectionState',
      state: 'error',
      error: typeof detail === 'string' && detail.length > 0 ? detail : undefined,
    };
  }
  return { type: 'connectionState', state: 'disconnected' };
}

export function sessionsLoadedMessage(sessions: readonly SessionSummary[]): WebviewBoundMessage {
  return {
    type: 'sessionsLoaded',
    sessions: sessions.slice(0, BRIDGE_LIMITS.maxPageEntries).map(sessionToUpstream),
  };
}

export function sessionStatusMessage(
  sessionId: string,
  status: 'idle' | 'busy' | 'retry' | 'offline',
): WebviewBoundMessage {
  return { type: 'sessionStatus', sessionID: sessionId, status };
}

export function errorMessage(message: string, sessionId?: string | null): WebviewBoundMessage {
  return {
    type: 'error',
    message: message.length > 2000 ? `${message.slice(0, 2000)}…` : message,
    sessionID: sessionId ?? undefined,
  };
}

export function todoUpdatedMessage(
  sessionId: string,
  task: TaskSummary,
): WebviewBoundMessage {
  const items = [
    ...task.completed.map((content, index) => ({
      id: `milestone-completed-${index}`,
      content,
      status: 'completed' as const,
    })),
    ...task.open.map((content, index) => ({
      id: `milestone-open-${index}`,
      content,
      status: 'pending' as const,
    })),
  ].slice(0, BRIDGE_LIMITS.maxPageEntries);
  return { type: 'todoUpdated', sessionID: sessionId, items };
}

// ------------------------------------------- additive Faktor panel (outbound)

/**
 * The additive `faktor*` namespace consumed by the companion panel. These
 * messages are emitted AFTER the frozen Kilo messages and never replace or
 * reshape them; every field is derived from validated store state and
 * bounded by count and characters. `present:false` is an explicit empty
 * state, never a fabricated payload.
 */

function clampPanel(value: string, max = 240): string {
  return value.length > max ? `${value.slice(0, max)}…` : value;
}

function clampPanelLines(values: readonly string[], max: number): string[] {
  return values.slice(0, max).map((value) => clampPanel(value));
}

/**
 * Deterministic serialized-byte budget for one additive frame: later entries
 * are dropped once the budget is spent (never truncated silently mid-entry).
 */
function withinFrameBudget<T>(items: readonly T[], budget: number): T[] {
  const out: T[] = [];
  let left = budget;
  for (const item of items) {
    let cost: number;
    try {
      cost = Buffer.byteLength(JSON.stringify(item) ?? '', 'utf8') + 1;
    } catch {
      cost = budget;
    }
    if (cost > left) {
      break;
    }
    left -= cost;
    out.push(item);
  }
  return out;
}

export function faktorTaskStateMessage(snapshot: FaktorSnapshot): WebviewBoundMessage {
  const sessionID = snapshot.session?.id ?? null;
  const task = snapshot.task;
  if (task === null) {
    return { type: 'faktorTaskState', sessionID, present: false };
  }
  const cockpit = snapshot.cockpit;
  return {
    type: 'faktorTaskState',
    sessionID,
    present: true,
    goal: clampPanel(task.goal, 512),
    state: clampPanel(task.state, 64),
    phase: task.phase !== null ? clampPanel(task.phase, 96) : null,
    acceptanceCriteria: clampPanelLines(task.acceptanceCriteria, BRIDGE_LIMITS.maxFaktorCockpitLines),
    milestones: {
      completed: clampPanelLines(task.completed, BRIDGE_LIMITS.maxFaktorCockpitLines),
      open: clampPanelLines(task.open, BRIDGE_LIMITS.maxFaktorCockpitLines),
    },
    tests: {
      run: clampPanelLines(task.testsRun, 32),
      failed: clampPanelLines(task.testsFailed, 32),
    },
    changedFiles: clampPanelLines(task.changedFiles, BRIDGE_LIMITS.maxFaktorCockpitLines),
    blockers: clampPanelLines(task.blockers, BRIDGE_LIMITS.maxFaktorCockpitLines),
    verification: cockpit
      ? {
          status: clampPanel(cockpit.verification.status, 48),
          criteriaPassed: cockpit.verification.criteriaPassed,
          criteriaTotal: cockpit.verification.criteriaTotal,
          checksFailed: cockpit.verification.checksFailed,
          owed: cockpit.verification.owed,
          failedChecks: cockpit.verification.failedChecks,
        }
      : null,
    budget: task.budget
      ? {
          maxTokens: task.budget.maxTokens,
          spentTokens: task.budget.spentTokens,
          maxCostMicro: task.budget.maxCostMicro,
          spentCostMicro: task.budget.spentCostMicro,
          openReservedMicro: task.budget.openReservedMicro,
        }
      : null,
  };
}

export function faktorAgentsMessage(snapshot: FaktorSnapshot): WebviewBoundMessage {
  const sessionID = snapshot.session?.id ?? null;
  const agents = withinFrameBudget(
    (snapshot.agents ?? []).slice(0, BRIDGE_LIMITS.maxFaktorAgents).map((agent) => ({
    agentId: agent.agentId,
    kind: agent.kind,
    state: clampPanel(agent.state, 64),
    goal: clampPanel(agent.goal, 512),
    model: agent.model !== null ? clampPanel(agent.model, 128) : null,
    provider: agent.provider !== null ? clampPanel(agent.provider, 128) : null,
    reasoning: agent.reasoning,
    thinking: agent.thinking,
    itemId: agent.itemId !== null ? clampPanel(agent.itemId, 128) : null,
    itemKind: agent.itemKind !== null ? clampPanel(agent.itemKind, 64) : null,
    itemIds: agent.itemIds.slice(0, 16).map((id) => clampPanel(id, 128)),
    sessionId: agent.sessionId,
    worktreeId: agent.worktreeId,
    ownership: clampPanel(agent.ownership, 64),
    budget: agent.budget,
    capabilities: agent.capabilities.slice(0, 16).map((capability) => clampPanel(capability, 128)),
    progress: agent.progress,
    result: agent.result,
    blockers: clampPanelLines(agent.blockers, 16),
    presentation: agent.presentation,
    // Deterministic pixel identity: the full 5x5 sprite + colors, so the
    // panel never has to recompute or guess.
    pixel: {
      childId: agent.pixel.childId,
      state: agent.pixel.state,
      animation: agent.pixel.animation,
      avatar: {
        hash: agent.pixel.avatar.hash,
        color: agent.pixel.avatar.color,
        accent: agent.pixel.avatar.accent,
        pixels: agent.pixel.avatar.pixels.slice(0, 25),
        version: agent.pixel.avatar.version,
      },
    },
    })),
    BRIDGE_LIMITS.maxFaktorFrameBytes,
  );
  return {
    type: 'faktorAgents',
    sessionID,
    agents,
    background: agents.filter((agent) => agent.presentation === 'background').length,
  };
}

export function faktorCockpitMessage(snapshot: FaktorSnapshot): WebviewBoundMessage {
  const sessionID = snapshot.session?.id ?? null;
  const sections = withinFrameBudget(
    (snapshot.cockpitSections ?? []).slice(0, 16).map((section) => ({
    key: section.key,
    title: clampPanel(section.title, 96),
    present: section.present,
    lines: clampPanelLines(section.lines, BRIDGE_LIMITS.maxFaktorCockpitLines),
    evidence: section.evidence.slice(0, BRIDGE_LIMITS.maxFaktorEvidenceRefs).map((ref) => ({
      id: ref.id,
      label: clampPanel(ref.label, 240),
    })),
    actions:
      section.actions !== undefined
        ? section.actions.map((action) => ({
            key: action.key,
            label: clampPanel(action.label, 96),
            enabled: action.enabled,
          }))
        : [],
    })),
    BRIDGE_LIMITS.maxFaktorFrameBytes,
  );
  return { type: 'faktorCockpit', sessionID, present: sections.length > 0, sections };
}

export function faktorTournamentMessage(snapshot: FaktorSnapshot): WebviewBoundMessage {
  const sessionID = snapshot.session?.id ?? null;
  const tournament = snapshot.tournament ?? null;
  if (tournament === null) {
    return { type: 'faktorTournament', sessionID, present: false, tournament: null };
  }
  return {
    type: 'faktorTournament',
    sessionID,
    present: true,
    tournament: {
      id: clampPanel(tournament.id, 128),
      state: clampPanel(tournament.state, 48),
      open: tournament.open,
      canDecide: tournament.canDecide,
      winner: tournament.winner !== null ? clampPanel(tournament.winner, 128) : null,
      criteria: clampPanelLines(tournament.criteria, 64),
      candidates: withinFrameBudget(
        tournament.candidates.slice(0, 8).map((candidate) => ({
        childId: clampPanel(candidate.childId, 128),
        state: clampPanel(candidate.state, 48),
        verification: candidate.verification,
        verificationPass: candidate.verificationPass,
        reviewRank: candidate.reviewRank !== null ? clampPanel(candidate.reviewRank, 64) : null,
        reviewer: candidate.reviewer !== null ? clampPanel(candidate.reviewer, 128) : null,
        costMicro: candidate.costMicro,
        wallMs: candidate.wallMs,
        winner: candidate.winner,
        })),
        BRIDGE_LIMITS.maxFaktorFrameBytes,
      ),
    },
  };
}

/** Evidence references available for expansion (pre-expansion frame). */
export function faktorEvidenceRefsMessage(snapshot: FaktorSnapshot): WebviewBoundMessage {
  const sessionID = snapshot.session?.id ?? null;
  const refs: Array<{ readonly id: number | null; readonly label: string }> = [];
  const seen = new Set<string>();
  for (const section of snapshot.cockpitSections ?? []) {
    for (const ref of section.evidence) {
      const key = `${ref.id ?? 'text'}:${ref.label}`;
      if (!seen.has(key) && refs.length < BRIDGE_LIMITS.maxFaktorEvidenceRefs) {
        seen.add(key);
        refs.push({ id: ref.id, label: clampPanel(ref.label, 240) });
      }
    }
  }
  return { type: 'faktorEvidence', sessionID, mode: 'refs', refs, evidence: null };
}

/** One expanded evidence artifact; never bytes in the prompt, text only. */
export function faktorEvidenceExpandedMessage(
  sessionId: string | null,
  evidenceId: number,
  text: string,
  truncated: boolean,
): WebviewBoundMessage {
  const bounded =
    text.length > BRIDGE_LIMITS.maxFaktorEvidenceChars
      ? text.slice(0, BRIDGE_LIMITS.maxFaktorEvidenceChars)
      : text;
  return {
    type: 'faktorEvidence',
    sessionID: sessionId,
    mode: 'expanded',
    refs: [],
    evidence: {
      id: evidenceId,
      text: bounded,
      truncated: truncated || bounded.length < text.length,
      bytes: Buffer.byteLength(bounded, 'utf8'),
    },
  };
}

export function faktorBoardStateMessage(snapshot: FaktorSnapshot): WebviewBoundMessage {
  const sessionID = snapshot.session?.id ?? null;
  const board = snapshot.board ?? null;
  if (board === null) {
    return {
      type: 'faktorBoardState',
      sessionID,
      present: false,
      available: false,
      source: 'none',
      revision: null,
      unread: null,
      posts: [],
      reason: 'the serving daemon exposes no coordination-board read',
    };
  }
  return {
    type: 'faktorBoardState',
    sessionID,
    present: true,
    available: board.available,
    source: clampPanel(board.source, 64),
    revision: board.revision,
    unread: board.unread,
    posts: withinFrameBudget(
      board.posts.slice(0, BRIDGE_LIMITS.maxPageEntries).map((post) => ({
      id: clampPanel(post.id, 128),
      author: clampPanel(post.author, 128),
      subject: clampPanel(post.subject, BRIDGE_LIMITS.maxBoardSubjectBytes),
      body: clampPanel(post.body, BRIDGE_LIMITS.maxBoardBodyBytes),
      refs: post.refs.slice(0, BRIDGE_LIMITS.maxBoardRefs).map((ref) => clampPanel(ref, 512)),
      revision: post.revision,
      createdMs: post.createdMs,
      })),
      BRIDGE_LIMITS.maxFaktorFrameBytes,
    ),
    reason: board.reason !== null ? clampPanel(board.reason, 240) : null,
  };
}

// ------------------------------------------------------ transcript mapping

function toolState(tool: TranscriptTool): Record<string, unknown> {
  const input = isRecord(tool.input) ? tool.input : {};
  if (tool.state === 'completed' && (tool.exitCode === null || tool.exitCode === 0)) {
    return {
      status: 'completed',
      input,
      output: tool.excerpt ?? '',
      title: tool.name,
    };
  }
  if (tool.exitCode !== null && tool.exitCode !== 0) {
    return {
      status: 'error',
      input,
      error: tool.excerpt ?? `${tool.name} exited with ${tool.exitCode}`,
    };
  }
  if (tool.state === 'running' || tool.state === 'pending') {
    return { status: tool.state === 'pending' ? 'pending' : 'running', input };
  }
  return { status: 'pending', input };
}

export function messageFromEntry(
  entry: TranscriptEntry,
  sessionId: string,
): Record<string, unknown> {
  const parts: WebviewBoundMessage[] = [];
  if (entry.text.length > 0) {
    parts.push({ type: 'text', id: `${entry.id}:text`, sessionID: sessionId, messageID: entry.id, text: entry.text });
  }
  if (entry.reasoning.length > 0) {
    parts.push({
      type: 'reasoning',
      id: `${entry.id}:reasoning`,
      sessionID: sessionId,
      messageID: entry.id,
      text: entry.reasoning,
    });
  }
  let toolParts = 0;
  for (const tool of entry.tools) {
    if (toolParts >= BRIDGE_LIMITS.maxToolPartsPerMessage) {
      break;
    }
    toolParts += 1;
    parts.push({
      type: 'tool',
      id: `${entry.id}:tool:${tool.toolCallId}`,
      sessionID: sessionId,
      messageID: entry.id,
      tool: tool.name,
      callID: tool.toolCallId.length > 0 ? tool.toolCallId : undefined,
      state: toolState(tool),
    });
  }
  return {
    id: entry.id,
    sessionID: sessionId,
    role: entry.role === 'user' ? 'user' : 'assistant',
    content: entry.text,
    createdAt: isoFromMs(entry.createdMs),
    time: { created: entry.createdMs },
    parts,
  };
}

/** Newest-first trim so a huge transcript can never produce a huge payload. */
function boundEntries(
  entries: readonly TranscriptEntry[],
  sessionId: string,
): { messages: Record<string, unknown>[]; hasMore: boolean } {
  const slice = entries.slice(Math.max(0, entries.length - BRIDGE_LIMITS.maxPageEntries));
  const messages = slice.map((entry) => messageFromEntry(entry, sessionId));
  let hasMore = slice.length < entries.length;
  if (messages.length > 1) {
    const sizes = messages.map((message) => serializedBytes(message));
    let total = sizes.reduce((sum, size) => sum + size, 0);
    let drop = 0;
    while (drop < messages.length - 1 && total > BRIDGE_LIMITS.maxOutboundBytes) {
      total -= sizes[drop];
      drop += 1;
    }
    if (drop > 0) {
      messages.splice(0, drop);
      hasMore = true;
    }
  }
  return { messages, hasMore };
}

export function messagesLoadedMessage(
  sessionId: string,
  entries: readonly TranscriptEntry[],
): WebviewBoundMessage {
  const { messages, hasMore } = boundEntries(entries, sessionId);
  return {
    type: 'messagesLoaded',
    sessionID: sessionId,
    messages,
    mode: 'replace',
    hasMore,
  };
}

/**
 * Translate one Faktor snapshot into the upstream message batch. This is the
 * authoritative native -> frozen-UI mapping; every message is derived from
 * validated store state, never from raw daemon bytes.
 */
export function snapshotToWebviewMessages(snapshot: FaktorSnapshot): WebviewBoundMessage[] {
  const out: WebviewBoundMessage[] = [
    connectionStateMessage(snapshot.daemon, snapshot.daemonDetail),
  ];
  if (snapshot.sessions.length > 0) {
    out.push(sessionsLoadedMessage(snapshot.sessions));
  }
  const sessionId = snapshot.session?.id ?? null;
  if (sessionId !== null) {
    out.push(sessionStatusMessage(sessionId, snapshot.busy ? 'busy' : 'idle'));
    out.push(messagesLoadedMessage(sessionId, snapshot.transcript));
    if (snapshot.task) {
      out.push(todoUpdatedMessage(sessionId, snapshot.task));
    }
  }
  if (snapshot.lastError !== null) {
    out.push(errorMessage(snapshot.lastError, sessionId));
  }
  // Additive Faktor tail: appended, never interleaved with the frozen
  // message order above. Each frame is bounded and carries its own
  // explicit empty state.
  out.push(faktorTaskStateMessage(snapshot));
  out.push(faktorAgentsMessage(snapshot));
  out.push(faktorCockpitMessage(snapshot));
  out.push(faktorTournamentMessage(snapshot));
  out.push(faktorEvidenceRefsMessage(snapshot));
  out.push(faktorBoardStateMessage(snapshot));
  return out;
}

// --------------------------------------------------- SSE event translation

function asRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function ssePartToUpstream(part: Record<string, unknown>): WebviewBoundMessage | null {
  const type = typeof part.type === 'string' ? part.type : null;
  if (type === 'text' || type === 'reasoning') {
    const text = typeof part.text === 'string' ? part.text : '';
    if (text.length === 0) {
      return null;
    }
    return { type, text };
  }
  if (type === 'tool_call') {
    return {
      type: 'tool',
      tool: typeof part.name === 'string' ? part.name : '',
      callID: typeof part.tool_call_id === 'string' ? part.tool_call_id : undefined,
      state: { status: 'running', input: isRecord(part.input) ? part.input : {} },
    };
  }
  if (type === 'tool_result') {
    const result = asRecord(part.result) ?? {};
    const exitCode = typeof result.exit_code === 'number' ? result.exit_code : null;
    return {
      type: 'tool',
      tool: '',
      callID: typeof part.tool_call_id === 'string' ? part.tool_call_id : undefined,
      state:
        exitCode !== null && exitCode !== 0
          ? { status: 'error', input: {}, error: typeof result.excerpt === 'string' ? result.excerpt : '' }
          : { status: 'completed', input: {}, output: typeof result.excerpt === 'string' ? result.excerpt : '', title: '' },
    };
  }
  return null;
}

/**
 * Translate one SSE journal frame into upstream messages. Unmappable events
 * return an empty array (the snapshot path remains authoritative).
 */
export function nativeEventToWebviewMessages(
  event: string,
  data: unknown,
  sessionId: string,
): WebviewBoundMessage[] {
  const object = asRecord(data);
  if (!object) {
    return [];
  }
  if (event === 'message_created') {
    const message = asRecord(object.message);
    if (!message) {
      return [];
    }
    const id = typeof message.id === 'string' ? message.id : null;
    if (id === null || id.length === 0) {
      return [];
    }
    const parts: WebviewBoundMessage[] = [];
    const rawParts = Array.isArray(message.parts) ? message.parts : [];
    for (const rawPart of rawParts.slice(0, BRIDGE_LIMITS.maxToolPartsPerMessage)) {
      const part = asRecord(rawPart);
      if (!part) {
        continue;
      }
      const mapped = ssePartToUpstream(part);
      if (mapped) {
        parts.push({ ...mapped, id: `${id}:${parts.length}`, sessionID: sessionId, messageID: id });
      }
    }
    return [
      {
        type: 'messageCreated',
        message: {
          id,
          sessionID: sessionId,
          role: typeof message.role === 'string' ? message.role : 'assistant',
          parts,
          createdAt: EPOCH_ISO,
        },
      },
    ];
  }
  if (event === 'message_part_updated') {
    const messageId = typeof object.message_id === 'string' ? object.message_id : null;
    const part = asRecord(object.part);
    if (messageId === null || part === null) {
      return [];
    }
    const mapped = ssePartToUpstream(part);
    if (mapped === null) {
      return [];
    }
    const update: WebviewBoundMessage = {
      type: 'partUpdated',
      sessionID: sessionId,
      messageID: messageId,
      part: { ...mapped, id: `${messageId}:part`, sessionID: sessionId, messageID: messageId },
    };
    if (mapped.type === 'text' && typeof mapped.text === 'string') {
      return [{ ...update, delta: { type: 'text-delta', textDelta: mapped.text } }];
    }
    return [update];
  }
  return [];
}

// -------------------------------------------------------- vendored HTML

export interface VendoredHtmlOptions {
  readonly cspSource: string;
  readonly nonce: string;
  readonly scriptUri: string;
  readonly styleUri: string;
  readonly iconsBaseUri: string;
  readonly workerUri: string;
  readonly title: string;
  readonly sidebar?: 'left' | 'right' | '';
  readonly topBar?: boolean;
  readonly extraStyles?: string;
  /**
   * Additive Faktor companion overlay URIs (staged under
   * `dist/overlay/`). When both are present the shell captures the single
   * `acquireVsCodeApi` handle for the panel and loads the panel after the
   * vendored bundle. Without them the shell is byte-identical to the
   * frozen bootstrap (two nonce scripts, no panel).
   */
  readonly companionScriptUri?: string;
  readonly companionStyleUri?: string;
}

/**
 * Capture the webview's single `acquireVsCodeApi` handle so the companion
 * panel (loaded after the vendored bundle) can post strictly validated
 * `faktor*` actions without calling `acquireVsCodeApi` a second time (VS
 * Code refuses that). Purely additive: the vendored bundle still receives
 * the real handle from the first call.
 */
const COMPANION_API_BOOTSTRAP =
  '(function(){var real=window.acquireVsCodeApi;var api=null;' +
  'window.acquireVsCodeApi=function(){if(api===null&&typeof real==="function"){' +
  'api=real.apply(window,arguments);}return api;};' +
  'window.__faktorVsCodeApi=function(){return api;};})();';

function jsUri(uri: string): string {
  return JSON.stringify(uri).replace(/</g, '\\u003c');
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Strict CSP for the vendored bundle: local webview resources only, nonce
 * scripts, no remote origin, no inline event handlers. The Faktor daemon is
 * reached by the extension host, never by the webview, so connect-src never
 * needs a localhost port.
 */
export function vendoredCsp(cspSource: string, nonce: string): string {
  return [
    "default-src 'none'",
    `style-src 'unsafe-inline' ${cspSource}`,
    `script-src 'nonce-${nonce}' 'wasm-unsafe-eval'`,
    `worker-src ${cspSource} blob:`,
    `font-src ${cspSource}`,
    `img-src ${cspSource} data:`,
    `connect-src ${cspSource}`,
  ].join('; ');
}

/**
 * HTML shell for the vendored webview bundle, mirroring the upstream
 * bootstrap contract (`#root`, ICONS_BASE_URI, shiki worker globals) with a
 * nonce-only script policy.
 */
export function buildVendoredWebviewHtml(options: VendoredHtmlOptions): string {
  const markdownWorkerUri = options.workerUri.replace(
    /shiki-worker\.js$/,
    'markdown-shiki-worker.js',
  );
  const csp = vendoredCsp(options.cspSource, options.nonce);
  const topBar = options.topBar !== false;
  const extraStyles = options.extraStyles !== undefined ? `\n    ${options.extraStyles}` : '';
  const companion =
    options.companionScriptUri !== undefined && options.companionStyleUri !== undefined
      ? { script: options.companionScriptUri, style: options.companionStyleUri }
      : null;
  const companionStyle =
    companion !== null
      ? `\n  <link rel="stylesheet" href="${escapeHtml(companion.style)}">`
      : '';
  const bootstrapPrelude = companion !== null ? `${COMPANION_API_BOOTSTRAP}\n  ` : '';
  const companionScript =
    companion !== null
      ? `\n  <script nonce="${options.nonce}" src="${escapeHtml(companion.script)}"></script>`
      : '';
  return `<!DOCTYPE html>
<html lang="en" data-theme="kilo-vscode" data-sidebar="${options.sidebar ?? ''}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <link rel="stylesheet" href="${escapeHtml(options.styleUri)}">${companionStyle}
  <title>${escapeHtml(options.title)}</title>
  <style>
    html {
      scrollbar-color: auto;
    }
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
      overflow: hidden;
    }
    body {
      background-color: var(--vscode-sideBar-background, var(--vscode-editor-background));
      color: var(--vscode-foreground);
      font-family: var(--vscode-font-family);
    }
    #root {
      height: 100%;
    }${extraStyles}
  </style>
</head>
<body>
  <div id="root"></div>
  <script nonce="${options.nonce}">${bootstrapPrelude}window.ICONS_BASE_URI = ${jsUri(options.iconsBaseUri)}; window.KILO_SHIKI_WORKER_URI = ${jsUri(options.workerUri)}; window.KILO_MARKDOWN_SHIKI_WORKER_URI = ${jsUri(markdownWorkerUri)}; window.KILO_TOP_BAR = ${topBar}; window.KILO_TOP_BAR_SURFACE = "sidebar_title"; window.KILO_AGENT_MANAGER_SETTINGS = false;</script>
  <script nonce="${options.nonce}" src="${escapeHtml(options.scriptUri)}"></script>${companionScript}
</body>
</html>`;
}

// ------------------------------------------------------- bundle discovery

/**
 * A located vendored bundle. Every path is an absolute local file path inside
 * `root`; nothing here is derived from webview-provided input.
 */
export interface VendoredBundle {
  /** Bundle root (contains `dist/` and optionally `assets/`). */
  readonly root: string;
  /** `index.html` when the build emitted one, else the esbuild entry pair. */
  readonly entry: 'index.html' | 'esbuild';
  readonly script: string;
  readonly style: string;
  /** Worker bootstrap assets when the build emitted them. */
  readonly worker: string | null;
  readonly markdownWorker: string | null;
  /** Icon base directory when the bundle (or vendored tree) ships one. */
  readonly icons: string | null;
  /**
   * The additive Faktor companion overlay when the staged bundle carries it
   * (`dist/overlay/faktor-companion.{js,css}`). Null for a plain pinned
   * bundle: the shell then stays byte-identical to the frozen bootstrap.
   */
  readonly companion: { readonly script: string; readonly style: string } | null;
}

const REMOTE_REF = /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i;

/**
 * Resolve one dist-relative local asset reference from a built index.html.
 * Remote origins, absolute paths, traversal segments, backslashes and query
 * smuggling are all refused so a tampered index.html can never widen the
 * resource roots the extension serves.
 */
function localAssetFrom(distDir: string, ref: string | undefined): string | null {
  if (ref === undefined) {
    return null;
  }
  const trimmed = ref.trim();
  if (
    trimmed.length === 0 ||
    REMOTE_REF.test(trimmed) ||
    trimmed.startsWith('/') ||
    trimmed.includes('\\') ||
    trimmed.includes('\0')
  ) {
    return null;
  }
  const clean = trimmed.split(/[?#]/)[0];
  if (clean.length === 0 || clean.split('/').some((segment) => segment === '..')) {
    return null;
  }
  const abs = resolve(distDir, clean);
  const rel = relative(distDir, abs);
  if (rel.length === 0 || rel.startsWith('..') || isAbsolute(rel) || !existsSync(abs)) {
    return null;
  }
  return abs;
}

function attr(tag: string, name: string): string | undefined {
  const match = tag.match(new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s>]+))`, 'i'));
  return match?.[1] ?? match?.[2] ?? match?.[3];
}

/**
 * Extract the entry script/style from a built `dist/index.html`. Returns null
 * (refuse the whole bundle) if any script/link reference is remote, escapes
 * `dist/`, or does not exist: an upstream build never does that, a tampered
 * one must not be served.
 */
function parseIndexHtml(html: string, distDir: string): { script: string; style: string } | null {
  let script: string | null = null;
  for (const tag of html.match(/<script\b[^>]*>/gi) ?? []) {
    const src = attr(tag, 'src');
    if (src === undefined) {
      continue; // inline bootstrap scripts are not the bundle entry
    }
    const asset = localAssetFrom(distDir, src);
    if (asset === null) {
      return null;
    }
    if (script === null && /\.m?js$/i.test(asset)) {
      script = asset;
    }
  }
  let style: string | null = null;
  for (const tag of html.match(/<link\b[^>]*>/gi) ?? []) {
    if (!/rel\s*=\s*(?:"[^"]*stylesheet[^"]*"|'[^']*stylesheet[^']*'|stylesheet)/i.test(tag)) {
      continue;
    }
    const asset = localAssetFrom(distDir, attr(tag, 'href'));
    if (asset === null) {
      return null;
    }
    if (style === null && /\.css$/i.test(asset)) {
      style = asset;
    }
  }
  if (script === null || style === null) {
    return null;
  }
  return { script, style };
}

function existingFile(candidates: readonly string[]): string | null {
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

/**
 * Locate the built webview bundle under `root` (the pinned vendored tree or a
 * FAKTOR_UI_BUNDLE override). Prefers a built `dist/index.html` entry when
 * present and falls back to the upstream esbuild pair
 * (`dist/webview.js` + `dist/webview.css`); returns null when the bundle is
 * absent or its entry is not strictly local.
 */
export function locateVendoredBundle(root: string): VendoredBundle | null {
  const distDir = join(root, 'dist');
  if (!existsSync(distDir)) {
    return null;
  }
  let entry: VendoredBundle['entry'];
  let script: string;
  let style: string;
  const htmlPath = join(distDir, 'index.html');
  if (existsSync(htmlPath)) {
    let parsed: { script: string; style: string } | null;
    try {
      parsed = parseIndexHtml(readFileSync(htmlPath, 'utf8'), distDir);
    } catch {
      return null;
    }
    if (parsed === null) {
      return null;
    }
    entry = 'index.html';
    script = parsed.script;
    style = parsed.style;
  } else {
    entry = 'esbuild';
    script = join(distDir, 'webview.js');
    style = join(distDir, 'webview.css');
    if (!existsSync(script) || !existsSync(style)) {
      return null;
    }
  }
  const worker = existingFile([join(distDir, 'shiki-worker.js')]);
  const markdownWorker = existingFile([join(distDir, 'markdown-shiki-worker.js')]);
  const icons = existingFile([
    join(distDir, 'assets', 'icons'),
    join(root, 'assets', 'icons'),
    join(distDir, 'assets'),
  ]);
  // The Faktor companion overlay is strictly optional: only when BOTH the
  // script and the stylesheet staged locally is it served, and each path
  // went through the same strict local-asset resolution as the bundle.
  const companionScript = localAssetFrom(distDir, 'overlay/faktor-companion.js');
  const companionStyle = localAssetFrom(distDir, 'overlay/faktor-companion.css');
  const companion =
    companionScript !== null && companionStyle !== null
      ? { script: companionScript, style: companionStyle }
      : null;
  return { root, entry, script, style, worker, markdownWorker, icons, companion };
}

/** Recorded notice when the vendored bundle is absent and the fallback serves. */
export function vendoredFallbackNotice(root: string): string {
  return (
    `[faktor-webview] vendored bundle not found under ${join(root, 'dist')} ` +
    '(expected dist/index.html or dist/webview.js + dist/webview.css); ' +
    'using the built-in fallback panel'
  );
}
