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
// The module is dependency-free (no vscode, no vendored imports) so
// scripts/selftest.mjs can drive every accept/reject path, and it contains no
// Faktor state of its own: it is a pure translation layer.

import { existsSync, readFileSync } from 'node:fs';
import { isAbsolute, join, relative, resolve } from 'node:path';

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
} as const;

const SUPPORTED_INBOUND: ReadonlySet<string> = new Set([
  'webviewReady',
  'sendMessage',
  'abort',
  'createSession',
  'loadSessions',
  'loadMessages',
  'openExternal',
]);

const MESSAGE_LOAD_MODES: ReadonlySet<string> = new Set([
  'replace',
  'prepend',
  'focus',
  'reconcile',
]);

export type BridgeCommand =
  | { readonly kind: 'ready' }
  | { readonly kind: 'sendMessage'; readonly text: string; readonly sessionId: string | null }
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

// --------------------------------------------------------------- inbound ABI

/**
 * Validate one message from the frozen UI. Returns a typed command, or a
 * `BridgeDrop` that the caller must log (never forward).
 */
export function ingestWebviewMessage(raw: unknown): BridgeIngest {
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
    if (isNonEmptyStructured(message.files)) {
      return drop(rawType, 'file attachments are not supported by the native bridge', bytes);
    }
    if (isNonEmptyStructured(message.review)) {
      return drop(rawType, 'review comments are not supported by the native bridge', bytes);
    }
    if (isNonEmptyStructured(message.agentManagerContext)) {
      return drop(rawType, 'agent-manager context is not supported by the native bridge', bytes);
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
    return { kind: 'sendMessage', text: message.text, sessionId };
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
      return { type: 'sendGoal', goal: command.text };
    case 'abort':
      return { type: 'cancelRun' };
    case 'createSession':
      return { type: 'refresh' };
    case 'loadSessions':
      return { type: 'refresh' };
    case 'loadMessages':
      return { type: 'refresh' };
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
}

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
  return `<!DOCTYPE html>
<html lang="en" data-theme="kilo-vscode" data-sidebar="${options.sidebar ?? ''}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <link rel="stylesheet" href="${escapeHtml(options.styleUri)}">
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
  <script nonce="${options.nonce}">window.ICONS_BASE_URI = ${jsUri(options.iconsBaseUri)}; window.KILO_SHIKI_WORKER_URI = ${jsUri(options.workerUri)}; window.KILO_MARKDOWN_SHIKI_WORKER_URI = ${jsUri(markdownWorkerUri)}; window.KILO_TOP_BAR = ${topBar}; window.KILO_TOP_BAR_SURFACE = "sidebar_title"; window.KILO_AGENT_MANAGER_SETTINGS = false;</script>
  <script nonce="${options.nonce}" src="${escapeHtml(options.scriptUri)}"></script>
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
  return { root, entry, script, style, worker, markdownWorker, icons };
}

/** Recorded notice when the vendored bundle is absent and the fallback serves. */
export function vendoredFallbackNotice(root: string): string {
  return (
    `[faktor-webview] vendored bundle not found under ${join(root, 'dist')} ` +
    '(expected dist/index.html or dist/webview.js + dist/webview.css); ' +
    'using the built-in fallback panel'
  );
}
