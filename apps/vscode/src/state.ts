// Minimal observable state store for the chat panel plus the transcript
// reducer that turns native message pages and SSE event frames into the
// rendered conversation. Dependency-free (no vscode import) so
// scripts/selftest.mjs can drive it directly.
//
// "Bounded everything": the transcript keeps at most MAX_TRANSCRIPT_ENTRIES
// entries and MAX_ENTRY_CHARS characters per text field; agent summaries
// bound every free-form JSON blob; deltas past the cap are truncated, never
// accumulated forever.

import { foldPixelPresence, pixelPresence } from './pixelAgents.ts';
import type { PixelPresence } from './pixelAgents.ts';
import type { CockpitSection, CockpitTournamentView, CockpitView } from './cockpit';
import type { NativeBoardPage } from './nativeClient.ts';

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export const MAX_TRANSCRIPT_ENTRIES = 500;
export const MAX_ENTRY_CHARS = 256 * 1024;

/** Bound of one summary JSON blob (progress/result/capabilities). */
export const MAX_SUMMARY_JSON_CHARS = 8 * 1024;
export const MAX_SUMMARY_ARRAY = 64;
export const MAX_SUMMARY_KEYS = 64;
export const MAX_SUMMARY_STRING = 512;

export interface TranscriptTool {
  readonly toolCallId: string;
  readonly name: string;
  readonly state: string;
  readonly input: Json;
  readonly excerpt: string | null;
  readonly exitCode: number | null;
  readonly artifact: string | null;
}

export interface TranscriptEntry {
  readonly id: string;
  readonly role: string;
  readonly seq: number;
  readonly createdMs: number;
  readonly text: string;
  readonly reasoning: string;
  readonly summary: string;
  readonly tools: readonly TranscriptTool[];
}

export interface RunSummary {
  readonly taskId: string;
  readonly runId: string;
  readonly mode: string;
  readonly state: string;
  readonly goal: string | null;
  readonly model: string | null;
}

export interface AgentModelInfo {
  readonly provider: string;
  readonly model: string;
  readonly reasoning: boolean;
  readonly thinking: boolean;
  readonly tools: boolean;
}

/**
 * The full child-inspection summary: nothing native is collapsed away.
 * Identity (item/child/worktree/session), ownership, capabilities,
 * progress, result, budget, state, model AND the catalog-derived
 * provider/reasoning/thinking metadata, plus the deterministic pixel
 * avatar for the agent panel. Every free-form field is bounded.
 */
export interface AgentSummary {
  readonly agentId: string;
  readonly kind: string;
  readonly runId: string;
  readonly state: string;
  readonly goal: string;
  readonly model: string | null;
  readonly provider: string | null;
  readonly reasoning: boolean | null;
  readonly thinking: boolean | null;
  readonly itemId: string | null;
  readonly itemKind: string | null;
  readonly itemIds: readonly string[];
  readonly sessionId: number | null;
  readonly worktreeId: number | null;
  readonly ownership: string;
  readonly capabilities: readonly string[];
  readonly progress: Json;
  readonly result: Json;
  readonly budget: number | null;
  readonly blockers: readonly string[];
  /** Durable presentation/attention state; background children render
   * dimmed and tucked. Absent on the wire means foreground. */
  readonly presentation: 'foreground' | 'background';
  readonly pixel: import('./pixelAgents').PixelPresence;
}

export interface TaskPlanStep {
  readonly id: string;
  readonly summary: string;
  readonly state: string;
  readonly dependsOn: readonly string[];
}

export interface BudgetSummary {
  readonly maxTokens: number | null;
  readonly spentTokens: number | null;
  readonly maxCostMicro: number | null;
  readonly spentCostMicro: number;
  readonly openReservedMicro: number;
}

/** One completion-contract step as the UI renders it. */
export interface TaskCompletionStepSummary {
  readonly step: string;
  readonly status: string;
  readonly detail: string | null;
}

/**
 * The PR/CI-fix completion contract of the active task. `source` names the
 * provenance of the step rows:
 *   - `daemon`     — the daemon served the durable per-step rows;
 *   - `derived`    — projected from the durable task-run state and the
 *                    submitted contract (pending, or all-succeeded only
 *                    because the durable gate certified the task);
 *   - `unavailable`— the contract is known but the serving daemon exposes
 *                    no per-step read (terminal non-certified run).
 * Step statuses are NEVER fabricated: a missing read is reported as
 * unavailable, not as success.
 */
export interface TaskCompletionSummary {
  readonly includeCommit: boolean;
  readonly includePush: boolean;
  readonly includePr: boolean;
  readonly steps: readonly TaskCompletionStepSummary[];
  readonly source: 'daemon' | 'derived' | 'unavailable';
  readonly reason: string | null;
}

export interface TaskSummary {
  readonly goal: string;
  readonly state: string;
  readonly completed: readonly string[];
  readonly open: readonly string[];
  readonly testsRun: readonly string[];
  readonly testsFailed: readonly string[];
  readonly changedFiles: readonly string[];
  readonly budget: BudgetSummary | null;
  /** Additive native fields, surfaced when the daemon serves them. */
  readonly acceptanceCriteria: readonly string[];
  readonly plan: readonly TaskPlanStep[];
  readonly blockers: readonly string[];
  readonly evidenceRefs: readonly string[];
  readonly phase: string | null;
  readonly progress: Json;
  /** The Task-mode completion contract + its durable step statuses. */
  readonly completion: TaskCompletionSummary | null;
}


export interface VerificationSummary {
  readonly owed: readonly {
    readonly opId: string;
    readonly tool: string;
    readonly status: string;
    readonly effectStatus: string | null;
  }[];
  readonly failedChecks: readonly { readonly id: string; readonly detail: string }[];
}

export interface UsageSummary {
  readonly tokens: number;
  readonly spentMicro: number;
  readonly maxMicro: number | null;
  readonly openMicro: number;
  readonly truncated: boolean;
}

/**
 * One coordination-board post as the Faktor panel renders it. The board is
 * a run-family durable surface; when the daemon of this compatibility
 * revision serves no board route the snapshot carries `available: false`
 * with the explicit reason — never fabricated posts.
 */
export interface BoardPostSummary {
  readonly id: string;
  readonly author: string;
  readonly subject: string;
  readonly body: string;
  readonly refs: readonly string[];
  readonly revision: number | null;
  readonly createdMs: number | null;
}

export interface BoardStateSummary {
  readonly available: boolean;
  readonly source: string;
  readonly revision: number | null;
  readonly unread: number | null;
  readonly posts: readonly BoardPostSummary[];
  readonly reason: string | null;
}

/** Mirrors of the daemon/bridge board bounds (host-side re-validation). */
export const MAX_BOARD_SUBJECT_BYTES = 512;
export const MAX_BOARD_BODY_BYTES = 16 * 1024;
export const MAX_BOARD_REFS = 32;
export const MAX_BOARD_REF_BYTES = 1024;
export const MAX_BOARD_PAGE = 100;

/** The explicitly UNAVAILABLE board: no posts are ever fabricated. */
export function unavailableBoardState(reason: string, source = 'none'): BoardStateSummary {
  return {
    available: false,
    source,
    revision: null,
    unread: null,
    posts: [],
    reason: reason.length > 240 ? `${reason.slice(0, 240)}…` : reason,
  };
}

/**
 * Project one validated native board page onto the snapshot vocabulary.
 * `seenRevision` is the host's read watermark: posts with a strictly newer
 * revision are unread. The watermark only moves on an explicit read/post —
 * automatic refreshes never mark posts read. The returned `watermark` is
 * the board revision at page-read time (resets consume a revision too).
 */
export function boardStateFromPage(
  page: NativeBoardPage,
  seenRevision: number | null,
): { readonly board: BoardStateSummary; readonly watermark: number } {
  const seen = seenRevision ?? 0;
  let unread = 0;
  for (const post of page.posts) {
    if (post.revision > seen) {
      unread += 1;
    }
  }
  return {
    board: {
      available: true,
      source: 'native',
      revision: page.revision,
      unread,
      posts: page.posts.slice(0, MAX_BOARD_PAGE).map((post) => ({
        id: String(post.id),
        author: post.author_child === null ? 'root' : `child:${post.author_child}`,
        subject: boundText(post.subject, MAX_BOARD_SUBJECT_BYTES),
        body: boundText(post.body, MAX_BOARD_BODY_BYTES),
        refs: post.refs.slice(0, MAX_BOARD_REFS).map((ref) => boundText(ref, MAX_BOARD_REF_BYTES)),
        revision: post.revision,
        createdMs: post.created_ms,
      })),
      reason: null,
    },
    watermark: page.revision,
  };
}

function boundText(value: string, maxBytes: number): string {
  if (Buffer.byteLength(value, 'utf8') <= maxBytes) {
    return value;
  }
  return `${value.slice(0, maxBytes)}…`;
}

/** One bounded `boardRead` host request, or a typed refusal reason. */
export function parseBoardReadRequest(
  rawSince: unknown,
  rawLimit: unknown,
): { readonly since: number | null; readonly limit: number | null } | { readonly reason: string } {
  let since: number | null = null;
  if (rawSince !== undefined && rawSince !== null) {
    if (
      typeof rawSince !== 'number' ||
      !Number.isInteger(rawSince) ||
      rawSince <= 0 ||
      !Number.isSafeInteger(rawSince)
    ) {
      return { reason: 'boardRead.since must be a positive integer cursor' };
    }
    since = rawSince;
  }
  let limit: number | null = null;
  if (rawLimit !== undefined && rawLimit !== null) {
    if (
      typeof rawLimit !== 'number' ||
      !Number.isInteger(rawLimit) ||
      rawLimit <= 0 ||
      rawLimit > MAX_BOARD_PAGE
    ) {
      return { reason: `boardRead.limit must be an integer in 1..${MAX_BOARD_PAGE}` };
    }
    limit = rawLimit;
  }
  return { since, limit };
}

/**
 * One bounded `boardPost` host request, or a typed refusal reason. The
 * durable authority additionally requires a non-empty subject AND body;
 * refusing here keeps the failure local and explicit instead of a raw 400.
 */
export function parseBoardPostRequest(
  raw: { readonly subject?: unknown; readonly body?: unknown; readonly refs?: unknown },
): { readonly subject: string; readonly body: string; readonly refs: string[] } | { readonly reason: string } {
  const subject = typeof raw.subject === 'string' ? raw.subject.trim() : '';
  if (subject.length === 0) {
    return { reason: 'boardPost.subject must be a non-empty string' };
  }
  if (Buffer.byteLength(subject, 'utf8') > MAX_BOARD_SUBJECT_BYTES) {
    return { reason: `boardPost.subject exceeds ${MAX_BOARD_SUBJECT_BYTES} bytes` };
  }
  if (typeof raw.body !== 'string') {
    return { reason: 'boardPost.body must be a string' };
  }
  const body = raw.body;
  if (body.trim().length === 0) {
    return { reason: 'boardPost.body must be non-empty' };
  }
  if (Buffer.byteLength(body, 'utf8') > MAX_BOARD_BODY_BYTES) {
    return { reason: `boardPost.body exceeds ${MAX_BOARD_BODY_BYTES} bytes` };
  }
  const refs: string[] = [];
  if (raw.refs !== undefined && raw.refs !== null) {
    if (!Array.isArray(raw.refs)) {
      return { reason: 'boardPost.refs must be an array of strings' };
    }
    if (raw.refs.length > MAX_BOARD_REFS) {
      return { reason: `boardPost.refs exceeds ${MAX_BOARD_REFS} entries` };
    }
    for (let index = 0; index < raw.refs.length; index += 1) {
      const ref = raw.refs[index];
      if (typeof ref !== 'string' || ref.trim().length === 0) {
        return { reason: `boardPost.refs[${index}] must be a non-empty string` };
      }
      if (Buffer.byteLength(ref, 'utf8') > MAX_BOARD_REF_BYTES) {
        return { reason: `boardPost.refs[${index}] exceeds ${MAX_BOARD_REF_BYTES} bytes` };
      }
      refs.push(ref.trim());
    }
  }
  return { subject, body, refs };
}

export interface SessionSummary {
  readonly id: string;
  readonly title: string;
  readonly provider: string;
  readonly model: string;
  readonly state: string;
}

export type DaemonStatus = 'stopped' | 'starting' | 'running' | 'error';

/** Run states that are terminal: any other tag is still active. */
export const TERMINAL_RUN_STATES: readonly string[] = ['done', 'failed', 'cancelled'];

export function isTerminalRunState(state: string): boolean {
  return TERMINAL_RUN_STATES.includes(state.trim().toLowerCase());
}

/**
 * The active-run id after one refresh: the tracked id survives ONLY while
 * its run is present AND non-terminal. A terminal run (Done/Failed/
 * Cancelled) clears busy/activeRunId — mere presence in the listing is not
 * liveness.
 */
export function activeRunIdAfter(
  activeRunId: string | null,
  runs: readonly RunSummary[],
): string | null {
  if (activeRunId === null) {
    return null;
  }
  const tracked = runs.find((run) => run.runId === activeRunId);
  if (!tracked || isTerminalRunState(tracked.state)) {
    return null;
  }
  return activeRunId;
}

/**
 * The run a cancel action may target: the tracked run when it is still
 * non-terminal, otherwise the first non-terminal run, otherwise null.
 * Terminal runs are NEVER cancelled (the server would refuse with a typed
 * 409; the client must not even attempt it).
 */
export function cancelRunTarget(
  activeRunId: string | null,
  runs: readonly RunSummary[],
): string | null {
  if (activeRunId !== null) {
    const tracked = runs.find((run) => run.runId === activeRunId);
    if (tracked) {
      return isTerminalRunState(tracked.state) ? null : tracked.runId;
    }
  }
  const live = runs.find((run) => !isTerminalRunState(run.state));
  return live ? live.runId : null;
}


export interface FaktorSnapshot {
  readonly daemon: DaemonStatus;
  readonly daemonDetail: string;
  readonly baseUrl: string | null;
  readonly session: SessionSummary | null;
  readonly machineState: string;
  readonly machineLabel: string;
  readonly sessions: readonly SessionSummary[];
  readonly runs: readonly RunSummary[];
  readonly activeRunId: string | null;
  readonly agents: readonly AgentSummary[];
  readonly task: TaskSummary | null;
  readonly verification: VerificationSummary | null;
  readonly usage: UsageSummary | null;
  /** The persistent Task cockpit assembled from every native section. */
  readonly cockpit: CockpitView | null;
  readonly cockpitSections: readonly CockpitSection[];
  /** The durable tournament of the session (proposal only; never auto-merged). */
  readonly tournament: CockpitTournamentView | null;
  /** The run-family coordination board surface (explicitly unavailable when
   * the serving daemon exposes no board read). */
  readonly board: BoardStateSummary | null;
  readonly transcript: readonly TranscriptEntry[];
  readonly streamStatus: string;
  readonly lastError: string | null;
  readonly busy: boolean;
}

export function emptySnapshot(): FaktorSnapshot {
  return {
    daemon: 'stopped',
    daemonDetail: '',
    baseUrl: null,
    session: null,
    machineState: 'unknown',
    machineLabel: 'Daemon stopped',
    sessions: [],
    runs: [],
    activeRunId: null,
    agents: [],
    task: null,
    verification: null,
    usage: null,
    cockpit: null,
    cockpitSections: [],
    tournament: null,
    board: null,
    transcript: [],
    streamStatus: 'stopped',
    lastError: null,
    busy: false,
  };
}

export type StoreListener = (snapshot: FaktorSnapshot) => void;

export class FaktorStore {
  private state: FaktorSnapshot = emptySnapshot();
  private readonly listeners = new Set<StoreListener>();

  subscribe(listener: StoreListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  snapshot(): FaktorSnapshot {
    return this.state;
  }

  patch(patch: Partial<FaktorSnapshot>): void {
    let changed = false;
    const next: Record<string, unknown> = { ...this.state };
    for (const [key, value] of Object.entries(patch)) {
      if (value !== undefined && next[key] !== value) {
        next[key] = value;
        changed = true;
      }
    }
    if (!changed) {
      return;
    }
    const previous = this.state;
    const nextState = next as unknown as FaktorSnapshot;
    // Additive Faktor-only panels never bleed across a session switch or a
    // daemon stop: they describe ONE session/run family. The host patches a
    // fixed field set, so the store owns this invariant (extension.ts is
    // deliberately not coupled to the panel fields).
    const sessionChanged =
      (nextState.session?.id ?? null) !== (previous.session?.id ?? null);
    const stopped = nextState.daemon === 'stopped' && previous.daemon !== 'stopped';
    if (sessionChanged || stopped) {
      next.tournament = null;
      next.board = null;
    }
    this.state = nextState;
    this.emit();
  }

  private emit(): void {
    for (const listener of [...this.listeners]) {
      listener(this.state);
    }
  }
}

// ------------------------------------------------------- transcript reducer

function clampText(existing: string, addition: string): string {
  if (existing.length >= MAX_ENTRY_CHARS) {
    return existing;
  }
  return (existing + addition).slice(0, MAX_ENTRY_CHARS);
}

function emptyEntry(id: string, role: string, seq: number, createdMs: number): TranscriptEntry {
  return { id, role, seq, createdMs, text: '', reasoning: '', summary: '', tools: [] };
}

function asObject(value: Json | undefined): Record<string, Json> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null;
  }
  return value;
}

function asString(value: Json | undefined): string | null {
  return typeof value === 'string' ? value : null;
}

function asInt(value: Json | undefined): number | null {
  return typeof value === 'number' && Number.isInteger(value) ? value : null;
}

/** Message ids are strings on SSE frames and integers on durable pages. */
function asId(value: Json | undefined): string | null {
  if (typeof value === 'string' && value.length > 0) {
    return value;
  }
  if (typeof value === 'number' && Number.isInteger(value)) {
    return String(value);
  }
  return null;
}

/** Normalize an SSE `Part` (`{type: ...}`) into a tool update, or null. */
function toolFromSsePart(part: Record<string, Json>): Partial<TranscriptTool> | null {
  const type = asString(part.type);
  if (type === 'tool_call') {
    return {
      toolCallId: asString(part.tool_call_id) ?? '',
      name: asString(part.name) ?? '',
      state: asString(part.state) ?? 'unknown',
      input: part.input ?? null,
    };
  }
  if (type === 'tool_result') {
    const result = asObject(part.result);
    return {
      toolCallId: asString(part.tool_call_id) ?? '',
      excerpt: result ? asString(result.excerpt) : null,
      exitCode: result ? asInt(result.exit_code) : null,
      artifact: result ? asString(result.artifact) : null,
    };
  }
  return null;
}

/** Normalize a durable part row (`{kind, data}` with a flat payload). */
function toolFromDurablePart(
  kind: string,
  payload: Record<string, Json>,
): Partial<TranscriptTool> | null {
  if (kind === 'tool_call') {
    return {
      toolCallId: asString(payload.tool_call_id) ?? '',
      name: asString(payload.name) ?? '',
      state: asString(payload.state) ?? 'unknown',
      input: payload.input ?? null,
    };
  }
  if (kind === 'tool_result') {
    return {
      toolCallId: asString(payload.tool_call_id) ?? '',
      excerpt: asString(payload.excerpt),
      exitCode: asInt(payload.exit_code),
      artifact: asString(payload.artifact),
    };
  }
  return null;
}

function mergeTool(
  tools: readonly TranscriptTool[],
  update: Partial<TranscriptTool>,
): TranscriptTool[] {
  const id = update.toolCallId ?? '';
  const existing = tools.find((tool) => tool.toolCallId === id);
  if (!existing) {
    const created: TranscriptTool = {
      toolCallId: id,
      name: update.name ?? '',
      state: update.state ?? 'running',
      input: update.input ?? null,
      excerpt: update.excerpt ?? null,
      exitCode: update.exitCode ?? null,
      artifact: update.artifact ?? null,
    };
    return [...tools, created];
  }
  const merged: TranscriptTool = {
    toolCallId: existing.toolCallId,
    name: update.name ?? existing.name,
    state: update.state ?? existing.state,
    input: update.input !== undefined ? update.input : existing.input,
    excerpt: update.excerpt !== undefined ? update.excerpt : existing.excerpt,
    exitCode: update.exitCode !== undefined ? update.exitCode : existing.exitCode,
    artifact: update.artifact !== undefined ? update.artifact : existing.artifact,
  };
  return tools.map((tool) => (tool.toolCallId === id ? merged : tool));
}

function applyTextPart(
  entry: TranscriptEntry,
  kind: string,
  text: string,
): TranscriptEntry {
  if (kind === 'reasoning') {
    return { ...entry, reasoning: clampText(entry.reasoning, text) };
  }
  if (kind === 'summary') {
    return { ...entry, summary: clampText(entry.summary, text) };
  }
  return { ...entry, text: clampText(entry.text, text) };
}

function entryFromMessage(id: string, message: Record<string, Json>): TranscriptEntry {
  let entry = emptyEntry(
    id,
    asString(message.role) ?? 'assistant',
    asInt(message.seq) ?? 0,
    asInt(message.created_ms) ?? 0,
  );
  // User prompt rows carry their text in the message payload (`data.text`,
  // e.g. `{files, text}`) with an empty parts list; assistant rows carry it
  // in parts. Both must render (parts win when both exist).
  const data = asObject(message.data);
  const dataText = data ? asString(data.text) : null;
  const parts = Array.isArray(message.parts) ? (message.parts as Json[]) : [];
  for (const rawPart of parts) {
    const part = asObject(rawPart);
    if (!part) {
      continue;
    }
    const type = asString(part.type) ?? asString(part.kind);
    // Durable part rows carry the flat payload under `data`
    // (`{kind: "text", data: {text}}`); SSE parts carry it inline.
    const payload = asObject(part.data);
    if (type === 'text' || type === 'reasoning' || type === 'summary') {
      const text = asString(part.text) ?? (payload ? asString(payload.text) : null) ?? '';
      if (text.length > 0) {
        entry = applyTextPart(entry, type, text);
      }
      continue;
    }
    const update =
      asString(part.type) !== null
        ? toolFromSsePart(part)
        : toolFromDurablePart(type ?? '', payload ?? part);
    if (update) {
      entry = { ...entry, tools: mergeTool(entry.tools, update) };
    }
  }
  if (entry.text.length === 0 && dataText !== null && dataText.length > 0) {
    entry = { ...entry, text: clampText(entry.text, dataText) };
  }
  return entry;
}

function upsertEntry(
  entries: readonly TranscriptEntry[],
  id: string,
  mutator: (entry: TranscriptEntry) => TranscriptEntry,
  fallback: () => TranscriptEntry,
): TranscriptEntry[] {
  const index = entries.findIndex((entry) => entry.id === id);
  if (index >= 0) {
    const current = entries[index] as TranscriptEntry;
    const updated = mutator(current);
    if (updated === current) {
      return entries as TranscriptEntry[];
    }
    const next = [...entries];
    next[index] = updated;
    return next;
  }
  return boundTranscript([...entries, mutator(fallback())]);
}

function boundTranscript(entries: TranscriptEntry[]): TranscriptEntry[] {
  if (entries.length <= MAX_TRANSCRIPT_ENTRIES) {
    return entries;
  }
  return entries.slice(entries.length - MAX_TRANSCRIPT_ENTRIES);
}

/** Rebuild the transcript from a newest-first native message page. */
export function transcriptFromMessages(messages: readonly Json[]): TranscriptEntry[] {
  const entries: TranscriptEntry[] = [];
  for (const raw of [...messages].reverse()) {
    const message = asObject(raw);
    if (!message) {
      continue;
    }
    const id = asId(message.id);
    if (!id) {
      continue;
    }
    entries.push(entryFromMessage(id, message));
  }
  return boundTranscript(entries);
}

/** Apply one SSE frame to the transcript; returns the same array if unused. */
export function applySseEvent(
  entries: readonly TranscriptEntry[],
  event: string,
  data: Json,
): TranscriptEntry[] {
  const object = asObject(data);
  if (!object) {
    return entries as TranscriptEntry[];
  }
  if (event === 'message_created') {
    const message = asObject(object.message);
    if (!message) {
      return entries as TranscriptEntry[];
    }
    const id = asId(message.id);
    if (!id) {
      return entries as TranscriptEntry[];
    }
    return upsertEntry(entries, id, () => entryFromMessage(id, message), () => emptyEntry(id, 'assistant', 0, 0));
  }
  if (event === 'message_part_updated') {
    const id = asString(object.message_id);
    if (!id) {
      return entries as TranscriptEntry[];
    }
    const part = asObject(object.part);
    if (!part) {
      return entries as TranscriptEntry[];
    }
    const type = asString(part.type);
    return upsertEntry(
      entries,
      id,
      (entry) => {
        if (type === 'text' || type === 'reasoning' || type === 'summary') {
          const text = asString(part.text) ?? '';
          return text.length === 0 ? entry : applyTextPart(entry, type, text);
        }
        const update = toolFromSsePart(part);
        return update ? { ...entry, tools: mergeTool(entry.tools, update) } : entry;
      },
      () => emptyEntry(id, 'assistant', 0, 0),
    );
  }
  if (event === 'tool_call_state') {
    const toolCallId = asString(object.tool_call_id);
    const state = asString(object.state);
    if (!toolCallId || !state) {
      return entries as TranscriptEntry[];
    }
    let changed = false;
    const next = entries.map((entry) => {
      if (!entry.tools.some((tool) => tool.toolCallId === toolCallId)) {
        return entry;
      }
      changed = true;
      return {
        ...entry,
        tools: entry.tools.map((tool) =>
          tool.toolCallId === toolCallId ? { ...tool, state } : tool,
        ),
      };
    });
    return changed ? (next as TranscriptEntry[]) : (entries as TranscriptEntry[]);
  }
  return entries as TranscriptEntry[];
}

/** Recursively bound a JSON value (depth, array/keys, string lengths). */
export function boundJson(value: Json, depth = 0): Json {
  if (value === null || typeof value === 'boolean' || typeof value === 'number') {
    return value;
  }
  if (typeof value === 'string') {
    return value.length > MAX_SUMMARY_STRING ? `${value.slice(0, MAX_SUMMARY_STRING)}…` : value;
  }
  if (depth >= 6) {
    return '[…]';
  }
  if (Array.isArray(value)) {
    return value.slice(0, MAX_SUMMARY_ARRAY).map((entry) => boundJson(entry, depth + 1));
  }
  const out: Record<string, Json> = {};
  let keys = 0;
  for (const [key, entry] of Object.entries(value)) {
    if (keys >= MAX_SUMMARY_KEYS) {
      out['…'] = `${Object.keys(value).length - keys} more`;
      break;
    }
    out[key] = boundJson(entry as Json, depth + 1);
    keys += 1;
  }
  return out;
}

function boundedJsonText(value: Json): string {
  try {
    const encoded = JSON.stringify(boundJson(value));
    if (encoded === undefined) {
      return 'null';
    }
    return encoded.length > MAX_SUMMARY_JSON_CHARS
      ? `${encoded.slice(0, MAX_SUMMARY_JSON_CHARS)}…`
      : encoded;
  } catch {
    return '[unserializable]';
  }
}

/** One capability entry as a bounded, renderable string. */
function capabilityOf(value: Json): string {
  if (typeof value === 'string') {
    return value.length > MAX_SUMMARY_STRING ? `${value.slice(0, MAX_SUMMARY_STRING)}…` : value;
  }
  return boundedJsonText(value);
}

/** Blockers carried by an agent entry, from the additive fields. */
function agentBlockers(agent: Record<string, Json>): string[] {
  const out: string[] = [];
  const push = (value: Json | undefined): void => {
    if (typeof value === 'string' && value.trim().length > 0) {
      out.push(value.trim().slice(0, MAX_SUMMARY_STRING));
    }
  };
  const pushObject = (value: Json | undefined): void => {
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      const record = value as Record<string, Json>;
      push(record.reason);
      push(record.detail);
      push(record.message);
      push(record.summary);
      const kind = typeof record.kind === 'string' ? record.kind : null;
      const resolution = typeof record.resolution === 'string' ? record.resolution : null;
      if (kind !== null && out.length > 0 && resolution !== null) {
        out[out.length - 1] = `${kind}: ${out[out.length - 1]} (${resolution})`;
      }
    }
  };
  pushObject(agent.blocker);
  push(agent.blocker);
  const blockers = agent.blockers;
  if (Array.isArray(blockers)) {
    for (const entry of blockers.slice(0, 16)) {
      if (typeof entry === 'string') {
        push(entry);
      } else {
        pushObject(entry);
      }
    }
  } else {
    pushObject(blockers);
  }
  const progress = agent.progress;
  if (typeof progress === 'object' && progress !== null && !Array.isArray(progress)) {
    const record = progress as Record<string, Json>;
    const progressBlockers = record.blockers ?? record.blocked_on ?? record.blocker;
    if (Array.isArray(progressBlockers)) {
      for (const entry of progressBlockers.slice(0, 16)) {
        if (typeof entry === 'string') {
          push(entry);
        } else {
          pushObject(entry);
        }
      }
    } else {
      pushObject(progressBlockers);
      push(progressBlockers);
    }
  }
  return out.slice(0, 16);
}

/**
 * Fold one native agent frame into the persistent per-ChildId presence map:
 * existing presences keep their identity (deterministic avatars), current
 * frames update state, and transiently missing children are retained.
 */
export function nextPixelPresence(
  previous: ReadonlyMap<string, PixelPresence>,
  agents: readonly Json[],
): Map<string, PixelPresence> {
  const frame: Array<{ agentId: string; state: string }> = [];
  for (const raw of agents) {
    const agent = asObject(raw);
    if (!agent) {
      continue;
    }
    const agentId = asString(agent.agent_id);
    if (agentId) {
      frame.push({ agentId, state: asString(agent.state) ?? 'unknown' });
    }
  }
  return foldPixelPresence(previous, frame);
}

/**
 * Human-readable summary of one durable agent listing entry, with every
 * native child field surfaced (bounded) plus catalog-derived model
 * metadata and the deterministic pixel presence. `presence` (when given)
 * is the persistent per-ChildId map from previous frames.
 */
export function summarizeAgents(
  agents: readonly Json[],
  catalog: readonly AgentModelInfo[] = [],
  presence?: ReadonlyMap<string, PixelPresence>,
): AgentSummary[] {
  const folded = presence !== undefined ? nextPixelPresence(presence, agents) : null;
  const out: AgentSummary[] = [];
  for (const raw of agents) {
    const agent = asObject(raw);
    if (!agent) {
      continue;
    }
    const agentId = asString(agent.agent_id);
    if (!agentId) {
      continue;
    }
    const state = asString(agent.state) ?? 'unknown';
    const model = asString(agent.model);
    // The child session's durable provider rides the wire; the
    // (provider, model) pair is the ONLY safe catalog join key because two
    // providers may expose the same model id with different capabilities.
    // A provider-less entry never guesses by model alone.
    const provider = asString(agent.provider);
    const info =
      provider !== null && model !== null
        ? catalog.find((entry) => entry.provider === provider && entry.model === model)
        : undefined;
    const itemIdsRaw = Array.isArray(agent.item_ids) ? (agent.item_ids as Json[]) : [];
    const capabilitiesRaw = Array.isArray(agent.capabilities) ? (agent.capabilities as Json[]) : [];
    const progress = agent.progress ?? null;
    const result = agent.result ?? null;
    out.push({
      agentId,
      kind: asString(agent.kind) ?? 'child',
      runId: asString(agent.run_id) ?? '',
      state,
      goal: (asString(agent.goal) ?? '').slice(0, MAX_ENTRY_CHARS),
      model,
      provider: provider ?? null,
      reasoning: info?.reasoning ?? null,
      thinking: info?.thinking ?? null,
      itemId: asString(agent.item_id),
      itemKind: asString(agent.item_kind),
      itemIds: itemIdsRaw
        .filter((entry): entry is string => typeof entry === 'string')
        .slice(0, MAX_SUMMARY_ARRAY),
      sessionId: asInt(agent.session_id),
      worktreeId: asInt(agent.worktree_id),
      ownership: asString(agent.ownership) ?? 'unknown',
      capabilities: capabilitiesRaw.slice(0, MAX_SUMMARY_ARRAY).map(capabilityOf),
      progress: boundJson(progress),
      result: boundJson(result),
      budget: asInt(agent.budget),
      blockers: agentBlockers(agent),
      presentation: agent.presentation === 'background' ? 'background' : 'foreground',
      pixel: folded?.get(agentId) ?? pixelPresence(agentId, state),
    });
  }
  return out;
}

