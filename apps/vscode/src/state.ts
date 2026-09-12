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
import type { CockpitSection, CockpitView } from './cockpit';

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
    this.state = next as unknown as FaktorSnapshot;
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
    const info = model !== null ? catalog.find((entry) => entry.model === model) : undefined;
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
      provider: info?.provider ?? null,
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
      pixel: folded?.get(agentId) ?? pixelPresence(agentId, state),
    });
  }
  return out;
}

