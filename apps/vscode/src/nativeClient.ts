// Typed client of the Faktor Native Protocol v1 (docs/native-protocol.md)
// plus the minimal SDK-shaped session surface (`/session/create`,
// `/session/list`) the native surface does not duplicate.
//
// Design rules:
//   - Dependency-free: no axios, no vscode import, no DOM lib. The fetch
//     implementation is injectable (`FetchLike`) so scripts/selftest.mjs can
//     drive every parse/reject path with a fake.
//   - Strict runtime validation of RESPONSES: every known field is checked
//     for presence and exact type. Unknown fields are IGNORED, per the v1
//     additive contract (docs/native-protocol.md): a newer daemon may add
//     optional fields and that must never break an older client. Requests
//     keep strict rejection — the daemon's `deny_unknown_fields` DTOs turn
//     any unknown request field into a loud 400; the client never widens or
//     rewrites a request body.
//   - Bounded bodies: responses are read through a streaming byte cap; an
//     oversized body is cancelled and rejected, never buffered unbounded.
//   - Bounded time: every request carries an abort-based timeout.

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export interface StreamReaderLike {
  read(): Promise<{ done: boolean; value?: Uint8Array }>;
  cancel(reason?: unknown): Promise<void>;
}

/** The structural subset of `fetch`'s Response the client uses. */
export interface ResponseLike {
  readonly status: number;
  readonly ok: boolean;
  readonly headers?: { get(name: string): string | null } | null;
  readonly body?: { getReader(): StreamReaderLike } | null;
  arrayBuffer(): Promise<ArrayBuffer>;
  text(): Promise<string>;
}

export interface FetchInitLike {
  method: string;
  headers: Record<string, string>;
  body?: string;
  signal?: AbortSignal;
}

export type FetchLike = (url: string, init: FetchInitLike) => Promise<ResponseLike>;

export const DEFAULT_TIMEOUT_MS = 15_000;
export const DEFAULT_MAX_BODY_BYTES = 4 * 1024 * 1024;
export const ERROR_BODY_BYTES = 64 * 1024;
export const EVIDENCE_MAX_BODY_BYTES = 32 * 1024 * 1024;

export interface NativeClientOptions {
  readonly baseUrl: string;
  /** Native auth form: `Authorization: Bearer <password>`. */
  readonly bearerToken: string;
  readonly fetch?: FetchLike;
  readonly timeoutMs?: number;
  readonly maxBodyBytes?: number;
}

export class NativeProtocolError extends Error {
  readonly path: string;
  readonly detail: string;

  constructor(path: string, detail: string) {
    super(`native protocol violation at ${path}: ${detail}`);
    this.name = 'NativeProtocolError';
    this.path = path;
    this.detail = detail;
  }
}

export class NativeApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;

  constructor(status: number, code: string, message: string, retryable: boolean) {
    super(`native API error ${status} ${code}: ${message}`);
    this.name = 'NativeApiError';
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

// ---------------------------------------------------------------- validation

type JsonObject = { [key: string]: Json };

function describe(value: Json): string {
  if (value === null) {
    return 'null';
  }
  if (Array.isArray(value)) {
    return 'an array';
  }
  return typeof value;
}

function fail(path: string, detail: string): never {
  throw new NativeProtocolError(path, detail);
}

export function isJsonObject(value: Json): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asObject(value: Json, path: string): JsonObject {
  if (!isJsonObject(value)) {
    fail(path, `expected an object, got ${describe(value)}`);
  }
  return value;
}

// Response validation: required fields must be present (their types are
// checked by the f* helpers at the call site); unknown fields are ignored.
// A newer daemon adding an optional field must never break a v1 client
// (docs/native-protocol.md additive contract). This helper is used ONLY by
// response validators — request bodies are passed through verbatim and the
// daemon's strict `deny_unknown_fields` DTOs reject unknown request fields
// with a loud 400.
function checkResponseKeys(object: JsonObject, path: string, required: readonly string[]): void {
  for (const key of required) {
    if (!Object.prototype.hasOwnProperty.call(object, key)) {
      fail(path, `missing required field ${key}`);
    }
  }
}

function field(object: JsonObject, key: string, path: string): Json {
  if (!Object.prototype.hasOwnProperty.call(object, key)) {
    fail(path, `missing required field ${key}`);
  }
  return object[key] as Json;
}

function fString(object: JsonObject, key: string, path: string): string {
  const value = field(object, key, path);
  if (typeof value !== 'string') {
    fail(`${path}.${key}`, `expected a string, got ${describe(value)}`);
  }
  return value;
}

function fNumber(object: JsonObject, key: string, path: string): number {
  const value = field(object, key, path);
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail(`${path}.${key}`, `expected a finite number, got ${describe(value)}`);
  }
  return value;
}

function fInt(object: JsonObject, key: string, path: string): number {
  const value = fNumber(object, key, path);
  if (!Number.isInteger(value)) {
    fail(`${path}.${key}`, `expected an integer, got ${value}`);
  }
  return value;
}

function fBool(object: JsonObject, key: string, path: string): boolean {
  const value = field(object, key, path);
  if (typeof value !== 'boolean') {
    fail(`${path}.${key}`, `expected a boolean, got ${describe(value)}`);
  }
  return value;
}

function fArray(object: JsonObject, key: string, path: string): Json[] {
  const value = field(object, key, path);
  if (!Array.isArray(value)) {
    fail(`${path}.${key}`, `expected an array, got ${describe(value)}`);
  }
  return value;
}

function fNullableString(object: JsonObject, key: string, path: string): string | null {
  const value = field(object, key, path);
  if (value === null) {
    return null;
  }
  if (typeof value !== 'string') {
    fail(`${path}.${key}`, `expected a string or null, got ${describe(value)}`);
  }
  return value;
}

function fNullableInt(object: JsonObject, key: string, path: string): number | null {
  const value = field(object, key, path);
  if (value === null) {
    return null;
  }
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    fail(`${path}.${key}`, `expected an integer or null, got ${describe(value)}`);
  }
  return value;
}

function fNullableObject(object: JsonObject, key: string, path: string): JsonObject | null {
  const value = field(object, key, path);
  if (value === null) {
    return null;
  }
  return asObject(value, `${path}.${key}`);
}

function fJson(object: JsonObject, key: string, path: string): Json {
  return field(object, key, path);
}

function fStringArray(object: JsonObject, key: string, path: string): string[] {
  return fArray(object, key, path).map((entry, index) => {
    if (typeof entry !== 'string') {
      fail(`${path}.${key}[${index}]`, `expected a string, got ${describe(entry)}`);
    }
    return entry;
  });
}

function fObjectArray(object: JsonObject, key: string, path: string): JsonObject[] {
  return fArray(object, key, path).map((entry, index) =>
    asObject(entry, `${path}.${key}[${index}]`),
  );
}

// -------------------------------------------------------------- result types

export interface NativeHealth {
  readonly ok: boolean;
  readonly version: string;
}

export interface NativeReady {
  readonly ready: boolean;
}

export interface NativeSessionCreated {
  readonly id: string;
  readonly title: string;
  /** Wire field name (snake_case by contract). */
  readonly created_ms: number;
}

export interface NativeSessionSummary {
  readonly id: string;
  readonly title: string;
  readonly provider: string;
  readonly model: string;
  readonly state: string;
}

export interface NativeModelInfo {
  readonly provider: string;
  readonly model: string;
  readonly context: number;
  readonly maxOutput: number;
  readonly tools: boolean;
  readonly parallelTools: boolean;
  readonly reasoning: boolean;
  readonly thinking: boolean;
  readonly vision: boolean;
  readonly structuredOutput: boolean;
  readonly embeddings: boolean;
  readonly streaming: boolean;
  readonly source: string;
}

export interface NativeActiveModel {
  readonly provider: string;
  readonly model: string;
  readonly variant: string | null;
}

export interface NativeActiveTool {
  readonly tool: string;
  readonly opId: string;
  readonly startedMs: number;
  readonly status: string;
}

export interface NativeProjection {
  readonly session: {
    readonly id: string;
    readonly title: string;
    readonly provider: string;
    readonly model: string;
    readonly lifecycle: string;
  };
  readonly state: {
    readonly machine: string;
    readonly label: string;
    readonly active: boolean;
    readonly terminal: boolean;
  };
  readonly activeModel: NativeActiveModel | null;
  readonly activeTool: NativeActiveTool | null;
  readonly progress: Json;
  readonly filesChanged: string[];
  readonly lastCheckpoint: {
    readonly sequence: number;
    readonly path: string;
    readonly createdMs: number;
    readonly restoredMs: number | null;
  } | null;
  readonly verification: Array<{
    readonly opId: string;
    readonly tool: string;
    readonly startedMs: number;
    readonly effectStatus: string | null;
  }>;
  readonly contextUsage: Json;
  readonly queued: number;
  readonly prefixStability: {
    readonly observations: number;
    readonly mean: number;
    readonly stdDev: number;
  } | null;
}

export interface NativeTurn {
  readonly opId: string;
  readonly status: string;
  readonly provider: string;
  readonly model: string;
  readonly variant: string | null;
  readonly toolMode: string | null;
  readonly startedAt: number;
  readonly updatedMs: number;
  readonly queueSeq: number | null;
  readonly promptMessageId: number | null;
}

export interface NativeTaskView {
  readonly goal: string;
  readonly constraints: string[];
  readonly state: string;
  readonly milestones: { readonly completed: string[]; readonly open: string[] };
  readonly decisions: string[];
  readonly failures: string[];
  readonly changedFiles: string[];
  readonly tests: { readonly run: string[]; readonly failed: string[] };
  readonly preferences: string[];
  readonly verification: NativeVerificationFact[];
  readonly progress: Json;
  readonly budget: NativeTaskBudget | null;
  /** Additive (served when present): acceptance criteria, plan/DAG steps,
   * blockers and explicit evidence refs. */
  readonly acceptanceCriteria: string[];
  readonly plan: NativePlanStep[];
  readonly blockers: NativeBlockerEntry[];
  readonly evidenceRefs: string[];
  readonly phase: string | null;
}

/** One plan/DAG step of a native task view (additive). */
export interface NativePlanStep {
  readonly id: string;
  readonly summary: string;
  readonly state: string;
  readonly dependsOn: string[];
}

/** One blocker of a native task view (additive). */
export interface NativeBlockerEntry {
  readonly id: string | null;
  readonly detail: string;
  readonly state: string | null;
}


export interface NativeTaskBudget {
  readonly maxTokens: number | null;
  readonly maxTurns: number | null;
  readonly spentTokens: number | null;
  readonly spentTurns: number | null;
  readonly maxCostMicro: number | null;
  readonly spentCostMicro: number;
  readonly openReservedMicro: number;
}

export interface NativeVerificationFact {
  readonly id: string;
  readonly detail: string;
  readonly status: string;
}

export interface NativeCheckpoint {
  readonly sequence: number;
  readonly path: string;
  readonly beforeHash: string | null;
  readonly afterHash: string | null;
  readonly beforeExists: boolean;
  readonly afterExists: boolean;
  readonly createdMs: number;
  readonly restoredMs: number | null;
}

export interface NativeVerificationView {
  readonly owed: Array<{
    readonly opId: string;
    readonly tool: string;
    readonly startedMs: number;
    readonly status: string;
    readonly effectStatus: string | null;
  }>;
  readonly failedChecks: NativeVerificationFact[];
}

export interface NativeTaskRun {
  /** Numeric durable task id (the daemon's `task_id` is a u64, not a string). */
  readonly task_id: number;
  readonly run_id: string;
  readonly mode: string;
  readonly state: string;
  readonly goal: string | null;
  readonly item_ids: string[];
  readonly model: string | null;
}

export interface NativeTaskRunStarted {
  readonly task_id: number;
  readonly run_id: string;
  readonly state: string;
}

export interface NativeTaskRunCancelled {
  readonly run_id: string;
  readonly cancelled: boolean;
}

export interface NativeAgentEntry {
  readonly agent_id: string;
  readonly kind: 'self' | 'child';
  readonly run_id: string;
  readonly session_id: number;
  readonly worktree_id: number;
  readonly goal: string;
  readonly state: string;
  readonly model: string | null;
  readonly budget: number | null;
  readonly ownership: string;
  readonly capabilities: Json[];
  readonly progress: Json;
  readonly result: Json;
  readonly item_ids: string[] | null;
  readonly item_id: string | null;
  readonly item_kind: string | null;
  /** Additive blocker fields, surfaced when the daemon serves them. */
  readonly blockers: Json | null;
  readonly blocker: Json | null;
}

export interface NativeAgentControlAck {
  readonly queuedSeq: number | null;
  readonly applied: boolean | null;
}

export interface NativeMessagePart {
  readonly kind: string;
  readonly createdMs: number;
  readonly data: Json;
}

export interface NativeMessage {
  readonly seq: number;
  /** Numeric durable message id (SSE `message_created` uses the wire string id). */
  readonly id: number;
  readonly role: string;
  readonly createdMs: number;
  readonly data: Json;
  readonly parts: NativeMessagePart[];
}

export interface NativeMessagePage {
  readonly sessionId: string;
  readonly messages: NativeMessage[];
  readonly hasMore: boolean;
  readonly nextBefore: number | null;
}

export interface NativeEvent {
  readonly seq: number;
  readonly kind: string;
  readonly state: string;
  readonly opId: string | null;
  readonly tsMs: number;
  readonly payload: Json;
}

export interface NativeEventPage {
  readonly sessionId: string;
  readonly events: NativeEvent[];
  readonly hasMore: boolean;
  readonly nextCursor: number | null;
}

export interface NativeReservationGroup {
  readonly count: number;
  readonly predictedMicro: number;
}

export interface NativeReservationSettled extends NativeReservationGroup {
  readonly spentMicro: number;
  readonly providerReportedMicro: number;
}

export interface NativeReservations {
  readonly open: NativeReservationGroup;
  readonly settled: NativeReservationSettled;
  readonly refunded: NativeReservationGroup;
  readonly uncertain: NativeReservationGroup;
  readonly routeDecisions: Json[];
  readonly truncated: boolean;
}

export interface NativeSessionUsage {
  readonly sessionId: string;
  readonly providerCalls: {
    readonly tokens: number;
    readonly prefixObservations: Array<{
      readonly rowId: number;
      readonly promptTokens: number;
      readonly stability: Json;
    }>;
  };
  readonly prefixStability: {
    readonly observations: number;
    readonly mean: number;
    readonly stdDev: number;
  } | null;
  readonly tasks: Array<{
    readonly taskId: string;
    readonly budget: NativeTaskBudget;
    readonly reservations: NativeReservations;
  }>;
}

export interface NativeUsageTotals {
  readonly sessions: number;
  readonly totals: { readonly budget: number; readonly spent: number };
  readonly perSession: Array<{
    readonly sessionId: string;
    readonly budget: number | null;
    readonly spent: number | null;
  }>;
  readonly durable: {
    readonly sessionsWithCalls: number;
    readonly providerCalls: {
      readonly tokens: number;
      readonly prefixObservations: number;
      readonly prefixTokens: number;
      readonly prefixStabilityObservations: number;
    };
    readonly taskSpend: { readonly settledCostMicro: number };
    readonly reservations: NativeReservations;
    readonly truncated: boolean;
  };
}

export interface NativeVerificationRecord {
  readonly recordId: string;
  readonly revision: string;
  readonly workspaceId: string;
  readonly worktreeId: string;
  readonly treeHash: string | null;
  readonly criteria: Array<{
    readonly criterionKey: string;
    readonly passed: boolean;
    readonly evidence: string | null;
  }>;
  readonly checks: Array<{
    readonly check: string;
    readonly program: string;
    readonly args: string[];
    readonly category: string;
    readonly required: boolean;
    readonly status: string;
    readonly startedMs: number;
    readonly finishedMs: number | null;
    readonly exit: number | null;
    readonly summary: string | null;
  }>;
  readonly changedFiles: Array<{
    readonly path: string;
    readonly digestHex: string;
    readonly size: number;
  }>;
  readonly unrelatedChanges: string[];
  readonly reviewer: Json;
  readonly status: string;
  readonly startedMs: number;
  readonly completedMs: number | null;
}

export interface NativeTaskVerification {
  readonly sessionId: string;
  readonly taskId: string;
  readonly records: NativeVerificationRecord[];
}

export interface NativeEvidence {
  readonly id: number;
  readonly kind: Json;
  readonly sessionId: number;
  readonly workspaceId: number;
  readonly taskId: Json;
  readonly sourceRevision: Json;
  readonly compressibility: Json;
  readonly backingCompleteness: Json;
  readonly backingRetained: boolean;
  readonly backingLen: number | null;
  readonly allowRanges: boolean;
  readonly allowSearch: boolean;
  readonly maxBytes: number;
}

export interface NativeEvidenceRetrieval {
  readonly id: number;
  readonly selector: Json;
  readonly bytesBase64: string;
  readonly byteLen: number;
  readonly truncatedByPolicy: boolean;
}

export type NativeEvidenceSelector =
  | { readonly selector: 'all' }
  | { readonly selector: 'byte_range'; readonly start: number; readonly end: number }
  | { readonly selector: 'line_range'; readonly start: number; readonly end: number }
  | { readonly selector: 'search'; readonly query: string; readonly max_hits: number }
  | { readonly selector: 'items'; readonly ids: number[] };

export interface NativeSemanticStatus {
  readonly configured: boolean;
  readonly providerCount: number;
  readonly providers: Array<{
    readonly id: string;
    readonly version: number;
    readonly capabilities: Json;
  }>;
  readonly fallback: { readonly id: string; readonly version: number; readonly capabilities: Json };
  readonly snapshotState: { readonly providers: string[]; readonly fallback: boolean };
}

export interface NativeAbortAck {
  readonly aborted: string[];
}

export interface StartTaskRunRequest {
  readonly goal: string;
  readonly criteria?: string[];
  readonly work_items?: Array<{
    readonly id: string;
    readonly kind: string;
    readonly summary?: string;
    readonly depends_on?: string[];
    readonly acceptance_checks?: string[];
  }>;
  readonly model?: string;
  readonly max_tokens?: number;
  readonly max_cost_micro?: number;
  readonly mutation_mode?: string;
}

// -------------------------------------------------------------- validators

export function validateHealth(json: Json): NativeHealth {
  const path = 'GET /native/health';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['ok', 'version']);
  return { ok: fBool(object, 'ok', path), version: fString(object, 'version', path) };
}

export function validateReady(json: Json): NativeReady {
  const path = 'GET /native/ready';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['ready']);
  return { ready: fBool(object, 'ready', path) };
}

export function validateSessionCreated(json: Json): NativeSessionCreated {
  const path = 'POST /session/create';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['id', 'title', 'created_ms']);
  return {
    id: fString(object, 'id', path),
    title: fString(object, 'title', path),
    created_ms: fInt(object, 'created_ms', path),
  };
}

export function validateSessionList(json: Json): NativeSessionSummary[] {
  const path = 'GET /session/list';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['sessions']);
  return fObjectArray(object, 'sessions', path).map((entry, index) =>
    validateSessionSummary(entry, `${path}.sessions[${index}]`),
  );
}

function validateSessionSummary(object: JsonObject, path: string): NativeSessionSummary {
  checkResponseKeys(object, path, ['id', 'title', 'provider', 'model', 'state']);
  return {
    id: fString(object, 'id', path),
    title: fString(object, 'title', path),
    provider: fString(object, 'provider', path),
    model: fString(object, 'model', path),
    state: fString(object, 'state', path),
  };
}

export function validateModelCatalog(json: Json): NativeModelInfo[] {
  const path = 'GET /models';
  if (!Array.isArray(json)) {
    fail(path, `expected an array, got ${describe(json)}`);
  }
  return json.map((entry, index) => {
    const itemPath = `${path}[${index}]`;
    const object = asObject(entry, itemPath);
    checkResponseKeys(object, itemPath, [
      'provider',
      'model',
      'context',
      'maxOutput',
      'tools',
      'parallelTools',
      'reasoning',
      'thinking',
      'vision',
      'structuredOutput',
      'embeddings',
      'streaming',
      'source',
    ]);
    return {
      provider: fString(object, 'provider', itemPath),
      model: fString(object, 'model', itemPath),
      context: fNumber(object, 'context', itemPath),
      maxOutput: fNumber(object, 'maxOutput', itemPath),
      tools: fBool(object, 'tools', itemPath),
      parallelTools: fBool(object, 'parallelTools', itemPath),
      reasoning: fBool(object, 'reasoning', itemPath),
      thinking: fBool(object, 'thinking', itemPath),
      vision: fBool(object, 'vision', itemPath),
      structuredOutput: fBool(object, 'structuredOutput', itemPath),
      embeddings: fBool(object, 'embeddings', itemPath),
      streaming: fBool(object, 'streaming', itemPath),
      source: fString(object, 'source', itemPath),
    };
  });
}

export function validateProjection(json: Json): NativeProjection {
  return validateProjectionAt(json, 'GET /session/{id}/projection');
}

function validateProjectionAt(json: Json, path: string): NativeProjection {
  const object = asObject(json, path);
  checkResponseKeys(object, path, [
    'session',
    'state',
    'activeModel',
    'activeTool',
    'progress',
    'filesChanged',
    'lastCheckpoint',
    'verification',
    'contextUsage',
    'queued',
  ]);

  const session = asObject(field(object, 'session', path), `${path}.session`);
  checkResponseKeys(session, `${path}.session`, ['id', 'title', 'provider', 'model', 'lifecycle']);
  const state = asObject(field(object, 'state', path), `${path}.state`);
  checkResponseKeys(state, `${path}.state`, ['machine', 'label', 'active', 'terminal']);

  const activeModelRaw = fNullableObject(object, 'activeModel', path);
  let activeModel: NativeActiveModel | null = null;
  if (activeModelRaw !== null) {
    checkResponseKeys(activeModelRaw, `${path}.activeModel`, ['provider', 'model', 'variant']);
    activeModel = {
      provider: fString(activeModelRaw, 'provider', `${path}.activeModel`),
      model: fString(activeModelRaw, 'model', `${path}.activeModel`),
      variant: fNullableString(activeModelRaw, 'variant', `${path}.activeModel`),
    };
  }

  const activeToolRaw = fNullableObject(object, 'activeTool', path);
  let activeTool: NativeActiveTool | null = null;
  if (activeToolRaw !== null) {
    checkResponseKeys(activeToolRaw, `${path}.activeTool`, ['tool', 'opId', 'startedMs', 'status']);
    activeTool = {
      tool: fString(activeToolRaw, 'tool', `${path}.activeTool`),
      opId: fString(activeToolRaw, 'opId', `${path}.activeTool`),
      startedMs: fInt(activeToolRaw, 'startedMs', `${path}.activeTool`),
      status: fString(activeToolRaw, 'status', `${path}.activeTool`),
    };
  }

  const checkpointRaw = fNullableObject(object, 'lastCheckpoint', path);
  let lastCheckpoint: NativeProjection['lastCheckpoint'] = null;
  if (checkpointRaw !== null) {
    const checkpointPath = `${path}.lastCheckpoint`;
    checkResponseKeys(checkpointRaw, checkpointPath, ['sequence', 'path', 'createdMs', 'restoredMs']);
    lastCheckpoint = {
      sequence: fInt(checkpointRaw, 'sequence', checkpointPath),
      path: fString(checkpointRaw, 'path', checkpointPath),
      createdMs: fInt(checkpointRaw, 'createdMs', checkpointPath),
      restoredMs: fNullableInt(checkpointRaw, 'restoredMs', checkpointPath),
    };
  }

  const verification = fObjectArray(object, 'verification', path).map((entry, index) => {
    const entryPath = `${path}.verification[${index}]`;
    checkResponseKeys(entry, entryPath, ['opId', 'tool', 'startedMs', 'effectStatus']);
    return {
      opId: fString(entry, 'opId', entryPath),
      tool: fString(entry, 'tool', entryPath),
      startedMs: fInt(entry, 'startedMs', entryPath),
      effectStatus: fNullableString(entry, 'effectStatus', entryPath),
    };
  });

  const prefixRaw =
    'prefixStability' in object ? fNullableObject(object, 'prefixStability', path) : null;
  let prefixStability: NativeProjection['prefixStability'] = null;
  if (prefixRaw !== null) {
    checkResponseKeys(prefixRaw, `${path}.prefixStability`, ['observations', 'mean', 'stdDev']);
    prefixStability = {
      observations: fInt(prefixRaw, 'observations', `${path}.prefixStability`),
      mean: fNumber(prefixRaw, 'mean', `${path}.prefixStability`),
      stdDev: fNumber(prefixRaw, 'stdDev', `${path}.prefixStability`),
    };
  }

  return {
    session: {
      id: fString(session, 'id', `${path}.session`),
      title: fString(session, 'title', `${path}.session`),
      provider: fString(session, 'provider', `${path}.session`),
      model: fString(session, 'model', `${path}.session`),
      lifecycle: fString(session, 'lifecycle', `${path}.session`),
    },
    state: {
      machine: fString(state, 'machine', `${path}.state`),
      label: fString(state, 'label', `${path}.state`),
      active: fBool(state, 'active', `${path}.state`),
      terminal: fBool(state, 'terminal', `${path}.state`),
    },
    activeModel,
    activeTool,
    progress: fJson(object, 'progress', path),
    filesChanged: fStringArray(object, 'filesChanged', path),
    lastCheckpoint,
    verification,
    contextUsage: fJson(object, 'contextUsage', path),
    queued: fInt(object, 'queued', path),
    prefixStability,
  };
}

export function validateTurns(json: Json): NativeTurn[] {
  const path = 'GET /native/session/{id}/turns';
  if (!Array.isArray(json)) {
    fail(path, `expected an array, got ${describe(json)}`);
  }
  return json.map((entry, index) => {
    const itemPath = `${path}[${index}]`;
    const object = asObject(entry, itemPath);
    checkResponseKeys(object, itemPath, [
      'opId',
      'status',
      'provider',
      'model',
      'variant',
      'toolMode',
      'startedAt',
      'updatedMs',
      'queueSeq',
      'promptMessageId',
    ]);
    return {
      opId: fString(object, 'opId', itemPath),
      status: fString(object, 'status', itemPath),
      provider: fString(object, 'provider', itemPath),
      model: fString(object, 'model', itemPath),
      variant: fNullableString(object, 'variant', itemPath),
      toolMode: fNullableString(object, 'toolMode', itemPath),
      startedAt: fInt(object, 'startedAt', itemPath),
      updatedMs: fInt(object, 'updatedMs', itemPath),
      queueSeq: fNullableInt(object, 'queueSeq', itemPath),
      promptMessageId: fNullableInt(object, 'promptMessageId', itemPath),
    };
  });
}

function validateBudget(object: JsonObject, path: string): NativeTaskBudget {
  checkResponseKeys(object, path, [
    'maxTokens',
    'maxTurns',
    'spentTokens',
    'spentTurns',
    'maxCostMicro',
    'spentCostMicro',
    'openReservedMicro',
  ]);
  return {
    maxTokens: fNullableInt(object, 'maxTokens', path),
    maxTurns: fNullableInt(object, 'maxTurns', path),
    spentTokens: fNullableInt(object, 'spentTokens', path),
    spentTurns: fNullableInt(object, 'spentTurns', path),
    maxCostMicro: fNullableInt(object, 'maxCostMicro', path),
    spentCostMicro: fInt(object, 'spentCostMicro', path),
    openReservedMicro: fInt(object, 'openReservedMicro', path),
  };
}

function validateVerificationFact(object: JsonObject, path: string): NativeVerificationFact {
  checkResponseKeys(object, path, ['id', 'detail', 'status']);
  return {
    id: fString(object, 'id', path),
    detail: fString(object, 'detail', path),
    status: fString(object, 'status', path),
  };
}

// ------------------------------------------------- additive task-view fields
// The server may add acceptance criteria / plan steps / blockers / evidence
// refs at any time (v1 additive contract). When PRESENT they are validated
// strictly enough to trust (bad known-shape entries fail loudly) and
// normalized to one client vocabulary; when ABSENT they are empty, never
// fabricated.

function optionalField(object: JsonObject, keys: readonly string[]): Json | undefined {
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(object, key)) {
      return object[key] as Json;
    }
  }
  return undefined;
}

function optionalString(object: JsonObject, keys: readonly string[], path: string): string | null {
  const value = optionalField(object, keys);
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value !== 'string') {
    fail(`${path}.${keys[0]}`, `expected a string or null, got ${describe(value)}`);
  }
  return value;
}

function optionalStringArray(
  object: JsonObject,
  keys: readonly string[],
  path: string,
): string[] {
  const value = optionalField(object, keys);
  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    fail(`${path}.${keys[0]}`, `expected an array, got ${describe(value)}`);
  }
  return value.slice(0, 64).map((entry, index) => {
    if (typeof entry !== 'string') {
      fail(`${path}.${keys[0]}[${index}]`, `expected a string, got ${describe(entry)}`);
    }
    return entry;
  });
}

function optionalPlanSteps(object: JsonObject, path: string): NativePlanStep[] {
  const value = optionalField(object, ['plan', 'plan_steps', 'planSteps']);
  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    fail(`${path}.plan`, `expected an array, got ${describe(value)}`);
  }
  return value.slice(0, 64).map((entry, index) => {
    const itemPath = `${path}.plan[${index}]`;
    const step = asObject(entry, itemPath);
    const rawId = optionalField(step, ['id']);
    const id =
      typeof rawId === 'string'
        ? rawId
        : typeof rawId === 'number' && Number.isInteger(rawId)
          ? String(rawId)
          : '';
    if (id.length === 0) {
      fail(`${itemPath}.id`, 'expected a non-empty string or integer id');
    }
    return {
      id,
      summary: optionalString(step, ['summary', 'title'], itemPath) ?? '',
      state: optionalString(step, ['state', 'status'], itemPath) ?? 'pending',
      dependsOn: optionalStringArray(step, ['depends_on', 'dependsOn'], itemPath),
    };
  });
}

function optionalBlockers(object: JsonObject, path: string): NativeBlockerEntry[] {
  const value = optionalField(object, ['blockers', 'blocked_on']);
  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    fail(`${path}.blockers`, `expected an array, got ${describe(value)}`);
  }
  return value.slice(0, 64).map((entry, index) => {
    if (typeof entry === 'string') {
      return { id: null, detail: entry, state: null };
    }
    const itemPath = `${path}.blockers[${index}]`;
    const blocker = asObject(entry, itemPath);
    const detail =
      optionalString(blocker, ['detail', 'message', 'summary', 'reason'], itemPath) ?? '';
    if (detail.length === 0) {
      fail(`${itemPath}.detail`, 'expected a non-empty blocker detail');
    }
    return {
      id: optionalString(blocker, ['id'], itemPath),
      detail,
      state: optionalString(blocker, ['state', 'status'], itemPath),
    };
  });
}


export function validateTaskViews(json: Json): NativeTaskView[] {
  const path = 'GET /native/session/{id}/tasks';
  if (!Array.isArray(json)) {
    fail(path, `expected an array, got ${describe(json)}`);
  }
  return json.map((entry, index) => {
    const itemPath = `${path}[${index}]`;
    const object = asObject(entry, itemPath);
    checkResponseKeys(object, itemPath, [
      'goal',
      'constraints',
      'state',
      'milestones',
      'decisions',
      'failures',
      'changedFiles',
      'tests',
      'preferences',
      'verification',
      'progress',
      'budget',
    ]);
    const milestones = asObject(field(object, 'milestones', itemPath), `${itemPath}.milestones`);
    checkResponseKeys(milestones, `${itemPath}.milestones`, ['completed', 'open']);
    const tests = asObject(field(object, 'tests', itemPath), `${itemPath}.tests`);
    checkResponseKeys(tests, `${itemPath}.tests`, ['run', 'failed']);
    const budgetRaw = fNullableObject(object, 'budget', itemPath);
    return {
      goal: fString(object, 'goal', itemPath),
      constraints: fStringArray(object, 'constraints', itemPath),
      state: fString(object, 'state', itemPath),
      milestones: {
        completed: fStringArray(milestones, 'completed', `${itemPath}.milestones`),
        open: fStringArray(milestones, 'open', `${itemPath}.milestones`),
      },
      decisions: fStringArray(object, 'decisions', itemPath),
      failures: fStringArray(object, 'failures', itemPath),
      changedFiles: fStringArray(object, 'changedFiles', itemPath),
      tests: {
        run: fStringArray(tests, 'run', `${itemPath}.tests`),
        failed: fStringArray(tests, 'failed', `${itemPath}.tests`),
      },
      preferences: fStringArray(object, 'preferences', itemPath),
      verification: fObjectArray(object, 'verification', itemPath).map((fact, factIndex) =>
        validateVerificationFact(fact, `${itemPath}.verification[${factIndex}]`),
      ),
      progress: fJson(object, 'progress', itemPath),
      budget: budgetRaw === null ? null : validateBudget(budgetRaw, `${itemPath}.budget`),
      acceptanceCriteria: optionalStringArray(
        object,
        ['acceptanceCriteria', 'acceptance_criteria'],
        itemPath,
      ),
      plan: optionalPlanSteps(object, itemPath),
      blockers: optionalBlockers(object, itemPath),
      evidenceRefs: optionalStringArray(object, ['evidenceRefs', 'evidence_refs'], itemPath),
      phase: optionalString(object, ['phase'], itemPath),
    };
  });
}

export function validateCheckpoints(json: Json): NativeCheckpoint[] {
  const path = 'GET /native/session/{id}/checkpoints';
  if (!Array.isArray(json)) {
    fail(path, `expected an array, got ${describe(json)}`);
  }
  return json.map((entry, index) => {
    const itemPath = `${path}[${index}]`;
    const object = asObject(entry, itemPath);
    checkResponseKeys(object, itemPath, [
      'sequence',
      'path',
      'beforeHash',
      'afterHash',
      'beforeExists',
      'afterExists',
      'createdMs',
      'restoredMs',
    ]);
    return {
      sequence: fInt(object, 'sequence', itemPath),
      path: fString(object, 'path', itemPath),
      beforeHash: fNullableString(object, 'beforeHash', itemPath),
      afterHash: fNullableString(object, 'afterHash', itemPath),
      beforeExists: fBool(object, 'beforeExists', itemPath),
      afterExists: fBool(object, 'afterExists', itemPath),
      createdMs: fInt(object, 'createdMs', itemPath),
      restoredMs: fNullableInt(object, 'restoredMs', itemPath),
    };
  });
}

export function validateVerificationView(json: Json): NativeVerificationView {
  const path = 'GET /native/session/{id}/verification';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['owed', 'failedChecks']);
  return {
    owed: fObjectArray(object, 'owed', path).map((entry, index) => {
      const itemPath = `${path}.owed[${index}]`;
      checkResponseKeys(entry, itemPath, ['opId', 'tool', 'startedMs', 'status', 'effectStatus']);
      return {
        opId: fString(entry, 'opId', itemPath),
        tool: fString(entry, 'tool', itemPath),
        startedMs: fInt(entry, 'startedMs', itemPath),
        status: fString(entry, 'status', itemPath),
        effectStatus: fNullableString(entry, 'effectStatus', itemPath),
      };
    }),
    failedChecks: fObjectArray(object, 'failedChecks', path).map((entry, index) =>
      validateVerificationFact(entry, `${path}.failedChecks[${index}]`),
    ),
  };
}

export function validateTaskRuns(json: Json): NativeTaskRun[] {
  const path = 'GET /native/session/{id}/task-runs';
  if (!Array.isArray(json)) {
    fail(path, `expected an array, got ${describe(json)}`);
  }
  return json.map((entry, index) => {
    const itemPath = `${path}[${index}]`;
    const object = asObject(entry, itemPath);
    checkResponseKeys(object, itemPath, [
      'task_id',
      'run_id',
      'mode',
      'state',
      'goal',
      'item_ids',
      'model',
    ]);
    return {
      task_id: fInt(object, 'task_id', itemPath),
      run_id: fString(object, 'run_id', itemPath),
      mode: fString(object, 'mode', itemPath),
      state: fString(object, 'state', itemPath),
      goal: fNullableString(object, 'goal', itemPath),
      item_ids: fStringArray(object, 'item_ids', itemPath),
      model: fNullableString(object, 'model', itemPath),
    };
  });
}

export function validateTaskRun(json: Json): NativeTaskRun {
  const entries = validateTaskRuns([json]);
  return entries[0] as NativeTaskRun;
}

export function validateTaskRunStarted(json: Json): NativeTaskRunStarted {
  const path = 'POST /native/session/{id}/task-runs';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['task_id', 'run_id', 'state']);
  return {
    task_id: fInt(object, 'task_id', path),
    run_id: fString(object, 'run_id', path),
    state: fString(object, 'state', path),
  };
}

export function validateTaskRunCancelled(json: Json): NativeTaskRunCancelled {
  const path = 'POST /native/session/{id}/task-runs/{run_id}/cancel';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['run_id', 'cancelled']);
  return { run_id: fString(object, 'run_id', path), cancelled: fBool(object, 'cancelled', path) };
}

export function validateAgents(json: Json): NativeAgentEntry[] {
  const path = 'GET /native/agents';
  if (!Array.isArray(json)) {
    fail(path, `expected an array, got ${describe(json)}`);
  }
  return json.map((entry, index) => {
    const itemPath = `${path}[${index}]`;
    const object = asObject(entry, itemPath);
    checkResponseKeys(
      object,
      itemPath,
      [
        'agent_id',
        'kind',
        'run_id',
        'session_id',
        'worktree_id',
        'goal',
        'state',
        'model',
        'budget',
        'ownership',
        'capabilities',
        'progress',
        'result',
      ],
    );
    const kind = fString(object, 'kind', itemPath);
    if (kind !== 'self' && kind !== 'child') {
      fail(`${itemPath}.kind`, `expected "self" or "child", got ${JSON.stringify(kind)}`);
    }
    const budget = field(object, 'budget', itemPath);
    if (budget !== null && (typeof budget !== 'number' || !Number.isInteger(budget))) {
      fail(`${itemPath}.budget`, `expected an integer or null, got ${describe(budget)}`);
    }
    return {
      agent_id: fString(object, 'agent_id', itemPath),
      kind,
      run_id: fString(object, 'run_id', itemPath),
      session_id: fInt(object, 'session_id', itemPath),
      worktree_id: fInt(object, 'worktree_id', itemPath),
      goal: fString(object, 'goal', itemPath),
      state: fString(object, 'state', itemPath),
      model: fNullableString(object, 'model', itemPath),
      budget: budget === null ? null : (budget as number),
      ownership: fString(object, 'ownership', itemPath),
      capabilities: fArray(object, 'capabilities', itemPath),
      progress: fJson(object, 'progress', itemPath),
      result: fJson(object, 'result', itemPath),
      item_ids: 'item_ids' in object ? fStringArray(object, 'item_ids', itemPath) : null,
      item_id: 'item_id' in object ? fNullableString(object, 'item_id', itemPath) : null,
      item_kind: 'item_kind' in object ? fNullableString(object, 'item_kind', itemPath) : null,
      blockers: 'blockers' in object ? fJson(object, 'blockers', itemPath) : null,
      blocker: 'blocker' in object ? fJson(object, 'blocker', itemPath) : null,
    };
  });
}

export function validateAgentControlAck(json: Json, path: string): NativeAgentControlAck {
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['queuedSeq', 'applied']);
  const queued = field(object, 'queuedSeq', path);
  const applied = field(object, 'applied', path);
  if (queued !== null && (typeof queued !== 'number' || !Number.isInteger(queued))) {
    fail(`${path}.queuedSeq`, `expected an integer or null, got ${describe(queued)}`);
  }
  if (applied !== null && typeof applied !== 'boolean') {
    fail(`${path}.applied`, `expected a boolean or null, got ${describe(applied)}`);
  }
  return { queuedSeq: queued === null ? null : (queued as number), applied };
}

export function validateMessagePage(json: Json): NativeMessagePage {
  const path = 'GET /native/messages';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['sessionId', 'messages', 'hasMore', 'nextBefore']);
  return {
    sessionId: fString(object, 'sessionId', path),
    messages: fObjectArray(object, 'messages', path).map((entry, index) => {
      const itemPath = `${path}.messages[${index}]`;
      checkResponseKeys(entry, itemPath, ['seq', 'id', 'role', 'createdMs', 'data', 'parts']);
      return {
        seq: fInt(entry, 'seq', itemPath),
        id: fInt(entry, 'id', itemPath),
        role: fString(entry, 'role', itemPath),
        createdMs: fInt(entry, 'createdMs', itemPath),
        data: fJson(entry, 'data', itemPath),
        parts: fObjectArray(entry, 'parts', itemPath).map((part, partIndex) => {
          const partPath = `${itemPath}.parts[${partIndex}]`;
          checkResponseKeys(part, partPath, ['kind', 'createdMs', 'data']);
          return {
            kind: fString(part, 'kind', partPath),
            createdMs: fInt(part, 'createdMs', partPath),
            data: fJson(part, 'data', partPath),
          };
        }),
      };
    }),
    hasMore: fBool(object, 'hasMore', path),
    nextBefore: fNullableInt(object, 'nextBefore', path),
  };
}

export function validateEventPage(json: Json): NativeEventPage {
  const path = 'GET /native/events';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['sessionId', 'events', 'hasMore', 'nextCursor']);
  return {
    sessionId: fString(object, 'sessionId', path),
    events: fObjectArray(object, 'events', path).map((entry, index) => {
      const itemPath = `${path}.events[${index}]`;
      checkResponseKeys(entry, itemPath, ['seq', 'kind', 'state', 'opId', 'tsMs', 'payload']);
      return {
        seq: fInt(entry, 'seq', itemPath),
        kind: fString(entry, 'kind', itemPath),
        state: fString(entry, 'state', itemPath),
        opId: fNullableString(entry, 'opId', itemPath),
        tsMs: fInt(entry, 'tsMs', itemPath),
        payload: fJson(entry, 'payload', itemPath),
      };
    }),
    hasMore: fBool(object, 'hasMore', path),
    nextCursor: fNullableInt(object, 'nextCursor', path),
  };
}

function validateReservationGroup(object: JsonObject, path: string): NativeReservationGroup {
  checkResponseKeys(object, path, ['count', 'predictedMicro']);
  return {
    count: fInt(object, 'count', path),
    predictedMicro: fInt(object, 'predictedMicro', path),
  };
}

function validateReservations(object: JsonObject, path: string): NativeReservations {
  // `routeDecisions` rides the per-task view; the cross-session aggregate
  // (`/native/usage.durable.reservations`) omits it by contract.
  checkResponseKeys(object, path, ['open', 'settled', 'refunded', 'uncertain']);
  const settled = asObject(field(object, 'settled', path), `${path}.settled`);
  checkResponseKeys(settled, `${path}.settled`, ['count', 'predictedMicro', 'spentMicro', 'providerReportedMicro']);
  return {
    open: validateReservationGroup(
      asObject(field(object, 'open', path), `${path}.open`),
      `${path}.open`,
    ),
    settled: {
      count: fInt(settled, 'count', `${path}.settled`),
      predictedMicro: fInt(settled, 'predictedMicro', `${path}.settled`),
      spentMicro: fInt(settled, 'spentMicro', `${path}.settled`),
      providerReportedMicro: fInt(settled, 'providerReportedMicro', `${path}.settled`),
    },
    refunded: validateReservationGroup(
      asObject(field(object, 'refunded', path), `${path}.refunded`),
      `${path}.refunded`,
    ),
    uncertain: validateReservationGroup(
      asObject(field(object, 'uncertain', path), `${path}.uncertain`),
      `${path}.uncertain`,
    ),
    routeDecisions:
      'routeDecisions' in object ? fArray(object, 'routeDecisions', path) : [],
    truncated: 'truncated' in object ? fBool(object, 'truncated', path) : false,
  };
}

export function validateSessionUsage(json: Json): NativeSessionUsage {
  const path = 'GET /native/session/{id}/usage';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['sessionId', 'providerCalls', 'prefixStability', 'tasks']);
  const calls = asObject(field(object, 'providerCalls', path), `${path}.providerCalls`);
  checkResponseKeys(calls, `${path}.providerCalls`, ['tokens', 'prefixObservations']);
  const prefixRaw = fNullableObject(object, 'prefixStability', path);
  let prefixStability: NativeSessionUsage['prefixStability'] = null;
  if (prefixRaw !== null) {
    checkResponseKeys(prefixRaw, `${path}.prefixStability`, ['observations', 'mean', 'stdDev']);
    prefixStability = {
      observations: fInt(prefixRaw, 'observations', `${path}.prefixStability`),
      mean: fNumber(prefixRaw, 'mean', `${path}.prefixStability`),
      stdDev: fNumber(prefixRaw, 'stdDev', `${path}.prefixStability`),
    };
  }
  return {
    sessionId: fString(object, 'sessionId', path),
    providerCalls: {
      tokens: fInt(calls, 'tokens', `${path}.providerCalls`),
      prefixObservations: fObjectArray(calls, 'prefixObservations', `${path}.providerCalls`).map(
        (entry, index) => {
          const itemPath = `${path}.providerCalls.prefixObservations[${index}]`;
          checkResponseKeys(entry, itemPath, ['rowId', 'promptTokens', 'stability']);
          return {
            rowId: fInt(entry, 'rowId', itemPath),
            promptTokens: fInt(entry, 'promptTokens', itemPath),
            stability: fJson(entry, 'stability', itemPath),
          };
        },
      ),
    },
    prefixStability,
    tasks: fObjectArray(object, 'tasks', path).map((entry, index) => {
      const itemPath = `${path}.tasks[${index}]`;
      checkResponseKeys(entry, itemPath, ['taskId', 'budget', 'reservations']);
      return {
        taskId: fString(entry, 'taskId', itemPath),
        budget: validateBudget(
          asObject(field(entry, 'budget', itemPath), `${itemPath}.budget`),
          `${itemPath}.budget`,
        ),
        reservations: validateReservations(
          asObject(field(entry, 'reservations', itemPath), `${itemPath}.reservations`),
          `${itemPath}.reservations`,
        ),
      };
    }),
  };
}

export function validateUsage(json: Json): NativeUsageTotals {
  const path = 'GET /native/usage';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['sessions', 'totals', 'perSession', 'durable']);
  const totals = asObject(field(object, 'totals', path), `${path}.totals`);
  checkResponseKeys(totals, `${path}.totals`, ['budget', 'spent']);
  const durable = asObject(field(object, 'durable', path), `${path}.durable`);
  checkResponseKeys(durable, `${path}.durable`, ['sessionsWithCalls', 'providerCalls', 'taskSpend', 'reservations']);
  const calls = asObject(field(durable, 'providerCalls', path), `${path}.durable.providerCalls`);
  checkResponseKeys(calls, `${path}.durable.providerCalls`, [
    'tokens',
    'prefixObservations',
    'prefixTokens',
    'prefixStabilityObservations',
  ]);
  const taskSpend = asObject(field(durable, 'taskSpend', path), `${path}.durable.taskSpend`);
  checkResponseKeys(taskSpend, `${path}.durable.taskSpend`, ['settledCostMicro']);
  return {
    sessions: fInt(object, 'sessions', path),
    totals: {
      budget: fInt(totals, 'budget', `${path}.totals`),
      spent: fInt(totals, 'spent', `${path}.totals`),
    },
    perSession: fObjectArray(object, 'perSession', path).map((entry, index) => {
      const itemPath = `${path}.perSession[${index}]`;
      checkResponseKeys(entry, itemPath, ['sessionId', 'budget', 'spent']);
      return {
        sessionId: fString(entry, 'sessionId', itemPath),
        budget: fNullableInt(entry, 'budget', itemPath),
        spent: fNullableInt(entry, 'spent', itemPath),
      };
    }),
    durable: {
      sessionsWithCalls: fInt(durable, 'sessionsWithCalls', `${path}.durable`),
      providerCalls: {
        tokens: fInt(calls, 'tokens', `${path}.durable.providerCalls`),
        prefixObservations: fInt(calls, 'prefixObservations', `${path}.durable.providerCalls`),
        prefixTokens: fInt(calls, 'prefixTokens', `${path}.durable.providerCalls`),
        prefixStabilityObservations: fInt(
          calls,
          'prefixStabilityObservations',
          `${path}.durable.providerCalls`,
        ),
      },
      taskSpend: {
        settledCostMicro: fInt(taskSpend, 'settledCostMicro', `${path}.durable.taskSpend`),
      },
      reservations: validateReservations(
        asObject(field(durable, 'reservations', path), `${path}.durable.reservations`),
        `${path}.durable.reservations`,
      ),
      truncated: 'truncated' in durable ? fBool(durable, 'truncated', `${path}.durable`) : false,
    },
  };
}

export function validateTaskVerification(json: Json): NativeTaskVerification {
  const path = 'GET /native/session/{id}/tasks/{task_id}/verification';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['sessionId', 'taskId', 'records']);
  return {
    sessionId: fString(object, 'sessionId', path),
    taskId: fString(object, 'taskId', path),
    records: fObjectArray(object, 'records', path).map((entry, index) =>
      validateVerificationRecord(entry, `${path}.records[${index}]`),
    ),
  };
}

function validateVerificationRecord(object: JsonObject, path: string): NativeVerificationRecord {
  checkResponseKeys(object, path, [
    'recordId',
    'revision',
    'workspaceId',
    'worktreeId',
    'treeHash',
    'criteria',
    'checks',
    'changedFiles',
    'unrelatedChanges',
    'reviewer',
    'status',
    'startedMs',
    'completedMs',
  ]);
  return {
    recordId: fString(object, 'recordId', path),
    revision: fString(object, 'revision', path),
    workspaceId: fString(object, 'workspaceId', path),
    worktreeId: fString(object, 'worktreeId', path),
    treeHash: fNullableString(object, 'treeHash', path),
    criteria: fObjectArray(object, 'criteria', path).map((entry, index) => {
      const itemPath = `${path}.criteria[${index}]`;
      checkResponseKeys(entry, itemPath, ['criterionKey', 'passed', 'evidence']);
      return {
        criterionKey: fString(entry, 'criterionKey', itemPath),
        passed: fBool(entry, 'passed', itemPath),
        evidence: fNullableString(entry, 'evidence', itemPath),
      };
    }),
    checks: fObjectArray(object, 'checks', path).map((entry, index) => {
      const itemPath = `${path}.checks[${index}]`;
      checkResponseKeys(entry, itemPath, [
        'check',
        'program',
        'args',
        'category',
        'required',
        'status',
        'startedMs',
        'finishedMs',
        'exit',
        'summary',
      ]);
      return {
        check: fString(entry, 'check', itemPath),
        program: fString(entry, 'program', itemPath),
        args: fStringArray(entry, 'args', itemPath),
        category: fString(entry, 'category', itemPath),
        required: fBool(entry, 'required', itemPath),
        status: fString(entry, 'status', itemPath),
        startedMs: fInt(entry, 'startedMs', itemPath),
        finishedMs: fNullableInt(entry, 'finishedMs', itemPath),
        exit: fNullableInt(entry, 'exit', itemPath),
        summary: fNullableString(entry, 'summary', itemPath),
      };
    }),
    changedFiles: fObjectArray(object, 'changedFiles', path).map((entry, index) => {
      const itemPath = `${path}.changedFiles[${index}]`;
      checkResponseKeys(entry, itemPath, ['path', 'digestHex', 'size']);
      return {
        path: fString(entry, 'path', itemPath),
        digestHex: fString(entry, 'digestHex', itemPath),
        size: fInt(entry, 'size', itemPath),
      };
    }),
    unrelatedChanges: fStringArray(object, 'unrelatedChanges', path),
    reviewer: fJson(object, 'reviewer', path),
    status: fString(object, 'status', path),
    startedMs: fInt(object, 'startedMs', path),
    completedMs: fNullableInt(object, 'completedMs', path),
  };
}

export function validateEvidence(json: Json): NativeEvidence {
  const path = 'GET /native/evidence/{id}';
  const object = asObject(json, path);
  checkResponseKeys(object, path, [
    'id',
    'kind',
    'sessionId',
    'workspaceId',
    'taskId',
    'sourceRevision',
    'compressibility',
    'backingCompleteness',
    'backingRetained',
    'backingLen',
    'allowRanges',
    'allowSearch',
    'maxBytes',
  ]);
  return {
    id: fInt(object, 'id', path),
    kind: fJson(object, 'kind', path),
    sessionId: fInt(object, 'sessionId', path),
    workspaceId: fInt(object, 'workspaceId', path),
    taskId: fJson(object, 'taskId', path),
    sourceRevision: fJson(object, 'sourceRevision', path),
    compressibility: fJson(object, 'compressibility', path),
    backingCompleteness: fJson(object, 'backingCompleteness', path),
    backingRetained: fBool(object, 'backingRetained', path),
    backingLen: fNullableInt(object, 'backingLen', path),
    allowRanges: fBool(object, 'allowRanges', path),
    allowSearch: fBool(object, 'allowSearch', path),
    maxBytes: fInt(object, 'maxBytes', path),
  };
}

export function validateEvidenceRetrieval(json: Json): NativeEvidenceRetrieval {
  const path = 'POST /native/evidence/{id}/retrieve';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['id', 'selector', 'bytesBase64', 'byteLen', 'truncatedByPolicy']);
  return {
    id: fInt(object, 'id', path),
    selector: fJson(object, 'selector', path),
    bytesBase64: fString(object, 'bytesBase64', path),
    byteLen: fInt(object, 'byteLen', path),
    truncatedByPolicy: fBool(object, 'truncatedByPolicy', path),
  };
}

export function validateSemanticStatus(json: Json): NativeSemanticStatus {
  const path = 'GET /native/semantic/status';
  const object = asObject(json, path);
  checkResponseKeys(object, path, [
    'configured',
    'providerCount',
    'providers',
    'fallback',
    'snapshotState',
  ]);
  const provider = (entry: JsonObject, itemPath: string): { id: string; version: number; capabilities: Json } => {
    checkResponseKeys(entry, itemPath, ['id', 'version', 'capabilities']);
    return {
      id: fString(entry, 'id', itemPath),
      version: fInt(entry, 'version', itemPath),
      capabilities: fJson(entry, 'capabilities', itemPath),
    };
  };
  const snapshot = asObject(field(object, 'snapshotState', path), `${path}.snapshotState`);
  checkResponseKeys(snapshot, `${path}.snapshotState`, ['providers', 'fallback']);
  return {
    configured: fBool(object, 'configured', path),
    providerCount: fInt(object, 'providerCount', path),
    providers: fObjectArray(object, 'providers', path).map((entry, index) =>
      provider(entry, `${path}.providers[${index}]`),
    ),
    fallback: provider(
      asObject(field(object, 'fallback', path), `${path}.fallback`),
      `${path}.fallback`,
    ),
    snapshotState: {
      providers: fStringArray(snapshot, 'providers', `${path}.snapshotState`),
      fallback: fBool(snapshot, 'fallback', `${path}.snapshotState`),
    },
  };
}

export function validateAbortAck(json: Json): NativeAbortAck {
  const path = 'POST /native/session/{id}/abort';
  const object = asObject(json, path);
  checkResponseKeys(object, path, ['aborted']);
  return { aborted: fStringArray(object, 'aborted', path) };
}

// -------------------------------------------------------------- the client

interface RequestOptions<T> {
  readonly query?: Record<string, string | number | undefined>;
  readonly body?: Json;
  readonly maxBytes?: number;
  readonly timeoutMs?: number;
  readonly validate: (json: Json, path: string) => T;
}

export class NativeClient {
  readonly baseUrl: string;
  readonly bearerToken: string;
  private readonly fetchImpl: FetchLike;
  private readonly timeoutMs: number;
  private readonly maxBodyBytes: number;

  constructor(options: NativeClientOptions) {
    const base = options.baseUrl.replace(/\/+$/, '');
    if (!/^https?:\/\//.test(base)) {
      throw new NativeProtocolError('NativeClient', `baseUrl must be http(s), got ${options.baseUrl}`);
    }
    this.baseUrl = base;
    this.bearerToken = options.bearerToken;
    const injected = options.fetch;
    if (injected) {
      this.fetchImpl = injected;
    } else {
      const globalFetch = (globalThis as unknown as { fetch?: FetchLike }).fetch;
      if (typeof globalFetch !== 'function') {
        throw new NativeProtocolError(
          'NativeClient',
          'no fetch implementation available; pass options.fetch',
        );
      }
      this.fetchImpl = (url, init) => globalFetch(url, init);
    }
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxBodyBytes = options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  }

  private async request<T>(method: string, path: string, options: RequestOptions<T>): Promise<T> {
    const url = new URL(this.baseUrl + path);
    if (options.query) {
      for (const [key, value] of Object.entries(options.query)) {
        if (value !== undefined) {
          url.searchParams.set(key, String(value));
        }
      }
    }
    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.bearerToken}`,
      Accept: 'application/json',
    };
    let body: string | undefined;
    if (options.body !== undefined) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(options.body);
    }
    const controller = new AbortController();
    const timeout = this.timeoutMs;
    const timer = setTimeout(() => controller.abort(), timeout);
    let response: ResponseLike;
    try {
      response = await this.fetchImpl(url.toString(), {
        method,
        headers,
        body,
        signal: controller.signal,
      });
    } catch (error) {
      if (controller.signal.aborted) {
        throw new NativeProtocolError(`${method} ${path}`, `request timed out after ${timeout}ms`);
      }
      throw new NativeProtocolError(
        `${method} ${path}`,
        `fetch failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      clearTimeout(timer);
    }
    const label = `${method} ${path}`;
    if (!response.ok) {
      const text = await readBounded(response, ERROR_BODY_BYTES, label);
      throw apiError(response.status, text, label);
    }
    const text = await readBounded(response, options.maxBytes ?? this.maxBodyBytes, label);
    let parsed: Json;
    try {
      parsed = JSON.parse(text) as Json;
    } catch {
      throw new NativeProtocolError(label, `response was not valid JSON (${text.length} bytes)`);
    }
    return options.validate(parsed, label);
  }

  health(): Promise<NativeHealth> {
    return this.request('GET', '/native/health', { validate: validateHealth });
  }

  ready(): Promise<NativeReady> {
    return this.request('GET', '/native/ready', { validate: validateReady });
  }

  async readyOrThrow(timeoutMs = 10_000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    let last = 'not ready';
    while (Date.now() < deadline) {
      try {
        const ready = await this.ready();
        if (ready.ready) {
          return;
        }
        last = 'ready:false';
      } catch (error) {
        last = error instanceof Error ? error.message : String(error);
      }
      await new Promise<void>((resolve) => setTimeout(resolve, 100));
    }
    throw new NativeProtocolError('GET /native/ready', `daemon never became ready (${last})`);
  }

  createSession(request: {
    provider: string;
    model: string;
    workspace?: string;
    title?: string;
  }): Promise<NativeSessionCreated> {
    return this.request('POST', '/session/create', {
      body: { ...request },
      validate: validateSessionCreated,
    });
  }

  listSessions(): Promise<NativeSessionSummary[]> {
    return this.request('GET', '/session/list', { validate: validateSessionList });
  }

  projection(sessionId: string): Promise<NativeProjection> {
    return this.request('GET', `/session/${encodeURIComponent(sessionId)}/projection`, {
      validate: validateProjection,
    });
  }

  modelCatalog(): Promise<NativeModelInfo[]> {
    return this.request('GET', '/models', { validate: validateModelCatalog });
  }

  turns(sessionId: string): Promise<NativeTurn[]> {
    return this.request('GET', `/native/session/${encodeURIComponent(sessionId)}/turns`, {
      validate: validateTurns,
    });
  }

  tasks(sessionId: string): Promise<NativeTaskView[]> {
    return this.request('GET', `/native/session/${encodeURIComponent(sessionId)}/tasks`, {
      validate: validateTaskViews,
    });
  }

  checkpoints(sessionId: string): Promise<NativeCheckpoint[]> {
    return this.request('GET', `/native/session/${encodeURIComponent(sessionId)}/checkpoints`, {
      validate: validateCheckpoints,
    });
  }

  verification(sessionId: string): Promise<NativeVerificationView> {
    return this.request('GET', `/native/session/${encodeURIComponent(sessionId)}/verification`, {
      validate: validateVerificationView,
    });
  }

  taskRuns(sessionId: string): Promise<NativeTaskRun[]> {
    return this.request('GET', `/native/session/${encodeURIComponent(sessionId)}/task-runs`, {
      validate: validateTaskRuns,
    });
  }

  taskRunState(sessionId: string, runId: string): Promise<NativeTaskRun> {
    return this.request(
      'GET',
      `/native/session/${encodeURIComponent(sessionId)}/task-runs/${encodeURIComponent(runId)}`,
      { validate: validateTaskRun },
    );
  }

  startTaskRun(sessionId: string, request: StartTaskRunRequest): Promise<NativeTaskRunStarted> {
    return this.request('POST', `/native/session/${encodeURIComponent(sessionId)}/task-runs`, {
      body: request as unknown as Json,
      validate: validateTaskRunStarted,
    });
  }

  cancelTaskRun(sessionId: string, runId: string): Promise<NativeTaskRunCancelled> {
    return this.request(
      'POST',
      `/native/session/${encodeURIComponent(sessionId)}/task-runs/${encodeURIComponent(runId)}/cancel`,
      { validate: validateTaskRunCancelled },
    );
  }

  agents(sessionId: string): Promise<NativeAgentEntry[]> {
    return this.request('GET', '/native/agents', {
      query: { session: sessionId },
      validate: validateAgents,
    });
  }

  pauseAgent(childId: string): Promise<NativeAgentControlAck> {
    return this.agentControlPost(childId, 'pause');
  }

  resumeAgent(childId: string): Promise<NativeAgentControlAck> {
    return this.agentControlPost(childId, 'resume');
  }

  cancelAgent(childId: string): Promise<NativeAgentControlAck> {
    return this.agentControlPost(childId, 'cancel');
  }

  retryAgent(childId: string): Promise<NativeAgentControlAck> {
    return this.agentControlPost(childId, 'retry');
  }

  steerAgent(childId: string, text: string): Promise<NativeAgentControlAck> {
    return this.request('POST', `/native/agents/${encodeURIComponent(childId)}/steer`, {
      body: { text },
      validate: (json, path) => validateAgentControlAck(json, path),
    });
  }

  setAgentModel(childId: string, model: string): Promise<NativeAgentControlAck> {
    return this.request('POST', `/native/agents/${encodeURIComponent(childId)}/model`, {
      body: { model },
      validate: (json, path) => validateAgentControlAck(json, path),
    });
  }

  setAgentBudget(
    childId: string,
    budget: { max_tokens?: number; max_cost_micro?: number },
  ): Promise<NativeAgentControlAck> {
    return this.request('POST', `/native/agents/${encodeURIComponent(childId)}/budget`, {
      body: { ...budget },
      validate: (json, path) => validateAgentControlAck(json, path),
    });
  }

  private agentControlPost(
    childId: string,
    action: 'pause' | 'resume' | 'cancel' | 'retry',
  ): Promise<NativeAgentControlAck> {
    return this.request('POST', `/native/agents/${encodeURIComponent(childId)}/${action}`, {
      validate: (json, path) => validateAgentControlAck(json, path),
    });
  }

  messages(
    sessionId: string,
    page: { before?: number; limit?: number } = {},
  ): Promise<NativeMessagePage> {
    return this.request('GET', '/native/messages', {
      query: { session: sessionId, before: page.before, limit: page.limit },
      validate: validateMessagePage,
    });
  }

  events(
    sessionId: string,
    page: { after?: number; limit?: number } = {},
  ): Promise<NativeEventPage> {
    return this.request('GET', '/native/events', {
      query: { session: sessionId, after: page.after, limit: page.limit },
      validate: validateEventPage,
    });
  }

  usage(): Promise<NativeUsageTotals> {
    return this.request('GET', '/native/usage', { validate: validateUsage });
  }

  sessionUsage(sessionId: string): Promise<NativeSessionUsage> {
    return this.request('GET', `/native/session/${encodeURIComponent(sessionId)}/usage`, {
      validate: validateSessionUsage,
    });
  }

  taskVerification(sessionId: string, taskId: string): Promise<NativeTaskVerification> {
    return this.request(
      'GET',
      `/native/session/${encodeURIComponent(sessionId)}/tasks/${encodeURIComponent(taskId)}/verification`,
      { validate: validateTaskVerification },
    );
  }

  evidence(sessionId: string, evidenceId: number): Promise<NativeEvidence> {
    return this.request('GET', `/native/evidence/${encodeURIComponent(String(evidenceId))}`, {
      query: { session: sessionId },
      validate: validateEvidence,
    });
  }

  retrieveEvidence(
    sessionId: string,
    evidenceId: number,
    selector: NativeEvidenceSelector,
  ): Promise<NativeEvidenceRetrieval> {
    return this.request(
      'POST',
      `/native/evidence/${encodeURIComponent(String(evidenceId))}/retrieve`,
      {
        query: { session: sessionId },
        body: selector as unknown as Json,
        maxBytes: EVIDENCE_MAX_BODY_BYTES,
        validate: validateEvidenceRetrieval,
      },
    );
  }

  semanticStatus(): Promise<NativeSemanticStatus> {
    return this.request('GET', '/native/semantic/status', { validate: validateSemanticStatus });
  }

  abortSession(sessionId: string, opId?: string): Promise<NativeAbortAck> {
    const body: Json = opId === undefined
      ? { session_id: sessionId }
      : { session_id: sessionId, op_id: opId };
    return this.request('POST', `/native/session/${encodeURIComponent(sessionId)}/abort`, {
      body,
      validate: validateAbortAck,
    });
  }
}

// ------------------------------------------------------------------ helpers

function apiError(status: number, body: string, label: string): NativeApiError {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body) as unknown;
  } catch {
    return new NativeApiError(status, 'http_error', body.slice(0, 200) || `HTTP ${status}`, false);
  }
  if (typeof parsed === 'object' && parsed !== null) {
    const error = (parsed as { error?: unknown }).error;
    if (typeof error === 'object' && error !== null) {
      const code = (error as { code?: unknown }).code;
      const message = (error as { message?: unknown }).message;
      const retryable = (error as { retryable?: unknown }).retryable;
      if (typeof code === 'string' && typeof message === 'string') {
        return new NativeApiError(status, code, message, retryable === true);
      }
    }
  }
  throw new NativeProtocolError(label, `non-JSON error body with HTTP ${status}`);
}

async function readBounded(
  response: ResponseLike,
  maxBytes: number,
  label: string,
): Promise<string> {
  const declared = Number(response.headers?.get?.('content-length') ?? Number.NaN);
  if (Number.isFinite(declared) && declared > maxBytes) {
    throw new NativeProtocolError(label, `declared body ${declared} bytes exceeds bound ${maxBytes}`);
  }
  const reader = response.body?.getReader();
  if (reader) {
    const chunks: Uint8Array[] = [];
    let size = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (!value) {
        continue;
      }
      size += value.byteLength;
      if (size > maxBytes) {
        try {
          await reader.cancel();
        } catch {
          // Cancellation is best-effort; the rejection below is the point.
        }
        throw new NativeProtocolError(label, `streamed body exceeded bound ${maxBytes} bytes`);
      }
      chunks.push(value);
    }
    return decodeUtf8(chunks, size);
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength > maxBytes) {
    throw new NativeProtocolError(label, `body ${buffer.byteLength} bytes exceeds bound ${maxBytes}`);
  }
  return decodeUtf8([new Uint8Array(buffer)], buffer.byteLength);
}

function decodeUtf8(chunks: readonly Uint8Array[], size: number): string {
  const joined = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder('utf-8', { fatal: false }).decode(joined);
}
