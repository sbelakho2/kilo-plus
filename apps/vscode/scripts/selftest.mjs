#!/usr/bin/env node
// Faktor VS Code extension selftest. Plain `node scripts/selftest.mjs`, no
// npm install, no test framework. It imports the real TypeScript modules
// (Node >= 23.6 strips types natively) and drives them with a fake fetch /
// fake SSE stream:
//
//   1. nativeClient accept paths for every endpoint the extension uses,
//      including the v1 additive contract: unknown RESPONSE fields are
//      ignored while known fields keep exact-type validation;
//   2. nativeClient reject paths: hostile shapes, bad types, API error
//      envelopes and oversized bodies all fail loudly;
//   3. eventStream: cursor resume, backoff, heartbeat tolerance, replay
//      suppression, malformed-frame reporting and bounded frames;
//   4. the state store and transcript reducer (durable pages + SSE frames);
//   5. daemon binary resolution / refusal.
//
// Prints one line per check and exits nonzero on any failure.

import * as nc from '../src/nativeClient.ts';
import * as es from '../src/eventStream.ts';
import * as st from '../src/state.ts';
import * as dm from '../src/daemon.ts';
import * as ts from '../src/taskStart.ts';
import * as wb from '../src/workspaceBinding.ts';
import * as px from '../src/pixelAgents.ts';
import * as cp from '../src/cockpit.ts';
import composerPolicy from '../media/composer-state.js';
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { bridgeTests } from './bridge-selftest.mjs';
import {
  bridgeCommandToHostMessage,
  ingestWebviewMessage,
  sendMessageFailedMessage,
} from '../src/kilo-bridge.ts';
import { stageBundle, verifyOverlay } from './prepare-vendored-webview.mjs';

// `--packaged <extension-dir>` additionally asserts the extracted VSIX layout
// (out/ + media/ + the pinned vendored webview closure) without a daemon.
const packagedIndex = process.argv.indexOf('--packaged');
const packagedDir = packagedIndex !== -1 ? process.argv[packagedIndex + 1] : null;

// ------------------------------------------------------------- test harness

let passed = 0;
let failed = 0;

function pass(label, detail) {
  passed += 1;
  console.log(`PASS  ${label}${detail ? ` — ${detail}` : ''}`);
}

async function test(label, fn) {
  try {
    await fn();
    pass(label);
  } catch (error) {
    failed += 1;
    console.error(`FAIL  ${label} — ${error && error.message ? error.message : String(error)}`);
  }
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message || 'assertion failed');
  }
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(
      `${message || 'assertEqual'}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`,
    );
  }
}

function assertDeepEqual(actual, expected, message) {
  const left = JSON.stringify(actual);
  const right = JSON.stringify(expected);
  if (left !== right) {
    throw new Error(`${message || 'assertDeepEqual'}: expected ${right}, got ${left}`);
  }
}

function assertProtocol(fn, needle) {
  try {
    fn();
  } catch (error) {
    assert(
      error instanceof nc.NativeProtocolError,
      `expected NativeProtocolError, got ${error && error.name}: ${error && error.message}`,
    );
    if (needle) {
      assert(
        error.message.includes(needle),
        `message ${JSON.stringify(error.message)} does not mention ${JSON.stringify(needle)}`,
      );
    }
    return;
  }
  throw new Error('expected a NativeProtocolError, nothing was thrown');
}

async function assertRejects(factory, predicate, label) {
  try {
    await factory();
  } catch (error) {
    if (predicate) {
      assert(
        predicate(error),
        `${label || 'rejection'}: predicate rejected ${error && error.name}: ${error && error.message}`,
      );
    }
    return;
  }
  throw new Error(`${label || 'rejection'}: expected the promise to reject`);
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

// ---------------------------------------------------------- payload builders

const healthJson = { ok: true, version: '9.9.9' };
const readyJson = { ready: true };
const sessionCreatedJson = { id: '7', title: 'selftest', created_ms: 1 };
const sessionSummaryJson = {
  id: '7',
  title: 'selftest',
  provider: 'fake',
  model: 'm',
  state: 'idle',
};
const modelInfoJson = {
  provider: 'fake',
  model: 'm',
  context: 8192,
  maxOutput: 2048,
  tools: true,
  parallelTools: false,
  reasoning: true,
  thinking: true,
  vision: false,
  structuredOutput: true,
  embeddings: false,
  streaming: true,
  source: 'providerCatalog',
};
const projectionJson = {
  session: { id: '7', title: 'selftest', provider: 'fake', model: 'm', lifecycle: 'open' },
  state: { machine: 'idle', label: 'Idle', active: false, terminal: false },
  activeModel: null,
  activeTool: null,
  progress: null,
  filesChanged: ['a.ts'],
  lastCheckpoint: null,
  verification: [{ opId: '1', tool: 'bash', startedMs: 2, effectStatus: 'unknown' }],
  contextUsage: null,
  queued: 0,
  prefixStability: null,
};
const budgetJson = {
  maxTokens: null,
  maxTurns: null,
  spentTokens: null,
  spentTurns: null,
  maxCostMicro: null,
  spentCostMicro: 12,
  openReservedMicro: 3,
};
const taskViewJson = {
  goal: 'ship it',
  constraints: [],
  state: 'running',
  milestones: { completed: [], open: ['main'] },
  decisions: [],
  failures: [],
  changedFiles: ['a.ts'],
  tests: { run: ['cargo test'], failed: ['cargo test'] },
  preferences: [],
  verification: [{ id: 'v1', detail: 'failed:cargo test', status: 'failed' }],
  progress: null,
  budget: budgetJson,
};
const checkpointJson = {
  sequence: 1,
  path: '/tmp/a.ts',
  beforeHash: null,
  afterHash: 'ab',
  beforeExists: false,
  afterExists: true,
  createdMs: 1,
  restoredMs: null,
};
const verificationViewJson = {
  owed: [
    { opId: '1', tool: 'bash', startedMs: 1, status: 'running', effectStatus: 'unknown' },
  ],
  failedChecks: [{ id: 'v1', detail: 'failed:cargo test', status: 'failed' }],
};
const taskRunJson = {
  task_id: 1,
  run_id: 'r1',
  mode: 'in_session',
  state: 'Running',
  goal: 'ship it',
  item_ids: ['main'],
  model: null,
};
const taskRunStartedJson = { task_id: 1, run_id: 'r1', state: 'Pending' };
const taskRunCancelledJson = { run_id: 'r1', cancelled: true };
const agentsJson = [
  {
    agent_id: 'r1',
    kind: 'self',
    run_id: 'r1',
    session_id: 7,
    worktree_id: 1,
    goal: 'ship it',
    state: 'Running',
    model: null,
    budget: null,
    ownership: 'self',
    capabilities: [],
    progress: null,
    result: null,
    item_ids: ['main'],
  },
  {
    agent_id: 'c1',
    kind: 'child',
    run_id: 'r1',
    session_id: 8,
    worktree_id: 2,
    goal: 'implement main',
    state: 'Running',
    model: 'm',
    provider: 'fake',
    budget: 1000,
    ownership: 'orchestrator',
    capabilities: ['ReadWorkspace'],
    progress: { phase: 'work' },
    result: null,
    item_id: 'main',
    item_kind: 'Implementation',
  },
];
const controlAckJson = { queuedSeq: 3, applied: null };
const presentationAckJson = { child_id: 'c1', presentation: 'background', changed: true };
const boardPostJson = {
  id: 3,
  board_id: 7,
  author_child: 8,
  author_session: 8,
  subject: 'handoff',
  body: 'main step ready',
  refs: ['evidence:41'],
  revision: 3,
  created_ms: 1700,
};
const boardRootPostJson = {
  id: 2,
  board_id: 7,
  author_child: null,
  author_session: 7,
  subject: 'root note',
  body: 'no children yet',
  refs: [],
  revision: 2,
  created_ms: 1600,
};
const boardPageJson = {
  board_id: 7,
  revision: 3,
  posts: [boardPostJson, boardRootPostJson],
  next_before_revision: 2,
  has_more: true,
};
const emptyBoardPageJson = {
  board_id: 7,
  revision: 0,
  posts: [],
  next_before_revision: null,
  has_more: false,
};
const tournamentJson = {
  id: 't-1',
  run_family: 'run-7',
  goal: 'pick winner',
  criteria: [{ id: 'c-1', spec: 'tests pass' }],
  candidates: [
    {
      child_id: 'child-0',
      worktree: '/tmp/w0',
      base_revision: 'abc',
      state: 'done',
      verification: 12,
      verification_pass: true,
      review: { rank: 'clean', reviewer: 'rev-1' },
      cost_micro: 100,
      wall_ms: 1000,
    },
    {
      child_id: 'child-1',
      worktree: '/tmp/w1',
      base_revision: 'abc',
      state: 'running',
      verification: null,
      verification_pass: null,
      review: null,
      cost_micro: 50,
      wall_ms: 900,
    },
  ],
  winner: null,
  state: 'open',
};
const tournamentStartedJson = {
  tournament_id: 't-1',
  run_id: 'run-7',
  candidates: ['child-0', 'child-1'],
  state: 'open',
  winner: null,
};
const tournamentSummariesJson = [
  { id: 't-1', state: 'open', candidate_count: 2, winner: null, decided_ms: null },
  { id: 't-0', state: 'decided', candidate_count: 2, winner: 'child-0', decided_ms: 123 },
];
const tournamentDecisionJson = {
  tournament_id: 't-1',
  winner: 'child-0',
  rationale: 'winner child-0 (verification=pass)',
  discarded: [{ child_id: 'child-1', reason: 'candidate ended failed' }],
};
const messagePageJson = {
  sessionId: '7',
  messages: [
    {
      seq: 2,
      id: 2,
      role: 'assistant',
      createdMs: 2,
      data: {},
      parts: [{ kind: 'text', createdMs: 2, data: { text: 'hi' } }],
    },
  ],
  hasMore: false,
  nextBefore: null,
};
const eventPageJson = {
  sessionId: '7',
  events: [
    { seq: 1, kind: 'SessionCreated', state: 'idle', opId: null, tsMs: 3, payload: null },
  ],
  hasMore: false,
  nextCursor: null,
};
const reservationsJson = {
  open: { count: 0, predictedMicro: 0 },
  settled: { count: 1, predictedMicro: 11, spentMicro: 12, providerReportedMicro: 12 },
  refunded: { count: 0, predictedMicro: 0 },
  uncertain: { count: 0, predictedMicro: 0 },
  routeDecisions: [],
  truncated: false,
};
const sessionUsageJson = {
  sessionId: '7',
  providerCalls: {
    tokens: 130,
    prefixObservations: [{ rowId: 1, promptTokens: 100, stability: null }],
  },
  prefixStability: null,
  tasks: [{ taskId: 't1', budget: budgetJson, reservations: reservationsJson }],
};
const usageTotalsJson = {
  sessions: 1,
  totals: { budget: 0, spent: 0 },
  perSession: [{ sessionId: '7', budget: 0, spent: 0 }],
  durable: {
    sessionsWithCalls: 1,
    providerCalls: {
      tokens: 130,
      prefixObservations: 1,
      prefixTokens: 100,
      prefixStabilityObservations: 0,
    },
    taskSpend: { settledCostMicro: 12 },
    reservations: reservationsJson,
    truncated: false,
  },
};
const verificationRecordJson = {
  recordId: 'rec1',
  revision: 'rev1',
  workspaceId: '1',
  worktreeId: '1',
  treeHash: null,
  criteria: [{ criterionKey: 'build', passed: true, evidence: null }],
  checks: [
    {
      check: 'cargo test',
      program: 'cargo',
      args: ['test'],
      category: 'test',
      required: true,
      status: 'passed',
      startedMs: 1,
      finishedMs: 2,
      exit: 0,
      summary: null,
    },
  ],
  changedFiles: [{ path: 'a.ts', digestHex: 'ab', size: 4 }],
  unrelatedChanges: [],
  reviewer: null,
  status: 'passed',
  startedMs: 1,
  completedMs: 2,
};
const taskVerificationJson = {
  sessionId: '7',
  taskId: 't1',
  records: [verificationRecordJson],
};
const evidenceJson = {
  id: 41,
  kind: 'terminal',
  sessionId: 7,
  workspaceId: 1,
  taskId: null,
  sourceRevision: null,
  compressibility: null,
  backingCompleteness: null,
  backingRetained: true,
  backingLen: 3,
  allowRanges: true,
  allowSearch: true,
  maxBytes: 1024,
};
const evidenceRetrievalJson = {
  id: 41,
  selector: { selector: 'all' },
  bytesBase64: Buffer.from('ok\n').toString('base64'),
  byteLen: 3,
  truncatedByPolicy: false,
};
const semanticStatusJson = {
  configured: false,
  providerCount: 0,
  providers: [],
  fallback: { id: 'fallback', version: 1, capabilities: { operations: {} } },
  snapshotState: { providers: [], fallback: true },
};
const abortAckJson = { aborted: ['1'] };

// --------------------------------------------------------------- fake fetch

function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function makeClient(routes, options = {}) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    const parsed = new URL(url);
    const key = `${init.method} ${parsed.pathname}`;
    calls.push({
      url,
      method: init.method,
      path: parsed.pathname,
      query: Object.fromEntries(parsed.searchParams.entries()),
      headers: init.headers,
      body: init.body === undefined ? undefined : JSON.parse(init.body),
    });
    const handler = routes[key];
    if (!handler) {
      throw new Error(`unexpected request ${key} (${url})`);
    }
    return handler(calls[calls.length - 1]);
  };
  const client = new nc.NativeClient({
    baseUrl: 'http://127.0.0.1:9',
    bearerToken: 'selftest-token',
    fetch: fetchImpl,
    timeoutMs: 500,
    ...options,
  });
  return { client, calls };
}

function findCall(calls, method, path) {
  const call = calls.find((entry) => entry.method === method && entry.path === path);
  assert(call, `no ${method} ${path} request was made (calls: ${calls.map((c) => `${c.method} ${c.path}`).join(', ')})`);
  return call;
}

function sseResponse(frames, status = 200) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      for (const frame of frames) {
        controller.enqueue(encoder.encode(frame));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    status,
    headers: { 'content-type': 'text/event-stream' },
  });
}

function frame(event, id, data) {
  const head = event === null ? '' : `event: ${event}\n`;
  const idLine = id === null ? '' : `id: ${id}\n`;
  return `${head}${idLine}data: ${data}\n\n`;
}

// ----------------------------------------------------- 1. validator accepts

async function validatorAccepts() {
  await test('validators accept the documented shapes', () => {
    assertEqual(nc.validateHealth(clone(healthJson)).version, '9.9.9');
    assertEqual(nc.validateReady(clone(readyJson)).ready, true);
    assertEqual(nc.validateSessionCreated(clone(sessionCreatedJson)).id, '7');
    assertEqual(nc.validateSessionList({ sessions: [clone(sessionSummaryJson)] }).length, 1);
    assertEqual(nc.validateModelCatalog([clone(modelInfoJson)]).length, 1);
    assertEqual(nc.validateProjection(clone(projectionJson)).filesChanged[0], 'a.ts');
    assertEqual(nc.validateTurns([{ opId: '1', status: 'completed', provider: 'p', model: 'm', variant: null, toolMode: null, startedAt: 1, updatedMs: 2, queueSeq: null, promptMessageId: null }]).length, 1);
    assertEqual(nc.validateTaskViews([clone(taskViewJson)])[0].budget.spentCostMicro, 12);
    assertEqual(nc.validateCheckpoints([clone(checkpointJson)])[0].sequence, 1);
    assertEqual(nc.validateVerificationView(clone(verificationViewJson)).owed.length, 1);
    assertEqual(nc.validateTaskRuns([clone(taskRunJson)])[0].run_id, 'r1');
    assertEqual(nc.validateTaskRun(clone(taskRunJson)).task_id, 1);
    assertEqual(nc.validateTaskRunStarted(clone(taskRunStartedJson)).state, 'Pending');
    assertEqual(
      nc.validateAttachmentId({
        digest: 'a'.repeat(64),
        mime: 'application/pdf',
        filename: null,
        size: 8,
      }).size,
      8,
    );
    assertEqual(nc.validateTaskRunCancelled(clone(taskRunCancelledJson)).cancelled, true);
    assertEqual(nc.validateAgents(clone(agentsJson)).length, 2);
    assertEqual(nc.validateAgents(clone(agentsJson))[1].presentation, 'foreground');
    const presented = clone(agentsJson);
    presented[1].presentation = 'background';
    assertEqual(nc.validateAgents(presented)[1].presentation, 'background');
    assertEqual(nc.validateAgentControlAck(clone(controlAckJson), 'test').queuedSeq, 3);
    assertEqual(nc.validateAgentPresentationAck(clone(presentationAckJson), 'test').presentation, 'background');
    assertEqual(nc.validateAgents(clone(agentsJson))[1].provider, 'fake');
    assertEqual(nc.validateAgents(clone(agentsJson))[0].provider, null, 'self entries carry no provider');
    assertEqual(nc.validateTournament(clone(tournamentJson)).candidates[0].reviewRank, 'clean');
    assertEqual(nc.validateTournament(clone(tournamentJson)).candidates[0].verificationPass, true);
    assertEqual(nc.validateTournament(clone(tournamentJson)).candidates[1].reviewRank, null);
    assertEqual(nc.validateTournamentStarted(clone(tournamentStartedJson)).candidates.length, 2);
    assertEqual(nc.validateTournamentSummaries(clone(tournamentSummariesJson)).length, 2);
    assertEqual(nc.validateTournamentDecision(clone(tournamentDecisionJson)).winner, 'child-0');
    // v1 additive: a pre-provider daemon entry validates with provider null.
    const legacyAgent = clone(agentsJson[1]);
    delete legacyAgent.provider;
    assertEqual(nc.validateAgents([legacyAgent])[0].provider, null);
    assertEqual(nc.validateMessagePage(clone(messagePageJson)).messages[0].parts[0].kind, 'text');
    assertEqual(nc.validateEventPage(clone(eventPageJson)).events[0].seq, 1);
    assertEqual(nc.validateSessionUsage(clone(sessionUsageJson)).tasks[0].taskId, 't1');
    assertEqual(nc.validateUsage(clone(usageTotalsJson)).durable.providerCalls.tokens, 130);
    const aggregateUsage = clone(usageTotalsJson);
    delete aggregateUsage.durable.reservations.routeDecisions;
    assertDeepEqual(nc.validateUsage(aggregateUsage).durable.reservations.routeDecisions, []);
    assertEqual(nc.validateTaskVerification(clone(taskVerificationJson)).records[0].checks[0].exit, 0);
    assertEqual(nc.validateBoardPage(clone(boardPageJson)).posts[0].revision, 3);
    assertEqual(nc.validateBoardPage(clone(boardPageJson)).posts[1].author_child, null);
    assertEqual(nc.validateBoardPage(clone(boardPageJson)).next_before_revision, 2);
    assertEqual(nc.validateBoardPage(clone(emptyBoardPageJson)).revision, 0, 'an empty board has revision 0');
    assertEqual(nc.validateBoardPost(clone(boardPostJson)).subject, 'handoff');
    assertEqual(nc.validateEvidence(clone(evidenceJson)).id, 41);
    assertEqual(nc.validateEvidenceRetrieval(clone(evidenceRetrievalJson)).byteLen, 3);
    assertEqual(nc.validateSemanticStatus(clone(semanticStatusJson)).providerCount, 0);
    assertEqual(nc.validateSemanticStatus(clone(semanticStatusJson)).fallback.version, 1);
    assertEqual(nc.validateAbortAck(clone(abortAckJson)).aborted[0], '1');

    // v1 additive contract: a newer daemon's unknown optional field is
    // ignored at every nesting level, never a rejection.
    assertEqual(nc.validateHealth({ ...clone(healthJson), daemon_build: 'future' }).version, '9.9.9');
    assertEqual(nc.validateReady({ ...clone(readyJson), queue_depth: 0 }).ready, true);
    assertEqual(
      nc.validateProjection({
        ...clone(projectionJson),
        state: { ...projectionJson.state, futureLabel: 'x' },
        futureRoot: { nested: true },
      }).queued,
      0,
    );
    assertEqual(
      nc.validateTaskViews([{ ...clone(taskViewJson), futureTaskField: [1, 2] }])[0].goal,
      'ship it',
    );
  });
}

// ------------------------------------------------------ 2. validator rejects

async function validatorRejects() {
  await test('validators reject missing fields and bad known-field types', () => {
    assertProtocol(() => nc.validateHealth({ ok: true }), 'missing required field version');
    assertProtocol(() => nc.validateHealth({ ok: true, version: 7 }), 'expected a string, got number');
    assertProtocol(() => nc.validateReady({ ready: 'yes' }), 'expected a boolean');
    assertProtocol(
      () => nc.validateModelCatalog([{ ...clone(modelInfoJson), context: '8192' }]),
      'expected a finite number',
    );
    assertProtocol(
      () => nc.validateProjection({ ...clone(projectionJson), state: { machine: 'idle', label: 'Idle', active: false, terminal: false, rogue: 1 }, queued: '0' }),
      'expected a finite number',
    );
    assertProtocol(() => nc.validateAgents([{ ...clone(agentsJson[1]), kind: 'parent' }]), 'expected "self" or "child"');
    assertProtocol(() => nc.validateAgents([{ ...clone(agentsJson[1]), budget: 1.5 }]), 'expected an integer or null');
    assertProtocol(
      () => nc.validateAgents([{ ...clone(agentsJson[1]), presentation: 'hidden' }]),
      'expected "foreground" or "background"',
    );
    assertProtocol(
      () => nc.validateAgents([{ ...clone(agentsJson[1]), provider: 7 }]),
      'expected a string or null',
    );
    assertProtocol(
      () =>
        nc.validateTournament({
          ...clone(tournamentJson),
          candidates: [{ ...tournamentJson.candidates[0], cost_micro: '100' }],
        }),
      'expected a finite number',
    );
    assertProtocol(
      () => nc.validateTournamentDecision({ tournament_id: 't', winner: 'c' }),
      'missing required field rationale',
    );
    assertProtocol(
      () => nc.validateAgentPresentationAck({ child_id: 'c1', presentation: null, changed: true }, 'test'),
      'expected "foreground" or "background"',
    );
    assertProtocol(() => nc.validateMessagePage({ ...clone(messagePageJson), messages: [{ seq: '2' }] }), 'missing required field id');
    assertProtocol(() => nc.validateEventPage({ ...clone(eventPageJson), events: [{ ...eventPageJson.events[0], opId: 7 }] }), 'expected a string or null');
    const missingDurable = clone(usageTotalsJson);
    delete missingDurable.durable;
    assertProtocol(() => nc.validateUsage(missingDurable), 'missing required field durable');
    assertProtocol(() => nc.validateSemanticStatus({ ...clone(semanticStatusJson), fallback: null }), 'expected an object');
    assertProtocol(() => nc.validateCheckpoints([{ ...clone(checkpointJson), beforeExists: 'nope' }]), 'expected a boolean');
    assertProtocol(
      () => nc.validateAttachmentId({ digest: 'ZZ', mime: 'application/pdf', filename: null, size: 8 }),
      '64-char lowercase hex digest',
    );
    assertProtocol(
      () => nc.validateAttachmentId({ digest: 'a'.repeat(64), mime: 'application/pdf', filename: null }),
      'missing required field size',
    );
    assertProtocol(() => nc.validateTaskViews([{ ...clone(taskViewJson), budget: { ...clone(budgetJson), spentCostMicro: '12' } }]), 'expected a finite number');
    // Board: absent fields, hostile types and phantom entries fail loudly.
    const missingBoardField = clone(boardPageJson);
    delete missingBoardField.has_more;
    assertProtocol(() => nc.validateBoardPage(missingBoardField), 'missing required field has_more');
    assertProtocol(
      () => nc.validateBoardPage({ ...clone(boardPageJson), posts: [{ ...clone(boardPostJson), revision: 0 }] }),
      'expected a positive integer',
    );
    assertProtocol(
      () => nc.validateBoardPage({ ...clone(boardPageJson), posts: [{ ...clone(boardPostJson), author_child: 'root' }] }),
      'expected an integer or null',
    );
    assertProtocol(
      () => nc.validateBoardPage({ ...clone(boardPageJson), revision: -1 }),
      'expected a non-negative integer',
    );
    assertProtocol(
      () => nc.validateBoardPage({ ...clone(boardPageJson), posts: [{ ...clone(boardPostJson), refs: [7] }] }),
      'expected a string',
    );
    assertProtocol(
      () => nc.validateBoardPost({ ...clone(boardPostJson), id: 3.5 }),
      'expected an integer',
    );
    assertProtocol(() => nc.validateBoardPost({}), 'missing required field id');
  });
}

// -------------------------------------------------------------- 3. client IO

async function clientAccepts() {
  await test('client speaks every native endpoint with strict validation', async () => {
    const routes = {
      'GET /native/health': () => jsonResponse(healthJson),
      'GET /native/ready': () => jsonResponse(readyJson),
      'POST /session/create': () => jsonResponse(sessionCreatedJson),
      'GET /session/list': () => jsonResponse({ sessions: [sessionSummaryJson] }),
      'GET /session/7/projection': () => jsonResponse(projectionJson),
      'GET /models': () => jsonResponse([modelInfoJson]),
      'GET /native/session/7/turns': () => jsonResponse([]),
      'GET /native/session/7/tasks': () => jsonResponse([taskViewJson]),
      'GET /native/session/7/checkpoints': () => jsonResponse([checkpointJson]),
      'GET /native/session/7/verification': () => jsonResponse(verificationViewJson),
      'GET /native/session/7/task-runs': () => jsonResponse([taskRunJson]),
      'GET /native/session/7/task-runs/r1': () => jsonResponse(taskRunJson),
      'POST /native/session/7/task-runs': () => jsonResponse(taskRunStartedJson),
      'POST /native/session/7/attachments': () =>
        jsonResponse({ digest: 'a'.repeat(64), mime: 'application/pdf', filename: 'spec.pdf', size: 8 }),
      'POST /native/session/7/task-runs/r1/cancel': () => jsonResponse(taskRunCancelledJson),
      'GET /native/agents': () => jsonResponse(agentsJson),
      'POST /native/agents/c1/pause': () => jsonResponse(controlAckJson),
      'POST /native/agents/c1/resume': () => jsonResponse(controlAckJson),
      'POST /native/agents/c1/cancel': () => jsonResponse(controlAckJson),
      'POST /native/agents/c1/retry': () => jsonResponse(controlAckJson),
      'POST /native/agents/c1/steer': () => jsonResponse(controlAckJson),
      'POST /native/agents/c1/model': () => jsonResponse(controlAckJson),
      'POST /native/agents/c1/budget': () => jsonResponse(controlAckJson),
      'POST /native/session/7/agents/c1/presentation': () => jsonResponse(presentationAckJson),
      'GET /native/session/7/tournaments': () => jsonResponse(tournamentSummariesJson),
      'GET /native/session/7/tournament/t-1': () => jsonResponse(tournamentJson),
      'POST /native/session/7/tournament': () => jsonResponse(tournamentStartedJson),
      'POST /native/session/7/tournaments/t-1/decide': () => jsonResponse(tournamentDecisionJson),
      'POST /native/session/7/tournaments/t-1/abort': () =>
        jsonResponse({ ...clone(tournamentJson), state: 'aborted' }),
      'GET /native/session/7/board': () => jsonResponse(boardPageJson),
      'POST /native/session/7/board': () => jsonResponse(boardPostJson),
      'GET /native/messages': () => jsonResponse(messagePageJson),
      'GET /native/events': () => jsonResponse(eventPageJson),
      'GET /native/usage': () => jsonResponse(usageTotalsJson),
      'GET /native/session/7/usage': () => jsonResponse(sessionUsageJson),
      'GET /native/session/7/tasks/t1/verification': () => jsonResponse(taskVerificationJson),
      'GET /native/evidence/41': () => jsonResponse(evidenceJson),
      'POST /native/evidence/41/retrieve': () => jsonResponse(evidenceRetrievalJson),
      'GET /native/semantic/status': () => jsonResponse(semanticStatusJson),
      'POST /native/session/7/abort': () => jsonResponse(abortAckJson),
    };
    const { client, calls } = makeClient(routes);

    assertEqual((await client.health()).ok, true);
    assertEqual((await client.ready()).ready, true);
    assertEqual((await client.listSessions())[0].id, '7');
    assertEqual((await client.modelCatalog())[0].model, 'm');
    assertEqual(
      (await client.createSession({ provider: 'fake', model: 'm', workspace: '/w', title: 'selftest' })).id,
      '7',
    );
    assertEqual((await client.projection('7')).queued, 0);
    assertDeepEqual(await client.turns('7'), []);
    assertEqual((await client.tasks('7'))[0].goal, 'ship it');
    assertEqual((await client.checkpoints('7'))[0].path, '/tmp/a.ts');
    assertEqual((await client.verification('7')).failedChecks.length, 1);
    assertEqual((await client.taskRuns('7'))[0].state, 'Running');
    assertEqual((await client.taskRunState('7', 'r1')).run_id, 'r1');    assertEqual((await client.startTaskRun('7', { goal: 'ship it' })).run_id, 'r1');
    assertEqual((await client.uploadAttachment('7', { mime: 'application/pdf', filename: 'spec.pdf', data_base64: 'eA==' })).digest, 'a'.repeat(64));
    assertEqual((await client.cancelTaskRun('7', 'r1')).cancelled, true);
    assertEqual((await client.agents('7')).length, 2);
    assertEqual((await client.pauseAgent('c1')).queuedSeq, 3);
    assertEqual((await client.resumeAgent('c1')).queuedSeq, 3);
    assertEqual((await client.cancelAgent('c1')).queuedSeq, 3);
    assertEqual((await client.retryAgent('c1')).queuedSeq, 3);
    assertEqual((await client.steerAgent('c1', 'focus')).queuedSeq, 3);
    assertEqual((await client.setAgentModel('c1', 'm')).queuedSeq, 3);
    assertEqual((await client.setAgentBudget('c1', { max_tokens: 1000 })).queuedSeq, 3);
    assertEqual(
      (await client.setAgentPresentation('7', 'c1', 'background')).presentation,
      'background',
    );
    assertEqual((await client.tournaments('7'))[0].id, 't-1');
    assertEqual((await client.tournamentState('7', 't-1')).candidates.length, 2);
    assertEqual(
      (await client.startTournament('7', { goal: 'pick', criteria: ['tests pass'], n: 2 })).state,
      'open',
    );
    assertEqual((await client.decideTournament('7', 't-1')).winner, 'child-0');
    assertEqual((await client.abortTournament('7', 't-1', 'smoke reason')).state, 'aborted');
    assertEqual((await client.board('7', { since: 9, limit: 2 })).posts[0].subject, 'handoff');
    assertEqual((await client.board('7')).next_before_revision, 2);
    assertEqual(
      (await client.boardPost('7', { subject: 'status', body: 'all green', refs: ['evidence:41'] }))
        .revision,
      3,
    );
    assertEqual((await client.messages('7', { before: 9, limit: 2 })).messages[0].id, 2);
    assertEqual((await client.events('7', { after: 7, limit: 3 })).events[0].seq, 1);
    assertEqual((await client.usage()).sessions, 1);
    assertEqual((await client.sessionUsage('7')).providerCalls.tokens, 130);
    assertEqual((await client.taskVerification('7', 't1')).records[0].recordId, 'rec1');
    assertEqual((await client.evidence('7', 41)).backingRetained, true);
    assertEqual((await client.retrieveEvidence('7', 41, { selector: 'all' })).byteLen, 3);
    assertEqual((await client.semanticStatus()).snapshotState.fallback, true);
    assertEqual((await client.abortSession('7', '3')).aborted[0], '1');

    // Request construction: auth, bodies, paths, cursor paging.
    const health = findCall(calls, 'GET', '/native/health');
    assertEqual(health.headers.Authorization, 'Bearer selftest-token');
    assertDeepEqual(findCall(calls, 'POST', '/session/create').body, {
      provider: 'fake',
      model: 'm',
      workspace: '/w',
      title: 'selftest',
    });
    assertDeepEqual(findCall(calls, 'POST', '/native/session/7/task-runs').body, { goal: 'ship it' });
    assertDeepEqual(findCall(calls, 'POST', '/native/session/7/attachments').body, {
      mime: 'application/pdf',
      filename: 'spec.pdf',
      data_base64: 'eA==',
    });
    assertDeepEqual(findCall(calls, 'GET', '/native/messages').query, {
      session: '7',
      before: '9',
      limit: '2',
    });
    assertDeepEqual(findCall(calls, 'GET', '/native/events').query, {
      session: '7',
      after: '7',
      limit: '3',
    });
    assertDeepEqual(findCall(calls, 'GET', '/native/agents').query, { session: '7' });
    assertDeepEqual(findCall(calls, 'POST', '/native/agents/c1/steer').body, { text: 'focus' });
    assertDeepEqual(findCall(calls, 'POST', '/native/agents/c1/model').body, { model: 'm' });
    assertDeepEqual(findCall(calls, 'POST', '/native/agents/c1/budget').body, { max_tokens: 1000 });
    assertDeepEqual(findCall(calls, 'POST', '/native/session/7/agents/c1/presentation').body, {
      state: 'background',
    });
    assertDeepEqual(findCall(calls, 'POST', '/native/session/7/tournament').body, {
      goal: 'pick',
      criteria: ['tests pass'],
      n: 2,
    });
    assertDeepEqual(findCall(calls, 'POST', '/native/session/7/tournaments/t-1/decide').body, {});
    assertDeepEqual(findCall(calls, 'POST', '/native/session/7/tournaments/t-1/abort').body, {
      reason: 'smoke reason',
    });
    assertDeepEqual(findCall(calls, 'POST', '/native/evidence/41/retrieve').body, {
      selector: 'all',
    });
    assertDeepEqual(findCall(calls, 'POST', '/native/evidence/41/retrieve').query, { session: '7' });
    assertDeepEqual(findCall(calls, 'GET', '/native/session/7/board').query, {
      since: '9',
      limit: '2',
    });
    assertDeepEqual(findCall(calls, 'POST', '/native/session/7/board').body, {
      subject: 'status',
      body: 'all green',
      refs: ['evidence:41'],
    });
    assertDeepEqual(findCall(calls, 'POST', '/native/session/7/abort').body, {
      session_id: '7',
      op_id: '3',
    });
  });
}

async function clientRejects() {
  await test('client accepts additive response fields and rejects known-field type drift', async () => {
    const additive = makeClient({
      'GET /native/health': () => jsonResponse({ ok: true, version: '1', future_optional: { hint: 'v2' } }),
    });
    assertEqual((await additive.client.health()).version, '1');

    const drift = makeClient({
      'GET /native/health': () => jsonResponse({ ok: 'yes', version: '1' }),
    });
    await assertRejects(
      () => drift.client.health(),
      (error) => error instanceof nc.NativeProtocolError && /expected a boolean/.test(error.message),
      'known-field type drift',
    );

    const missing = makeClient({ 'GET /native/health': () => jsonResponse({ ok: true }) });
    await assertRejects(
      () => missing.client.health(),
      (error) => error instanceof nc.NativeProtocolError && /missing required field version/.test(error.message),
      'missing known field',
    );
  });

  await test('client maps API error envelopes and rejects malformed ones', async () => {
    const api = makeClient({
      'GET /native/health': () =>
        new Response(
          JSON.stringify({ error: { code: 'unauthorized', message: 'nope', retryable: false } }),
          { status: 401 },
        ),
    });
    await assertRejects(
      () => api.client.health(),
      (error) => error instanceof nc.NativeApiError && error.code === 'unauthorized' && error.status === 401,
      'api error envelope',
    );
    const garbage = makeClient({
      'GET /native/health': () => new Response('not json at all', { status: 500 }),
    });
    await assertRejects(
      () => garbage.client.health(),
      (error) => error instanceof nc.NativeApiError && error.code === 'http_error',
      'non-JSON error body',
    );
    const envelopeOnly = makeClient({
      'GET /native/health': () => new Response(JSON.stringify({ error: { code: 'x' } }), { status: 500 }),
    });
    await assertRejects(
      () => envelopeOnly.client.health(),
      (error) => error instanceof nc.NativeProtocolError,
      'error envelope without message',
    );
  });

  await test('client bounds response bodies and request time', async () => {
    const { client } = makeClient(
      { 'GET /native/health': () => jsonResponse({ ok: true, version: 'x'.repeat(4096) }) },
      { maxBodyBytes: 64 },
    );
    await assertRejects(
      () => client.health(),
      (error) => error instanceof nc.NativeProtocolError && /exceeded bound/.test(error.message),
      'streamed body bound',
    );

    const declared = {
      status: 200,
      ok: true,
      headers: { get: (name) => (name.toLowerCase() === 'content-length' ? '100000' : null) },
      body: null,
      arrayBuffer: async () => new ArrayBuffer(0),
      text: async () => '',
    };
    const declaredFetch = new nc.NativeClient({
      baseUrl: 'http://127.0.0.1:9',
      bearerToken: 't',
      fetch: async () => declared,
      maxBodyBytes: 32,
    });
    await assertRejects(
      () => declaredFetch.health(),
      (error) => error instanceof nc.NativeProtocolError && /declared body/.test(error.message),
      'declared body bound',
    );

    const hanging = new nc.NativeClient({
      baseUrl: 'http://127.0.0.1:9',
      bearerToken: 't',
      timeoutMs: 20,
      fetch: (url, init) =>
        new Promise((resolve, reject) => {
          init.signal.addEventListener('abort', () => reject(new Error('aborted')));
        }),
    });
    await assertRejects(
      () => hanging.health(),
      (error) => error instanceof nc.NativeProtocolError && /timed out/.test(error.message),
      'request timeout',
    );
  });
}

// ------------------------------------------------------------ 4. eventStream

async function eventStreamTests() {
  await test('eventStream tolerates heartbeats, suppresses replay, reports bad frames', async () => {
    const urls = [];
    const delivered = [];
    const errors = [];
    const statuses = [];
    let stream = null;
    const fetchImpl = async (url) => {
      urls.push(url);
      return sseResponse([
        frame('message_created', 1, '{"event":"message_created","session_id":"5","message":{"id":"1"}}'),
        frame('heartbeat', null, '{}'),
        ': keep-alive\n\n',
        frame('agent_state_changed', 2, '{"event":"agent_state_changed","session_id":"5","state":"streaming","label":"streaming"}'),
        frame('agent_state_changed', 2, '{"event":"agent_state_changed","session_id":"5","state":"streaming","label":"streaming"}'),
        frame('error', 3, 'not-json'),
        frame('error', 4, '{"event":"agent_state_changed","session_id":"5"}'),
        frame('heartbeat', 5, '{}'),
      ]);
    };
    stream = new es.EventStream({
      baseUrl: 'http://127.0.0.1:9/',
      bearerToken: 'tok',
      sessionId: '5',
      cursor: 0,
      fetch: fetchImpl,
      sleep: async () => {},
      onEvent: (event) => delivered.push(event),
      onStatus: (status, detail) => {
        statuses.push(status);
        if (status === 'retrying' && String(detail).startsWith('stream ended')) {
          stream.stop();
        }
      },
      onError: (error) => errors.push(error),
    });
    stream.start();
    await stream.whenStopped();
    assertDeepEqual(delivered.map((event) => event.id), [1, 2]);
    assertEqual(stream.cursor, 5, 'heartbeat ids must advance the cursor');
    assert(statuses.includes('open'), `statuses included open: ${statuses.join(',')}`);
    assertEqual(errors.length, 2);
    assert(
      errors.every((error) => error instanceof es.EventStreamProtocolError),
      `protocol errors expected: ${errors.map((e) => e.message).join(' | ')}`,
    );
    assertEqual(new URL(urls[0]).searchParams.get('events_after'), '0');
    assertEqual(new URL(urls[0]).searchParams.get('session'), null);
  });

  await test('eventStream resumes from the cursor with backoff', async () => {
    const urls = [];
    const delivered = [];
    const sleeps = [];
    let call = 0;
    let stream = null;
    const fetchImpl = async (url) => {
      urls.push(url);
      call += 1;
      if (call === 1) {
        return sseResponse([frame('heartbeat', 7, '{}')]);
      }
      return sseResponse([
        frame('agent_state_changed', 7, '{"event":"agent_state_changed","session_id":"5","state":"x","label":"x"}'),
        frame('agent_state_changed', 8, '{"event":"agent_state_changed","session_id":"5","state":"y","label":"y"}'),
      ]);
    };
    stream = new es.EventStream({
      baseUrl: 'http://127.0.0.1:9',
      bearerToken: 'tok',
      sessionId: '5',
      cursor: 6,
      fetch: fetchImpl,
      minBackoffMs: 5,
      maxBackoffMs: 50,
      jitter: () => 0,
      sleep: async (ms) => {
        sleeps.push(ms);
      },
      onEvent: (event) => {
        delivered.push(event.id);
        if (event.id === 8) {
          stream.stop();
        }
      },
    });
    stream.start();
    await stream.whenStopped();
    assertEqual(urls.length, 2, 'one reconnect was expected');
    assertEqual(new URL(urls[0]).searchParams.get('events_after'), '6');
    assertEqual(new URL(urls[1]).searchParams.get('events_after'), '7');
    assertDeepEqual(delivered, [8], 'the replayed frame behind the cursor must not redeliver');
    assert(sleeps.length >= 1 && sleeps[0] >= 5, `backoff slept: ${JSON.stringify(sleeps)}`);
    assertEqual(stream.status, 'stopped');
  });

  await test('eventStream surfaces transport rejection and bounds frames', async () => {
    const errors = [];
    let stream = null;
    stream = new es.EventStream({
      baseUrl: 'http://127.0.0.1:9',
      bearerToken: 'tok',
      sessionId: '5',
      fetch: async () => new Response('denied', { status: 401 }),
      sleep: async () => {},
      onEvent: () => {},
      onError: (error) => {
        errors.push(error);
        stream.stop();
      },
    });
    stream.start();
    await stream.whenStopped();
    assertEqual(errors.length, 1);
    assert(errors[0].message.includes('HTTP 401'), errors[0].message);

    const frameErrors = [];
    let bounded = null;
    bounded = new es.EventStream({
      baseUrl: 'http://127.0.0.1:9',
      bearerToken: 'tok',
      sessionId: '5',
      maxFrameBytes: 32,
      fetch: async () => sseResponse(['data: ' + 'x'.repeat(256)]),
      sleep: async () => {},
      onEvent: () => {},
      onError: (error) => {
        frameErrors.push(error);
        bounded.stop();
      },
    });
    bounded.start();
    await bounded.whenStopped();
    assertEqual(frameErrors.length, 1);
    assert(/unterminated frame exceeded/.test(frameErrors[0].message), frameErrors[0].message);
  });
}

// ------------------------------------------------------------ 5. state store

async function stateTests() {
  await test('store notifies subscribers exactly once per change', () => {
    const store = new st.FaktorStore();
    let calls = 0;
    const unsubscribe = store.subscribe(() => {
      calls += 1;
    });
    store.patch({ daemon: 'starting' });
    assertEqual(calls, 1);
    store.patch({ daemon: 'starting' });
    assertEqual(calls, 1, 'same-value patches must not notify');
    store.patch({ daemon: 'running', daemonDetail: 'up' });
    assertEqual(calls, 2);
    assertEqual(store.snapshot().daemon, 'running');
    unsubscribe();
    store.patch({ daemon: 'stopped' });
    assertEqual(calls, 2, 'unsubscribed listeners must not fire');
  });

  await test('transcript reducer renders durable message pages and SSE frames', () => {
    const messages = [
      {
        seq: 2,
        id: 2,
        role: 'assistant',
        createdMs: 2,
        data: {},
        parts: [
          { kind: 'text', createdMs: 2, data: { text: 'hello' } },
          {
            kind: 'tool_call',
            createdMs: 2,
            data: { tool_call_id: 'c1', name: 'bash', input: { cmd: 'ls' }, state: 'running' },
          },
          {
            kind: 'tool_result',
            createdMs: 2,
            data: { tool_call_id: 'c1', excerpt: 'ok', exit_code: 0, artifact: 'evidence:41' },
          },
        ],
      },
      {
        seq: 1,
        id: 1,
        role: 'user',
        createdMs: 1,
        data: { files: [], text: 'do it' },
        parts: [],
      },
    ];
    const entries = st.transcriptFromMessages(messages);
    assertEqual(entries.length, 2);
    assertEqual(entries[0].role, 'user');
    assertEqual(entries[0].text, 'do it');
    assertEqual(entries[1].text, 'hello', 'durable text parts must render');
    assertEqual(entries[1].tools.length, 1);
    assertEqual(entries[1].tools[0].name, 'bash');
    assertEqual(entries[1].tools[0].state, 'running');
    assertEqual(entries[1].tools[0].excerpt, 'ok');
    assertEqual(entries[1].tools[0].exitCode, 0);
    assertEqual(entries[1].tools[0].artifact, 'evidence:41');

    let sse = st.applySseEvent([], 'message_created', {
      event: 'message_created',
      session_id: '7',
      message: {
        id: '9',
        role: 'assistant',
        seq: 9,
        created_ms: 1,
        parts: [{ type: 'text', text: 'hi' }],
      },
    });
    assertEqual(sse.length, 1);
    sse = st.applySseEvent(sse, 'message_part_updated', {
      message_id: '9',
      part: { type: 'text', text: ' there' },
    });
    assertEqual(sse[0].text, 'hi there');
    sse = st.applySseEvent(sse, 'message_part_updated', {
      message_id: '9',
      part: { type: 'tool_call', tool_call_id: 'c1', name: 'bash', input: {}, state: 'running' },
    });
    assertEqual(sse[0].tools[0].state, 'running');
    sse = st.applySseEvent(sse, 'tool_call_state', { tool_call_id: 'c1', state: 'completed' });
    assertEqual(sse[0].tools[0].state, 'completed');
    assert(st.applySseEvent(sse, 'unrelated_event', {}) === sse, 'unrelated events must not clone');
  });

  await test('transcript is bounded to MAX_TRANSCRIPT_ENTRIES', () => {
    const messages = [];
    for (let i = 0; i < st.MAX_TRANSCRIPT_ENTRIES + 25; i += 1) {
      messages.push({ seq: i, id: String(i), role: 'user', createdMs: i, data: {}, parts: [] });
    }
    const entries = st.transcriptFromMessages(messages);
    assertEqual(entries.length, st.MAX_TRANSCRIPT_ENTRIES);
    assertEqual(entries[entries.length - 1].seq, 0, 'newest messages survive the bound');
  });
}

// ---------------------------------------------------------------- 6. daemon

async function daemonTests() {
  await test('daemon resolves an explicit binary path', () => {
    assertEqual(
      dm.findBinary({ workspaceRoot: '/nonexistent', binaryPath: '/tmp/fake-faktor-cli' }),
      '/tmp/fake-faktor-cli',
    );
  });

  await test('daemon refuses to start without a binary', async () => {
    const saved = process.env.FAKTOR_BIN;
    delete process.env.FAKTOR_BIN;
    try {
      await assertRejects(
        () => dm.startDaemon({ workspaceRoot: '/nonexistent-root-for-selftest' }),
        (error) => error instanceof Error && /binary not found/.test(error.message),
        'missing binary',
      );
    } finally {
      if (saved !== undefined) {
        process.env.FAKTOR_BIN = saved;
      }
    }
  });
}

// ---------------------------------------- 7. VS Code product defect fixes

async function shadowDefaultTests() {
  await test('P0 shadow default: empty setting inherits the daemon, never direct_compat', () => {
    const base = ts.startTaskRequest('goal', { mutationMode: '', maxTokens: 0, maxCostMicro: 0 });
    assert(!('mutation_mode' in base), `empty setting must omit mutation_mode: ${JSON.stringify(base)}`);
    assertDeepEqual(
      ts.startTaskRequest('goal', { mutationMode: 'shadow', maxTokens: 10, maxCostMicro: 5 }),
      { goal: 'goal', max_tokens: 10, max_cost_micro: 5, mutation_mode: 'shadow' },
    );
    assertEqual(
      ts.startTaskRequest('goal', { mutationMode: 'direct_compat', maxTokens: 0, maxCostMicro: 0 }).mutation_mode,
      'direct_compat',
    );
    const manifest = JSON.parse(
      readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
    );
    assertEqual(
      manifest.contributes.configuration.properties['faktor.mutationMode'].default,
      '',
      'the setting default must be inherit-daemon',
    );
  });

  await test('409 shadow refusal is typed/actionable and attempted exactly once (no downgrade)', async () => {
    const calls = [];
    const conflict = new nc.NativeApiError(
      409,
      'conflict',
      'session 7 has no registered worktree row; shadowed mutating runs need a real owner worktree',
      false,
    );
    const client = {
      startTaskRun: async (sessionId, request) => {
        calls.push({ sessionId, request });
        throw conflict;
      },
    };
    const failures = [];
    const outcome = await ts.startTaskRun({
      client,
      sessionId: '7',
      goal: 'ship it',
      settings: { mutationMode: '', maxTokens: 0, maxCostMicro: 0 },
      onStarted: () => {
        throw new Error('must not start');
      },
      onFailure: (failure) => failures.push(failure),
    });
    assertEqual(calls.length, 1, 'a 409 must never be retried (no silent downgrade)');
    assert(!('mutation_mode' in calls[0].request), 'the refused request must carry no fabricated mode');
    assertEqual(outcome.ok, false);
    assertEqual(failures.length, 1);
    assertEqual(failures[0].kind, 'shadow_unregistered');
    assert(
      failures[0].message.includes('direct_compat'),
      `message must name the opt-in: ${failures[0].message}`,
    );
    assert(failures[0].message.includes('native API error 409'), failures[0].message);
  });

  await test('start failures classify 4xx/5xx/transport and never start; success acks once', async () => {
    const classify = async (error) => {
      const client = {
        startTaskRun: async () => {
          throw error;
        },
      };
      const failures = [];
      const outcome = await ts.startTaskRun({
        client,
        sessionId: '7',
        goal: 'g',
        settings: { mutationMode: '', maxTokens: 0, maxCostMicro: 0 },
        onStarted: () => {},
        onFailure: (failure) => failures.push(failure),
      });
      assertEqual(outcome.ok, false);
      assertEqual(failures.length, 1);
      return failures[0];
    };
    assertEqual((await classify(new nc.NativeApiError(400, 'malformed', 'bad body', false))).kind, 'validation');
    assertEqual((await classify(new nc.NativeApiError(500, 'internal', 'boom', false))).kind, 'server');
    assertEqual((await classify(new nc.NativeApiError(401, 'unauthorized', 'nope', false))).kind, 'auth');
    assertEqual((await classify(new Error('socket closed'))).kind, 'transport');

    const started = [];
    const outcome = await ts.startTaskRun({
      client: { startTaskRun: async () => taskRunStartedJson },
      sessionId: '7',
      goal: 'g',
      settings: { mutationMode: 'shadow', maxTokens: 0, maxCostMicro: 0 },
      onStarted: (run) => started.push(run),
      onFailure: () => {
        throw new Error('must not fail');
      },
    });
    assertEqual(outcome.ok, true);
    assertEqual(outcome.runId, 'r1');
    assertEqual(started.length, 1);
  });
}

// ------------------------- 7b. Task-mode completion contract + file forwarding

async function completionContractTests() {
  await test('completion contract parsing is strict; all-false is the default path', () => {
    assertDeepEqual(ts.completionContractSetting(undefined), null);
    assertDeepEqual(ts.completionContractSetting(null), null);
    assertDeepEqual(
      ts.completionContractSetting({
        include_commit: false,
        include_push: false,
        include_pr: false,
      }),
      null,
      'all-false is the default behavior and must not reach the wire',
    );
    assertDeepEqual(
      ts.completionContractSetting({
        include_commit: true,
        include_push: false,
        include_pr: true,
      }),
      { include_commit: true, include_push: false, include_pr: true },
    );
    // Hostile shapes are refused to null, never coerced or partially applied.
    for (const hostile of [
      'commit',
      ['include_commit'],
      true,
      42,
      {},
      { include_commit: true },
      { include_commit: 'yes', include_push: false, include_pr: false },
      { include_commit: 1, include_push: 0, include_pr: 0 },
      { include_commit: true, include_push: false, include_pr: false, include_release: true },
      Object.assign(
        Object.create({ include_commit: true, include_push: false, include_pr: false }),
        { include_pr: true, include_commit: true },
      ),
    ]) {
      assertDeepEqual(
        ts.completionContractSetting(hostile),
        null,
        `hostile contract must be refused: ${JSON.stringify(hostile)}`,
      );
    }
    // Inherited-only members cannot smuggle a contract through the parser.
    const inherited = Object.create({
      include_commit: true,
      include_push: false,
      include_pr: false,
    });
    assertDeepEqual(ts.completionContractSetting(inherited), null);
  });

  await test('a non-default contract starts an explicit work item; the default stays byte-identical', () => {
    const base = ts.startTaskRequest('goal', { mutationMode: '', maxTokens: 0, maxCostMicro: 0 });
    assertDeepEqual(base, { goal: 'goal' });
    assert(
      !('work_items' in base) && !('completion_contract' in base) && !('files' in base),
      'the default path carries no contract seam and no files',
    );
    assertDeepEqual(
      ts.startTaskRequest('goal', {
        mutationMode: '',
        maxTokens: 0,
        maxCostMicro: 0,
        files: ['src/a.ts', 'docs/b.md'],
      }),
      { goal: 'goal', files: ['src/a.ts', 'docs/b.md'] },
      'files-only task keeps the plain-prompt shape',
    );
    const requested = ts.startTaskRequest('goal', {
      mutationMode: 'shadow',
      maxTokens: 10,
      maxCostMicro: 5,
      files: ['src/a.ts'],
      completionContract: { include_commit: true, include_push: false, include_pr: true },
    });
    assertDeepEqual(requested, {
      goal: 'goal',
      max_tokens: 10,
      max_cost_micro: 5,
      files: ['src/a.ts'],
      work_items: [
        {
          id: 'main',
          kind: 'Implementation',
          summary: 'goal',
          ownership: 'isolated_worktree',
        },
      ],
      completion_contract: { include_commit: true, include_push: false, include_pr: true },
      mutation_mode: 'shadow',
    });
    assertDeepEqual(
      ts.startTaskRequest('goal', {
        mutationMode: '',
        maxTokens: 0,
        maxCostMicro: 0,
        completionContract: { include_commit: false, include_push: false, include_pr: false },
      }),
      { goal: 'goal' },
      'an all-false contract is still the default path',
    );
  });

  await test('the task view parses the additive durable completion block and rejects malformed shapes', () => {
    const absent = nc.validateTaskViews([clone(taskViewJson)])[0];
    assertEqual(absent.completion, null, 'absent completion stays null, never fabricated');
    const served = nc.validateTaskViews([
      {
        ...clone(taskViewJson),
        completion: {
          contract: { include_commit: true, include_push: true, include_pr: false },
          steps: [
            { step: 'commit', status: 'succeeded', detail: 'committed abc', seq: 7, at_ms: 11 },
            { step: 'push', status: 'skipped', detail: 'no remote', seq: 8 },
          ],
        },
      },
    ])[0];
    assertEqual(served.completion.contract.include_push, true);
    assertEqual(served.completion.steps.length, 2);
    assertEqual(served.completion.steps[0].status, 'succeeded');
    assertEqual(served.completion.steps[0].detail, 'committed abc');
    assertEqual(served.completion.steps[0].atMs, 11);
    assertEqual(served.completion.steps[1].atMs, null);
    for (const hostile of [
      { completion: { contract: { include_commit: true, include_push: false }, steps: [] } },
      {
        completion: {
          contract: { include_commit: 'yes', include_push: false, include_pr: false },
          steps: [],
        },
      },
      {
        completion: {
          contract: { include_commit: true, include_push: false, include_pr: false },
          steps: 'none',
        },
      },
      {
        completion: {
          contract: { include_commit: true, include_push: false, include_pr: false },
          steps: [{ step: 'commit', status: 'failed' }],
        },
      },
    ]) {
      assertProtocol(() => nc.validateTaskViews([{ ...clone(taskViewJson), ...hostile }]));
    }
  });

  await test('cockpit renders the completion contract with explicit provenance', () => {
    const task = {
      goal: 'ship it',
      state: 'running',
      completed: [],
      open: ['main'],
      testsRun: [],
      testsFailed: [],
      changedFiles: [],
      budget: null,
      acceptanceCriteria: [],
      plan: [],
      blockers: [],
      evidenceRefs: [],
      phase: null,
      progress: null,
      completion: {
        includeCommit: true,
        includePush: false,
        includePr: true,
        steps: [
          { step: 'commit', status: 'pending', detail: 'awaiting the gate' },
          { step: 'pr', status: 'pending', detail: null },
        ],
        source: 'derived',
        reason: null,
      },
    };
    const view = cp.buildCockpit({
      task,
      agents: [],
      verification: null,
      usage: null,
      taskVerification: null,
    });
    const section = cp.cockpitSections(view).find((entry) => entry.key === 'completion');
    assert(section && section.present, 'the completion section must be present');
    assert(section.lines[0].includes('commit, pr'), JSON.stringify(section.lines));
    assert(section.lines[1].includes('[pending] commit'), JSON.stringify(section.lines));
    assert(section.lines[2].includes('[pending] pr'), JSON.stringify(section.lines));
    assert(
      section.lines.some((line) => line.includes('status source: derived')),
      JSON.stringify(section.lines),
    );
    // No contract: the section is explicitly empty, never fabricated.
    const bare = cp.buildCockpit({
      task: { ...task, completion: null },
      agents: [],
      verification: null,
      usage: null,
      taskVerification: null,
    });
    const empty = cp.cockpitSections(bare).find((entry) => entry.key === 'completion');
    assert(empty && empty.present === false && empty.lines[0].includes('none'));
  });

  await test('the built-in Task composer posts the checked contract and displays its statuses', () => {
    const contractTask = {
      goal: 'ship it',
      state: 'running',
      completed: [],
      open: [],
      testsRun: [],
      testsFailed: [],
      changedFiles: [],
      budget: null,
      acceptanceCriteria: [],
      plan: [],
      blockers: [],
      evidenceRefs: [],
      phase: null,
      progress: null,
      completion: {
        includeCommit: true,
        includePush: false,
        includePr: false,
        steps: [{ step: 'commit', status: 'unknown', detail: 'run ended' }],
        source: 'unavailable',
        reason: 'no native completion read',
      },
    };
    const snapshot = { ...webviewSnapshot([]), task: contractTask };
    const { posted, dom, deliver } = runChatWebview(snapshot);
    // Plain start: no contract field at all (chat never carries one).
    dom.document.getElementById('goal').value = 'plain goal';
    dom.document.getElementById('composer').dispatch('submit', { preventDefault() {} });
    assertDeepEqual(posted[posted.length - 1], { type: 'sendGoal', goal: 'plain goal' });
    // Checked boxes: the exact strict contract rides this task start only.
    dom.document.getElementById('goal').value = 'contracted goal';
    dom.document.getElementById('contract-commit').checked = true;
    dom.document.getElementById('contract-pr').checked = true;
    dom.document.getElementById('composer').dispatch('submit', { preventDefault() {} });
    assertDeepEqual(posted[posted.length - 1], {
      type: 'sendGoal',
      goal: 'contracted goal',
      completionContract: { include_commit: true, include_push: false, include_pr: true },
    });
    // A successful start resets the controls (per-start contract).
    deliver({
      type: 'startResult',
      goal: 'contracted goal',
      ok: true,
    });
    assert(
      dom.document.getElementById('contract-commit').checked === false &&
        dom.document.getElementById('contract-pr').checked === false,
      'a success ack clears the completion controls',
    );
    // The task card renders the durable status + its explicit source/reason.
    const completionNode = dom.document.getElementById('task-completion');
    assert(
      findFake(completionNode, (node) => String(node.textContent).includes('[unknown] commit')),
      JSON.stringify(completionNode.children.map((child) => child.textContent)),
    );
    assert(
      findFake(completionNode, (node) => String(node.textContent).includes('status source: unavailable')),
    );
    assert(
      findFake(completionNode, (node) => String(node.textContent).includes('no native completion read')),
    );
  });
}


// ------------------- 7b-2. pending submission envelope + admission restore

function pendingEnvelope(overrides = {}) {
  return {
    text: 'ship the screenshot',
    sessionId: '7',
    draftId: 'draft-1',
    messageId: 'msg-1',
    files: [{ url: 'data:image/png;base64,QUJD', mime: 'image/png', filename: 'shot.png' }],
    attachments: [],
    ...overrides,
  };
}

function binaryAttachment(overrides = {}) {
  return {
    mime: 'application/pdf',
    filename: 'spec.pdf',
    bytes: 3,
    dataBase64: Buffer.from('%PDF').toString('base64'),
    isImage: false,
    ...overrides,
  };
}

async function pendingSubmissionTests() {
  await test('typed attachments ride the task start beside workspace paths', () => {
    const id = { digest: 'a'.repeat(64), mime: 'application/pdf', filename: 'spec.pdf', size: 4 };
    assertDeepEqual(
      ts.startTaskRequest('goal', {
        mutationMode: '',
        maxTokens: 0,
        maxCostMicro: 0,
        files: ['src/a.ts'],
        attachments: [id],
      }),
      { goal: 'goal', files: ['src/a.ts'], attachments: [id] },
    );
    assertDeepEqual(
      ts.startTaskRequest('goal', { mutationMode: '', maxTokens: 0, maxCostMicro: 0 }),
      { goal: 'goal' },
      'the attachment-free path stays byte-identical',
    );
  });

  await test('admission uploads bytes first and starts with the durable ids', async () => {
    const calls = [];
    const restores = [];
    const started = [];
    const client = {
      uploadAttachment: async (sessionId, request) => {
        calls.push({ kind: 'upload', sessionId, request });
        return { digest: 'b'.repeat(64), mime: request.mime, filename: request.filename ?? null, size: 3 };
      },
      startTaskRun: async (sessionId, request) => {
        calls.push({ kind: 'start', sessionId, request });
        return taskRunStartedJson;
      },
    };
    const outcome = await ts.admitPendingSubmission({
      client,
      sessionId: '7',
      pending: pendingEnvelope({ attachments: [binaryAttachment()] }),
      settings: { mutationMode: '', maxTokens: 0, maxCostMicro: 0 },
      onStarted: (run) => started.push(run),
      onFailure: () => {},
      restore: (failure) => restores.push(failure),
    });
    assertEqual(outcome.ok, true);
    assertEqual(restores.length, 0, 'success never restores');
    assertEqual(calls.length, 2, 'one upload then exactly one start');
    assertEqual(calls[0].kind, 'upload');
    assertDeepEqual(calls[0].request, {
      mime: 'application/pdf',
      filename: 'spec.pdf',
      data_base64: Buffer.from('%PDF').toString('base64'),
    });
    assertEqual(calls[1].kind, 'start');
    assertDeepEqual(calls[1].request.attachments, [
      { digest: 'b'.repeat(64), mime: 'application/pdf', filename: 'spec.pdf', size: 3 },
    ]);
    assertEqual(started.length, 1);
  });

  await test('every start failure restores the Kilo identity and never leaves a partial admission', async () => {
    const failures = [
      ['validation', new nc.NativeApiError(400, 'malformed', 'bad body', false)],
      ['unavailable model', new nc.NativeApiError(400, 'unknown_model', 'model "x" is not available', false)],
      ['existing run', new nc.NativeApiError(409, 'conflict', 'session already has a live run', false)],
      ['daemon loss', new Error('socket closed')],
    ];
    for (const [label, error] of failures) {
      const calls = [];
      const restores = [];
      const client = {
        uploadAttachment: async () => {
          calls.push('upload');
          return { digest: 'c'.repeat(64), mime: 'application/pdf', filename: null, size: 3 };
        },
        startTaskRun: async (sessionId, request) => {
          calls.push('start');
          throw error;
        },
      };
      const envelope = pendingEnvelope({ attachments: [binaryAttachment()] });
      const snapshot = JSON.stringify(envelope);
      const outcome = await ts.admitPendingSubmission({
        client,
        sessionId: '7',
        pending: envelope,
        settings: { mutationMode: '', maxTokens: 0, maxCostMicro: 0 },
        onStarted: () => {
          throw new Error(`${label}: must not start`);
        },
        onFailure: () => {},
        restore: (failure) => restores.push(failure),
      });
      assertEqual(outcome.ok, false, label);
      assertEqual(outcome.runId, null, `${label}: no durable run id may leak`);
      assertDeepEqual(calls, ['upload', 'start'], `${label}: exactly one attempt each, no retry`);
      assertEqual(restores.length, 1, `${label}: restore exactly once`);
      // The ORIGINAL envelope identity/files survive verbatim; the Kilo
      // restore message carries them back to the composer.
      assertEqual(JSON.stringify(envelope), snapshot, `${label}: envelope untouched`);
      const failed = sendMessageFailedMessage(envelope, restores[0].message);
      assertEqual(failed.text, envelope.text);
      assertEqual(failed.sessionID, '7');
      assertEqual(failed.draftID, 'draft-1');
      assertEqual(failed.messageID, 'msg-1');
      assertEqual(failed.files[0].url, envelope.files[0].url, `${label}: images restored`);
      assertEqual(failed.files[0].mime, 'image/png');
    }
  });

  await test('an upload failure restores the draft and never issues a task start', async () => {
    const calls = [];
    const restores = [];
    const client = {
      uploadAttachment: async () => {
        calls.push('upload');
        throw new nc.NativeApiError(413, 'oversized', 'attachment exceeds the bound', false);
      },
      startTaskRun: async () => {
        calls.push('start');
        throw new Error('must not start');
      },
    };
    const outcome = await ts.admitPendingSubmission({
      client,
      sessionId: '7',
      pending: pendingEnvelope({ attachments: [binaryAttachment()] }),
      settings: { mutationMode: '', maxTokens: 0, maxCostMicro: 0 },
      onStarted: () => {},
      onFailure: () => {},
      restore: (failure) => restores.push(failure),
    });
    assertEqual(outcome.ok, false);
    assertDeepEqual(calls, ['upload'], 'no start after an upload failure');
    assertEqual(restores.length, 1);
    assertEqual(restores[0].kind, 'upload');
    assert(restores[0].message.includes('413') || restores[0].message.includes('upload'), restores[0].message);
  });

  await test('image submission is refused loudly and the draft/images remain', async () => {
    const calls = [];
    const restores = [];
    const client = {
      uploadAttachment: async () => {
        calls.push('upload');
        throw new Error('must not upload an image');
      },
      startTaskRun: async () => {
        calls.push('start');
        throw new Error('must not start an image submission');
      },
    };
    const image = binaryAttachment({
      mime: 'image/png',
      filename: 'shot.png',
      isImage: true,
      bytes: 4,
      dataBase64: Buffer.from([137, 80, 78, 71]).toString('base64'),
    });
    const envelope = pendingEnvelope({ attachments: [image] });
    const outcome = await ts.admitPendingSubmission({
      client,
      sessionId: '7',
      pending: envelope,
      settings: { mutationMode: '', maxTokens: 0, maxCostMicro: 0 },
      onStarted: () => {},
      onFailure: () => {},
      restore: (failure) => restores.push(failure),
    });
    assertEqual(outcome.ok, false);
    assertDeepEqual(calls, [], 'images are refused before any upload/start request');
    assertEqual(restores.length, 1);
    assertEqual(restores[0].kind, 'image_unsupported');
    assert(
      restores[0].message.includes('provider media/content parts are not wired'),
      restores[0].message,
    );
    const failed = sendMessageFailedMessage(envelope, restores[0].message);
    assertEqual(failed.text, envelope.text);
    assertEqual(failed.files[0].url, envelope.files[0].url, 'the image draft payload is restored');
    assertEqual(outcome.attachmentIds.length, 0);
  });

  await test('pending envelopes are re-validated strictly at the host boundary', () => {
    const valid = ts.parsePendingSubmission(pendingEnvelope());
    assert(valid !== null && valid.draftId === 'draft-1' && valid.messageId === 'msg-1');
    assertEqual(ts.parsePendingSubmission(pendingEnvelope()).files.length, 1);
    for (const hostile of [
      null,
      'nope',
      [],
      { ...pendingEnvelope(), text: '   ' },
      { ...pendingEnvelope(), messageId: 'x'.repeat(4097) },
      { ...pendingEnvelope(), files: 'not-an-array' },
      { ...pendingEnvelope(), attachments: 'not-an-array' },
      { ...pendingEnvelope(), attachments: [{ mime: 'application/pdf' }] },
      {
        ...pendingEnvelope(),
        attachments: [{ ...binaryAttachment(), dataBase64: 'x'.repeat(10 * 1024 * 1024) }],
      },
      { ...pendingEnvelope(), attachments: [{ ...binaryAttachment(), bytes: -1 }] },
    ]) {
      assertEqual(ts.parsePendingSubmission(hostile), null, JSON.stringify(hostile).slice(0, 120));
    }
  });

  await test('the bridge output feeds the host admission flow end to end', async () => {
    const image = `data:image/png;base64,${Buffer.from([137, 80, 78, 71]).toString('base64')}`;
    const raw = {
      type: 'sendMessage',
      text: 'inspect these',
      sessionID: '7',
      messageID: 'msg-9',
      draftID: 'draft-9',
      files: [
        { url: 'data:application/pdf;base64,JVBERi0xLjQ=', mime: 'application/pdf', filename: 'spec.pdf' },
        { url: image, mime: 'image/png', filename: 'shot.png' },
      ],
    };
    const command = ingestWebviewMessage(raw, { workspaceDirectory: '/w' });
    assertEqual(command.kind, 'sendMessage');
    const host = bridgeCommandToHostMessage(command);
    const pending = ts.parsePendingSubmission(host.pending);
    assert(pending !== null, 'the host must accept the bridge envelope');
    assertEqual(pending.attachments.length, 2, 'binary refs ride the pending envelope');
    assertEqual(pending.attachments[0].mime, 'application/pdf');
    assertEqual(pending.attachments[0].dataBase64, 'JVBERi0xLjQ=');
    assertEqual(pending.attachments[0].isImage, false);
    assertEqual(pending.attachments[1].isImage, true);

    // An image refuses the WHOLE submission loudly before any upload/start.
    const calls = [];
    const restores = [];
    await ts.admitPendingSubmission({
      client: {
        uploadAttachment: async () => {
          calls.push('upload');
          throw new Error('images must never upload');
        },
        startTaskRun: async () => {
          calls.push('start');
          throw new Error('images must never start');
        },
      },
      sessionId: '7',
      pending,
      settings: { mutationMode: '', maxTokens: 0, maxCostMicro: 0 },
      onStarted: () => {},
      onFailure: () => {},
      restore: (failure) => restores.push(failure),
    });
    assertDeepEqual(calls, [], 'no daemon calls for an image submission');
    assertEqual(restores.length, 1);
    assertEqual(restores[0].kind, 'image_unsupported');
    const failed = sendMessageFailedMessage(pending, restores[0].message);
    assertEqual(failed.messageID, 'msg-9');
    assertEqual(failed.files[1].url, image, 'the image bytes return to the composer');

    // Without the image the same flow uploads the exact bytes and starts
    // with the durable typed id.
    const textOnly = ts.parsePendingSubmission(
      bridgeCommandToHostMessage(
        ingestWebviewMessage({
          type: 'sendMessage',
          text: 'attach spec',
          files: [{ url: 'data:application/pdf;base64,JVBERi0xLjQ=', mime: 'application/pdf', filename: 'spec.pdf' }],
        }),
      ).pending,
    );
    const calls2 = [];
    let startedRequest = null;
    const outcome = await ts.admitPendingSubmission({
      client: {
        uploadAttachment: async (sessionId, request) => {
          calls2.push({ kind: 'upload', request });
          return { digest: 'd'.repeat(64), mime: request.mime, filename: request.filename ?? null, size: 8 };
        },
        startTaskRun: async (sessionId, request) => {
          calls2.push({ kind: 'start' });
          startedRequest = request;
          return taskRunStartedJson;
        },
      },
      sessionId: '7',
      pending: textOnly,
      settings: { mutationMode: '', maxTokens: 0, maxCostMicro: 0 },
      onStarted: () => {},
      onFailure: () => {},
      restore: () => {
        throw new Error('must not restore a successful admission');
      },
    });
    assertEqual(outcome.ok, true);
    assertDeepEqual(calls2[0].request, {
      mime: 'application/pdf',
      filename: 'spec.pdf',
      data_base64: 'JVBERi0xLjQ=',
    });
    assertDeepEqual(startedRequest.attachments, [
      { digest: 'd'.repeat(64), mime: 'application/pdf', filename: 'spec.pdf', size: 8 },
    ]);
  });
}

// -------------- 7c. board state projection + webview forwarding hardening

async function boardAndForwardingTests() {
  await test('board pages project posts, unread and the read watermark', () => {
    const first = st.boardStateFromPage(clone(boardPageJson), null);
    assertEqual(first.board.available, true);
    assertEqual(first.board.source, 'native');
    assertEqual(first.board.revision, 3);
    assertEqual(first.board.unread, 2, 'a null watermark counts every page post as unread');
    assertEqual(first.board.posts[0].id, '3');
    assertEqual(first.board.posts[0].author, 'child:8');
    assertEqual(first.board.posts[1].author, 'root');
    assertDeepEqual(first.board.posts[0].refs, ['evidence:41']);
    assertEqual(first.board.posts[0].createdMs, 1700);
    assertEqual(first.watermark, 3);

    // The watermark acknowledges exactly the revisions at-or-below it.
    assertEqual(st.boardStateFromPage(clone(boardPageJson), 2).board.unread, 1);
    assertEqual(st.boardStateFromPage(clone(boardPageJson), 3).board.unread, 0);
    assertEqual(st.boardStateFromPage(clone(emptyBoardPageJson), 3).board.unread, 0);
    assertEqual(st.boardStateFromPage(clone(emptyBoardPageJson), 3).board.revision, 0);

    // Unavailable is explicit and never fabricates posts.
    const unavailable = st.unavailableBoardState('route absent (HTTP 404 not_found)');
    assertEqual(unavailable.available, false);
    assertEqual(unavailable.posts.length, 0);
    assertEqual(unavailable.unread, null);
    assert(unavailable.reason.includes('404'), unavailable.reason);

    // Host re-validation: board gestures are bounded before the wire.
    assertDeepEqual(st.parseBoardReadRequest(undefined, undefined), { since: null, limit: null });
    assertDeepEqual(st.parseBoardReadRequest(5, 10), { since: 5, limit: 10 });
    for (const [since, limit] of [
      [0, null],
      [-1, null],
      [1.5, null],
      ['5', null],
      [null, 0],
      [null, 101],
      [null, 1.5],
    ]) {
      const parsed = st.parseBoardReadRequest(since, limit);
      assert('reason' in parsed, `hostile board read must be refused: ${since}/${limit}`);
    }
    assertDeepEqual(st.parseBoardPostRequest({ subject: ' s ', body: ' b ', refs: [' r '] }), {
      subject: 's',
      body: ' b ',
      refs: ['r'],
    });
    for (const hostile of [
      {},
      { subject: '   ', body: 'b' },
      { subject: 's' },
      { subject: 's', body: '' },
      { subject: 's', body: 'b', refs: 'nope' },
      { subject: 's', body: 'b', refs: [''] },
      { subject: 's'.repeat(513), body: 'b' },
      { subject: 's', body: 'b'.repeat(16 * 1024 + 1) },
    ]) {
      const parsed = st.parseBoardPostRequest(hostile);
      assert('reason' in parsed, `hostile board post must be refused: ${JSON.stringify(hostile)}`);
    }
  });

  await test('the client refuses empty board posts locally (no request is made)', async () => {
    const { client, calls } = makeClient({});
    assertProtocol(() => client.boardPost('7', { subject: ' ', body: 'b' }));
    assertProtocol(() => client.boardPost('7', { subject: 's', body: ' ' }));
    assertEqual(calls.length, 0, 'invalid board posts must never reach the wire');
  });

  await test('composer files are bounded per-entry and forwarded to the native task-run', async () => {
    const { files, refused } = ts.boundedWebviewFiles([
      'src/a.ts',
      '  docs/b.md  ',
      '',
      '  ',
      7,
      null,
      '../escape.txt',
      'dir/../up.txt',
      'ctrl\u0000name',
      '/abs/escape.rs',
      'C:\\abs\\escape.rs',
      'data:image/png;base64,AAAA',
      'file:///etc/passwd',
      'x'.repeat(4097),
      ...Array.from({ length: 70 }, (_, i) => `f${i}.ts`),
    ]);
    assertEqual(files.length, 64, 'the accepted list is capped at MAX_WEBVIEW_FILES');
    assertEqual(files[0], 'src/a.ts');
    assertEqual(files[1], 'docs/b.md', 'accepted paths are trimmed');
    assert(
      refused.some((entry) => entry.reason.includes('traverses outside the workspace')),
      JSON.stringify(refused),
    );
    assert(
      refused.some((entry) => entry.reason.includes('control characters')),
      JSON.stringify(refused),
    );
    assert(
      refused.some((entry) => entry.reason.includes('4096')),
      JSON.stringify(refused),
    );
    assert(
      refused.some((entry) => entry.reason.includes('more than 64')),
      JSON.stringify(refused),
    );
    assert(
      refused.some((entry) => entry.reason.includes('workspace-relative paths only')),
      JSON.stringify(refused),
    );
    assert(
      refused.some((entry) => entry.reason.includes('must be workspace-relative')),
      JSON.stringify(refused),
    );
    assertDeepEqual(ts.boundedWebviewFiles(undefined), { files: [], refused: [] });
    assertEqual(ts.boundedWebviewFiles('nope').refused[0].index, 0);

    // The accepted subset reaches ONE native task-run request alongside the
    // submitted contract (an explicit work item), never a silent drop.
    const captured = [];
    const outcome = await ts.startTaskRun({
      client: {
        startTaskRun: async (_sessionId, request) => {
          captured.push(request);
          return taskRunStartedJson;
        },
      },
      sessionId: '7',
      goal: 'ship it',
      settings: {
        mutationMode: '',
        maxTokens: 0,
        maxCostMicro: 0,
        files,
        completionContract: { include_commit: true, include_push: false, include_pr: true },
      },
      onStarted: () => {},
      onFailure: () => {
        throw new Error('must not fail');
      },
    });
    assertEqual(outcome.ok, true);
    assertEqual(captured.length, 1);
    assertEqual(captured[0].files.length, 64);
    assertDeepEqual(captured[0].completion_contract, {
      include_commit: true,
      include_push: false,
      include_pr: true,
    });
    assertEqual(captured[0].work_items[0].id, 'main');
    assertEqual(captured[0].work_items[0].kind, 'Implementation');
  });

  await test('a malformed completion contract is refused with a typed reason', () => {
    assertDeepEqual(ts.parseCompletionContract(undefined), { contract: null });
    assertDeepEqual(ts.parseCompletionContract(null), { contract: null });
    assertDeepEqual(
      ts.parseCompletionContract({
        include_commit: false,
        include_push: false,
        include_pr: false,
      }),
      { contract: null },
      'all-false is the default path, not a contract',
    );
    assertDeepEqual(
      ts.parseCompletionContract({
        include_commit: true,
        include_push: false,
        include_pr: true,
      }),
      { contract: { include_commit: true, include_push: false, include_pr: true } },
    );
    for (const hostile of [
      'commit',
      ['include_commit'],
      true,
      42,
      {},
      { include_commit: true },
      { include_commit: 'yes', include_push: false, include_pr: false },
      { include_commit: true, include_push: false, include_pr: false, include_release: true },
    ]) {
      const parsed = ts.parseCompletionContract(hostile);
      assert(
        'reason' in parsed && parsed.reason.length > 0,
        `hostile contract must carry a typed reason: ${JSON.stringify(hostile)}`,
      );
    }
    // Inherited members cannot smuggle a contract through the parser.
    const inherited = Object.create({
      include_commit: true,
      include_push: false,
      include_pr: false,
    });
    const parsed = ts.parseCompletionContract(inherited);
    assert('reason' in parsed, 'inherited-only members must be refused');
  });
}

async function draftPreservationTests() {
  await test('composer clears the draft only after a successful start', () => {
    assertEqual(composerPolicy.afterStart('my draft', 'my draft', false), 'my draft');
    assertEqual(composerPolicy.afterStart('my draft', 'my draft', true), '');
    assertEqual(composerPolicy.afterStart('newer text', 'submitted', true), 'newer text');
    assertEqual(composerPolicy.keepDraft('  spaced  ', 'spaced', true), false);
    assertEqual(composerPolicy.keepDraft('spaced', 'spaced', false), true);
  });
}

async function runStateTests() {
  const make = (state, runId = 'r1') => ({
    taskId: '1',
    runId,
    mode: 'in_session',
    state,
    goal: null,
    model: null,
  });

  await test('busy/activeRunId derive from run STATE (terminal => not busy)', () => {
    assertEqual(st.isTerminalRunState('Done'), true);
    assertEqual(st.isTerminalRunState('Failed'), true);
    assertEqual(st.isTerminalRunState('Cancelled'), true);
    assertEqual(st.isTerminalRunState('Running'), false);
    assertEqual(st.activeRunIdAfter('r1', [make('Running')]), 'r1');
    assertEqual(st.activeRunIdAfter('r1', [make('Done')]), null, 'a listed Done run is not busy');
    assertEqual(st.activeRunIdAfter('r1', [make('Failed')]), null);
    assertEqual(st.activeRunIdAfter('r1', [make('Cancelled')]), null);
    assertEqual(st.activeRunIdAfter('r1', []), null, 'a vanished run is not busy');
    assertEqual(st.activeRunIdAfter(null, [make('Running')]), null);
  });

  await test('cancel targets only active runs; terminal runs are never attempted', () => {
    assertEqual(st.cancelRunTarget('r1', [make('Done')]), null, 'terminal tracked run => no cancel attempt');
    assertEqual(st.cancelRunTarget('r1', [make('Cancelled')]), null);
    assertEqual(st.cancelRunTarget('r1', [make('Running')]), 'r1');
    assertEqual(st.cancelRunTarget(null, [make('Done'), make('Running', 'r2')]), 'r2');
    assertEqual(st.cancelRunTarget(null, [make('Done'), make('Failed')]), null, 'all-terminal => no cancel attempt');
    assertEqual(st.cancelRunTarget('stale', [make('Running', 'r2')]), 'r2');
  });

  await test('a server 409 on cancel surfaces as a typed NativeApiError', async () => {
    const { client } = makeClient({
      'POST /native/session/7/task-runs/r1/cancel': () =>
        new Response(
          JSON.stringify({ error: { code: 'conflict', message: 'terminal runs refuse cancel', retryable: false } }),
          { status: 409 },
        ),
    });
    await assertRejects(
      () => client.cancelTaskRun('7', 'r1'),
      (error) => error instanceof nc.NativeApiError && error.status === 409 && error.code === 'conflict',
      'typed terminal-cancel refusal',
    );
  });
}

async function workspaceBindingTests() {
  await test('session binding uses the canonical workspace identity, never sessions[0]', () => {
    const sessions = [
      { id: 's1', title: 'workspace A', provider: 'p', model: 'm', state: 'idle' },
      { id: 's2', title: 'workspace B', provider: 'p', model: 'm', state: 'idle' },
    ];
    const keyA = wb.canonicalWorkspaceKey('file:///repo/a/');
    const keyB = wb.canonicalWorkspaceKey('file:///repo/b');
    assertEqual(keyA, 'file:///repo/a');
    assertEqual(wb.boundSessionFor(keyB, sessions, { [keyB]: 's2' }), 's2');
    assertEqual(
      wb.boundSessionFor(keyA, sessions, { [keyB]: 's2' }),
      null,
      'no binding must create, not reuse sessions[0]',
    );
    assertEqual(
      wb.boundSessionFor(keyB, sessions, { [keyB]: 's9' }),
      null,
      'a stale binding must not fall back to sessions[0]',
    );
    assertEqual(wb.boundSessionFor(null, sessions, { '': 's2' }), null);
    assertEqual(wb.windowWorkspaceKey([]), null);
    assertEqual(wb.windowWorkspaceKey(['file:///repo/b/', 'file:///repo/a']), 'file:///repo/b');
    const bound = wb.withBinding({}, keyB, 's2');
    assertEqual(bound[keyB], 's2');
    assertDeepEqual(wb.pruneBindings({ [keyA]: 's1', [keyB]: 'gone' }, [sessions[0]]), {
      [keyA]: 's1',
    });
    let many = {};
    for (let i = 0; i < wb.MAX_SESSION_BINDINGS + 5; i += 1) {
      many = wb.withBinding(many, `file:///w/${i}`, `s${i}`);
    }
    assertEqual(Object.keys(many).length, wb.MAX_SESSION_BINDINGS, 'bindings stay bounded');
  });
}

async function childInspectionTests() {
  await test('child summaries surface identity/worktree/ownership/capabilities/progress/result/budget/model metadata', () => {
    const catalog = [{ provider: 'fake', model: 'm', reasoning: true, thinking: true, tools: true }];
    const summaries = st.summarizeAgents(clone(agentsJson), catalog);
    const child = summaries.find((agent) => agent.agentId === 'c1');
    assertEqual(child.itemId, 'main');
    assertEqual(child.itemKind, 'Implementation');
    assertEqual(child.worktreeId, 2);
    assertEqual(child.sessionId, 8);
    assertEqual(child.ownership, 'orchestrator');
    assertDeepEqual(child.capabilities, ['ReadWorkspace']);
    assertDeepEqual(child.progress, { phase: 'work' });
    assertEqual(child.result, null);
    assertEqual(child.budget, 1000);
    assertEqual(child.model, 'm');
    assertEqual(child.provider, 'fake');
    assertEqual(child.reasoning, true);
    assertEqual(child.thinking, true);
    assertEqual(child.presentation, 'foreground');
    assertEqual(child.pixel.childId, 'c1');
    const self = summaries.find((agent) => agent.agentId === 'r1');
    assertDeepEqual(self.itemIds, ['main']);
    assertEqual(self.provider, null);
  });

  await test('same model two providers: the catalog join is (provider, model), never model alone', () => {
    const catalog = [
      { provider: 'alpha', model: 'm', reasoning: true, thinking: false, tools: true },
      { provider: 'beta', model: 'm', reasoning: false, thinking: true, tools: false },
    ];
    const frame = (agentId, provider) => ({
      ...clone(agentsJson[1]),
      agent_id: agentId,
      provider,
    });
    const summaries = st.summarizeAgents([frame('a', 'alpha'), frame('b', 'beta')], catalog);
    assertEqual(summaries[0].provider, 'alpha');
    assertEqual(summaries[0].reasoning, true, 'alpha child keeps alpha reasoning');
    assertEqual(summaries[0].thinking, false);
    assertEqual(summaries[1].provider, 'beta');
    assertEqual(summaries[1].reasoning, false, 'beta child must never inherit alpha metadata');
    assertEqual(summaries[1].thinking, true);
    // A provider-less entry never guesses metadata by model alone.
    const bare = clone(agentsJson[1]);
    delete bare.provider;
    const [legacy] = st.summarizeAgents([bare], catalog);
    assertEqual(legacy.provider, null);
    assertEqual(legacy.reasoning, null);
    assertEqual(legacy.thinking, null);
  });

  await test('background presentation survives the summary and defaults to foreground', () => {
    const [background] = st.summarizeAgents([
      { ...clone(agentsJson[1]), presentation: 'background' },
    ]);
    assertEqual(background.presentation, 'background');
    const [missing] = st.summarizeAgents([clone(agentsJson[1])]);
    assertEqual(missing.presentation, 'foreground');
    const [hostile] = st.summarizeAgents([
      { ...clone(agentsJson[1]), presentation: 'invisible' },
    ]);
    assertEqual(hostile.presentation, 'foreground', 'unknown tags never fabricate background');
  });

  await test('presentation transitions use the session-scoped route and surface typed 409s', async () => {
    const { client, calls } = makeClient({
      'POST /native/session/7/agents/c1/presentation': () => jsonResponse(presentationAckJson),
    });
    const ack = await client.setAgentPresentation('7', 'c1', 'background');
    assertEqual(ack.child_id, 'c1');
    assertEqual(ack.changed, true);
    assertDeepEqual(findCall(calls, 'POST', '/native/session/7/agents/c1/presentation').body, {
      state: 'background',
    });
    const terminal = makeClient({
      'POST /native/session/7/agents/c1/presentation': () =>
        new Response(
          JSON.stringify({
            error: { code: 'conflict', message: 'terminal child', retryable: false },
          }),
          { status: 409 },
        ),
    });
    await assertRejects(
      () => terminal.client.setAgentPresentation('7', 'c1', 'foreground'),
      (error) =>
        error instanceof nc.NativeApiError && error.status === 409 && error.code === 'conflict',
      'typed terminal presentation refusal',
    );
  });

  await test('child summary fields are bounded and blocker fields surface when present', () => {
    const huge = {
      ...clone(agentsJson[1]),
      blockers: [{ id: 'b1', detail: 'dependency x not done' }],
      progress: { phase: 'work', note: 'x'.repeat(10_000) },
    };
    const [child] = st.summarizeAgents([huge]);
    assertEqual(child.blockers[0], 'dependency x not done');
    assertEqual(child.progress.note.length < 1000, true, 'progress strings must be bounded');
    // The durable blocker object shape (kind/reason/resolution) surfaces too.
    const blocked = {
      ...clone(agentsJson[1]),
      blocker: {
        kind: 'permission',
        reason: 'waiting for a pending permission decision',
        resolution: 'resolve the pending permission request',
      },
    };
    const [durable] = st.summarizeAgents([blocked]);
    assert(
      durable.blockers[0].includes('permission: waiting for a pending permission decision'),
      durable.blockers[0],
    );
    assert(nc.validateAgents([blocked])[0].blocker.kind === 'permission', 'object blocker must validate');
  });

  await test('agent Retry uses the server guard: a typed 409 surfaces, never a silent no-op', async () => {
    const { client } = makeClient({
      'POST /native/agents/c1/retry': () =>
        new Response(
          JSON.stringify({ error: { code: 'conflict', message: 'only Failed children retry', retryable: false } }),
          { status: 409 },
        ),
    });
    await assertRejects(
      () => client.retryAgent('c1'),
      (error) => error instanceof nc.NativeApiError && error.status === 409,
      'typed retry refusal',
    );
  });
}

async function pixelAgentTests() {
  await test('pixel avatars are deterministic per ChildId with per-state animation', () => {
    const a1 = px.pixelAvatar('c1');
    const a2 = px.pixelAvatar('c1');
    assertDeepEqual(a1, a2);
    assertEqual(a1.pixels.length, 25);
    assertEqual(a1.pixels.filter((bit) => bit === 1).length > 0, true, 'sprite must have pixels');
    assert(
      a1.hash !== px.pixelAvatar('c2').hash || a1.color !== px.pixelAvatar('c2').color,
      'different children must differ deterministically',
    );
    const mapping = [
      ['Running', 'running'],
      ['Paused', 'paused'],
      ['Waiting', 'waiting'],
      ['Blocked', 'blocked'],
      ['Done', 'done'],
      ['Failed', 'failed'],
      ['Cancelled', 'cancelled'],
      ['Idle', 'waiting'],
    ];
    for (const [native, expected] of mapping) {
      assertEqual(px.pixelStateOf(native), expected, native);
    }
    assertEqual(px.pixelAnimation('running'), 'pixel-running');
    assertEqual(px.pixelAnimation('blocked'), 'pixel-blocked');
  });

  await test('pixel presence transitions over native mock frames and survives gaps', () => {
    const frame = (state) => [
      {
        agent_id: 'c1',
        kind: 'child',
        run_id: 'r1',
        session_id: 8,
        worktree_id: 2,
        goal: 'g',
        state,
        model: 'm',
        budget: 1,
        ownership: 'isolated_worktree',
        capabilities: [],
        progress: null,
        result: null,
        item_id: 'main',
        item_kind: 'Implementation',
      },
    ];
    let presence = new Map();
    const timeline = [];
    for (const state of ['Running', 'Paused', 'Waiting', 'Blocked', 'Done']) {
      presence = px.foldPixelPresence(presence, frame(state));
      const entry = presence.get('c1');
      timeline.push(entry.state);
      assertDeepEqual(entry.avatar, px.pixelAvatar('c1'), 'avatar is stable across frames');
      assertEqual(entry.animation, `pixel-${entry.state}`);
    }
    assertDeepEqual(timeline, ['running', 'paused', 'waiting', 'blocked', 'done']);
    presence = px.foldPixelPresence(presence, []);
    assertEqual(presence.get('c1').state, 'done', 'a frame gap keeps the last presence');
  });
}

async function cockpitTests() {
  await test('validateTaskViews surfaces additive criteria/plan/blockers/evidence/phase fields', () => {
    const payload = {
      ...clone(taskViewJson),
      acceptance_criteria: ['build passes', 'tests pass'],
      plan: [{ id: 'main', summary: 'implement', state: 'running', depends_on: ['analysis'] }],
      blockers: [{ id: 'b1', detail: 'waiting on analysis', state: 'open' }],
      evidence_refs: ['evidence:41'],
      phase: 'implementation',
    };
    const view = nc.validateTaskViews([payload])[0];
    assertDeepEqual(view.acceptanceCriteria, ['build passes', 'tests pass']);
    assertEqual(view.plan[0].id, 'main');
    assertDeepEqual(view.plan[0].dependsOn, ['analysis']);
    assertEqual(view.blockers[0].detail, 'waiting on analysis');
    assertDeepEqual(view.evidenceRefs, ['evidence:41']);
    assertEqual(view.phase, 'implementation');
    const bare = nc.validateTaskViews([clone(taskViewJson)])[0];
    assertDeepEqual(bare.acceptanceCriteria, []);
    assertDeepEqual(bare.plan, []);
    assertDeepEqual(bare.blockers, []);
    assertEqual(bare.phase, null);
    assertProtocol(
      () => nc.validateTaskViews([{ ...clone(taskViewJson), plan: [{ id: 'x', depends_on: [7] }] }]),
      'expected a string',
    );
    assertProtocol(() => nc.validateTaskViews([{ ...clone(taskViewJson), blockers: 'nope' }]), 'expected an array');
  });

  await test('task cockpit renders every section from a mock native payload', () => {
    const taskView = nc.validateTaskViews([
      {
        ...clone(taskViewJson),
        acceptance_criteria: ['build passes', 'tests pass'],
        plan: [
          { id: 'analysis', summary: 'analyze', state: 'done' },
          { id: 'main', summary: 'implement', state: 'running', depends_on: ['analysis'] },
        ],
        blockers: [{ id: 'b1', detail: 'waiting on analysis', state: 'open' }],
        evidence_refs: [],
        phase: 'implementation',
      },
    ])[0];
    const task = {
      goal: taskView.goal,
      state: taskView.state,
      completed: taskView.milestones.completed,
      open: taskView.milestones.open,
      testsRun: taskView.tests.run,
      testsFailed: taskView.tests.failed,
      changedFiles: taskView.changedFiles,
      budget: taskView.budget,
      acceptanceCriteria: taskView.acceptanceCriteria,
      plan: taskView.plan,
      blockers: taskView.blockers.map((blocker) => blocker.detail),
      evidenceRefs: taskView.evidenceRefs,
      phase: taskView.phase,
      progress: taskView.progress,
    };
    const catalog = [{ provider: 'fake', model: 'm', reasoning: true, thinking: true, tools: true }];
    const agents = st.summarizeAgents(
      clone(agentsJson).map((agent) =>
        agent.kind === 'child' ? { ...agent, state: 'Blocked' } : agent,
      ),
      catalog,
    );
    const verification = clone(verificationViewJson);
    const usage = clone(sessionUsageJson);
    const taskVerificationRecord = clone(verificationRecordJson);
    taskVerificationRecord.criteria[0].evidence = 'evidence:41';
    const view = cp.buildCockpit({
      task,
      agents,
      verification,
      usage: {
        tokens: usage.providerCalls.tokens,
        spentMicro: 12,
        maxMicro: null,
        openMicro: 0,
        truncated: false,
      },
      taskVerification: { records: [taskVerificationRecord] },
    });
    assert(view, 'a task cockpit must build from the mock payload');
    const sections = cp.cockpitSections(view);
    assertDeepEqual(
      sections.map((section) => section.key),
      ['acceptance', 'plan', 'completion', 'children', 'tournament', 'phase', 'blockers', 'verification', 'evidence', 'spend'],
    );
    const byKey = Object.fromEntries(sections.map((section) => [section.key, section]));
    assert(byKey.acceptance.present && byKey.acceptance.lines.some((line) => line.includes('build passes')));
    assert(
      byKey.plan.present &&
        byKey.plan.lines[0].includes('analysis') &&
        byKey.plan.lines[1].includes('after analysis'),
      JSON.stringify(byKey.plan.lines),
    );
    assert(
      byKey.children.present &&
        byKey.children.lines[0].includes('c1') &&
        byKey.children.lines[0].includes('worktree 2') &&
        byKey.children.lines[0].includes('ownership orchestrator'),
      JSON.stringify(byKey.children.lines),
    );
    assert(byKey.phase.present && byKey.phase.lines[0].includes('implementation'));
    assert(byKey.blockers.present && byKey.blockers.lines[0].includes('waiting on analysis'));
    assert(byKey.verification.present && byKey.verification.lines[0].includes('status passed'));
    assert(
      byKey.evidence.present && byKey.evidence.evidence[0].id === 41,
      JSON.stringify(byKey.evidence),
    );
    assert(byKey.spend.present && byKey.spend.lines[0].includes('tokens 130'));
    assertEqual(cp.evidenceRefOf('evidence:41').id, 41);
    assertEqual(cp.evidenceRefOf('plain text').id, null);
    assertEqual(cp.phaseOf({ stage: 'verify' }), 'verify');
    assertEqual(cp.buildCockpit({ task: null, agents: [], verification: null, usage: null, taskVerification: null }), null);
  });

  await test('cockpit tucks background children last and marks them dimmed', () => {
    const [backgroundChild] = st.summarizeAgents([
      { ...clone(agentsJson[1]), presentation: 'background' },
    ]);
    const [foregroundChild] = st.summarizeAgents([clone(agentsJson[1])]);
    const view = cp.buildCockpit({
      task: null,
      agents: [backgroundChild, foregroundChild],
      verification: null,
      usage: null,
      taskVerification: null,
    });
    assertDeepEqual(
      view.children.map((child) => child.presentation),
      ['foreground', 'background'],
      'background children are ordered after the foreground ones',
    );
    const children = cp.cockpitSections(view).find((section) => section.key === 'children');
    assert(
      children.lines[1].includes('background (dimmed)'),
      JSON.stringify(children.lines),
    );
  });

  await test('cockpit tournament block is state-gated: decide waits, abort is open-only', () => {
    const candidate = (childId, state) => ({
      childId,
      state,
      verification: state === 'done' ? 12 : null,
      verificationPass: state === 'done' ? true : null,
      reviewRank: state === 'done' ? 'clean' : null,
      reviewer: null,
      costMicro: 1,
      wallMs: 2,
    });
    const native = (state, winner, candidates) => ({
      id: 't-1',
      state,
      winner,
      criteria: [{ id: 'c-1', spec: 'tests pass' }],
      candidates,
    });
    const running = cp.tournamentViewOf(
      native('open', null, [candidate('child-0', 'done'), candidate('child-1', 'running')]),
    );
    assertEqual(running.open, true);
    assertEqual(running.canDecide, false, 'decide waits until every candidate settled');
    const settled = cp.tournamentViewOf(
      native('open', null, [candidate('child-0', 'done'), candidate('child-1', 'done')]),
    );
    assertEqual(settled.canDecide, true);
    const decided = cp.tournamentViewOf(
      native('decided', 'child-0', [candidate('child-0', 'done'), candidate('child-1', 'discarded')]),
    );
    assertEqual(decided.open, false);
    assertEqual(decided.canDecide, false, 'a decided tournament exposes no controls');

    // A tournament ALONE builds a cockpit and carries its section + actions.
    const view = cp.buildCockpit({
      task: null,
      agents: [],
      verification: null,
      usage: null,
      taskVerification: null,
      tournament: running,
    });
    assert(view, 'a tournament alone must build a cockpit');
    assert(view.tournament, 'the cockpit must carry the tournament block');
    const section = cp.cockpitSections(view).find((entry) => entry.key === 'tournament');
    assert(section.present, 'the tournament section is present');
    assert(
      section.lines.some((line) => line.includes('t-1') && line.includes('[open]')),
      JSON.stringify(section.lines),
    );
    const decide = section.actions.find((action) => action.key === 'decide');
    const abort = section.actions.find((action) => action.key === 'abort');
    assertEqual(decide.enabled, false, 'decide action is disabled while running');
    assertEqual(abort.enabled, true, 'abort action stays available while open');
    // The decided view's gating survives the section projection.
    const decidedView = cp.buildCockpit({
      task: null,
      agents: [],
      verification: null,
      usage: null,
      taskVerification: null,
      tournament: decided,
    });
    const decidedSection = cp.cockpitSections(decidedView).find(
      (entry) => entry.key === 'tournament',
    );
    assertEqual(
      decidedSection.actions.every((action) => action.enabled === false),
      true,
      'terminal tournaments expose only disabled controls',
    );
  });
}

// ----------------------------------------- presentation webview (fake DOM)

function makeFakeDom() {
  const nodesById = new Map();
  function makeNode(tagName) {
    const node = {
      tagName,
      children: [],
      parentNode: null,
      attributes: {},
      listeners: {},
      className: '',
      textContent: '',
      scrollTop: 0,
      scrollHeight: 0,
      hidden: false,
      value: '',
      type: '',
    };
    node.appendChild = (child) => {
      node.children.push(child);
      child.parentNode = node;
      return child;
    };
    node.removeChild = (child) => {
      const index = node.children.indexOf(child);
      if (index >= 0) {
        node.children.splice(index, 1);
      }
      return child;
    };
    node.setAttribute = (key, value) => {
      node.attributes[key] = String(value);
    };
    node.getAttribute = (key) => (key in node.attributes ? node.attributes[key] : null);
    node.addEventListener = (type, callback) => {
      if (!node.listeners[type]) {
        node.listeners[type] = [];
      }
      node.listeners[type].push(callback);
    };
    node.dispatch = (type, event) => {
      for (const callback of node.listeners[type] || []) {
        callback(event || {});
      }
    };
    node.click = () => node.dispatch('click', {});
    node.cloneNode = () => {
      const copy = makeNode(tagName);
      copy.className = node.className;
      copy.textContent = node.textContent;
      return copy;
    };
    Object.defineProperty(node, 'firstChild', { get: () => node.children[0] || null });
    Object.defineProperty(node, 'childNodes', { get: () => node.children });
    return node;
  }
  const document = {
    getElementById(id) {
      if (!nodesById.has(id)) {
        nodesById.set(id, makeNode('div'));
      }
      return nodesById.get(id);
    },
    createElement: makeNode,
    createElementNS: (namespace, tagName) => makeNode(tagName),
    querySelectorAll: () => [],
  };
  return { document, nodesById, makeNode };
}

function walkFake(root, visit) {
  visit(root);
  for (const child of root.children || []) {
    walkFake(child, visit);
  }
}

function findFake(root, predicate) {
  let found = null;
  walkFake(root, (node) => {
    if (found === null && predicate(node)) {
      found = node;
    }
  });
  return found;
}

function webviewSnapshot(agents) {
  return {
    daemon: 'running',
    daemonDetail: 'selftest',
    baseUrl: 'http://127.0.0.1:9',
    session: clone(sessionSummaryJson),
    machineState: 'idle',
    machineLabel: 'Idle',
    sessions: [clone(sessionSummaryJson)],
    runs: [],
    activeRunId: null,
    agents,
    task: null,
    verification: null,
    usage: null,
    cockpit: null,
    cockpitSections: [],
    tournament: null,
    transcript: [],
    streamStatus: 'open',
    lastError: null,
    busy: false,
  };
}

function runChatWebview(snapshot) {
  const source = readFileSync(new URL('../media/chat.js', import.meta.url), 'utf8');
  const posted = [];
  const dom = makeFakeDom();
  let messageHandler = null;
  const sandbox = {
    document: dom.document,
    window: {
      addEventListener(type, callback) {
        if (type === 'message') {
          messageHandler = callback;
        }
      },
    },
    acquireVsCodeApi: () => ({ postMessage: (message) => posted.push(message) }),
    setTimeout: () => 0,
    clearTimeout: () => {},
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);
  assert(messageHandler, 'chat.js must register a window message listener');
  messageHandler({ data: { type: 'snapshot', snapshot } });
  return {
    posted,
    dom,
    deliver: (message) => messageHandler({ data: message.data ? message.data : message }),
  };
}

async function presentationWebviewTests() {
  await test('background children render dimmed+grouped and the toggle posts the next state', () => {
    const backgroundWire = clone(agentsJson).map((agent) =>
      agent.kind === 'child' ? { ...agent, presentation: 'background' } : agent,
    );
    const agents = st.summarizeAgents(nc.validateAgents(backgroundWire));
    const { posted, dom } = runChatWebview(webviewSnapshot(agents));
    const list = dom.document.getElementById('agent-list');
    const dimmed = findFake(list, (node) => String(node.className).includes('agent-background'));
    assert(dimmed, 'a background child must carry the agent-background class');
    const group = findFake(list, (node) => node.className === 'agent-group-label');
    assert(
      group && /Background \(1\)/.test(group.textContent),
      `background children must be grouped: ${group && group.textContent}`,
    );
    const title = findFake(dimmed, (node) => String(node.className).includes('agent-title'));
    assert(
      title && title.textContent.includes('background'),
      `the dimmed marker must surface in the title: ${title && title.textContent}`,
    );
    const toggle = findFake(
      dimmed,
      (node) => node.tagName === 'button' && node.textContent === 'Foreground',
    );
    assert(toggle, 'a background child must offer a Foreground toggle');
    toggle.click();
    assertDeepEqual(posted[posted.length - 1], {
      type: 'agentControl',
      agentId: 'c1',
      action: 'presentation',
      state: 'foreground',
    });

    const foreground = st.summarizeAgents(nc.validateAgents(clone(agentsJson)));
    const second = runChatWebview(webviewSnapshot(foreground));
    const secondList = second.dom.document.getElementById('agent-list');
    assert(
      !findFake(secondList, (node) => String(node.className).includes('agent-background')),
      'foreground children are never dimmed',
    );
    assert(
      !findFake(secondList, (node) => node.className === 'agent-group-label'),
      'no background group without background children',
    );
    const forward = findFake(
      secondList,
      (node) => node.tagName === 'button' && node.textContent === 'Background',
    );
    assert(forward, 'a foreground child must offer a Background toggle');
    forward.click();
    assertDeepEqual(second.posted[second.posted.length - 1], {
      type: 'agentControl',
      agentId: 'c1',
      action: 'presentation',
      state: 'background',
    });
  });
}

// ----------------------------------- tournament cockpit + reduced motion

function tournamentCockpitSnapshot(tournament) {
  const cockpit = cp.buildCockpit({
    task: null,
    agents: [],
    verification: null,
    usage: null,
    taskVerification: null,
    tournament,
  });
  const snapshot = webviewSnapshot([]);
  snapshot.cockpit = cockpit;
  snapshot.cockpitSections = cp.cockpitSections(cockpit);
  snapshot.tournament = tournament;
  return snapshot;
}

async function tournamentWebviewTests() {
  await test('cockpit Decide/Abort are state-gated and post only when enabled', () => {
    const candidate = (childId, state) => ({
      childId,
      state,
      verification: null,
      verificationPass: null,
      reviewRank: null,
      reviewer: null,
      costMicro: 0,
      wallMs: 0,
    });
    const wire = {
      id: 't-1',
      state: 'open',
      winner: null,
      criteria: [{ id: 'c-1', spec: 'tests pass' }],
      candidates: [candidate('child-0', 'done'), candidate('child-1', 'running')],
    };
    const running = cp.tournamentViewOf(wire);
    const first = runChatWebview(tournamentCockpitSnapshot(running));
    const cockpit = first.dom.document.getElementById('cockpit');
    const decide = findFake(
      cockpit,
      (node) => node.tagName === 'button' && node.textContent === 'Decide winner',
    );
    const abort = findFake(cockpit, (node) => node.tagName === 'button' && node.textContent === 'Abort');
    assert(decide && abort, 'the tournament section must render Decide/Abort controls');
    assertEqual(decide.disabled, true, 'decide is disabled until every candidate settles');
    assertEqual(abort.disabled, false, 'abort is enabled while the tournament is open');
    const before = first.posted.length;
    decide.click();
    assertEqual(first.posted.length, before, 'a disabled Decide must never post');
    abort.click();
    assertDeepEqual(first.posted[first.posted.length - 1], {
      type: 'tournamentControl',
      tournamentId: 't-1',
      action: 'abort',
    });

    // Every candidate settled: decide enables and posts the exact control.
    const settled = cp.tournamentViewOf({
      ...wire,
      candidates: [candidate('child-0', 'done'), candidate('child-1', 'done')],
    });
    const second = runChatWebview(tournamentCockpitSnapshot(settled));
    const secondCockpit = second.dom.document.getElementById('cockpit');
    const decideNow = findFake(
      secondCockpit,
      (node) => node.tagName === 'button' && node.textContent === 'Decide winner',
    );
    assertEqual(decideNow.disabled, false, 'decide enables once all candidates settle');
    decideNow.click();
    assertDeepEqual(second.posted[second.posted.length - 1], {
      type: 'tournamentControl',
      tournamentId: 't-1',
      action: 'decide',
    });

    // A terminal tournament renders disabled controls only.
    const decided = cp.tournamentViewOf({
      ...wire,
      state: 'decided',
      winner: 'child-0',
      candidates: [candidate('child-0', 'done'), candidate('child-1', 'discarded')],
    });
    const third = runChatWebview(tournamentCockpitSnapshot(decided));
    const thirdCockpit = third.dom.document.getElementById('cockpit');
    const buttons = [];
    walkFake(thirdCockpit, (node) => {
      if (node.tagName === 'button' && (node.textContent === 'Abort' || node.textContent === 'Decide winner')) {
        buttons.push(node);
      }
    });
    assertEqual(buttons.length, 2, 'terminal tournaments still render both controls');
    assertEqual(
      buttons.every((button) => button.disabled === true),
      true,
      'terminal tournament controls are all disabled',
    );
  });
}

async function reducedMotionTests() {
  await test('prefers-reduced-motion disables every .pixel-* animation with static cues', () => {
    const css = readFileSync(new URL('../media/chat.css', import.meta.url), 'utf8');
    const match = /@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{([\s\S]*?)\n\}/.exec(css);
    assert(match, 'chat.css must carry a prefers-reduced-motion block');
    const block = match[1];
    const pixelClasses = [
      'pixel',
      'pixel-running',
      'pixel-paused',
      'pixel-waiting',
      'pixel-blocked',
      'pixel-done',
      'pixel-failed',
      'pixel-cancelled',
    ];
    for (const cls of pixelClasses) {
      assert(
        new RegExp(`\\.${cls}(?![\\w-])`).test(block),
        `reduced-motion block must cover .${cls}`,
      );
    }
    assert(/animation:\s*none/.test(block), 'reduced motion must disable animations');
    assert(/transform:\s*none/.test(block), 'reduced motion must disable transforms');
    for (const cls of pixelClasses.filter((entry) => entry !== 'pixel')) {
      assert(
        new RegExp(`\\.${cls}(?![\\w-])\\s*\\{[^}]*opacity:`).test(block),
        `reduced motion must keep a static opacity cue for .${cls}`,
      );
    }
  });
}


// ------------------------------------------- vendored webview packaging (P0)

function sha256File(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function walkPackagedFiles(root) {
  const out = [];
  const visit = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const rel = prefix.length > 0 ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isSymbolicLink()) {
        throw new Error(`symlink not allowed in the packaged webview: ${rel}`);
      }
      if (entry.isDirectory()) {
        visit(join(dir, entry.name), rel);
      } else if (entry.isFile()) {
        out.push(rel);
      }
    }
  };
  visit(root, '');
  return out;
}

async function vendoredResolutionTests() {
  await test('vendored resolution never leaves extensionUri (no checkout escape)', () => {
    const source = readFileSync(new URL('../src/webview.ts', import.meta.url), 'utf8');
    assert(
      source.includes("'media', 'kilo-v756-webview'"),
      'vendoredRoot must join extensionUri with media/kilo-v756-webview',
    );
    assert(!/'\.\.',\s*'\.\.'/.test(source), 'checkout-relative ../.. resolution must be gone');
    assert(source.includes('FAKTOR_UI_BUNDLE'), 'the explicit dev override must stay documented');
    assert(source.includes('vendoredFallbackNotice'), 'the missing-bundle notice path must stay wired');
  });
}

async function packagedLayoutTests(dir) {
  await test(`packaged VSIX layout is self-contained (${dir})`, () => {
    assert(existsSync(dir), `packaged dir does not exist: ${dir}`);
    for (const rel of [
      'out/extension.js',
      'out/webview.js',
      'out/kilo-bridge.js',
      'media/chat.js',
      'media/chat.css',
      'media/composer-state.js',
      'media/faktor.svg',
      'media/kilo-v756-webview/dist/webview.js',
      'media/kilo-v756-webview/dist/webview.css',
      'media/kilo-v756-webview/dist/shiki-worker.js',
    ]) {
      assert(existsSync(join(dir, ...rel.split('/'))), `packaged extension is missing ${rel}`);
    }
    const webviewRoot = join(dir, 'media', 'kilo-v756-webview');
    const files = walkPackagedFiles(webviewRoot);
    assert(files.length >= 25, `packaged vendored webview must hold >= 25 files, found ${files.length}`);

    const built = readFileSync(join(dir, 'out', 'webview.js'), 'utf8');
    assert(
      built.includes("'media', 'kilo-v756-webview'"),
      'compiled webview.js must resolve inside extensionUri/media/kilo-v756-webview',
    );
    assert(!/'\.\.',\s*'\.\.'/.test(built), 'compiled webview.js must not escape the extension');

    const manifestPath = fileURLToPath(
      new URL('../../../ui/kilo-v756-webview/dist/build-manifest.json', import.meta.url),
    );
    assert(existsSync(manifestPath), `pinned manifest must exist: ${manifestPath}`);
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    let verified = 0;
    for (const file of manifest.vendored ?? []) {
      const packaged = join(webviewRoot, ...file.path.split('/'));
      assert(existsSync(packaged), `packaged vendored file missing: ${file.path}`);
      const stat = statSync(packaged);
      assert(
        stat.size === file.size && sha256File(packaged) === file.sha256,
        `packaged vendored file diverges from the pin: ${file.path}`,
      );
      verified += 1;
    }
    assert(
      verified === (manifest.vendored ?? []).length && verified > 0,
      `expected the full pinned closure, verified ${verified}`,
    );

    // Additive Faktor overlay in the packaged layout: pinned by the overlay
    // manifest, recorded in the staged build manifest, markers intact.
    const packagedOverlayManifest = fileURLToPath(
      new URL('../../../ui/kilo-v756-webview/dist/toolchain/overlay/overlay-manifest.json', import.meta.url),
    );
    assert(existsSync(packagedOverlayManifest), 'the overlay manifest must exist for packaged verification');
    const overlayManifest = JSON.parse(readFileSync(packagedOverlayManifest, 'utf8'));
    let overlayVerified = 0;
    for (const file of overlayManifest.files ?? []) {
      const packaged = join(webviewRoot, 'dist', 'overlay', ...file.path.split('/'));
      assert(existsSync(packaged), `packaged overlay file missing: ${file.path}`);
      const stat = statSync(packaged);
      assert(
        stat.size === file.size && sha256File(packaged) === file.sha256,
        `packaged overlay file diverges: ${file.path}`,
      );
      overlayVerified += 1;
    }
    assert(
      overlayVerified === (overlayManifest.files ?? []).length && overlayVerified > 0,
      `expected both overlay files, verified ${overlayVerified}`,
    );
    const stagedManifestPath = join(webviewRoot, 'dist', 'build-manifest.json');
    assert(existsSync(stagedManifestPath), 'the staged build manifest must ship with the bundle');
    const stagedManifest = JSON.parse(readFileSync(stagedManifestPath, 'utf8'));
    assertEqual(
      (stagedManifest.faktorOverlay?.files ?? []).length,
      (overlayManifest.files ?? []).length,
      'the staged manifest must record every merged overlay hash',
    );
    assertDeepEqual(
      stagedManifest.vendored,
      manifest.vendored,
      'the packaged pinned vendored list must stay byte-identical',
    );
    const panel = readFileSync(
      join(webviewRoot, 'dist', 'overlay', 'faktor-companion.js'),
      'utf8',
    );
    for (const marker of [
      'faktorTaskState',
      'faktorAgents',
      'faktorCockpit',
      'faktorTournament',
      'faktorEvidence',
      'faktorBoardState',
      'faktorAgentAction',
      'faktorTournamentAction',
      'faktorEvidenceExpand',
      'faktorBoardAction',
    ]) {
      assert(panel.includes(marker), `packaged companion panel must consume/host ${marker}`);
    }
  });
}

// ------------------------------------------ Faktor companion overlay build

const OVERLAY_DIR = fileURLToPath(
  new URL('../../../ui/kilo-v756-webview/dist/toolchain/overlay', import.meta.url),
);
const OVERLAY_MANIFEST_PATH = join(OVERLAY_DIR, 'overlay-manifest.json');
const WEBVIEW_BUILD_MANIFEST = fileURLToPath(
  new URL('../../../ui/kilo-v756-webview/dist/build-manifest.json', import.meta.url),
);
const UPSTREAM_MANIFEST = fileURLToPath(new URL('../../../ui/upstream.json', import.meta.url));

async function overlayBuildTests() {
  await test('overlay manifest verifies clean and refuses tampered/extra files', () => {
    const clean = verifyOverlay();
    assert(clean.ok, `clean overlay must verify: ${clean.errors.join('; ')}`);
    assertEqual(clean.checked, 2, 'both panel files must be pinned');
    const dir = mkdtempSync(join(tmpdir(), 'faktor-overlay-tamper-'));
    try {
      cpSync(OVERLAY_DIR, dir, { recursive: true });
      writeFileSync(join(dir, 'faktor-companion.js'), '// tampered panel');
      const tampered = verifyOverlay(dir);
      assert(!tampered.ok, 'tampering must fail the overlay verification');
      assert(
        tampered.errors.some((error) => error.includes('hash mismatch')),
        `expected a hash mismatch: ${tampered.errors.join('; ')}`,
      );
      writeFileSync(join(dir, 'intruder.js'), '// extra');
      const extra = verifyOverlay(dir);
      assert(
        extra.errors.some((error) => error.includes('unexpected overlay file: intruder.js')),
        `expected an unexpected-file error: ${extra.errors.join('; ')}`,
      );
      rmSync(join(dir, 'intruder.js'));
      rmSync(join(dir, 'faktor-companion.js'));
      const missing = verifyOverlay(dir);
      assert(!missing.ok && missing.errors.some((error) => error.includes('missing')), 'missing file must fail');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  await test('staging merges the companion panel and keeps upstream bytes pinned', async () => {
    const manifestBefore = sha256File(WEBVIEW_BUILD_MANIFEST);
    const pinned = JSON.parse(readFileSync(WEBVIEW_BUILD_MANIFEST, 'utf8'));
    const overlay = JSON.parse(readFileSync(OVERLAY_MANIFEST_PATH, 'utf8'));
    const tmp = mkdtempSync(join(tmpdir(), 'faktor-overlay-stage-'));
    try {
      const stats = await stageBundle(tmp);
      assertEqual(stats.files, pinned.vendored.length, 'every pinned file must be staged');
      assertEqual(stats.overlayFiles, overlay.files.length, 'every overlay file must be merged');
      let verified = 0;
      for (const file of pinned.vendored) {
        const staged = join(tmp, ...file.path.split('/'));
        assert(existsSync(staged), `staged pinned file missing: ${file.path}`);
        const stat = statSync(staged);
        assert(
          stat.size === file.size && sha256File(staged) === file.sha256,
          `staged upstream file diverges byte-for-byte: ${file.path}`,
        );
        verified += 1;
      }
      assert(verified > 0, 'the pinned closure must not be empty');
      for (const file of overlay.files) {
        const staged = join(tmp, 'dist', 'overlay', ...file.path.split('/'));
        assert(existsSync(staged), `staged overlay file missing: ${file.path}`);
        const stat = statSync(staged);
        assert(
          stat.size === file.size && sha256File(staged) === file.sha256,
          `staged overlay file diverges: ${file.path}`,
        );
      }
      const stagedManifest = JSON.parse(readFileSync(join(tmp, 'dist', 'build-manifest.json'), 'utf8'));
      assertEqual(stagedManifest.faktorOverlay.files.length, overlay.files.length);
      assertDeepEqual(
        stagedManifest.vendored,
        pinned.vendored,
        'the staged manifest must keep the pinned vendored list byte-identical',
      );
      for (const entry of stagedManifest.faktorOverlay.files) {
        assert(
          entry.path.startsWith('dist/overlay/'),
          `merged overlay must land under dist/overlay: ${entry.path}`,
        );
      }
      assertEqual(
        sha256File(WEBVIEW_BUILD_MANIFEST),
        manifestBefore,
        'staging must never mutate the pinned source manifest',
      );
      const upstream = JSON.parse(readFileSync(UPSTREAM_MANIFEST, 'utf8'));
      const upstreamPaths = Object.keys(upstream.file_hashes ?? {});
      assert(
        upstreamPaths.every((path) => !path.includes('overlay/faktor-companion')),
        'the overlay must never enter the upstream pin',
      );
      const panel = readFileSync(join(OVERLAY_DIR, 'faktor-companion.js'), 'utf8');
      for (const marker of [
        'faktorTaskState',
        'faktorAgents',
        'faktorCockpit',
        'faktorTournament',
        'faktorEvidence',
        'faktorBoardState',
        'faktor-companion',
      ]) {
        assert(panel.includes(marker), `companion panel must consume/host ${marker}`);
      }
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });
}

// --------------------------------------- companion panel (vm + fake DOM)

function makePanelDom() {
  function makeNode(tagName) {
    const node = {
      tagName,
      id: '',
      className: '',
      textContent: '',
      value: '',
      rows: 0,
      maxLength: 0,
      type: '',
      placeholder: '',
      disabled: false,
      style: {},
      children: [],
      parentNode: null,
      attributes: {},
      listeners: {},
    };
    node.appendChild = (child) => {
      node.children.push(child);
      child.parentNode = node;
      return child;
    };
    node.removeChild = (child) => {
      const index = node.children.indexOf(child);
      if (index >= 0) node.children.splice(index, 1);
      return child;
    };
    node.insertBefore = (child, reference) => {
      const index = node.children.indexOf(reference);
      if (index < 0) node.children.push(child);
      else node.children.splice(index, 0, child);
      child.parentNode = node;
      return child;
    };
    node.setAttribute = (key, value) => {
      node.attributes[key] = String(value);
    };
    node.getAttribute = (key) => (key in node.attributes ? node.attributes[key] : null);
    node.addEventListener = (type, callback) => {
      if (!node.listeners[type]) node.listeners[type] = [];
      node.listeners[type].push(callback);
    };
    node.dispatch = (type, event) => {
      for (const callback of node.listeners[type] || []) {
        callback(event || { preventDefault() {} });
      }
    };
    node.click = () => node.dispatch('click', {});
    Object.defineProperty(node, 'firstChild', { get: () => node.children[0] || null });
    return node;
  }
  function walk(root, visit) {
    visit(root);
    for (const child of root.children || []) walk(child, visit);
  }
  const root = makeNode('div');
  root.id = 'root';
  const body = makeNode('body');
  body.appendChild(root);
  const document = {
    readyState: 'complete',
    body,
    getElementById(id) {
      let found = null;
      walk(body, (node) => {
        if (found === null && node.id === id) found = node;
      });
      return found;
    },
    createElement: makeNode,
    createElementNS: (_namespace, tagName) => makeNode(tagName),
    addEventListener() {},
    querySelectorAll: () => [],
  };
  return { document, body, walk };
}

function runCompanionPanel() {
  const source = readFileSync(join(OVERLAY_DIR, 'faktor-companion.js'), 'utf8');
  const posted = [];
  const dom = makePanelDom();
  const sandbox = {
    document: dom.document,
    window: {
      addEventListener(type, callback) {
        if (type === 'message') sandbox._message = callback;
      },
      __faktorVsCodeApi: () => ({ postMessage: (message) => posted.push(message) }),
    },
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);
  const companion = sandbox.window.__faktorCompanion;
  assert(companion, 'the panel must expose __faktorCompanion for the webview host');
  assert(sandbox._message, 'the panel must register a window message listener');
  const panel = dom.document.getElementById('faktor-companion');
  assert(panel, 'the panel must mount as #faktor-companion next to #root');
  return { posted, dom, panel, companion };
}

function findAll(root, predicate, walk) {
  const out = [];
  walk(root, (node) => {
    if (predicate(node)) out.push(node);
  });
  return out;
}

async function companionPanelTests() {
  await test('companion panel renders Faktor frames and posts state-gated actions', () => {
    const { posted, dom, panel, companion } = runCompanionPanel();
    const { walk } = dom;

    companion.handle({
      type: 'faktorTaskState',
      present: true,
      goal: 'ship the cockpit',
      state: 'running',
      phase: 'implementation',
      acceptanceCriteria: ['build passes'],
      milestones: { completed: [], open: ['main'] },
      tests: { run: [], failed: ['cargo test'] },
      blockers: ['waiting on analysis'],
      verification: { status: 'pending', criteriaPassed: 0, criteriaTotal: 1, checksFailed: 0, owed: 1, failedChecks: 1 },
      budget: null,
    });
    const taskText = findAll(panel, (node) => node.textContent === 'ship the cockpit', walk);
    assert(taskText.length === 1, 'the task goal must render');
    assert(
      findAll(panel, (node) => node.textContent === '! waiting on analysis', walk).length === 1,
      'blockers must render',
    );

    companion.handle({
      type: 'faktorAgents',
      agents: [
        {
          agentId: 'c1',
          kind: 'child',
          state: 'Failed',
          goal: 'implement main',
          model: 'm',
          provider: 'p',
          presentation: 'background',
          blockers: ['dependency x'],
          pixel: {
            childId: 'c1',
            state: 'failed',
            animation: 'pixel-failed',
            avatar: { pixels: new Array(25).fill(1), color: 'red', accent: 'pink', hash: 1, version: 1 },
          },
        },
        {
          agentId: 'c2',
          kind: 'child',
          state: 'Running',
          goal: 'verify',
          presentation: 'foreground',
        },
      ],
    });
    const cards = findAll(panel, (node) => node.tagName === 'article' && node.className === 'faktor-agent', walk);
    assertEqual(cards.length, 2, 'both agent cards must render');
    assertEqual(cards[0].getAttribute('data-presentation'), 'background');
    assertEqual(
      findAll(cards[0], (node) => String(node.className).includes('faktor-pixel-bit'), walk).length,
      25,
      'the 5x5 pixel identity must render',
    );
    const retry = findAll(cards[0], (node) => node.tagName === 'button' && node.textContent === 'Retry', walk)[0];
    assert(retry, 'a failed child must offer Retry');
    retry.click();
    assertDeepEqual(posted[posted.length - 1], {
      type: 'faktorAgentAction',
      agentId: 'c1',
      action: 'retry',
    });
    const pause = findAll(cards[1], (node) => node.tagName === 'button' && node.textContent === 'Pause', walk)[0];
    assert(pause, 'a running child must offer Pause');
    pause.click();
    assertEqual(posted[posted.length - 1].action, 'pause');

    // Steer (P2 UI parity): a bounded inline textbox plus the button posts
    // the EXACT faktorAgentAction body; an empty box falls back to the bare
    // host-prompt request.
    const steerInput = findAll(
      cards[1],
      (node) => node.tagName === 'input' && node.getAttribute('aria-label') === 'steer note',
      walk,
    )[0];
    assert(steerInput, 'a child card must offer the inline steer textbox');
    assertEqual(steerInput.maxLength, 500, 'the steer textbox must be bounded at 500 chars');
    const steerButton = findAll(
      cards[1],
      (node) => node.tagName === 'button' && node.textContent === 'Steer',
      walk,
    )[0];
    assert(steerButton, 'a child card must offer Steer');
    steerInput.value = '  focus on the parser  ';
    steerButton.click();
    assertDeepEqual(posted[posted.length - 1], {
      type: 'faktorAgentAction',
      agentId: 'c2',
      action: 'steer',
      text: 'focus on the parser',
    });
    steerInput.value = '   ';
    steerButton.click();
    assertDeepEqual(posted[posted.length - 1], {
      type: 'faktorAgentAction',
      agentId: 'c2',
      action: 'steer',
    });
    const toggle = findAll(
      cards[0],
      (node) => node.tagName === 'button' && node.textContent === 'Foreground',
      walk,
    )[0];
    toggle.click();
    assertDeepEqual(posted[posted.length - 1], {
      type: 'faktorAgentAction',
      agentId: 'c1',
      action: 'presentation',
      state: 'foreground',
    });

    // Tournament: Decide gates on canDecide; Abort gates on open.
    const tournament = (canDecide, open, id = 't-1') => ({
      type: 'faktorTournament',
      present: true,
      tournament: { id, state: open ? 'open' : 'decided', open, canDecide, winner: null, criteria: ['tests pass'], candidates: [] },
    });
    companion.handle(tournament(false, true));
    let decide = findAll(
      panel,
      (node) => node.tagName === 'button' && node.textContent === 'Decide winner',
      walk,
    )[0];
    assert(decide && decide.disabled === true, 'decide must be disabled until every candidate settled');
    companion.handle(tournament(true, true));
    decide = findAll(panel, (node) => node.tagName === 'button' && node.textContent === 'Decide winner', walk)[0];
    assert(decide && decide.disabled === false, 'decide must enable when canDecide');
    decide.click();
    assertDeepEqual(posted[posted.length - 1], {
      type: 'faktorTournamentAction',
      tournamentId: 't-1',
      action: 'decide',
    });
    companion.handle(tournament(true, false));
    const abort = findAll(panel, (node) => node.tagName === 'button' && node.textContent === 'Abort', walk)[0];
    assert(abort && abort.disabled === true, 'abort must be disabled on a terminal tournament');

    // Evidence: refs open an expansion; the expansion renders as text.
    companion.handle({ type: 'faktorEvidence', mode: 'refs', refs: [{ id: 41, label: 'evidence:41' }] });
    const ref = findAll(panel, (node) => node.tagName === 'button' && node.textContent === 'evidence:41', walk)[0];
    assert(ref, 'an evidence ref must be a button');
    ref.click();
    assertDeepEqual(posted[posted.length - 1], { type: 'faktorEvidenceExpand', evidenceId: 41 });
    companion.handle({
      type: 'faktorEvidence',
      mode: 'expanded',
      evidence: { id: 41, text: 'artifact text', truncated: false },
    });
    const pre = findAll(panel, (node) => node.tagName === 'pre', walk)[0];
    assert(pre && pre.textContent === 'artifact text', 'the expanded artifact must render as text');

    // Board: explicit unavailable, then posts + composer.
    companion.handle({
      type: 'faktorBoardState',
      available: false,
      source: 'none',
      revision: null,
      unread: null,
      posts: [],
      reason: 'no board route',
    });
    assert(
      findAll(panel, (node) => node.textContent === 'no board route', walk).length === 1,
      'an unavailable board must render its explicit reason',
    );
    companion.handle({
      type: 'faktorBoardState',
      available: true,
      source: 'transcript',
      revision: 3,
      unread: 2,
      posts: [{ id: 'p1', author: 'parent', subject: 'handoff', body: 'ready', refs: [] }],
      reason: null,
    });
    assert(
      findAll(panel, (node) => node.textContent === '2 unread', walk).length === 1,
      'unread count must render',
    );
    const subject = findAll(
      panel,
      (node) => node.tagName === 'input' && node.getAttribute('aria-label') === 'board subject',
      walk,
    )[0];
    const body = findAll(
      panel,
      (node) => node.tagName === 'textarea' && node.getAttribute('aria-label') === 'board body',
      walk,
    )[0];
    assert(subject && body, 'the board composer must render when available');
    subject.value = 'status';
    body.value = 'all green';
    const postButton = findAll(
      panel,
      (node) => node.tagName === 'button' && node.textContent === 'Post',
      walk,
    )[0];
    postButton.click();
    assertDeepEqual(posted[posted.length - 1], {
      type: 'faktorBoardAction',
      action: 'post',
      subject: 'status',
      body: 'all green',
    });
  });
}

// -------------------------------------------------------------------- main

async function main() {
  await validatorAccepts();
  await validatorRejects();
  await clientAccepts();
  await clientRejects();
  await eventStreamTests();
  await stateTests();
  await daemonTests();
  await shadowDefaultTests();
  await completionContractTests();
  await pendingSubmissionTests();
  await boardAndForwardingTests();
  await draftPreservationTests();
  await runStateTests();
  await workspaceBindingTests();
  await childInspectionTests();
  await pixelAgentTests();
  await cockpitTests();
  await presentationWebviewTests();
  await tournamentWebviewTests();
  await reducedMotionTests();
  await overlayBuildTests();
  await companionPanelTests();
  await vendoredResolutionTests();
  if (packagedDir !== null && packagedDir !== undefined) {
    await packagedLayoutTests(packagedDir);
  }
  for (const { label, fn } of bridgeTests) {
    await test(`bridge: ${label}`, fn);
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) {
    process.exit(1);
  }
  console.log('SELFTEST OK');
}

main().catch((error) => {
  console.error(`FATAL: ${error && error.stack ? error.stack : error}`);
  process.exit(1);
});
