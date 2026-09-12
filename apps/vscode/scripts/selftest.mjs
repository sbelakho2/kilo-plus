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
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { bridgeTests } from './bridge-selftest.mjs';

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
    assertEqual(nc.validateTaskRunCancelled(clone(taskRunCancelledJson)).cancelled, true);
    assertEqual(nc.validateAgents(clone(agentsJson)).length, 2);
    assertEqual(nc.validateAgents(clone(agentsJson))[1].presentation, 'foreground');
    const presented = clone(agentsJson);
    presented[1].presentation = 'background';
    assertEqual(nc.validateAgents(presented)[1].presentation, 'background');
    assertEqual(nc.validateAgentControlAck(clone(controlAckJson), 'test').queuedSeq, 3);
    assertEqual(nc.validateAgentPresentationAck(clone(presentationAckJson), 'test').presentation, 'background');
    assertEqual(nc.validateMessagePage(clone(messagePageJson)).messages[0].parts[0].kind, 'text');
    assertEqual(nc.validateEventPage(clone(eventPageJson)).events[0].seq, 1);
    assertEqual(nc.validateSessionUsage(clone(sessionUsageJson)).tasks[0].taskId, 't1');
    assertEqual(nc.validateUsage(clone(usageTotalsJson)).durable.providerCalls.tokens, 130);
    const aggregateUsage = clone(usageTotalsJson);
    delete aggregateUsage.durable.reservations.routeDecisions;
    assertDeepEqual(nc.validateUsage(aggregateUsage).durable.reservations.routeDecisions, []);
    assertEqual(nc.validateTaskVerification(clone(taskVerificationJson)).records[0].checks[0].exit, 0);
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
    assertProtocol(() => nc.validateTaskViews([{ ...clone(taskViewJson), budget: { ...clone(budgetJson), spentCostMicro: '12' } }]), 'expected a finite number');
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
    assertEqual((await client.taskRunState('7', 'r1')).run_id, 'r1');
    assertEqual((await client.startTaskRun('7', { goal: 'ship it' })).run_id, 'r1');
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
    assertDeepEqual(findCall(calls, 'POST', '/native/evidence/41/retrieve').body, {
      selector: 'all',
    });
    assertDeepEqual(findCall(calls, 'POST', '/native/evidence/41/retrieve').query, { session: '7' });
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
      ['acceptance', 'plan', 'children', 'phase', 'blockers', 'verification', 'evidence', 'spend'],
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
  return { posted, dom };
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
  await draftPreservationTests();
  await runStateTests();
  await workspaceBindingTests();
  await childInspectionTests();
  await pixelAgentTests();
  await cockpitTests();
  await presentationWebviewTests();
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
