#!/usr/bin/env node
// Faktor master integration test — a zero-dependency Node ESM protocol client.
//
// This harness IS the frozen v7.5.6 client for integration purposes: it
// launches the real faktor-cli daemon, performs the full wire flow, and fails
// loudly (nonzero exit) on ANY incompatibility with the frozen contract in
// crates/protocol/src/v756/wire.rs, crates/server/src/api.rs, and the
// fixtures in compat/kilo-v756/.
//
// Usage: node apps/vscode/harness/client.mjs [path-to-faktor-cli]
//   (the binary path defaults to $FAKTOR_BIN, then ../target/debug or
//    ../target/release relative to the repo root)
//
// Only node:http, node:child_process, node:crypto, node:fs, node:path —
// no npm dependencies, no npm install required.
//
// Notes on intentional deviations (harness-only, documented):
//   * The harness passes `--data-dir <fresh temp dir>` so repeated local runs
//     and CI never touch ~/.faktor and never inherit stale sessions. The
//     extension itself spawns `faktor-cli serve --port 0` without it.
//   * The daemon has NO real providers (the CLI registers none by default):
//     the message send runs the turn and fails at provider lookup. The turn
//     fix (crates/agent) lands the session on `failed_recoverable`, which is
//     promptable, so step 7 accepts `ready_for_next_turn` OR `completed` OR
//     `failed_recoverable` — the wire surface (message accepted, paging,
//     SSE) is what is proven here, not model output.
//   * SSE frames are `id: <n>\ndata: <json>\n\n` with an optional `event:`
//     field; heartbeats (`event: heartbeat`, `data: {}`, and axum keep-alive
//     text frames) are parsed but ignored.

import http from 'node:http';
import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import os from 'node:os';

const HARNESS_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HARNESS_DIR, '..', '..', '..');
const COMPAT_DIR = join(REPO_ROOT, 'compat', 'kilo-v756');

const STARTUP_TIMEOUT_MS = 5_000;
const TURN_POLL_TIMEOUT_MS = 10_000;
const SSE_TIMEOUT_MS = 8_000;
const OVERALL_BUDGET_MS = 30_000;
const TURN_POLL_INTERVAL_MS = 100;

// The exact frozen startup line (compat/kilo-v756/startup_line.json).
const STARTUP_LINE_RE = /^faktor server listening on http:\/\/127\.0\.0\.1:(\d+)$/;

// Frozen SSE payload discriminators (faktor_protocol::v756::GlobalEventPayload).
const FROZEN_EVENT_TYPES = new Set([
  'session_created',
  'session_turn_open',
  'session_turn_close',
  'session_queue_changed',
  'background_process_updated',
  'interactive_terminal_data',
  'sandbox_status_changed',
  'indexing_status',
  'message_part_updated',
  'session_next_text_delta',
  'session_next_reasoning_delta',
  'session_next_tool_called',
  'session_state_changed',
  'error',
]);

// Frozen wire part types (faktor_protocol::v756::wire::WirePart).
const FROZEN_PART_TYPES = new Set([
  'text',
  'subtask',
  'reasoning',
  'file',
  'tool',
  'stepStart',
  'stepFinish',
  'snapshot',
  'patch',
  'agent',
  'retry',
  'compaction',
]);

// --------------------------------------------------------------------------
// small test framework: bounded, loud, exit-code-driven
// --------------------------------------------------------------------------

let failures = 0;
let startedAt = Date.now();

function remainingBudget() {
  return Math.max(1, OVERALL_BUDGET_MS - (Date.now() - startedAt));
}

function checkBudget(label) {
  if (Date.now() - startedAt > OVERALL_BUDGET_MS) {
    console.error(`FAIL: overall ${OVERALL_BUDGET_MS}ms budget exceeded at step "${label}"`);
    process.exit(1);
  }
}

function pass(label, detail) {
  console.log(`PASS  ${label}${detail ? ` — ${detail}` : ''}`);
}

function fail(label, detail) {
  failures += 1;
  console.error(`FAIL  ${label}${detail ? ` — ${detail}` : ''}`);
}

// Assert that every key of `obj` is one of `allowed` (exact names). Mirrors
// the server's deny_unknown_fields loudly on the client side.
function assertSubsetKeys(obj, allowed, what) {
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
    throw new Error(`${what}: expected an object, got ${JSON.stringify(obj)}`);
  }
  const extra = Object.keys(obj).filter((k) => !allowed.includes(k));
  if (extra.length > 0) {
    throw new Error(
      `${what}: unexpected field(s) ${extra.join(', ')} (allowed: ${allowed.join(', ')})`,
    );
  }
}

function loadFixture(name) {
  return JSON.parse(readFileSync(join(COMPAT_DIR, name), 'utf8'));
}

// --------------------------------------------------------------------------
// daemon lifecycle
// --------------------------------------------------------------------------

function generatePassword() {
  return crypto.randomBytes(32).toString('hex');
}

function findBinary() {
  if (process.argv[2]) {
    return process.argv[2];
  }
  if (process.env.FAKTOR_BIN) {
    return process.env.FAKTOR_BIN;
  }
  for (const rel of ['target/debug/faktor-cli', 'target/release/faktor-cli']) {
    const candidate = join(REPO_ROOT, rel);
    try {
      readFileSync(candidate);
      return candidate;
    } catch {
      // keep looking
    }
  }
  throw new Error('faktor-cli binary not found (set FAKTOR_BIN or pass a path)');
}

function launchDaemon(binPath) {
  return new Promise((resolve, reject) => {
    const password = generatePassword();
    // Isolated store: repeated runs and CI never pollute ~/.faktor or inherit
    // stale sessions (harness-only deviation, see header).
    const dataDir = mkdtempSync(join(os.tmpdir(), 'faktor-harness-'));
    const child = spawn(binPath, ['serve', '--port', '0', '--data-dir', dataDir], {
      env: { ...process.env, FAKTOR_SERVER_PASSWORD: password },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let buffer = '';
    let settled = false;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        child.kill('SIGTERM');
        reject(new Error(`timed out waiting ${STARTUP_TIMEOUT_MS}ms for the startup line`));
      }
    }, STARTUP_TIMEOUT_MS);
    child.on('error', (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(err);
      }
    });
    child.on('exit', (code, signal) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(
          new Error(
            `daemon exited before the startup line (code=${code ?? 'null'} signal=${signal ?? 'null'})`,
          ),
        );
      }
    });
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk) => {
      buffer += chunk;
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).replace(/\r$/, '');
        buffer = buffer.slice(idx + 1);
        if (line.length === 0) {
          continue;
        }
        const match = STARTUP_LINE_RE.exec(line);
        if (match) {
          if (!settled) {
            settled = true;
            clearTimeout(timer);
            resolve({ port: Number(match[1]), password, child, dataDir });
          }
          return;
        }
        // Nothing else may be printed on stdout (startup_line.json notes).
        reject(
          new Error(
            `daemon printed an unexpected stdout line: ${JSON.stringify(line)} ` +
              `(only the frozen startup line is allowed)`,
          ),
        );
        child.kill('SIGTERM');
        settled = true;
        clearTimeout(timer);
        return;
      }
    });
  });
}

function killDaemon(daemon) {
  try {
    if (daemon.child && daemon.child.exitCode === null && daemon.child.signalCode === null) {
      daemon.child.kill('SIGTERM');
    }
  } catch {
    // already gone
  }
  try {
    rmSync(daemon.dataDir, { recursive: true, force: true });
  } catch {
    // best effort
  }
}

// --------------------------------------------------------------------------
// HTTP + SSE client primitives
// --------------------------------------------------------------------------

function basicAuthHeader(password) {
  return 'Basic ' + Buffer.from(`kilo:${password}`).toString('base64');
}

function request(port, password, method, path, body, extraHeaders = {}) {
  return new Promise((resolve, reject) => {
    const headers = { Authorization: basicAuthHeader(password), ...extraHeaders };
    let payload = null;
    if (body !== undefined && body !== null) {
      payload = JSON.stringify(body);
      headers['Content-Type'] = 'application/json';
      headers['Content-Length'] = Buffer.byteLength(payload);
    }
    const req = http.request(
      { host: '127.0.0.1', port, path, method, headers },
      (res) => {
        let data = '';
        res.setEncoding('utf8');
        res.on('data', (c) => {
          data += c;
        });
        res.on('end', () => {
          const status = res.statusCode ?? 0;
          let parsed = null;
          if (data.length > 0) {
            try {
              parsed = JSON.parse(data);
            } catch (err) {
              reject(
                new Error(
                  `${method} ${path} -> HTTP ${status}: response is not JSON: ${JSON.stringify(data.slice(0, 200))}`,
                ),
              );
              return;
            }
          }
          if (status < 200 || status >= 300) {
            const err = new Error(`${method} ${path} -> HTTP ${status}`);
            err.status = status;
            err.body = parsed;
            reject(err);
            return;
          }
          resolve(parsed);
        });
      },
    );
    req.on('error', (err) => reject(err));
    if (payload !== null) {
      req.write(payload);
    }
    req.end();
  });
}

// Subscribe to /global/event, parse `id: <n>\ndata: <json>\n\n` frames, and
// call onEvent({id, event, data}) for every parseable frame. Resolves when
// onEvent returns true; rejects on timeout (bounded, never unbounded) or on
// transport/protocol errors (loud). Errors thrown by onEvent reject the
// promise instead of crashing the process mid-stream.
function sse(port, password, path, onEvent, timeoutMs) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timer = null;
    const settleResolve = (value) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        resolve(value);
      }
    };
    const settleReject = (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        req.destroy();
        reject(err);
      }
    };
    const req = http.request(
      {
        host: '127.0.0.1',
        port,
        path,
        method: 'GET',
        headers: { Authorization: basicAuthHeader(password), Accept: 'text/event-stream' },
      },
      (res) => {
        if ((res.statusCode ?? 0) !== 200) {
          res.resume();
          res.on('end', () => {
            settleReject(new Error(`SSE ${path} -> HTTP ${res.statusCode}`));
          });
          return;
        }
        let buffer = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          buffer += chunk;
          let idx;
          while ((idx = buffer.indexOf('\n\n')) >= 0) {
            const frame = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            try {
              handleFrame(frame, onEvent, settleResolve);
            } catch (err) {
              settleReject(err);
            }
          }
        });
        res.on('error', (err) => settleReject(err));
      },
    );
    req.on('error', (err) => settleReject(err));
    timer = setTimeout(() => {
      settleReject(new Error(`SSE ${path} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    req.end();
  });
}

function handleFrame(frame, onEvent, settleResolve) {
  let id = null;
  let event = null;
  const dataLines = [];
  for (const rawLine of frame.split('\n')) {
    const line = rawLine.startsWith('\r') ? rawLine.slice(1) : rawLine;
    if (line.startsWith(':')) {
      continue; // comment / keep-alive
    }
    if (line.startsWith('id:')) {
      id = line.slice(3).trim();
    } else if (line.startsWith('event:')) {
      event = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).replace(/^ /, ''));
    }
  }
  if (dataLines.length === 0) {
    return;
  }
  const raw = dataLines.join('\n');
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    // Heartbeat/keep-alive text frames are not JSON; ignore them.
    return;
  }
  const done = onEvent({ id, event, data });
  if (done) {
    settleResolve();
  }
}

// --------------------------------------------------------------------------
// fixture-contract validation (compat/kilo-v756/*.json)
// --------------------------------------------------------------------------

function validateFixtureContract() {
  const steps = [];
  const startup = loadFixture('startup_line.json');
  const template = startup.template.replace('{port}', '49152');
  const m = /^faktor server listening on http:\/\/127\.0\.0\.1:(\d+)$/.exec(template);
  if (!m) {
    throw new Error(`startup_line.json template no longer matches the frozen startup regex: ${template}`);
  }
  steps.push('startup_line.json template matches the startup regex');

  const basic = loadFixture('basic_auth.json');
  if (basic.username !== 'kilo') {
    throw new Error(`basic_auth.json username drift: ${basic.username}`);
  }
  const examplePw = basic.example_password;
  if (!/^[0-9a-f]{64}$/.test(examplePw)) {
    throw new Error(`basic_auth.json example password is not 64 hex chars`);
  }
  const exampleHeader = `Authorization: ${basic.basic_scheme} ${Buffer.from(
    `${basic.username}:${examplePw}`,
  ).toString('base64')}`;
  // The fixture's example_header is the full `Authorization: Basic <payload>`.
  if (exampleHeader !== basic.example_header) {
    throw new Error(
      'basic_auth.json example_header does not match base64("kilo:"+password); ' +
        `got ${exampleHeader}`,
    );
  }
  steps.push('basic_auth.json header format matches base64("kilo:"+password)');

  const createFixture = loadFixture('wire_session_create.json').request;
  const sendFixture = loadFixture('wire_message_send.json').request;
  const partUnion = loadFixture('wire_part_union.json');
  const partsByName = new Map(partUnion.map((e) => [e.name, e.part]));

  // The exact request JSON the client will emit (step 3).
  const createReq = { title: 'harness', model: { id: 'm', providerID: 'fake' } };
  assertSubsetKeys(createReq, Object.keys(createFixture), 'session create request');
  assertSubsetKeys(createReq.model, Object.keys(createFixture.model), 'session create model');

  // The exact request JSON the client will emit (steps 6 and 11).
  const sendReq = {
    model: { providerID: 'fake', modelID: 'm' },
    parts: [{ type: 'text', text: 'hi' }],
  };
  assertSubsetKeys(sendReq, Object.keys(sendFixture), 'message send request');
  assertSubsetKeys(sendReq.model, Object.keys(sendFixture.model), 'message send model');
  for (const part of sendReq.parts) {
    const fixturePart = partsByName.get(part.type);
    if (!fixturePart) {
      throw new Error(`part type ${part.type} missing from wire_part_union.json`);
    }
    assertSubsetKeys(part, Object.keys(fixturePart), `part "${part.type}"`);
  }
  steps.push('client request shapes are subsets of the frozen fixture requests');

  // The frozen part-union and event-type sets must be mutually consistent.
  for (const entry of partUnion) {
    if (!FROZEN_PART_TYPES.has(entry.name)) {
      throw new Error(`wire_part_union.json type "${entry.name}" missing from the frozen set`);
    }
  }
  const eventFixture = loadFixture('global_event.json');
  for (const entry of eventFixture) {
    if (!FROZEN_EVENT_TYPES.has(entry.event.payload.type)) {
      throw new Error(
        `global_event.json type "${entry.event.payload.type}" missing from the frozen set`,
      );
    }
  }
  steps.push('fixtures are consistent with the frozen part/event type sets');

  return steps;
}

// --------------------------------------------------------------------------
// the master flow: 11 steps
// --------------------------------------------------------------------------

async function main() {
  console.log(`Faktor harness: repo=${REPO_ROOT}`);
  const fixtureChecks = validateFixtureContract();
  for (const check of fixtureChecks) {
    pass(`fixture contract`, check);
  }

  const binPath = findBinary();
  console.log(`binary: ${binPath}`);

  // Hard overall budget: never run past 30s.
  const budgetTimer = setTimeout(() => {
    console.error('FAIL: overall 30s budget exceeded');
    process.exit(1);
  }, OVERALL_BUDGET_MS);
  budgetTimer.unref();

  let daemon = null;
  try {
    // ---- step 1: launch + health with Basic auth ------------------------
    checkBudget('1');
    daemon = await launchDaemon(binPath);
    const health = await request(daemon.port, daemon.password, 'GET', '/global/health');
    assertSubsetKeys(health, ['ok', 'version', 'protocol'], 'health response');
    if (health.ok !== true || !health.version || !health.protocol) {
      throw new Error(`health response malformed: ${JSON.stringify(health)}`);
    }
    pass(
      '1. health with Basic auth',
      `daemon on :${daemon.port}, ok=${health.ok}, protocol=${health.protocol}`,
    );

    // ---- step 2: health without auth → 401 ------------------------------
    checkBudget('2');
    try {
      await request(daemon.port, 'wrong-password', 'GET', '/global/health');
      throw new Error('expected 401 for missing/wrong auth');
    } catch (err) {
      if (err.status !== 401) {
        throw new Error(`expected 401, got HTTP ${err.status ?? 'no status'} (${err.message})`);
      }
      // ApiError JSON is wrapped in an `error` envelope (compat errors.json).
      if (err.body?.error?.code !== 'unauthorized') {
        throw new Error(
          `401 body must carry code "unauthorized", got ${JSON.stringify(err.body)}`,
        );
      }
    }
    pass('2. health without auth → 401', 'code=unauthorized');

    // ---- step 3: create session ------------------------------------------
    checkBudget('3');
    const createResp = await request(
      daemon.port,
      daemon.password,
      'POST',
      '/session',
      { title: 'harness', model: { id: 'm', providerID: 'fake' } },
      { 'x-faktor-directory': REPO_ROOT },
    );
    assertSubsetKeys(createResp, ['sessionID', 'title', 'createdMs'], 'session create response');
    if (!createResp.sessionID || typeof createResp.sessionID !== 'string') {
      throw new Error(`session create response missing sessionID: ${JSON.stringify(createResp)}`);
    }
    if (createResp.title !== 'harness') {
      throw new Error(`session title drift: ${JSON.stringify(createResp.title)}`);
    }
    const sessionID = createResp.sessionID;
    pass('3. POST /session', `sessionID=${sessionID}`);

    // ---- step 4: session list contains it ---------------------------------
    checkBudget('4');
    const listResp = await request(daemon.port, daemon.password, 'GET', '/session');
    assertSubsetKeys(listResp, ['sessions'], 'session list response');
    if (!Array.isArray(listResp.sessions)) {
      throw new Error(`session list malformed: ${JSON.stringify(listResp)}`);
    }
    for (const s of listResp.sessions) {
      assertSubsetKeys(
        s,
        ['sessionID', 'title', 'state', 'createdMs', 'updatedMs'],
        'session summary',
      );
    }
    if (!listResp.sessions.some((s) => s.sessionID === sessionID)) {
      throw new Error(`session list does not contain ${sessionID}`);
    }
    pass('4. GET /session list contains it', `${listResp.sessions.length} session(s)`);

    // ---- step 5: subscribe to /global/event BEFORE sending ---------------
    checkBudget('5');
    const seenEvents = [];
    const sseDone = sse(
      daemon.port,
      daemon.password,
      '/global/event?after=0',
      (frame) => {
        if (!frame.data || typeof frame.data !== 'object') {
          return false;
        }
        assertSubsetKeys(
          frame.data,
          ['directory', 'project', 'workspace', 'payload'],
          'global event envelope',
        );
        const payload = frame.data.payload;
        if (payload && typeof payload === 'object' && payload.type) {
          seenEvents.push(frame.data);
        }
        // Turn-end markers: error (failed turn) or session_turn_close.
        if (payload?.type === 'error' || payload?.type === 'session_turn_close') {
          return true;
        }
        return false;
      },
      SSE_TIMEOUT_MS,
    ).then(
      () => undefined,
      (err) => {
        if (String(err.message).startsWith('SSE')) {
          // Timeout is tolerated here: step 9 asserts on what DID arrive.
          return undefined;
        }
        throw err;
      },
    );
    pass('5. subscribed to /global/event?after=0', 'before the message send');

    // ---- step 6: send a message ------------------------------------------
    checkBudget('6');
    const sendResp = await request(daemon.port, daemon.password, 'POST', `/session/${sessionID}/message`, {
      model: { providerID: 'fake', modelID: 'm' },
      parts: [{ type: 'text', text: 'hi' }],
    });
    assertSubsetKeys(sendResp, ['messageID', 'accepted', 'queued'], 'message send response');
    if (sendResp.accepted !== true) {
      throw new Error(`message not accepted: ${JSON.stringify(sendResp)}`);
    }
    pass('6. POST message', `messageID=${sendResp.messageID}, accepted=true`);

    // ---- step 7: wait for the turn to finish (bounded) --------------------
    checkBudget('7');
    const deadline = Date.now() + Math.min(TURN_POLL_TIMEOUT_MS, remainingBudget());
    let finalState = null;
    while (Date.now() < deadline) {
      const summary = await request(daemon.port, daemon.password, 'GET', `/session/${sessionID}`);
      assertSubsetKeys(
        summary,
        ['sessionID', 'title', 'state', 'createdMs', 'updatedMs'],
        'session summary',
      );
      finalState = summary.state;
      // The daemon has no providers: the turn fails at lookup and lands on
      // failed_recoverable (promptable). ready_for_next_turn / completed are
      // accepted for provider-having daemons.
      if (
        finalState === 'ready_for_next_turn' ||
        finalState === 'completed' ||
        finalState === 'failed_recoverable'
      ) {
        break;
      }
      await new Promise((r) => setTimeout(r, TURN_POLL_INTERVAL_MS));
    }
    if (!finalState) {
      throw new Error(`turn never started (no state observed)`);
    }
    if (
      finalState !== 'ready_for_next_turn' &&
      finalState !== 'completed' &&
      finalState !== 'failed_recoverable'
    ) {
      throw new Error(
        `turn did not finish in ${TURN_POLL_TIMEOUT_MS}ms; last state=${finalState} ` +
          `(expected ready_for_next_turn|completed|failed_recoverable)`,
      );
    }
    pass('7. turn finished', `state=${finalState}`);

    // ---- step 8: messages page, wire-valid parts --------------------------
    checkBudget('8');
    const page = await request(
      daemon.port,
      daemon.password,
      'GET',
      `/session/${sessionID}/message?limit=5`,
    );
    assertSubsetKeys(page, ['sessionID', 'messages', 'hasMore'], 'messages page');
    if (page.sessionID !== sessionID) {
      throw new Error(`page sessionID drift: ${page.sessionID} != ${sessionID}`);
    }
    if (!Array.isArray(page.messages) || page.messages.length === 0) {
      throw new Error(`no messages in page: ${JSON.stringify(page)}`);
    }
    let sawUserText = false;
    for (const msg of page.messages) {
      assertSubsetKeys(
        msg,
        ['sessionID', 'messageID', 'role', 'parts', 'createdMs', 'providerID', 'modelID'],
        'wire message',
      );
      if (msg.role === 'user') {
        sawUserText = true;
      }
      if (!Array.isArray(msg.parts)) {
        throw new Error(`message ${msg.messageID} has no parts array`);
      }
      for (const part of msg.parts) {
        if (typeof part.type !== 'string' || !FROZEN_PART_TYPES.has(part.type)) {
          throw new Error(
            `part type ${JSON.stringify(part.type)} is not wire-valid ` +
              `(frozen: ${[...FROZEN_PART_TYPES].join(', ')})`,
          );
        }
      }
    }
    if (!sawUserText) {
      throw new Error('user message missing from the page');
    }
    pass('8. messages page wire-valid', `${page.messages.length} message(s), parts validated`);

    // ---- step 9: SSE delivered ≥1 frozen-type GlobalEvent -----------------
    checkBudget('9');
    await Promise.race([
      sseDone,
      new Promise((r) => setTimeout(r, Math.min(2_000, remainingBudget()))),
    ]);
    if (seenEvents.length === 0) {
      throw new Error('SSE stream delivered no GlobalEvent with a payload type');
    }
    for (const ev of seenEvents) {
      if (!FROZEN_EVENT_TYPES.has(ev.payload.type)) {
        throw new Error(`SSE payload type ${ev.payload.type} not in the frozen set`);
      }
    }
    pass('9. SSE delivered frozen-type GlobalEvent', `types=${[...new Set(seenEvents.map((e) => e.payload.type))].join(', ')}`);

    // ---- step 10: abort ----------------------------------------------------
    checkBudget('10');
    const abortResp = await request(daemon.port, daemon.password, 'POST', `/session/${sessionID}/abort`, {});
    assertSubsetKeys(abortResp, ['aborted'], 'abort response');
    if (!Array.isArray(abortResp.aborted)) {
      throw new Error(`abort response malformed: ${JSON.stringify(abortResp)}`);
    }
    pass('10. abort', `aborted=${JSON.stringify(abortResp.aborted)}`);

    // ---- step 11: same session still usable --------------------------------
    checkBudget('11');
    const again = await request(
      daemon.port,
      daemon.password,
      'POST',
      `/session/${sessionID}/message`,
      {
        model: { providerID: 'fake', modelID: 'm' },
        parts: [{ type: 'text', text: 'still here' }],
      },
    );
    assertSubsetKeys(again, ['messageID', 'accepted', 'queued'], 'second message send response');
    if (again.accepted !== true) {
      throw new Error(`second message not accepted: ${JSON.stringify(again)}`);
    }
    const summary2 = await request(daemon.port, daemon.password, 'GET', `/session/${sessionID}`);
    if (summary2.sessionID !== sessionID) {
      throw new Error(`session summary drift after second message`);
    }
    pass('11. session usable after failure', `messageID=${again.messageID}, accepted=true`);
  } catch (err) {
    fail('flow', err.stack ?? String(err));
  } finally {
    if (daemon) {
      killDaemon(daemon);
    }
  }

  if (failures > 0) {
    console.error(`FAIL: ${failures} check(s) failed`);
    process.exit(1);
  }
  console.log('ALL 11 STEPS PASS');
  process.exit(0);
}

main().catch((err) => {
  console.error(`FATAL: ${err.stack ?? err}`);
  process.exit(1);
});
