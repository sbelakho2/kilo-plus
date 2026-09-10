// Bridge tests over mock messages: strict inbound acceptance, loud drops for
// unknown/hostile kinds, bounded payloads, native -> frozen-UI translation,
// and the vendored HTML shell invariants. No test framework: the cases are
// exported so scripts/selftest.mjs can run them under its own harness, and
// the standalone runner below uses a minimal local harness.

import {
  BRIDGE_LIMITS,
  bridgeCommandToHostMessage,
  buildVendoredWebviewHtml,
  connectionStateMessage,
  ingestWebviewMessage,
  messageFromEntry,
  messagesLoadedMessage,
  nativeEventToWebviewMessages,
  readyMessage,
  sessionToUpstream,
  snapshotToWebviewMessages,
  vendoredCsp,
} from '../src/kilo-bridge.ts';
import { pathToFileURL } from 'node:url';

function isDrop(result) {
  return result.dropped === true;
}

function assertDrop(result, needle, label) {
  if (!isDrop(result)) {
    throw new Error(`${label}: expected a drop, got ${JSON.stringify(result)}`);
  }
  if (!result.reason.includes(needle)) {
    throw new Error(`${label}: reason ${JSON.stringify(result.reason)} does not mention ${JSON.stringify(needle)}`);
  }
}

function makeEntry(overrides = {}) {
  return {
    id: '1',
    role: 'assistant',
    seq: 1,
    createdMs: 1000,
    text: '',
    reasoning: '',
    summary: '',
    tools: [],
    ...overrides,
  };
}

function makeSnapshot(overrides = {}) {
  return {
    daemon: 'running',
    daemonDetail: '9.9.9 on port 1234',
    baseUrl: 'http://127.0.0.1:1234',
    session: { id: '7', title: 'selftest', provider: 'p', model: 'm', state: 'idle' },
    machineState: 'idle',
    machineLabel: 'Idle',
    sessions: [{ id: '7', title: 'selftest', provider: 'p', model: 'm', state: 'idle' }],
    runs: [],
    activeRunId: null,
    agents: [],
    task: null,
    verification: null,
    usage: null,
    transcript: [],
    streamStatus: 'open',
    lastError: null,
    busy: false,
    ...overrides,
  };
}

const ctx = { extensionVersion: '0.1.0', workspaceDirectory: '/w', daemonVersion: '9.9.9', port: 1234 };

export const bridgeTests = [
  {
    label: 'bridge accepts the supported inbound command set',
    fn: () => {
      const ready = ingestWebviewMessage({ type: 'webviewReady' });
      if (ready.kind !== 'ready') throw new Error('webviewReady not accepted');
      const send = ingestWebviewMessage({ type: 'sendMessage', text: 'ship it', sessionID: '7' });
      if (send.kind !== 'sendMessage' || send.text !== 'ship it' || send.sessionId !== '7') {
        throw new Error(`sendMessage mapped wrong: ${JSON.stringify(send)}`);
      }
      const abort = ingestWebviewMessage({ type: 'abort', sessionID: '7' });
      if (abort.kind !== 'abort' || abort.sessionId !== '7') throw new Error('abort mapped wrong');
      if (ingestWebviewMessage({ type: 'createSession' }).kind !== 'createSession') throw new Error('createSession');
      if (ingestWebviewMessage({ type: 'loadSessions' }).kind !== 'loadSessions') throw new Error('loadSessions');
      const load = ingestWebviewMessage({
        type: 'loadMessages',
        sessionID: '7',
        before: '99',
        limit: 25,
        mode: 'prepend',
      });
      if (load.kind !== 'loadMessages' || load.limit !== 25 || load.before !== '99' || load.mode !== 'prepend') {
        throw new Error(`loadMessages mapped wrong: ${JSON.stringify(load)}`);
      }
      const external = ingestWebviewMessage({ type: 'openExternal', url: 'https://example.com/a?b=1' });
      if (external.kind !== 'openExternal' || external.url !== 'https://example.com/a?b=1') {
        throw new Error(`openExternal mapped wrong: ${JSON.stringify(external)}`);
      }
      const noLimit = ingestWebviewMessage({ type: 'loadMessages', sessionID: '7' });
      if (noLimit.limit !== BRIDGE_LIMITS.defaultMessagesLimit || noLimit.mode !== 'replace') {
        throw new Error('loadMessages defaults wrong');
      }
    },
  },
  {
    label: 'bridge drops unknown and malformed message kinds loudly',
    fn: () => {
      assertDrop(ingestWebviewMessage({ type: 'login' }), 'unsupported message kind "login"', 'unknown kind');
      assertDrop(ingestWebviewMessage({ type: '' }), 'non-empty string', 'empty type');
      assertDrop(ingestWebviewMessage({ type: 'x'.repeat(80) }), 'non-empty string', 'oversized type');
      assertDrop(ingestWebviewMessage({}), 'non-empty string', 'missing type');
      assertDrop(ingestWebviewMessage(null), 'JSON object', 'null');
      assertDrop(ingestWebviewMessage([{ type: 'webviewReady' }]), 'JSON object', 'array');
      assertDrop(ingestWebviewMessage('webviewReady'), 'JSON object', 'string');
      assertDrop(ingestWebviewMessage({ type: 'sendMessage', text: '   ' }), 'non-empty string', 'blank text');
      assertDrop(ingestWebviewMessage({ type: 'sendMessage', text: 42 }), 'non-empty string', 'non-string text');
      assertDrop(ingestWebviewMessage({ type: 'abort' }), 'abort.sessionID', 'missing abort session');
      assertDrop(
        ingestWebviewMessage({ type: 'sendMessage', text: 'hi', files: [{ mime: 'text/plain', url: 'data:,' }] }),
        'file attachments',
        'attachments refused',
      );
    },
  },
  {
    label: 'bridge bounds inbound payloads and identifiers',
    fn: () => {
      const bigText = ingestWebviewMessage({ type: 'sendMessage', text: 'x'.repeat(BRIDGE_LIMITS.maxTextChars + 1) });
      assertDrop(bigText, 'character bound', 'text bound');
      const envelope = ingestWebviewMessage({ type: 'sendMessage', text: 'x'.repeat(300 * 1024), pad: 'y'.repeat(300 * 1024) });
      assertDrop(envelope, 'byte bound', 'envelope bound');
      assertDrop(
        ingestWebviewMessage({ type: 'abort', sessionID: 's'.repeat(BRIDGE_LIMITS.maxStringChars + 1) }),
        'abort.sessionID',
        'identifier bound',
      );
      assertDrop(ingestWebviewMessage({ type: 'loadMessages', sessionID: '7', limit: 0 }), '1..', 'limit zero');
      assertDrop(ingestWebviewMessage({ type: 'loadMessages', sessionID: '7', limit: 1.5 }), 'integer', 'fractional limit');
      assertDrop(
        ingestWebviewMessage({ type: 'loadMessages', sessionID: '7', limit: BRIDGE_LIMITS.maxMessagesLimit + 1 }),
        '1..',
        'limit too large',
      );
      assertDrop(ingestWebviewMessage({ type: 'loadMessages', sessionID: '7', mode: 'focus-typo' }), 'not a known mode', 'mode');
      assertDrop(ingestWebviewMessage({ type: 'openExternal', url: 'http://example.com' }), 'non-https', 'http refused');
      assertDrop(ingestWebviewMessage({ type: 'openExternal', url: 'javascript:alert(1)' }), 'non-https', 'javascript refused');
      assertDrop(ingestWebviewMessage({ type: 'openExternal', url: 'not a url' }), 'not a valid URL', 'garbage url');
    },
  },
  {
    label: 'bridge maps commands onto the extension-host vocabulary',
    fn: () => {
      const cases = [
        [{ kind: 'ready' }, { type: 'ready' }],
        [{ kind: 'sendMessage', text: 'go', sessionId: null }, { type: 'sendGoal', goal: 'go' }],
        [{ kind: 'abort', sessionId: '7' }, { type: 'cancelRun' }],
        [{ kind: 'createSession' }, { type: 'refresh' }],
        [{ kind: 'loadSessions' }, { type: 'refresh' }],
        [{ kind: 'loadMessages', sessionId: '7', before: null, limit: 10, mode: 'replace' }, { type: 'refresh' }],
      ];
      for (const [command, expected] of cases) {
        const mapped = bridgeCommandToHostMessage(command);
        if (JSON.stringify(mapped) !== JSON.stringify(expected)) {
          throw new Error(`mapping ${command.kind} -> ${JSON.stringify(mapped)} != ${JSON.stringify(expected)}`);
        }
      }
      if (bridgeCommandToHostMessage({ kind: 'openExternal', url: 'https://example.com' }) !== null) {
        throw new Error('openExternal must be handled by the webview layer, not the host');
      }
    },
  },
  {
    label: 'bridge translates native snapshots into frozen-UI messages',
    fn: () => {
      const entry = makeEntry({
        id: '2',
        text: 'hello',
        reasoning: 'because',
        tools: [
          {
            toolCallId: 'c1',
            name: 'bash',
            state: 'completed',
            input: { cmd: 'ls' },
            excerpt: 'ok',
            exitCode: 0,
            artifact: null,
          },
        ],
      });
      const task = {
        goal: 'ship it',
        state: 'running',
        completed: ['a'],
        open: ['b'],
        testsRun: [],
        testsFailed: [],
        changedFiles: [],
        budget: null,
      };
      const messages = snapshotToWebviewMessages(makeSnapshot({ transcript: [entry], task }));
      const types = messages.map((message) => message.type);
      if (types.join(',') !== 'connectionState,sessionsLoaded,sessionStatus,messagesLoaded,todoUpdated') {
        throw new Error(`unexpected message order: ${types.join(',')}`);
      }
      const loaded = messages[3];
      if (loaded.sessionID !== '7' || loaded.messages.length !== 1) {
        throw new Error('messagesLoaded payload wrong');
      }
      const mapped = loaded.messages[0];
      if (mapped.role !== 'assistant' || mapped.content !== 'hello' || mapped.parts.length !== 3) {
        throw new Error(`message mapping wrong: ${JSON.stringify(mapped)}`);
      }
      if (mapped.parts[2].state.status !== 'completed' || mapped.parts[2].state.output !== 'ok') {
        throw new Error('tool state mapping wrong');
      }
      const todo = messages[4];
      if (todo.items[0].status !== 'completed' || todo.items[1].status !== 'pending') {
        throw new Error('todo mapping wrong');
      }
      const errored = snapshotToWebviewMessages(makeSnapshot({ daemon: "error", daemonDetail: "boom", lastError: "bad" }));
      if (errored[0].state !== 'error' || errored[0].error !== 'boom') {
        throw new Error('daemon error mapping wrong');
      }
    },
  },
  {
    label: 'bridge bounds outbound pages by entries and bytes',
    fn: () => {
      const many = [];
      for (let i = 0; i < BRIDGE_LIMITS.maxPageEntries + 50; i += 1) {
        many.push(makeEntry({ id: String(i), seq: i, createdMs: i, text: `m${i}` }));
      }
      const page = messagesLoadedMessage('7', many);
      if (page.messages.length !== BRIDGE_LIMITS.maxPageEntries || page.hasMore !== true) {
        throw new Error(`page not bounded: ${page.messages.length} hasMore=${page.hasMore}`);
      }
      if (page.messages[page.messages.length - 1].id !== String(many.length - 1)) {
        throw new Error('newest entry must survive the page bound');
      }

      const fat = [];
      for (let i = 0; i < 12; i += 1) {
        fat.push(makeEntry({ id: `fat${i}`, seq: i, createdMs: i, text: 'y'.repeat(100_000) }));
      }
      const bytes = JSON.stringify(fat.map((entry) => messageFromEntry(entry, '7'))).length;
      if (bytes <= BRIDGE_LIMITS.maxOutboundBytes) {
        throw new Error('test setup: fat page must exceed the byte bound');
      }
      const bounded = messagesLoadedMessage('7', fat);
      if (bounded.messages.length >= fat.length || bounded.hasMore !== true) {
        throw new Error(`byte bound not enforced: ${bounded.messages.length} hasMore=${bounded.hasMore}`);
      }
    },
  },
  {
    label: 'bridge maps SSE frames and ignores unmappable events',
    fn: () => {
      const created = nativeEventToWebviewMessages(
        'message_created',
        {
          message: {
            id: '9',
            role: 'assistant',
            parts: [
              { type: 'text', text: 'hi' },
              { type: 'tool_call', tool_call_id: 'c1', name: 'bash', input: {}, state: 'running' },
            ],
          },
        },
        '7',
      );
      if (created.length !== 1 || created[0].type !== 'messageCreated' || created[0].message.parts.length !== 2) {
        throw new Error(`messageCreated mapping wrong: ${JSON.stringify(created)}`);
      }
      const updated = nativeEventToWebviewMessages(
        'message_part_updated',
        { message_id: '9', part: { type: 'text', text: ' there' } },
        '7',
      );
      if (
        updated.length !== 1 ||
        updated[0].type !== 'partUpdated' ||
        updated[0].delta.textDelta !== ' there'
      ) {
        throw new Error(`partUpdated mapping wrong: ${JSON.stringify(updated)}`);
      }
      if (nativeEventToWebviewMessages('unrelated_event', {}, '7').length !== 0) {
        throw new Error('unmappable event must produce no messages');
      }
      if (nativeEventToWebviewMessages('message_created', { message: null }, '7').length !== 0) {
        throw new Error('malformed frame must produce no messages');
      }
    },
  },
  {
    label: 'bridge maps connection and session states deterministically',
    fn: () => {
      if (connectionStateMessage('running').state !== 'connected') throw new Error('running');
      if (connectionStateMessage('starting').state !== 'connecting') throw new Error('starting');
      if (connectionStateMessage('stopped').state !== 'disconnected') throw new Error('stopped');
      const error = connectionStateMessage('error', 'boom');
      if (error.state !== 'error' || error.error !== 'boom') throw new Error('error state');
      const session = sessionToUpstream({ id: '7', title: 't', provider: 'p', model: 'm', state: 'idle' });
      if (session.createdAt !== new Date(0).toISOString() || session.updatedAt !== session.createdAt) {
        throw new Error('session timestamps must be the deterministic epoch, not fabricated now');
      }
      const ready = readyMessage(ctx);
      if (ready.type !== 'ready' || ready.extensionVersion !== '0.1.0' || ready.serverInfo.port !== 1234) {
        throw new Error('ready mapping wrong');
      }
      const noServer = readyMessage({ extensionVersion: '0.0.0', workspaceDirectory: '' });
      if (noServer.serverInfo !== undefined) throw new Error('ready must omit serverInfo without a port');
    },
  },
  {
    label: 'hostile message properties cannot smuggle values through the bridge',
    fn: () => {
      const proto = Object.create({ type: 'webviewReady' });
      assertDrop(ingestWebviewMessage(proto), 'non-empty string', 'inherited type must not count');
      const command = ingestWebviewMessage({ type: 'sendMessage', text: 'ok', sessionID: ' 7 ' });
      if (command.kind !== 'sendMessage' || command.sessionId !== '7') {
        throw new Error('identifiers must be trimmed');
      }
      const weird = ingestWebviewMessage({ type: 'openExternal', url: new URL('https://example.com') });
      assertDrop(weird, 'bounded string', 'non-string url');
    },
  },
  {
    label: 'vendored HTML shell carries a strict nonce-only policy',
    fn: () => {
      const nonce = 'deadbeef';
      const html = buildVendoredWebviewHtml({
        cspSource: 'vscode-webview://abc',
        nonce,
        scriptUri: 'vscode-webview://abc/dist/webview.js',
        styleUri: 'vscode-webview://abc/dist/webview.css',
        iconsBaseUri: 'vscode-webview://abc/assets/icons',
        workerUri: 'vscode-webview://abc/dist/shiki-worker.js',
        title: '<Faktor & Co>',
        sidebar: '',
        topBar: false,
      });
      if (!html.includes(`id="root"`)) throw new Error('missing root element');
      if (!html.includes(`script-src 'nonce-${nonce}'`)) throw new Error('missing nonce script-src');
      const scriptSrc = html.match(/script-src[^;]*/)?.[0] ?? '';
      if (scriptSrc.includes('unsafe-inline')) throw new Error('script-src must be nonce-only');
      if ((html.match(/nonce="deadbeef"/g) ?? []).length !== 2) throw new Error('both inline scripts need the nonce');
      if (/src="https?:/.test(html) || /href="https?:/.test(html)) throw new Error('remote resources are forbidden');
      if (!html.includes('markdown-shiki-worker.js')) throw new Error('markdown worker URI not derived');
      if (!html.includes('&lt;Faktor &amp; Co&gt;')) throw new Error('title not HTML-escaped');
      if (!vendoredCsp('vscode-webview://abc', nonce).startsWith("default-src 'none'")) {
        throw new Error('CSP must default-deny');
      }
    },
  },
];

async function standalone() {
  let failed = 0;
  for (const { label, fn } of bridgeTests) {
    try {
      await fn();
      console.log(`PASS  ${label}`);
    } catch (error) {
      failed += 1;
      console.error(`FAIL  ${label} — ${error && error.message ? error.message : error}`);
    }
  }
  console.log(`\n${bridgeTests.length - failed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  standalone();
}
