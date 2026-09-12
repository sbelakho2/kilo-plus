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
  faktorEvidenceExpandedMessage,
  ingestWebviewMessage,
  locateVendoredBundle,
  mapKiloFiles,
  messageFromEntry,
  messagesLoadedMessage,
  nativeEventToWebviewMessages,
  readyMessage,
  sessionToUpstream,
  snapshotToWebviewMessages,
  vendoredCsp,
  vendoredFallbackNotice,
} from '../src/kilo-bridge.ts';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
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
    cockpit: null,
    cockpitSections: [],
    tournament: null,
    board: null,
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
      // Tournament decide/abort ride the same bridge (Faktor companion
      // vocabulary): decide takes no operator input, abort may carry a
      // bounded reason, and both map onto the host tournament control.
      const decide = ingestWebviewMessage({
        type: 'faktorTournamentAction',
        tournamentId: 't-1',
        action: 'decide',
      });
      if (decide.kind !== 'tournamentDecide' || decide.tournamentId !== 't-1') {
        throw new Error(`tournament decide mapped wrong: ${JSON.stringify(decide)}`);
      }
      const decideHost = bridgeCommandToHostMessage(decide);
      if (
        decideHost === null ||
        decideHost.type !== 'tournamentControl' ||
        decideHost.action !== 'decide' ||
        decideHost.tournamentId !== 't-1'
      ) {
        throw new Error(`tournament decide host mapping wrong: ${JSON.stringify(decideHost)}`);
      }
      const abortTournament = ingestWebviewMessage({
        type: 'faktorTournamentAction',
        tournamentId: 't-1',
        action: 'abort',
        reason: 'operator abort',
      });
      if (abortTournament.kind !== 'tournamentAbort' || abortTournament.reason !== 'operator abort') {
        throw new Error(`tournament abort mapped wrong: ${JSON.stringify(abortTournament)}`);
      }
      const abortHost = bridgeCommandToHostMessage(abortTournament);
      if (
        abortHost === null ||
        abortHost.type !== 'tournamentControl' ||
        abortHost.action !== 'abort' ||
        abortHost.reason !== 'operator abort'
      ) {
        throw new Error(`tournament abort host mapping wrong: ${JSON.stringify(abortHost)}`);
      }
      const bareAbort = ingestWebviewMessage({
        type: 'faktorTournamentAction',
        tournamentId: 't-1',
        action: 'abort',
      });
      if (bareAbort.kind !== 'tournamentAbort' || bareAbort.reason !== null) {
        throw new Error('an absent abort reason must stay null');
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
      // Malformed attachment payloads are dropped; well-formed ones map to
      // bounded references (never inline bytes) on the command.
      assertDrop(
        ingestWebviewMessage({ type: 'sendMessage', text: 'hi', files: 'nope' }),
        'files must be an array',
        'non-array files',
      );
      const attached = ingestWebviewMessage({
        type: 'sendMessage',
        text: 'hi',
        files: [{ mime: 'text/plain', url: 'data:,' }],
      });
      if (attached.kind !== 'sendMessage' || attached.attachments.length !== 1) {
        throw new Error(`well-formed attachment must map to a bounded ref: ${JSON.stringify(attached)}`);
      }
      if (!attached.attachments[0].ref.startsWith('faktor-attachment:sha256:')) {
        throw new Error(`attachment ref must be content-addressed: ${attached.attachments[0].ref}`);
      }
      assertDrop(
        ingestWebviewMessage({ type: 'faktorTournamentAction', action: 'decide' }),
        'tournamentId',
        'tournament without id',
      );
      assertDrop(
        ingestWebviewMessage({
          type: 'faktorTournamentAction',
          tournamentId: 't-1',
          action: 'delete',
        }),
        'must be "decide" or "abort"',
        'unknown tournament action',
      );
      assertDrop(
        ingestWebviewMessage({
          type: 'faktorTournamentAction',
          tournamentId: 't-1',
          action: 'decide',
          reason: 'not allowed',
        }),
        'decide takes no reason',
        'decide with reason',
      );
      assertDrop(
        ingestWebviewMessage({
          type: 'faktorTournamentAction',
          tournamentId: 't-1',
          action: 'abort',
          reason: 'x'.repeat(513),
        }),
        'reason exceeds',
        'oversized abort reason',
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
        [
          {
            kind: 'sendMessage',
            text: 'go',
            sessionId: null,
            files: [],
            attachments: [],
            refusedAttachments: [],
          },
          { type: 'sendGoal', goal: 'go' },
        ],
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
        acceptanceCriteria: [],
        plan: [],
        blockers: [],
        evidenceRefs: [],
        phase: null,
        progress: null,
      };
      const messages = snapshotToWebviewMessages(makeSnapshot({ transcript: [entry], task }));
      const types = messages.map((message) => message.type);
      // The frozen Kilo prefix is byte-identical and order-stable; the
      // additive Faktor panel tail is appended after it.
      const frozen = 'connectionState,sessionsLoaded,sessionStatus,messagesLoaded,todoUpdated';
      if (types.slice(0, 5).join(',') !== frozen) {
        throw new Error(`frozen message order changed: ${types.join(',')}`);
      }
      const additive =
        'faktorTaskState,faktorAgents,faktorCockpit,faktorTournament,faktorEvidence,faktorBoardState';
      if (types.slice(5).join(',') !== additive) {
        throw new Error(`additive Faktor tail wrong: ${types.join(',')}`);
      }
      const taskState = messages[5];
      if (taskState.present !== true || taskState.goal !== 'ship it') {
        throw new Error(`faktorTaskState payload wrong: ${JSON.stringify(taskState)}`);
      }
      if (messages[8].present !== false || messages[8].tournament !== null) {
        throw new Error('an absent tournament must be an explicit empty frame');
      }
      if (messages[10].available !== false || !String(messages[10].reason).includes('board')) {
        throw new Error('an absent board must be an explicit unavailable frame');
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
    label: 'bridge accepts Faktor panel actions and maps them to the host vocabulary',
    fn: () => {
      const retry = ingestWebviewMessage({ type: 'faktorAgentAction', agentId: 'c1', action: 'retry' });
      if (retry.kind !== 'faktorAgentAction' || retry.action !== 'retry' || retry.state !== null) {
        throw new Error(`agent retry mapped wrong: ${JSON.stringify(retry)}`);
      }
      const retryHost = bridgeCommandToHostMessage(retry);
      if (JSON.stringify(retryHost) !== JSON.stringify({ type: 'agentControl', agentId: 'c1', action: 'retry' })) {
        throw new Error(`agent retry host mapping wrong: ${JSON.stringify(retryHost)}`);
      }
      const presentation = ingestWebviewMessage({
        type: 'faktorAgentAction',
        agentId: 'c1',
        action: 'presentation',
        state: 'background',
      });
      if (presentation.kind !== 'faktorAgentAction' || presentation.state !== 'background') {
        throw new Error(`presentation mapped wrong: ${JSON.stringify(presentation)}`);
      }
      const steer = ingestWebviewMessage({ type: 'faktorAgentAction', agentId: 'c1', action: 'steer', text: 'focus' });
      if (steer.kind !== 'faktorAgentAction') throw new Error('steer not accepted');
      if (ingestWebviewMessage({ type: 'faktorAgentAction', agentId: 'c1', action: 'model' }).kind !== 'faktorAgentAction') {
        throw new Error('model action requires no inline value (the host prompts)');
      }
      if (
        ingestWebviewMessage({ type: 'faktorAgentAction', agentId: 'c1', action: 'budget', maxTokens: 1000 }).kind !==
        'faktorAgentAction'
      ) {
        throw new Error('budget action not accepted');
      }

      const evidence = ingestWebviewMessage({ type: 'faktorEvidenceExpand', evidenceId: 41 });
      if (evidence.kind !== 'faktorEvidenceExpand' || evidence.evidenceId !== 41) {
        throw new Error(`evidence expand mapped wrong: ${JSON.stringify(evidence)}`);
      }
      const evidenceHost = bridgeCommandToHostMessage(evidence);
      if (JSON.stringify(evidenceHost) !== JSON.stringify({ type: 'retrieveEvidence', evidenceId: 41 })) {
        throw new Error(`evidence host mapping wrong: ${JSON.stringify(evidenceHost)}`);
      }
      const expanded = faktorEvidenceExpandedMessage('7', 41, 'hello', false);
      if (expanded.mode !== 'expanded' || expanded.evidence.id !== 41 || expanded.evidence.text !== 'hello') {
        throw new Error(`expanded evidence frame wrong: ${JSON.stringify(expanded)}`);
      }

      const read = ingestWebviewMessage({ type: 'faktorBoardAction', action: 'read', since: 3, limit: 10 });
      if (read.kind !== 'faktorBoardAction' || read.limit !== 10 || read.since !== 3) {
        throw new Error(`board read mapped wrong: ${JSON.stringify(read)}`);
      }
      if (JSON.stringify(bridgeCommandToHostMessage(read)) !== JSON.stringify({ type: 'boardRead', since: 3, limit: 10 })) {
        throw new Error('board read host mapping wrong');
      }
      const post = ingestWebviewMessage({
        type: 'faktorBoardAction',
        action: 'post',
        subject: 's',
        body: 'b',
        refs: ['evidence:41'],
      });
      if (post.kind !== 'faktorBoardAction' || post.subject !== 's') {
        throw new Error(`board post mapped wrong: ${JSON.stringify(post)}`);
      }
      if (
        JSON.stringify(bridgeCommandToHostMessage(post)) !==
        JSON.stringify({ type: 'boardPost', subject: 's', body: 'b', refs: ['evidence:41'] })
      ) {
        throw new Error('board post host mapping wrong');
      }
    },
  },
  {
    label: 'bridge drops hostile Faktor panel actions with reasons',
    fn: () => {
      assertDrop(ingestWebviewMessage({ type: 'faktorAgentAction', agentId: 'c1' }), 'action must be one', 'missing action');
      assertDrop(ingestWebviewMessage({ type: 'faktorAgentAction', action: 'retry' }), 'agentId', 'missing agent id');
      assertDrop(
        ingestWebviewMessage({ type: 'faktorAgentAction', agentId: 'c1', action: 'presentation', state: 'hidden' }),
        'foreground',
        'bad presentation state',
      );
      assertDrop(
        ingestWebviewMessage({ type: 'faktorAgentAction', agentId: 'c1', action: 'steer' }),
        'non-empty string for steer',
        'steer without text',
      );
      assertDrop(
        ingestWebviewMessage({
          type: 'faktorAgentAction',
          agentId: 'c1',
          action: 'steer',
          text: 'x'.repeat(BRIDGE_LIMITS.maxSteerChars + 1),
        }),
        'character bound',
        'oversized steer',
      );
      assertDrop(
        ingestWebviewMessage({ type: 'faktorAgentAction', agentId: 'c1', action: 'retry', state: 'background' }),
        'not valid for action',
        'state smuggled onto a non-presentation action',
      );
      assertDrop(
        ingestWebviewMessage({ type: 'faktorAgentAction', agentId: 'c1', action: 'budget', maxTokens: 0 }),
        'positive integer',
        'zero budget',
      );
      assertDrop(ingestWebviewMessage({ type: 'faktorEvidenceExpand' }), 'positive integer', 'missing evidence id');
      assertDrop(ingestWebviewMessage({ type: 'faktorEvidenceExpand', evidenceId: 0 }), 'positive integer', 'zero evidence id');
      assertDrop(
        ingestWebviewMessage({ type: 'faktorEvidenceExpand', evidenceId: 1.5 }),
        'positive integer',
        'fractional evidence id',
      );
      assertDrop(ingestWebviewMessage({ type: 'faktorBoardAction', action: 'delete' }), 'read', 'unknown board action');
      assertDrop(
        ingestWebviewMessage({ type: 'faktorBoardAction', action: 'read', limit: 0 }),
        '1..',
        'zero board limit',
      );
      assertDrop(
        ingestWebviewMessage({ type: 'faktorBoardAction', action: 'read', since: -1 }),
        'non-negative',
        'negative board cursor',
      );
      assertDrop(ingestWebviewMessage({ type: 'faktorBoardAction', action: 'post' }), 'subject', 'post without subject');
      assertDrop(
        ingestWebviewMessage({ type: 'faktorBoardAction', action: 'post', subject: 's', refs: 'nope' }),
        'array of strings',
        'non-array refs',
      );
      const tooManyRefs = [];
      for (let i = 0; i < BRIDGE_LIMITS.maxBoardRefs + 1; i += 1) tooManyRefs.push(`ref-${i}`);
      assertDrop(
        ingestWebviewMessage({ type: 'faktorBoardAction', action: 'post', subject: 's', refs: tooManyRefs }),
        'exceeds',
        'ref count bound',
      );
      assertDrop(
        ingestWebviewMessage({
          type: 'faktorBoardAction',
          action: 'post',
          subject: 's',
          refs: ['r'.repeat(BRIDGE_LIMITS.maxBoardRefBytes + 1)],
        }),
        'bytes',
        'oversized ref',
      );
    },
  },
  {
    label: 'bridge maps Kilo file attachments to bounded paths and binary references',
    fn: () => {
      const png = `data:image/png;base64,${Buffer.from([137, 80, 78, 71]).toString('base64')}`;
      const mapping = mapKiloFiles(
        [
          { mime: 'text/plain', url: 'file:///w/src/a.ts' },
          { url: 'src/b.ts' },
          { url: 'file:///w/src/c.ts', source: { path: '/w/src/c.ts' } },
          { url: png, mime: 'image/png', filename: 'shot.png' },
          null,
          { url: 'https://evil.example/x' },
          { url: '/etc/passwd' },
          { url: '../../outside' },
          { url: `src/${'x'.repeat(4096)}.ts` },
          { url: 'src/ctrl\u0001.ts' },
        ],
        '/w',
      );
      if (JSON.stringify(mapping.files) !== JSON.stringify(['src/a.ts', 'src/b.ts', 'src/c.ts'])) {
        throw new Error(`path mapping wrong: ${JSON.stringify(mapping.files)}`);
      }
      if (
        mapping.attachments.length !== 1 ||
        !mapping.attachments[0].ref.startsWith('faktor-attachment:sha256:') ||
        mapping.attachments[0].mime !== 'image/png' ||
        mapping.attachments[0].filename !== 'shot.png' ||
        mapping.attachments[0].bytes !== 4
      ) {
        throw new Error(`binary reference wrong: ${JSON.stringify(mapping.attachments)}`);
      }
      if (mapping.refused.length !== 6) {
        throw new Error(`mixed list refusals wrong: ${JSON.stringify(mapping.refused)}`);
      }
      for (const refusal of mapping.refused) {
        if (typeof refusal.reason !== 'string' || refusal.reason.length === 0) {
          throw new Error(`a refusal must carry a reason: ${JSON.stringify(refusal)}`);
        }
      }

      // The literal required mapping: sendGoal with the bounded file paths.
      const command = ingestWebviewMessage(
        { type: 'sendMessage', text: 'goal', files: [{ url: 'file:///w/src/a.ts' }] },
        { workspaceDirectory: '/w' },
      );
      const host = bridgeCommandToHostMessage(command);
      if (JSON.stringify(host) !== JSON.stringify({ type: 'sendGoal', goal: 'goal', files: ['src/a.ts'] })) {
        throw new Error(`sendGoal mapping wrong: ${JSON.stringify(host)}`);
      }

      // A data URL only ever travels as a content-addressed reference: the
      // prompt text is untouched and no payload reaches `files`.
      const dataCommand = ingestWebviewMessage(
        { type: 'sendMessage', text: 'see image', files: [{ url: png, mime: 'image/png' }] },
        { workspaceDirectory: '/w' },
      );
      const dataHost = bridgeCommandToHostMessage(dataCommand);
      if (dataHost.files !== undefined || dataHost.attachments.length !== 1) {
        throw new Error(`data URL mapping wrong: ${JSON.stringify(dataHost)}`);
      }
      if (String(dataHost.goal).includes('base64')) {
        throw new Error('binary bytes must never enter the prompt text');
      }

      const many = [];
      for (let i = 0; i < BRIDGE_LIMITS.maxFilesPerPrompt + 3; i += 1) many.push({ url: `src/f${i}.ts` });
      const capped = mapKiloFiles(many, '/w');
      if (capped.files.length !== BRIDGE_LIMITS.maxFilesPerPrompt || capped.refused.length !== 3) {
        throw new Error(`file count bound wrong: ${capped.files.length}/${capped.refused.length}`);
      }

      const noWorkspace = mapKiloFiles([{ url: '/w/a.ts' }, { url: 'rel/a.ts' }], null);
      if (JSON.stringify(noWorkspace.files) !== JSON.stringify(['rel/a.ts'])) {
        throw new Error(`workspace-relative mapping wrong: ${JSON.stringify(noWorkspace)}`);
      }
      if (!noWorkspace.refused[0].reason.includes('workspace')) {
        throw new Error('an absolute path without a workspace root must be refused');
      }

      const badBase64 = mapKiloFiles([{ url: 'data:image/png;base64,@@@@' }], '/w');
      if (badBase64.refused.length !== 1 || !badBase64.refused[0].reason.includes('base64')) {
        throw new Error(`malformed base64 must be refused: ${JSON.stringify(badBase64)}`);
      }
    },
  },
  {
    label: 'completion contract is strict, per-start, and maps onto sendGoal',
    fn: () => {
      const valid = ingestWebviewMessage({
        type: 'sendMessage',
        text: 'goal',
        completionContract: { include_commit: true, include_push: false, include_pr: true },
      });
      if (isDrop(valid) || valid.kind !== 'sendMessage') {
        throw new Error(`a valid contract must be accepted: ${JSON.stringify(valid)}`);
      }
      if (
        JSON.stringify(valid.completionContract) !==
        JSON.stringify({ include_commit: true, include_push: false, include_pr: true })
      ) {
        throw new Error(`contract must round-trip: ${JSON.stringify(valid.completionContract)}`);
      }
      const host = bridgeCommandToHostMessage(valid);
      if (
        JSON.stringify(host) !==
        JSON.stringify({
          type: 'sendGoal',
          goal: 'goal',
          completionContract: { include_commit: true, include_push: false, include_pr: true },
        })
      ) {
        throw new Error(`completion contract mapping wrong: ${JSON.stringify(host)}`);
      }
      // The all-false default never reaches the host (byte-identical path).
      const allFalse = ingestWebviewMessage({
        type: 'sendMessage',
        text: 'goal',
        completionContract: { include_commit: false, include_push: false, include_pr: false },
      });
      if (isDrop(allFalse) || allFalse.completionContract !== null) {
        throw new Error(`all-false must normalize to null: ${JSON.stringify(allFalse)}`);
      }
      if ('completionContract' in bridgeCommandToHostMessage(allFalse)) {
        throw new Error('all-false must not add a host field');
      }
      // Hostile contracts are loud drops, never a silently contract-free task.
      for (const bad of [
        'commit',
        ['include_commit'],
        {},
        { include_commit: true },
        { include_commit: 'yes', include_push: false, include_pr: false },
        { include_commit: true, include_push: false, include_pr: false, include_release: true },
      ]) {
        assertDrop(
          ingestWebviewMessage({ type: 'sendMessage', text: 'goal', completionContract: bad }),
          'completionContract',
          `hostile contract ${JSON.stringify(bad)}`,
        );
      }
      const inherited = Object.create({
        include_commit: true,
        include_push: false,
        include_pr: false,
      });
      assertDrop(
        ingestWebviewMessage({ type: 'sendMessage', text: 'goal', completionContract: inherited }),
        'completionContract',
        'inherited-only contract members',
      );
    },
  },
  {
    label: 'vendored shell mounts the companion overlay only when staged',
    fn: () => {
      const nonce = 'companionnonce';
      const base = {
        cspSource: 'vscode-webview://x',
        nonce,
        scriptUri: 'vscode-webview://x/dist/webview.js',
        styleUri: 'vscode-webview://x/dist/webview.css',
        iconsBaseUri: 'vscode-webview://x/assets/icons',
        workerUri: 'vscode-webview://x/dist/shiki-worker.js',
        title: 'Faktor',
      };
      const plain = buildVendoredWebviewHtml(base);
      if ((plain.match(new RegExp(`nonce="${nonce}"`, 'g')) ?? []).length !== 2) {
        throw new Error('the frozen shell must keep exactly two nonce scripts');
      }
      if (plain.includes('faktor-companion')) {
        throw new Error('no companion overlay may be referenced without staged URIs');
      }
      const html = buildVendoredWebviewHtml({
        ...base,
        companionScriptUri: 'vscode-webview://x/dist/overlay/faktor-companion.js',
        companionStyleUri: 'vscode-webview://x/dist/overlay/faktor-companion.css',
      });
      if ((html.match(new RegExp(`nonce="${nonce}"`, 'g')) ?? []).length !== 3) {
        throw new Error('the companion shell must add exactly one nonce script');
      }
      if (!html.includes('faktor-companion.js') || !html.includes('faktor-companion.css')) {
        throw new Error('the companion URIs must be referenced');
      }
      if (!html.includes('__faktorVsCodeApi')) {
        throw new Error('the single acquireVsCodeApi handle must be captured for the panel');
      }
      const bundleScriptAt = html.indexOf(base.scriptUri);
      const bootstrapAt = html.indexOf('__faktorVsCodeApi');
      const companionAt = html.indexOf('faktor-companion.js');
      if (!(bootstrapAt < bundleScriptAt && bundleScriptAt < companionAt)) {
        throw new Error('boot order must be: capture handle, frozen bundle, companion panel');
      }

      const dir = mkdtempSync(join(tmpdir(), 'faktor-bridge-companion-'));
      try {
        const dist = join(dir, 'dist');
        mkdirSync(join(dist, 'overlay'), { recursive: true });
        writeFileSync(join(dist, 'webview.js'), '// entry');
        writeFileSync(join(dist, 'webview.css'), '/* entry */');
        const noOverlay = locateVendoredBundle(dir);
        if (noOverlay === null || noOverlay.companion !== null) {
          throw new Error('a plain bundle must not fabricate a companion overlay');
        }
        writeFileSync(join(dist, 'overlay', 'faktor-companion.js'), '// panel');
        const halfOverlay = locateVendoredBundle(dir);
        if (halfOverlay === null || halfOverlay.companion !== null) {
          throw new Error('half an overlay must not be served');
        }
        writeFileSync(join(dist, 'overlay', 'faktor-companion.css'), '/* panel */');
        const bundle = locateVendoredBundle(dir);
        if (
          bundle === null ||
          bundle.companion === null ||
          !bundle.companion.script.endsWith('faktor-companion.js') ||
          !bundle.companion.style.endsWith('faktor-companion.css')
        ) {
          throw new Error('a staged overlay must be discovered');
        }
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
  },
  {
    label: 'bridge bounds every additive Faktor frame',
    fn: () => {
      const long = 'x'.repeat(5000);
      const agents = [];
      for (let i = 0; i < BRIDGE_LIMITS.maxFaktorAgents + 20; i += 1) {
        agents.push({
          agentId: `c${i}`,
          kind: 'child',
          state: 'Running',
          goal: long,
          model: 'm',
          provider: 'p',
          reasoning: true,
          thinking: false,
          itemId: null,
          itemKind: null,
          itemIds: [],
          sessionId: i,
          worktreeId: null,
          ownership: 'orchestrator',
          capabilities: [],
          progress: null,
          result: null,
          budget: null,
          blockers: [],
          presentation: 'foreground',
          pixel: {
            childId: `c${i}`,
            state: 'running',
            animation: 'pixel-running',
            avatar: { hash: i, color: 'red', accent: 'pink', pixels: new Array(25).fill(1), version: 1 },
          },
        });
      }
      const sections = [];
      for (let i = 0; i < 40; i += 1) {
        sections.push({
          key: `s${i}`,
          title: long,
          present: true,
          lines: [long, long],
          evidence: [],
        });
      }
      const posts = [];
      for (let i = 0; i < BRIDGE_LIMITS.maxPageEntries + 20; i += 1) {
        posts.push({
          id: `p${i}`,
          author: 'a',
          subject: long,
          body: long,
          refs: [],
          revision: i,
          createdMs: i,
        });
      }
      const snapshot = makeSnapshot({
        agents,
        cockpitSections: sections,
        board: {
          available: true,
          source: 'transcript',
          revision: 1,
          unread: 3,
          posts,
          reason: null,
        },
        tournament: {
          id: long,
          state: 'open',
          open: true,
          canDecide: true,
          winner: null,
          criteria: [long],
          candidates: new Array(80).fill({
            childId: 'c1',
            state: 'done',
            verification: 1,
            verificationPass: true,
            reviewRank: 'clean',
            reviewer: 'r',
            costMicro: 1,
            wallMs: 1,
            winner: false,
          }),
        },
      });
      const frames = {};
      for (const message of snapshotToWebviewMessages(snapshot)) {
        if (typeof message.type === 'string' && message.type.startsWith('faktor')) {
          frames[message.type] = message;
        }
      }
      const expected = [
        'faktorTaskState',
        'faktorAgents',
        'faktorCockpit',
        'faktorTournament',
        'faktorEvidence',
        'faktorBoardState',
      ];
      for (const type of expected) {
        if (frames[type] === undefined) throw new Error(`missing additive frame ${type}`);
      }
      if (frames.faktorAgents.agents.length !== BRIDGE_LIMITS.maxFaktorAgents) {
        throw new Error(`agent frame must cap at ${BRIDGE_LIMITS.maxFaktorAgents}`);
      }
      if (frames.faktorAgents.agents[0].goal.length > 600) {
        throw new Error('agent strings must be clamped');
      }
      if (frames.faktorCockpit.sections.length > 16) {
        throw new Error('cockpit sections must be capped');
      }
      if (frames.faktorBoardState.posts.length > BRIDGE_LIMITS.maxPageEntries) {
        throw new Error('board posts must be capped');
      }
      if (frames.faktorTournament.tournament.candidates.length > 8) {
        throw new Error('tournament candidates must be capped');
      }
      if (frames.faktorTournament.tournament.id.length > 200) {
        throw new Error('tournament strings must be clamped');
      }
      for (const type of expected) {
        const bytes = Buffer.byteLength(JSON.stringify(frames[type]), 'utf8');
        if (bytes > BRIDGE_LIMITS.maxInboundBytes) {
          throw new Error(`${type} frame exceeds the inbound byte bound: ${bytes}`);
        }
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
  {
    label: 'vendored bundle discovery prefers dist/index.html and refuses remote entries',
    fn: () => {
      const dir = mkdtempSync(join(tmpdir(), 'faktor-bridge-bundle-'));
      try {
        const dist = join(dir, 'dist');
        mkdirSync(join(dist, 'assets'), { recursive: true });
        writeFileSync(
          join(dist, 'index.html'),
          '<link rel="stylesheet" href="./assets/index-abc.css"><script type="module" src="./assets/index-abc.js"></script>',
        );
        writeFileSync(join(dist, 'assets', 'index-abc.js'), '// entry');
        writeFileSync(join(dist, 'assets', 'index-abc.css'), '/* entry */');
        const bundle = locateVendoredBundle(dir);
        if (bundle === null || bundle.entry !== 'index.html') {
          throw new Error('local index.html entry was not discovered');
        }
        if (!bundle.script.endsWith('index-abc.js') || !bundle.style.endsWith('index-abc.css')) {
          throw new Error(`entry assets wrong: ${bundle.script} / ${bundle.style}`);
        }
        writeFileSync(
          join(dist, 'index.html'),
          '<link rel="stylesheet" href="./assets/index-abc.css"><script src="https://evil.example/x.js"></script>',
        );
        if (locateVendoredBundle(dir) !== null) throw new Error('remote script must refuse the bundle');
        writeFileSync(
          join(dist, 'index.html'),
          '<link rel="stylesheet" href="../outside.css"><script src="./assets/index-abc.js"></script>',
        );
        if (locateVendoredBundle(dir) !== null) throw new Error('traversal href must refuse the bundle');
        writeFileSync(
          join(dist, 'index.html'),
          '<link rel="stylesheet" href="./assets/missing.css"><script src="./assets/index-abc.js"></script>',
        );
        if (locateVendoredBundle(dir) !== null) throw new Error('missing referenced asset must refuse the bundle');
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
  },
  {
    label: 'missing vendored dist falls back with a recorded notice',
    fn: () => {
      const dir = mkdtempSync(join(tmpdir(), 'faktor-bridge-missing-'));
      try {
        if (locateVendoredBundle(dir) !== null) throw new Error('empty root must not locate a bundle');
        const notice = vendoredFallbackNotice(dir);
        if (!notice.includes('fallback') || !notice.includes(join(dir, 'dist'))) {
          throw new Error(`notice does not record the fallback path: ${notice}`);
        }
        mkdirSync(join(dir, 'dist'));
        writeFileSync(join(dir, 'dist', 'webview.js'), '// entry');
        if (locateVendoredBundle(dir) !== null) throw new Error('half a bundle must not be served');
        writeFileSync(join(dir, 'dist', 'webview.css'), '/* entry */');
        const esbuild = locateVendoredBundle(dir);
        if (esbuild === null || esbuild.entry !== 'esbuild') {
          throw new Error('esbuild entry pair was not discovered');
        }
        if (esbuild.worker !== null || esbuild.markdownWorker !== null) {
          throw new Error('absent workers must stay null');
        }
      } finally {
        rmSync(dir, { recursive: true, force: true });
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
