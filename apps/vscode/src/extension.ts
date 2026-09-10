// The Faktor VS Code product surface over the native daemon:
//
//   - daemon.ts        spawns and owns the faktor-cli process
//   - nativeClient.ts  strict typed fetch client of /native/* (+ the
//                      minimal /session/create|/session/list surface)
//   - eventStream.ts   SSE journal stream with cursor resume
//   - state.ts         the observable snapshot store + transcript reducer
//   - webview.ts       the chat panel (strict CSP, nonce, no remote code)
//
// Commands: start/stop the daemon, open the chat, start a task, cancel the
// active run. The status bar mirrors daemon + task state. Every daemon
// string that reaches the UI passes through the strict native validators
// first; nothing is rendered from an unvalidated response.

import * as vscode from 'vscode';
import { resolve } from 'node:path';
import { DaemonHandle, startDaemon, stopDaemon } from './daemon';
import {
  FetchLike,
  NativeApiError,
  NativeClient,
  NativeEvidenceSelector,
  NativeMessagePage,
  NativeSessionUsage,
  NativeTaskRun,
  NativeTaskRunStarted,
  NativeTaskView,
  NativeVerificationView,
  ResponseLike,
  StartTaskRunRequest,
} from './nativeClient';
import {
  EventStream,
  EventStreamStatus,
  SseFetchLike,
  SseResponseLike,
} from './eventStream';
import {
  FaktorStore,
  Json,
  RunSummary,
  TaskSummary,
  TranscriptEntry,
  UsageSummary,
  VerificationSummary,
  applySseEvent,
  summarizeAgents,
  transcriptFromMessages,
} from './state';
import { ChatMessage, ChatViewProvider } from './webview';

const HISTORY_PAGE_LIMIT = 100;
const MAX_EVIDENCE_PREVIEW_BYTES = 256 * 1024;

interface ActiveSession {
  daemon: DaemonHandle | null;
  client: NativeClient | null;
  stream: EventStream | null;
  sessionId: string | null;
  activeRunId: string | null;
  refreshing: boolean;
  refreshTimer: NodeJS.Timeout | null;
}

const active: ActiveSession = {
  daemon: null,
  client: null,
  stream: null,
  sessionId: null,
  activeRunId: null,
  refreshing: false,
  refreshTimer: null,
};

const store = new FaktorStore();
let chatProvider: ChatViewProvider | null = null;
let statusBar: vscode.StatusBarItem | null = null;

// ------------------------------------------------------------------ helpers

function config<T>(key: string, fallback: T): T {
  return vscode.workspace.getConfiguration('faktor').get<T>(key, fallback);
}

function workspaceRoot(context: vscode.ExtensionContext): string {
  const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (folder) {
    return folder;
  }
  // apps/vscode -> repository root.
  return resolve(context.extensionUri.fsPath, '..', '..');
}

function fetchAdapter(): FetchLike {
  return (url, init) => fetch(url, init) as unknown as Promise<ResponseLike>;
}

function sseAdapter(): SseFetchLike {
  return (url, init) => fetch(url, init) as unknown as Promise<SseResponseLike>;
}

function messageOf(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error);
  return raw.length > 500 ? `${raw.slice(0, 500)}…` : raw;
}

function reportError(error: unknown): void {
  const message = messageOf(error);
  if (store.snapshot().daemon === 'starting') {
    store.patch({ daemon: 'error', daemonDetail: message });
  }
  store.patch({ lastError: message });
  chatProvider?.postNotice('error', message);
  void vscode.window.showErrorMessage(`Faktor: ${message}`);
}

// ------------------------------------------------------------ daemon + session

async function startServer(context: vscode.ExtensionContext): Promise<void> {
  if (active.daemon && active.daemon.alive() && active.client) {
    if (!active.sessionId) {
      await ensureSession(active.client);
      startStream();
    }
    return;
  }
  store.patch({ daemon: 'starting', daemonDetail: 'locating faktor-cli', lastError: null });
  const binaryPath = config('binaryPath', '');
  const dataDir = config('dataDir', '');
  const extraArgs = config<string[]>('extraArgs', []);
  const startupTimeoutMs = config('startupTimeoutMs', 10_000);
  const daemon = await startDaemon({
    workspaceRoot: workspaceRoot(context),
    binaryPath: binaryPath.length > 0 ? binaryPath : undefined,
    dataDir: dataDir.length > 0 ? dataDir : undefined,
    extraArgs,
    startupTimeoutMs,
  });
  try {
    const client = new NativeClient({
      baseUrl: daemon.baseUrl,
      bearerToken: daemon.bearerToken,
      fetch: fetchAdapter(),
    });
    const health = await client.health();
    active.daemon = daemon;
    active.client = client;
    store.patch({
      daemon: 'running',
      daemonDetail: `${health.version} on port ${daemon.port}`,
      baseUrl: daemon.baseUrl,
      lastError: null,
    });
    await ensureSession(client);
    startStream();
    scheduleRefresh(0);
    chatProvider?.postNotice('info', `daemon ${health.version} ready at ${daemon.baseUrl}`);
  } catch (error) {
    // Never leak a spawned daemon when post-spawn setup fails.
    stopDaemon(daemon);
    active.daemon = null;
    active.client = null;
    store.patch({ daemon: 'error', daemonDetail: messageOf(error), baseUrl: null });
    throw error;
  }
}

function stopServer(): void {
  if (active.refreshTimer !== null) {
    clearTimeout(active.refreshTimer);
    active.refreshTimer = null;
  }
  active.stream?.stop();
  active.stream = null;
  stopDaemon(active.daemon);
  active.daemon = null;
  active.client = null;
  active.sessionId = null;
  active.activeRunId = null;
  active.refreshing = false;
  store.patch({
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
    transcript: [],
    streamStatus: 'stopped',
    lastError: null,
    busy: false,
  });
}

async function ensureSession(client: NativeClient): Promise<string> {
  if (active.sessionId) {
    return active.sessionId;
  }
  try {
    const sessions = await client.listSessions();
    store.patch({ sessions });
    if (sessions.length > 0) {
      active.sessionId = sessions[0]!.id;
      store.patch({ session: sessions[0]! });
    }
  } catch (error) {
    store.patch({ lastError: messageOf(error) });
  }
  if (!active.sessionId) {
    let provider = config('defaultProvider', '');
    let model = config('defaultModel', '');
    if (provider.length === 0 || model.length === 0) {
      try {
        const catalog = await client.modelCatalog();
        if (catalog.length > 0) {
          provider = provider || catalog[0]!.provider;
          model = model || catalog[0]!.model;
        }
      } catch {
        // The catalog is optional; session creation below still applies.
      }
    }
    const created = await client.createSession({
      provider: provider || 'faktor',
      model: model || 'default',
      workspace: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
      title: 'Faktor',
    });
    active.sessionId = created.id;
    store.patch({
      session: {
        id: created.id,
        title: created.title,
        provider: provider || '',
        model: model || '',
        state: 'unknown',
      },
    });
  }
  const sessionId = active.sessionId;
  try {
    const page = await client.messages(sessionId, { limit: HISTORY_PAGE_LIMIT });
    store.patch({ transcript: transcriptOf(page) });
  } catch (error) {
    store.patch({ lastError: messageOf(error) });
  }
  return sessionId;
}

/** Cheap structural signature: refresh the transcript only when it changed. */
function transcriptSignature(entries: readonly TranscriptEntry[]): string {
  let out = '';
  for (const entry of entries) {
    out += `${entry.id}:${entry.text.length}:${entry.reasoning.length}:${entry.summary.length}:`;
    for (const tool of entry.tools) {
      out += `${tool.state}:${tool.excerpt?.length ?? 0}:${tool.exitCode ?? 'n'};`;
    }
    out += '|';
  }
  return out;
}

function transcriptOf(page: NativeMessagePage): TranscriptEntry[] {
  return transcriptFromMessages(page.messages as unknown as Json[]);
}

function startStream(): void {
  const daemon = active.daemon;
  const sessionId = active.sessionId;
  if (!daemon || !sessionId) {
    return;
  }
  active.stream?.stop();
  const stream = new EventStream({
    baseUrl: daemon.baseUrl,
    bearerToken: daemon.bearerToken,
    sessionId,
    fetch: sseAdapter(),
    onEvent: (frame) => {
      const snapshot = store.snapshot();
      const transcript = applySseEvent(snapshot.transcript, frame.event, frame.data);
      if (transcript !== snapshot.transcript) {
        store.patch({ transcript });
      }
      scheduleRefresh();
    },
    onStatus: (status: EventStreamStatus) => {
      store.patch({ streamStatus: status });
      updateStatusBar();
    },
    onError: (error) => {
      store.patch({ lastError: messageOf(error) });
    },
  });
  active.stream = stream;
  stream.start();
  store.patch({ streamStatus: 'connecting' });
}

// -------------------------------------------------------------------- refresh

function scheduleRefresh(delayMs = 250): void {
  if (active.refreshTimer !== null) {
    clearTimeout(active.refreshTimer);
  }
  active.refreshTimer = setTimeout(() => {
    active.refreshTimer = null;
    void refresh();
  }, delayMs);
}

async function refresh(): Promise<void> {
  const client = active.client;
  const sessionId = active.sessionId;
  if (!client || !sessionId || active.refreshing) {
    return;
  }
  active.refreshing = true;
  try {
    const [projection, tasks, verification, usage, agents, runs, sessions, messages] =
      await Promise.all([
        client.projection(sessionId),
        client.tasks(sessionId),
        client.verification(sessionId),
        client.sessionUsage(sessionId),
        client.agents(sessionId),
        client.taskRuns(sessionId),
        client.listSessions(),
        client.messages(sessionId, { limit: HISTORY_PAGE_LIMIT }),
      ]);
    const task = tasks.length > 0 ? taskSummary(tasks[0]!) : null;
    const stillRunning =
      active.activeRunId !== null && runs.some((run) => run.run_id === active.activeRunId);
    if (!stillRunning) {
      active.activeRunId = null;
    }
    store.patch({
      sessions,
      machineState: projection.state.machine,
      machineLabel: projection.state.label,
      task,
      verification: verificationSummary(verification),
      usage: usageSummary(usage),
      agents: summarizeAgents(agents as unknown as Json[]),
      runs: runs.map(runSummary),
      activeRunId: active.activeRunId,
      busy: stillRunning,
      lastError: null,
    });
    // Assistant/status/tool lines are durable message rows; re-render the
    // bounded newest page only when its structure actually changed.
    const transcript = transcriptOf(messages);
    if (transcriptSignature(transcript) !== transcriptSignature(store.snapshot().transcript)) {
      store.patch({ transcript });
    }
  } catch (error) {
    store.patch({ lastError: messageOf(error) });
  } finally {
    active.refreshing = false;
    updateStatusBar();
  }
}

function taskSummary(view: NativeTaskView): TaskSummary {
  return {
    goal: view.goal,
    state: view.state,
    completed: view.milestones.completed,
    open: view.milestones.open,
    testsRun: view.tests.run,
    testsFailed: view.tests.failed,
    changedFiles: view.changedFiles,
    budget: view.budget
      ? {
          maxTokens: view.budget.maxTokens,
          spentTokens: view.budget.spentTokens,
          maxCostMicro: view.budget.maxCostMicro,
          spentCostMicro: view.budget.spentCostMicro,
          openReservedMicro: view.budget.openReservedMicro,
        }
      : null,
  };
}

function verificationSummary(view: NativeVerificationView): VerificationSummary {
  return {
    owed: view.owed.map((entry) => ({
      opId: entry.opId,
      tool: entry.tool,
      status: entry.status,
      effectStatus: entry.effectStatus,
    })),
    failedChecks: view.failedChecks.map((entry) => ({ id: entry.id, detail: entry.detail })),
  };
}

function usageSummary(usage: NativeSessionUsage): UsageSummary {
  let spentMicro = 0;
  let openMicro = 0;
  let maxMicro: number | null = null;
  let truncated = false;
  for (const task of usage.tasks) {
    spentMicro += task.budget.spentCostMicro;
    openMicro += task.budget.openReservedMicro;
    if (task.budget.maxCostMicro !== null) {
      maxMicro = (maxMicro ?? 0) + task.budget.maxCostMicro;
    }
    if (task.reservations.truncated) {
      truncated = true;
    }
  }
  return {
    tokens: usage.providerCalls.tokens,
    spentMicro,
    maxMicro,
    openMicro,
    truncated,
  };
}

function runSummary(run: NativeTaskRun): RunSummary {
  return {
    taskId: String(run.task_id),
    runId: run.run_id,
    mode: run.mode,
    state: run.state,
    goal: run.goal,
    model: run.model,
  };
}

// ---------------------------------------------------------------- task actions

async function startTask(goal: string, context: vscode.ExtensionContext): Promise<void> {
  try {
    if (!active.client || !active.sessionId) {
      await startServer(context);
    }
    const client = active.client;
    const sessionId = active.sessionId;
    if (!client || !sessionId) {
      return;
    }
    const maxTokens = config('budgetTokens', 0);
    const maxCostMicro = config('budgetCostMicro', 0);
    const mutationMode = config<'direct_compat' | 'shadow' | ''>('mutationMode', 'direct_compat');
    const request: StartTaskRunRequest = {
      goal,
      ...(maxTokens > 0 ? { max_tokens: maxTokens } : {}),
      ...(maxCostMicro > 0 ? { max_cost_micro: maxCostMicro } : {}),
      ...(mutationMode === 'shadow' || mutationMode === 'direct_compat'
        ? { mutation_mode: mutationMode }
        : {}),
    };
    let started: NativeTaskRunStarted;
    try {
      started = await client.startTaskRun(sessionId, request);
    } catch (error) {
      // A shadowed mutating run needs a daemon-registered worktree row; a
      // session created over HTTP has none (typed 409). Retry once as a
      // direct run and say so — never silently claim a different mode.
      if (
        error instanceof NativeApiError &&
        error.status === 409 &&
        request.mutation_mode !== 'direct_compat'
      ) {
        const directRequest: StartTaskRunRequest = { ...request, mutation_mode: 'direct_compat' };
        started = await client.startTaskRun(sessionId, directRequest);
        chatProvider?.postNotice(
          'info',
          'session has no daemon worktree row; task started as a direct run',
        );
      } else {
        throw error;
      }
    }
    active.activeRunId = started.run_id;
    store.patch({ activeRunId: started.run_id, busy: true, lastError: null });
    chatProvider?.postNotice('info', `task run ${started.run_id} started (${started.state})`);
    scheduleRefresh(0);
  } catch (error) {
    reportError(error);
  }
}

async function cancelActiveRun(): Promise<void> {
  const client = active.client;
  const sessionId = active.sessionId;
  const runId =
    active.activeRunId ??
    store.snapshot().runs.find((run) => run.state !== 'Done' && run.state !== 'Failed' && run.state !== 'Cancelled')?.runId ??
    null;
  if (!client || !sessionId || runId === null) {
    chatProvider?.postNotice('info', 'no active task run to cancel');
    return;
  }
  try {
    const ack = await client.cancelTaskRun(sessionId, runId);
    chatProvider?.postNotice('info', `run ${ack.run_id} cancel requested`);
    active.activeRunId = null;
    store.patch({ activeRunId: null, busy: false });
    scheduleRefresh(0);
  } catch (error) {
    reportError(error);
  }
}

async function controlAgent(message: ChatMessage): Promise<void> {
  const client = active.client;
  const agentId = typeof message.agentId === 'string' ? message.agentId : '';
  const action = typeof message.action === 'string' ? message.action : '';
  if (!client || agentId.length === 0) {
    return;
  }
  try {
    if (action === 'pause') {
      await client.pauseAgent(agentId);
    } else if (action === 'resume') {
      await client.resumeAgent(agentId);
    } else if (action === 'cancel') {
      await client.cancelAgent(agentId);
    } else if (action === 'steer') {
      const text = await vscode.window.showInputBox({
        title: `Faktor: steer ${agentId}`,
        prompt: 'Instruction delivered at the agent’s next safe boundary',
        ignoreFocusOut: true,
      });
      if (text === undefined || text.trim().length === 0) {
        return;
      }
      await client.steerAgent(agentId, text.trim());
    } else if (action === 'model') {
      const model = await vscode.window.showInputBox({
        title: `Faktor: model for ${agentId}`,
        prompt: 'Model id from the daemon catalog',
        ignoreFocusOut: true,
      });
      if (model === undefined || model.trim().length === 0) {
        return;
      }
      await client.setAgentModel(agentId, model.trim());
    } else if (action === 'budget') {
      const raw = await vscode.window.showInputBox({
        title: `Faktor: budget for ${agentId}`,
        prompt: 'Max tokens (empty keeps the budget unchanged)',
        ignoreFocusOut: true,
      });
      if (raw === undefined) {
        return;
      }
      const trimmed = raw.trim();
      const tokens = trimmed.length === 0 ? undefined : Number(trimmed);
      if (tokens !== undefined && (!Number.isInteger(tokens) || tokens <= 0)) {
        reportError(new Error('budget must be a positive integer token count'));
        return;
      }
      await client.setAgentBudget(agentId, tokens === undefined ? {} : { max_tokens: tokens });
    } else {
      return;
    }
    chatProvider?.postNotice('info', `${action} queued for ${agentId}`);
    scheduleRefresh(0);
  } catch (error) {
    reportError(error);
  }
}

async function retrieveEvidence(message: ChatMessage): Promise<void> {
  const client = active.client;
  const sessionId = active.sessionId;
  const evidenceId = message.evidenceId;
  if (!client || !sessionId || typeof evidenceId !== 'number' || !Number.isInteger(evidenceId)) {
    return;
  }
  try {
    const selector: NativeEvidenceSelector = { selector: 'all' };
    const retrieval = await client.retrieveEvidence(sessionId, evidenceId, selector);
    const bytes = Buffer.from(retrieval.bytesBase64, 'base64');
    const truncated =
      retrieval.truncatedByPolicy || bytes.byteLength > MAX_EVIDENCE_PREVIEW_BYTES;
    const text = bytes.subarray(0, MAX_EVIDENCE_PREVIEW_BYTES).toString('utf8');
    chatProvider?.postEvidence(evidenceId, text, truncated);
  } catch (error) {
    reportError(error);
  }
}

// --------------------------------------------------------------- status bar

function updateStatusBar(): void {
  if (!statusBar) {
    return;
  }
  const snapshot = store.snapshot();
  const icons: Record<string, string> = {
    running: '$(pulse)',
    starting: '$(sync~spin)',
    error: '$(error)',
    stopped: '$(circle-slash)',
  };
  const icon = icons[snapshot.daemon] ?? '$(circle-slash)';
  const task = snapshot.task ? ` · task ${snapshot.task.state}` : '';
  statusBar.text = `${icon} Faktor: ${snapshot.daemon}${task}`;
  statusBar.tooltip = snapshot.baseUrl
    ? `Faktor daemon ${snapshot.daemon} at ${snapshot.baseUrl}${snapshot.session ? ` · session ${snapshot.session.title}` : ''}`
    : 'Faktor daemon stopped';
  statusBar.show();
}

// ------------------------------------------------------------- webview bridge

async function handleWebviewMessage(
  message: ChatMessage,
  context: vscode.ExtensionContext,
): Promise<void> {
  switch (message.type) {
    case 'ready':
      chatProvider?.postSnapshot(store.snapshot());
      return;
    case 'startDaemon':
      try {
        await startServer(context);
      } catch (error) {
        reportError(error);
      }
      return;
    case 'stopDaemon':
      stopServer();
      return;
    case 'refresh':
      await refresh();
      return;
    case 'sendGoal': {
      const goal = typeof message.goal === 'string' ? message.goal.trim() : '';
      if (goal.length > 0) {
        await startTask(goal, context);
      }
      return;
    }
    case 'newTask': {
      const goal = await vscode.window.showInputBox({
        title: 'Faktor: new task',
        prompt: 'Goal for the task run',
        ignoreFocusOut: true,
      });
      if (goal !== undefined && goal.trim().length > 0) {
        await startTask(goal.trim(), context);
      }
      return;
    }
    case 'cancelRun':
      await cancelActiveRun();
      return;
    case 'agentControl':
      await controlAgent(message);
      return;
    case 'retrieveEvidence':
      await retrieveEvidence(message);
      return;
    default:
      return;
  }
}

// ------------------------------------------------------------------- lifecycle

export function activate(context: vscode.ExtensionContext): void {
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBar.command = 'faktor.openChat';
  context.subscriptions.push(statusBar);
  updateStatusBar();

  chatProvider = new ChatViewProvider(context.extensionUri, {
    handle: (message) => handleWebviewMessage(message, context),
  });

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, chatProvider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
    vscode.commands.registerCommand('faktor.startServer', async () => {
      try {
        await startServer(context);
      } catch (error) {
        reportError(error);
      }
    }),
    vscode.commands.registerCommand('faktor.stopServer', () => {
      stopServer();
    }),
    vscode.commands.registerCommand('faktor.openChat', () => {
      chatProvider?.focus();
    }),
    vscode.commands.registerCommand('faktor.newTask', async () => {
      const goal = await vscode.window.showInputBox({
        title: 'Faktor: new task',
        prompt: 'Goal for the task run',
        ignoreFocusOut: true,
      });
      if (goal !== undefined && goal.trim().length > 0) {
        await startTask(goal.trim(), context);
      }
    }),
    vscode.commands.registerCommand('faktor.cancelTask', () => cancelActiveRun()),
    vscode.commands.registerCommand('faktor.refresh', () => refresh()),
    {
      dispose: store.subscribe((snapshot) => {
        chatProvider?.postSnapshot(snapshot);
        updateStatusBar();
      }),
    },
    { dispose: () => stopServer() },
  );

  if (config('autoStart', false)) {
    void startServer(context).catch(reportError);
  }
}

export function deactivate(): void {
  stopServer();
}
