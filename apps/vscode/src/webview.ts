// The Faktor chat webview. Two render modes over one message transport:
//
//   - vendored (preferred): when the pinned Kilo v7.5.6 bundle exists at
//     <repo>/ui/kilo-v756-webview/dist (built `dist/index.html` entry when
//     present, else the upstream esbuild pair dist/webview.js +
//     dist/webview.css; FAKTOR_UI_BUNDLE overrides the root), its HTML shell
//     is served with a strict CSP + script nonce, and the kilo-bridge
//     translates between the frozen UI message ABI and the native snapshot.
//   - built-in (fallback): the hand-written HTML surface over native state
//     (`media/chat.js`), used when the vendored bundle is absent.
//
// This module owns ONLY VS Code webview plumbing — HTML generation, CSP,
// message transport and bridge routing — and delegates every daemon action to
// an injected host (extension.ts). No remote scripts, no eval, no inline
// handlers; the bundled media/chat.js or the vendored bundle is the only
// script, and every daemon-derived string is rendered as text.

import * as vscode from 'vscode';
import { randomBytes } from 'node:crypto';
import { join } from 'node:path';
import {
  buildVendoredWebviewHtml,
  bridgeCommandToHostMessage,
  ingestWebviewMessage,
  locateVendoredBundle,
  readyMessage,
  snapshotToWebviewMessages,
  vendoredFallbackNotice,
  VendoredBundle,
} from './kilo-bridge';
import { FaktorSnapshot } from './state';

/** Messages the webview sends to the extension host. */
export interface ChatMessage {
  readonly type: string;
  readonly [key: string]: unknown;
}

/** The extension-side handler the provider delegates every message to. */
export interface ChatHost {
  handle(message: ChatMessage): void | Promise<void>;
}

const MAX_LOUD_DROPS = 20;

function vendoredRoot(extensionUri: vscode.Uri): string {
  const override = process.env.FAKTOR_UI_BUNDLE;
  if (override !== undefined && override.length > 0) {
    return override;
  }
  // apps/vscode -> repository root -> ui/kilo-v756-webview
  return join(extensionUri.fsPath, '..', '..', 'ui', 'kilo-v756-webview');
}

function locateVendoredUi(extensionUri: vscode.Uri): VendoredBundle | null {
  const root = vendoredRoot(extensionUri);
  const bundle = locateVendoredBundle(root);
  if (bundle === null) {
    // Recorded once per webview resolution; the built-in fallback serves.
    console.warn(vendoredFallbackNotice(root));
  }
  return bundle;
}

export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'faktor.chat';

  private view: vscode.WebviewView | null = null;
  private snapshot: FaktorSnapshot | null = null;
  private vendored: VendoredBundle | null = null;
  private dropCount = 0;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly host: ChatHost,
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    this.vendored = locateVendoredUi(this.extensionUri);
    const roots: vscode.Uri[] = [vscode.Uri.joinPath(this.extensionUri, 'media')];
    if (this.vendored !== null) {
      roots.push(vscode.Uri.file(this.vendored.root));
    }
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: roots,
    };
    view.webview.html =
      this.vendored !== null
        ? this.renderVendored(view.webview, this.vendored)
        : this.render(view.webview);
    view.webview.onDidReceiveMessage((message: ChatMessage) => {
      this.route(message);
    });
    view.onDidDispose(() => {
      this.view = null;
    });
    if (this.snapshot !== null && this.vendored === null) {
      this.post({ type: 'snapshot', snapshot: this.snapshot });
    }
  }

  /** Push a full state snapshot; the webview re-renders from it. */
  postSnapshot(snapshot: FaktorSnapshot): void {
    this.snapshot = snapshot;
    if (this.vendored !== null) {
      for (const message of snapshotToWebviewMessages(snapshot)) {
        this.post(message);
      }
      return;
    }
    this.post({ type: 'snapshot', snapshot });
  }

  /** Deliver the decoded bytes of one expanded evidence artifact. */
  postEvidence(id: number, text: string, truncated: boolean): void {
    if (this.vendored !== null) {
      // The frozen UI has no evidence-expansion message; never fabricate one.
      console.log(`[faktor-bridge] evidence ${id} retrieved (${text.length} chars, truncated=${truncated}); no vendored-UI mapping`);
      return;
    }
    this.post({ type: 'evidence', id, text, truncated });
  }

  /** One transient notice line (last error, control ack, ...). */
  postNotice(level: 'info' | 'error', message: string): void {
    if (this.vendored !== null) {
      if (level === 'error') {
        this.post({ type: 'error', message });
      } else {
        console.log(`[faktor-bridge] notice: ${message}`);
      }
      return;
    }
    this.post({ type: 'notice', level, message });
  }

  focus(): void {
    void vscode.commands.executeCommand('faktor.chat.focus');
  }

  /** True when the pinned vendored bundle is being served in this session. */
  usingVendoredUi(): boolean {
    return this.vendored !== null;
  }

  private route(message: ChatMessage): void {
    if (this.vendored === null) {
      void this.host.handle(message);
      return;
    }
    const result = ingestWebviewMessage(message);
    if ('dropped' in result) {
      this.dropCount += 1;
      console.error(
        `[faktor-bridge] dropped ${result.type ?? 'unnamed'} message: ${result.reason} (${result.bytes} bytes)`,
      );
      if (this.dropCount <= MAX_LOUD_DROPS) {
        this.postNotice('error', `frozen UI message dropped: ${result.reason}`);
      }
      return;
    }
    if (result.kind === 'openExternal') {
      void vscode.env.openExternal(vscode.Uri.parse(result.url));
      return;
    }
    if (result.kind === 'ready') {
      this.post(readyMessage(this.bridgeContext()));
    }
    const hostMessage = bridgeCommandToHostMessage(result);
    if (hostMessage !== null) {
      void this.host.handle(hostMessage);
    }
  }

  private bridgeContext(): {
    extensionVersion: string;
    workspaceDirectory: string;
    daemonVersion: string | null;
    port: number | null;
  } {
    const self = vscode.extensions.all.find(
      (extension) => extension.extensionUri.fsPath === this.extensionUri.fsPath,
    );
    const version = self?.packageJSON?.version;
    const snapshot = this.snapshot;
    let port: number | null = null;
    if (snapshot?.baseUrl) {
      try {
        const parsed = new URL(snapshot.baseUrl);
        const candidate = Number(parsed.port);
        if (Number.isInteger(candidate) && candidate > 0) {
          port = candidate;
        }
      } catch {
        port = null;
      }
    }
    const detail = snapshot?.daemonDetail ?? '';
    const daemonVersion = detail.match(/^(\S+)\s+on\s+port/)?.[1] ?? null;
    return {
      extensionVersion: typeof version === 'string' ? version : '0.0.0',
      workspaceDirectory: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? '',
      daemonVersion,
      port,
    };
  }

  private post(message: unknown): void {
    void this.view?.webview.postMessage(message);
  }

  private renderVendored(webview: vscode.Webview, ui: VendoredBundle): string {
    const nonce = randomBytes(16).toString('hex');
    const resource = (path: string): string =>
      webview.asWebviewUri(vscode.Uri.file(path)).toString();
    return buildVendoredWebviewHtml({
      cspSource: webview.cspSource,
      nonce,
      scriptUri: resource(ui.script),
      styleUri: resource(ui.style),
      // Icons live under dist/assets/icons in the built bundle; the shell
      // falls back to the bundle root only for layout purposes (the icon
      // lookup itself is a webview-relative URL, never a filesystem read).
      iconsBaseUri: resource(ui.icons ?? ui.root),
      workerUri: ui.worker !== null ? resource(ui.worker) : '',
      title: 'Faktor',
      sidebar: '',
      topBar: false,
    });
  }

  private render(webview: vscode.Webview): string {
    const nonce = randomBytes(16).toString('hex');
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.js'),
    );
    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.css'),
    );
    const csp = [
      "default-src 'none'",
      `style-src ${webview.cspSource}`,
      `font-src ${webview.cspSource}`,
      `img-src ${webview.cspSource} data:`,
      `script-src 'nonce-${nonce}'`,
      "connect-src 'none'",
    ].join('; ');
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="${styleUri}" rel="stylesheet">
<title>Faktor</title>
</head>
<body>
<div id="app">
  <header>
    <span id="daemon-dot" class="dot dot-stopped" aria-hidden="true"></span>
    <span id="daemon-text">stopped</span>
    <span class="spacer"></span>
    <button id="btn-refresh" type="button" title="Refresh state">Refresh</button>
    <button id="btn-stop" type="button" title="Stop the daemon">Stop</button>
    <button id="btn-start" type="button" title="Start the daemon">Start</button>
  </header>
  <section id="meta">
    <div class="meta-row"><span class="meta-key">Session</span><span id="session-title">none</span></div>
    <div class="meta-row"><span class="meta-key">State</span><span id="machine-label">daemon stopped</span></div>
    <div class="meta-row"><span class="meta-key">Stream</span><span id="stream-status">stopped</span></div>
  </section>
  <section id="task-card" class="card" hidden>
    <h2>Task</h2>
    <div class="task-head"><span id="task-state" class="badge">—</span><button id="btn-cancel-run" type="button">Cancel run</button></div>
    <div id="task-goal" class="goal"></div>
    <div id="task-milestones" class="lines"></div>
    <div id="task-tests" class="lines"></div>
    <div id="task-files" class="lines"></div>
    <div id="task-verification" class="lines"></div>
    <div id="task-budget" class="lines"></div>
  </section>
  <section id="agents-card" class="card" hidden>
    <h2>Agents</h2>
    <ul id="agent-list"></ul>
  </section>
  <section id="transcript-card" class="card">
    <h2>Conversation</h2>
    <div id="entries"></div>
  </section>
  <section id="notices" aria-live="polite"></section>
  <form id="composer">
    <textarea id="goal" rows="3" placeholder="Describe the goal. It starts a task run."></textarea>
    <div class="composer-actions">
      <button id="btn-send" type="submit">Run task</button>
      <button id="btn-new-task" type="button">New task…</button>
    </div>
  </form>
</div>
<script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }
}
