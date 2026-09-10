// The Faktor chat webview: one hand-written HTML surface over the native
// daemon API. This module owns ONLY the VS Code webview plumbing — HTML
// generation, strict CSP + script nonce, message transport — and delegates
// every daemon action to an injected host (extension.ts). No remote
// scripts, no eval, no inline handlers; the bundled media/chat.js is the
// only script, and every daemon-derived string is rendered as text.

import * as vscode from 'vscode';
import { randomBytes } from 'node:crypto';

/** Messages the webview sends to the extension host. */
export interface ChatMessage {
  readonly type: string;
  readonly [key: string]: unknown;
}

/** The extension-side handler the provider delegates every message to. */
export interface ChatHost {
  handle(message: ChatMessage): void | Promise<void>;
}

export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'faktor.chat';

  private view: vscode.WebviewView | null = null;
  private snapshot: unknown = null;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly host: ChatHost,
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'media')],
    };
    view.webview.html = this.render(view.webview);
    view.webview.onDidReceiveMessage((message: ChatMessage) => {
      void this.host.handle(message);
    });
    view.onDidDispose(() => {
      this.view = null;
    });
    if (this.snapshot !== null) {
      this.post({ type: 'snapshot', snapshot: this.snapshot });
    }
  }

  /** Push a full state snapshot; the webview re-renders from it. */
  postSnapshot(snapshot: unknown): void {
    this.snapshot = snapshot;
    this.post({ type: 'snapshot', snapshot });
  }

  /** Deliver the decoded bytes of one expanded evidence artifact. */
  postEvidence(id: number, text: string, truncated: boolean): void {
    this.post({ type: 'evidence', id, text, truncated });
  }

  /** One transient notice line (last error, control ack, ...). */
  postNotice(level: 'info' | 'error', message: string): void {
    this.post({ type: 'notice', level, message });
  }

  focus(): void {
    void vscode.commands.executeCommand('faktor.chat.focus');
  }

  private post(message: unknown): void {
    void this.view?.webview.postMessage(message);
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
