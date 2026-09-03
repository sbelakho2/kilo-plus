// The frozen-client-equivalent launcher for the Faktor daemon (mirrors the
// behavior of Kilo's server-manager.ts):
//
//  1. Find the platform binary (env FAKTOR_PLUS_BIN, else target/debug or
//     target/release relative to the workspace root).
//  2. Generate a 64-hex FAKTOR_SERVER_PASSWORD and spawn
//     `faktor-cli serve --port 0` with it in the environment.
//  3. Read stdout line-by-line until the EXACT frozen startup line
//     `/kilo server listening on http:\/\/127\.0\.1:(\d+)/`, resolve the
//     port, and build the Basic auth header
//     (`Basic base64("kilo:" + password)`).
//  4. Expose health() against GET /global/health.
//
// Deliberately dependency-free (node:http only; no axios). The daemon never
// prints the password; the extension never logs it.

import * as vscode from 'vscode';
import * as http from 'node:http';
import * as crypto from 'node:crypto';
import { ChildProcess, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const STARTUP_LINE = /kilo server listening on http:\/\/127\.0\.0\.1:(\d+)/;
const STARTUP_TIMEOUT_MS = 10_000;

export interface KilopClient {
  readonly port: number;
  readonly password: string;
  readonly authHeader: string;
  health(): Promise<void>;
}

let activeChild: ChildProcess | null = null;
let activeClient: KilopClient | null = null;

function workspaceRoot(): string {
  const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (folder) {
    return folder;
  }
  // Fall back to the repository root (apps/vscode -> repo root).
  return resolve(__dirname, '..', '..', '..');
}

function findBinary(): string {
  const env = process.env.FAKTOR_PLUS_BIN;
  if (env && env.length > 0) {
    return env;
  }
  const root = workspaceRoot();
  const candidates = [
    join(root, 'target', 'debug', 'faktor-cli'),
    join(root, 'target', 'release', 'faktor-cli'),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(
    `faktor-cli binary not found (looked for ${candidates.join(', ')}; set FAKTOR_PLUS_BIN to override)`,
  );
}

export async function startServer(context: vscode.ExtensionContext): Promise<KilopClient> {
  if (activeChild && activeClient) {
    return activeClient;
  }
  const bin = findBinary();
  const password = crypto.randomBytes(32).toString('hex');
  const child = spawn(bin, ['serve', '--port', '0'], {
    env: { ...process.env, FAKTOR_SERVER_PASSWORD: password },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const port = await readStartupPort(child);
  const authHeader =
    'Basic ' + Buffer.from(`kilo:${password}`).toString('base64');
  const client: KilopClient = {
    port,
    password,
    authHeader,
    health: () => health(port, authHeader),
  };
  activeChild = child;
  activeClient = client;
  context.subscriptions.push({
    dispose: () => stopServer(),
  });
  void vscode.window.showInformationMessage(
    `Faktor server listening on http://127.0.0.1:${port}`,
  );
  return client;
}

export function stopServer(): void {
  const child = activeChild;
  activeChild = null;
  activeClient = null;
  if (child && child.exitCode === null && child.signalCode === null) {
    child.kill('SIGTERM');
  }
}

function readStartupPort(child: ChildProcess): Promise<number> {
  return new Promise<number>((resolvePort, reject) => {
    let buffer = '';
    let settled = false;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error(`timed out waiting for the daemon startup line (${STARTUP_TIMEOUT_MS}ms)`));
        child.kill('SIGTERM');
      }
    }, STARTUP_TIMEOUT_MS);
    const finish = (err: Error | null, port?: number): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (err) {
        reject(err);
        child.kill('SIGTERM');
      } else if (port !== undefined) {
        resolvePort(port);
      }
    };
    child.on('error', (err) => finish(err));
    child.on('exit', (code, signal) => {
      if (!settled) {
        finish(new Error(`daemon exited before the startup line (code=${code ?? 'null'} signal=${signal ?? 'null'})`));
      }
    });
    child.stdout?.setEncoding('utf8');
    child.stdout?.on('data', (chunk: string) => {
      buffer += chunk;
      let idx: number;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).replace(/\r$/, '');
        buffer = buffer.slice(idx + 1);
        const match = STARTUP_LINE.exec(line);
        if (match) {
          finish(null, Number(match[1]));
          return;
        }
      }
    });
    child.stderr?.on('data', (_chunk: Buffer | string) => {
      // stderr is never the startup line; diagnostics go to the daemon log.
    });
  });
}

function health(port: number, authHeader: string): Promise<void> {
  return new Promise<void>((resolveHealth, reject) => {
    const req = http.request(
      {
        host: '127.0.0.1',
        port,
        path: '/global/health',
        method: 'GET',
        headers: { Authorization: authHeader },
      },
      (res) => {
        res.resume();
        res.on('end', () => {
          if (res.statusCode === 200) {
            resolveHealth();
          } else if (res.statusCode === 401) {
            reject(new Error('health check failed: 401 unauthorized'));
          } else {
            reject(new Error(`health check failed: HTTP ${res.statusCode ?? 'unknown'}`));
          }
        });
      },
    );
    req.on('error', (err) => reject(err));
    req.end();
  });
}

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand('faktor-plus.startServer', async () => {
      await startServer(context);
    }),
    vscode.commands.registerCommand('faktor-plus.stopServer', () => {
      stopServer();
    }),
  );
}

export function deactivate(): void {
  stopServer();
}
