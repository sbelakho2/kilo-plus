// The Faktor daemon lifecycle (extracted from the launcher-only
// extension.ts):
//
//  1. Find the platform binary (env FAKTOR_BIN, else target/debug or
//     target/release relative to the workspace root).
//  2. Generate a 64-hex FAKTOR_SERVER_PASSWORD and spawn
//     `faktor-cli serve --port 0` with it in the environment.
//  3. Read stdout line-by-line until the EXACT frozen startup line
//     `/faktor server listening on http:\/\/127\.0\.0\.1:(\d+)/`, resolve
//     the port, and build both accepted auth forms:
//       - the frozen v7.5.6 Basic header
//         (`Basic base64("kilo:" + password)`), and
//       - the native bearer form (`Bearer <password>`).
//  4. Expose health() against GET /global/health.
//
// Deliberately dependency-free (node:http only; no axios, no vscode
// import — the caller supplies the workspace root). The daemon never
// prints the password; the lifecycle never logs it.

import * as http from 'node:http';
import * as crypto from 'node:crypto';
import { ChildProcess, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

export const STARTUP_LINE = /faktor server listening on http:\/\/127\.0\.0\.1:(\d+)/;
const DEFAULT_STARTUP_TIMEOUT_MS = 10_000;
const HEALTH_TIMEOUT_MS = 5_000;
const STDERR_TAIL_BYTES = 8 * 1024;
const KILL_ESCALATION_MS = 5_000;
const MAX_HEALTH_BODY_BYTES = 64 * 1024;

export interface DaemonOptions {
  /** Workspace root used to locate target/debug|release/faktor-cli. */
  readonly workspaceRoot: string;
  /** Explicit binary override (config `faktor.binaryPath` or FAKTOR_BIN). */
  readonly binaryPath?: string;
  /** Optional `--data-dir` for the daemon (config `faktor.dataDir`). */
  readonly dataDir?: string;
  /** Extra argv appended after `serve --port 0`. */
  readonly extraArgs?: readonly string[];
  readonly startupTimeoutMs?: number;
  readonly env?: NodeJS.ProcessEnv;
}

export interface DaemonHealth {
  readonly ok: boolean;
  readonly version: string;
}

export interface DaemonHandle {
  readonly port: number;
  readonly password: string;
  readonly baseUrl: string;
  /** Frozen v7.5.6 wire form: `Basic base64("kilo:" + password)`. */
  readonly authHeader: string;
  /** Native form: `Bearer <password>`. */
  readonly bearerToken: string;
  readonly pid: number | undefined;
  health(): Promise<DaemonHealth>;
  /** Bounded tail of the daemon's stderr (diagnostics only). */
  stderrTail(): string;
  /** Whether the child process is still alive. */
  alive(): boolean;
  stop(): void;
}

let activeChild: ChildProcess | null = null;
let activeHandle: DaemonHandle | null = null;

export function findBinary(options: DaemonOptions): string {
  const env = options.binaryPath ?? process.env.FAKTOR_BIN;
  if (env && env.length > 0) {
    return env;
  }
  const root = options.workspaceRoot;
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
    `faktor-cli binary not found (looked for ${candidates.join(', ')}; set FAKTOR_BIN or faktor.binaryPath to override)`,
  );
}

export function isRunning(): boolean {
  return activeHandle !== null && activeHandle.alive();
}

export function currentDaemon(): DaemonHandle | null {
  return activeHandle;
}

/**
 * Start (or return the already running) daemon. The returned handle owns
 * every child process it created; `stop()` is the only shutdown path.
 */
export async function startDaemon(options: DaemonOptions): Promise<DaemonHandle> {
  if (activeHandle && activeHandle.alive()) {
    return activeHandle;
  }
  const bin = findBinary(options);
  const password = crypto.randomBytes(32).toString('hex');
  const args = ['serve', '--port', '0'];
  if (options.dataDir && options.dataDir.length > 0) {
    args.push('--data-dir', options.dataDir);
  }
  if (options.extraArgs) {
    args.push(...options.extraArgs);
  }
  const child = spawn(bin, args, {
    env: { ...process.env, ...options.env, FAKTOR_SERVER_PASSWORD: password },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const stderr = new BoundedTail(STDERR_TAIL_BYTES);
  child.stderr?.setEncoding('utf8');
  child.stderr?.on('data', (chunk: string) => stderr.push(chunk));
  const port = await readStartupPort(child, options.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS);
  const authHeader = 'Basic ' + Buffer.from(`kilo:${password}`).toString('base64');
  const handle: DaemonHandle = {
    port,
    password,
    baseUrl: `http://127.0.0.1:${port}`,
    authHeader,
    bearerToken: password,
    pid: child.pid,
    health: () => health(port, authHeader),
    stderrTail: () => stderr.text(),
    alive: () => child.exitCode === null && child.signalCode === null,
    stop: () => stopChild(child),
  };
  child.once('exit', () => {
    if (activeChild === child) {
      activeChild = null;
      activeHandle = null;
    }
  });
  activeChild = child;
  activeHandle = handle;
  return handle;
}

/** SIGTERM, escalating to SIGKILL only when the child ignores it. */
export function stopDaemon(handle?: DaemonHandle | null): void {
  const target = handle ?? activeHandle;
  if (target) {
    target.stop();
    return;
  }
  if (activeChild) {
    stopChild(activeChild);
  }
}

function stopChild(child: ChildProcess): void {
  if (child.exitCode !== null || child.signalCode !== null) {
    if (activeChild === child) {
      activeChild = null;
      activeHandle = null;
    }
    return;
  }
  try {
    child.kill('SIGTERM');
  } catch {
    // Already gone.
  }
  const escalate = setTimeout(() => {
    if (child.exitCode === null && child.signalCode === null) {
      try {
        child.kill('SIGKILL');
      } catch {
        // Already gone.
      }
    }
  }, KILL_ESCALATION_MS);
  escalate.unref?.();
  child.once('exit', () => {
    clearTimeout(escalate);
    if (activeChild === child) {
      activeChild = null;
      activeHandle = null;
    }
  });
}

function readStartupPort(child: ChildProcess, timeoutMs: number): Promise<number> {
  return new Promise<number>((resolvePort, reject) => {
    let buffer = '';
    let settled = false;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error(`timed out waiting for the daemon startup line (${timeoutMs}ms)`));
        stopChild(child);
      }
    }, timeoutMs);
    const finish = (err: Error | null, port?: number): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (err) {
        reject(err);
        stopChild(child);
      } else if (port !== undefined) {
        resolvePort(port);
      }
    };
    child.on('error', (err) => finish(err));
    child.on('exit', (code, signal) => {
      if (!settled) {
        finish(
          new Error(
            `daemon exited before the startup line (code=${code ?? 'null'} signal=${signal ?? 'null'})`,
          ),
        );
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
  });
}

function health(port: number, authHeader: string): Promise<DaemonHealth> {
  return new Promise<DaemonHealth>((resolveHealth, reject) => {
    const req = http.request(
      {
        host: '127.0.0.1',
        port,
        path: '/global/health',
        method: 'GET',
        headers: { Authorization: authHeader },
        timeout: HEALTH_TIMEOUT_MS,
      },
      (res) => {
        const chunks: Buffer[] = [];
        let size = 0;
        res.on('data', (chunk: Buffer) => {
          size += chunk.length;
          if (size > MAX_HEALTH_BODY_BYTES) {
            res.destroy();
            reject(new Error('health check failed: response body exceeded the bound'));
            return;
          }
          chunks.push(chunk);
        });
        res.on('end', () => {
          const status = res.statusCode ?? 0;
          if (status === 401) {
            reject(new Error('health check failed: 401 unauthorized'));
            return;
          }
          if (status !== 200) {
            reject(new Error(`health check failed: HTTP ${status || 'unknown'}`));
            return;
          }
          let parsed: unknown;
          try {
            parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          } catch {
            reject(new Error('health check failed: response was not JSON'));
            return;
          }
          if (
            typeof parsed !== 'object' ||
            parsed === null ||
            typeof (parsed as { ok?: unknown }).ok !== 'boolean' ||
            typeof (parsed as { version?: unknown }).version !== 'string'
          ) {
            reject(new Error('health check failed: unexpected response shape'));
            return;
          }
          resolveHealth({
            ok: (parsed as { ok: boolean }).ok,
            version: (parsed as { version: string }).version,
          });
        });
      },
    );
    req.on('timeout', () => {
      req.destroy(new Error('health check failed: timeout'));
    });
    req.on('error', (err) => reject(err));
    req.end();
  });
}

/** A byte-bounded ring buffer (no unbounded stderr in RAM). */
class BoundedTail {
  private readonly chunks: string[] = [];
  private size = 0;
  private readonly limit: number;

  constructor(limit: number) {
    this.limit = limit;
  }

  push(chunk: string): void {
    this.chunks.push(chunk);
    this.size += chunk.length;
    while (this.size > this.limit && this.chunks.length > 1) {
      const dropped = this.chunks.shift();
      if (dropped === undefined) {
        break;
      }
      this.size -= dropped.length;
    }
    if (this.chunks.length === 1 && this.chunks[0]!.length > this.limit) {
      this.chunks[0] = this.chunks[0]!.slice(-this.limit);
      this.size = this.chunks[0]!.length;
    }
  }

  text(): string {
    return this.chunks.join('');
  }
}
