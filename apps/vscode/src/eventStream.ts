// SSE client for the daemon's journal event stream
// (`GET /api/session/{id}/events?events_after=<cursor>`), the stream the
// native server projects from the durable journal (faktor_protocol::sse).
//
// Contract: every frame carries `event:` (type), `id:` (the journal
// sequence — the resume cursor) and one JSON `data:` line. Heartbeats
// (`event: heartbeat`) and comment/keep-alive lines are ignored. On any
// disconnect the client reconnects with exponential backoff and resumes
// from the last valid frame id, so a reconnect can neither duplicate nor
// skip events. Frame size and error bodies are bounded.
//
// Dependency-free and fetch-injectable (scripts/selftest.mjs drives it with
// a fake fetch + ReadableStream); no vscode import.

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export interface SseReaderLike {
  read(): Promise<{ done: boolean; value?: Uint8Array }>;
  cancel(reason?: unknown): Promise<void>;
}

export interface SseResponseLike {
  readonly status: number;
  readonly ok: boolean;
  readonly headers?: { get(name: string): string | null } | null;
  readonly body?: { getReader(): SseReaderLike } | null;
  text(): Promise<string>;
}

export interface SseInitLike {
  method: string;
  headers: Record<string, string>;
  signal?: AbortSignal;
}

export type SseFetchLike = (url: string, init: SseInitLike) => Promise<SseResponseLike>;

export interface SseFrame {
  /** `id:` — the journal sequence, monotonic per session. */
  readonly id: number;
  /** `event:` — the projection type (`agent_state_changed`, ...). */
  readonly event: string;
  readonly data: Json;
}

export type EventStreamStatus = 'connecting' | 'open' | 'retrying' | 'stopped';

export interface EventStreamOptions {
  readonly baseUrl: string;
  readonly bearerToken: string;
  readonly sessionId: string;
  /** Journal sequence to resume from (0 = from the beginning). */
  readonly cursor?: number;
  readonly fetch?: SseFetchLike;
  readonly onEvent: (frame: SseFrame) => void;
  readonly onStatus?: (status: EventStreamStatus, detail?: string) => void;
  readonly onError?: (error: Error) => void;
  readonly minBackoffMs?: number;
  readonly maxBackoffMs?: number;
  /** A frame larger than this is rejected (bounded everything). */
  readonly maxFrameBytes?: number;
  /** Injectable clock for tests. */
  readonly sleep?: (ms: number) => Promise<void>;
  readonly jitter?: () => number;
}

export const DEFAULT_MIN_BACKOFF_MS = 250;
export const DEFAULT_MAX_BACKOFF_MS = 15_000;
export const DEFAULT_MAX_FRAME_BYTES = 1024 * 1024;
const ERROR_BODY_BYTES = 64 * 1024;

export class EventStreamProtocolError extends Error {
  constructor(detail: string) {
    super(`event stream protocol violation: ${detail}`);
    this.name = 'EventStreamProtocolError';
  }
}

export class EventStream {
  private readonly options: EventStreamOptions;
  private readonly fetchImpl: SseFetchLike;
  private readonly minBackoffMs: number;
  private readonly maxBackoffMs: number;
  private readonly maxFrameBytes: number;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly jitter: () => number;
  private cursorValue: number;
  private stopped = true;
  private controller: AbortController | null = null;
  private loopPromise: Promise<void> | null = null;
  private statusValue: EventStreamStatus = 'stopped';

  constructor(options: EventStreamOptions) {
    this.options = options;
    const injected = options.fetch;
    if (injected) {
      this.fetchImpl = injected;
    } else {
      const globalFetch = (globalThis as unknown as { fetch?: SseFetchLike }).fetch;
      if (typeof globalFetch !== 'function') {
        throw new EventStreamProtocolError('no fetch implementation available; pass options.fetch');
      }
      this.fetchImpl = (url, init) => globalFetch(url, init);
    }
    this.cursorValue = options.cursor ?? 0;
    this.minBackoffMs = options.minBackoffMs ?? DEFAULT_MIN_BACKOFF_MS;
    this.maxBackoffMs = options.maxBackoffMs ?? DEFAULT_MAX_BACKOFF_MS;
    this.maxFrameBytes = options.maxFrameBytes ?? DEFAULT_MAX_FRAME_BYTES;
    this.sleep = options.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    this.jitter = options.jitter ?? (() => Math.floor(Math.random() * 100));
  }

  get cursor(): number {
    return this.cursorValue;
  }

  get status(): EventStreamStatus {
    return this.statusValue;
  }

  /** Seed the resume cursor (e.g. from a paged `/native/events` read). */
  setCursor(cursor: number): void {
    if (Number.isInteger(cursor) && cursor >= 0) {
      this.cursorValue = cursor;
    }
  }

  /** Start streaming; idempotent while running. */
  start(): void {
    if (!this.stopped) {
      return;
    }
    this.stopped = false;
    this.controller = new AbortController();
    this.loopPromise = this.loop(this.controller.signal);
  }

  /** Stop streaming and abort the in-flight connection. Idempotent. */
  stop(): void {
    if (this.stopped) {
      return;
    }
    this.stopped = true;
    this.controller?.abort();
    this.controller = null;
    this.setStatus('stopped');
  }

  /** Resolves when the run loop exits (tests use this). */
  whenStopped(): Promise<void> {
    return this.loopPromise ?? Promise.resolve();
  }

  private async loop(signal: AbortSignal): Promise<void> {
    let attempt = 0;
    while (!this.stopped && !signal.aborted) {
      if (attempt > 0) {
        const backoff = Math.min(
          this.maxBackoffMs,
          this.minBackoffMs * 2 ** Math.min(attempt - 1, 16),
        );
        const wait = backoff + this.jitter();
        this.setStatus('retrying', `reconnect in ${wait}ms`);
        await this.sleep(wait);
        if (this.stopped || signal.aborted) {
          break;
        }
      }
      this.setStatus(attempt === 0 ? 'connecting' : 'retrying', `from cursor ${this.cursorValue}`);
      try {
        await this.connectOnce(signal);
        // The connection served at least one response; retry at the base
        // backoff (a flapping stream must not busy-loop).
        attempt = 1;
        this.setStatus('retrying', `stream ended at cursor ${this.cursorValue}`);
      } catch (error) {
        if (this.stopped || signal.aborted) {
          break;
        }
        attempt = Math.min(attempt + 1, 20);
        this.reportError(error);
      }
    }
    this.setStatus('stopped');
  }

  /** Connects and pumps frames until the stream ends. Throws on connect failure. */
  private async connectOnce(signal: AbortSignal): Promise<void> {
    const base = this.options.baseUrl.replace(/\/+$/, '');
    const url = `${base}/api/session/${encodeURIComponent(this.options.sessionId)}/events?events_after=${this.cursorValue}`;
    const response = await this.fetchImpl(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${this.options.bearerToken}`,
        Accept: 'text/event-stream',
        'Last-Event-ID': String(this.cursorValue),
        'Cache-Control': 'no-cache',
      },
      signal,
    });
    if (!response.ok) {
      const detail = await readErrorBody(response);
      throw new EventStreamProtocolError(`stream rejected with HTTP ${response.status}${detail}`);
    }
    const reader = response.body?.getReader();
    if (!reader) {
      throw new EventStreamProtocolError('stream response carried no body');
    }
    this.setStatus('open', `cursor ${this.cursorValue}`);
    const decoder = new TextDecoder('utf-8', { fatal: false });
    let buffer = '';
    let eventName: string | null = null;
    let frameId: number | null = null;
    let dataLines: string[] = [];
    try {
      for (;;) {
        if (this.stopped || signal.aborted) {
          return;
        }
        const { done, value } = await reader.read();
        if (done) {
          return;
        }
        if (!value) {
          continue;
        }
        buffer += decoder.decode(value, { stream: true });
        let newline: number;
        while ((newline = buffer.indexOf('\n')) >= 0) {
          const rawLine = buffer.slice(0, newline);
          buffer = buffer.slice(newline + 1);
          const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
          if (line === '') {
            this.dispatch(eventName, frameId, dataLines);
            eventName = null;
            frameId = null;
            dataLines = [];
            continue;
          }
          if (line.startsWith(':')) {
            continue;
          }
          if (line.startsWith('event:')) {
            eventName = line.slice(6).trim();
          } else if (line.startsWith('id:')) {
            const parsed = Number(line.slice(3).trim());
            if (Number.isInteger(parsed) && parsed >= 0) {
              frameId = parsed;
            }
          } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).replace(/^ /, ''));
          }
        }
        if (buffer.length > this.maxFrameBytes) {
          throw new EventStreamProtocolError(
            `unterminated frame exceeded ${this.maxFrameBytes} bytes`,
          );
        }
      }
    } finally {
      try {
        await reader.cancel();
      } catch {
        // Best effort.
      }
    }
  }

  private dispatch(eventName: string | null, frameId: number | null, dataLines: string[]): void {
    if (dataLines.length === 0) {
      return;
    }
    const raw = dataLines.join('\n');
    let data: Json;
    try {
      data = JSON.parse(raw) as Json;
    } catch {
      this.reportError(new EventStreamProtocolError(`frame ${frameId ?? '?'} data is not JSON`));
      return;
    }
    if (typeof data !== 'object' || data === null || Array.isArray(data)) {
      this.reportError(new EventStreamProtocolError(`frame ${frameId ?? '?'} data is not an object`));
      return;
    }
    const declared = (data as { event?: unknown }).event;
    // Daemon keep-alives are `event: heartbeat` + `data: {}` (no
    // discriminator, often no id): tolerate them instead of rejecting a
    // healthy stream. When both are present they must agree.
    const tagged = typeof declared === 'string' ? declared : eventName;
    if (typeof tagged !== 'string') {
      this.reportError(
        new EventStreamProtocolError(`frame ${frameId ?? '?'} carries no event discriminator`),
      );
      return;
    }
    if (typeof declared === 'string' && eventName !== null && eventName !== declared) {
      this.reportError(
        new EventStreamProtocolError(
          `frame ${frameId ?? '?'} event field ${eventName} disagrees with data discriminator ${declared}`,
        ),
      );
      return;
    }
    if (tagged === 'heartbeat') {
      // A heartbeat may carry an id (advances the resume cursor) or not
      // (pure keep-alive). It is never delivered to the UI.
      if (frameId !== null) {
        this.cursorValue = Math.max(this.cursorValue, frameId);
      }
      return;
    }
    if (frameId === null) {
      this.reportError(
        new EventStreamProtocolError(`frame (${tagged}) carries no id cursor; skipping`),
      );
      return;
    }
    if (frameId <= this.cursorValue) {
      // Replayed frame behind the cursor; never redeliver.
      return;
    }
    this.cursorValue = frameId;
    this.options.onEvent({ id: frameId, event: tagged, data });
  }

  private setStatus(status: EventStreamStatus, detail?: string): void {
    this.statusValue = status;
    this.options.onStatus?.(status, detail);
  }

  private reportError(error: unknown): void {
    const wrapped =
      error instanceof Error ? error : new EventStreamProtocolError(String(error));
    this.options.onError?.(wrapped);
  }
}

async function readErrorBody(response: SseResponseLike): Promise<string> {
  try {
    const reader = response.body?.getReader();
    if (reader) {
      const chunks: Uint8Array[] = [];
      let size = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done || !value) {
          break;
        }
        const remaining = ERROR_BODY_BYTES - size;
        if (remaining <= 0) {
          break;
        }
        chunks.push(value.byteLength > remaining ? value.subarray(0, remaining) : value);
        size += Math.min(value.byteLength, remaining);
        if (size >= ERROR_BODY_BYTES) {
          break;
        }
      }
      try {
        await reader.cancel();
      } catch {
        // Best effort.
      }
      return snippetOf(new TextDecoder('utf-8', { fatal: false }).decode(concat(chunks)));
    }
    return snippetOf(await response.text());
  } catch {
    return '';
  }
}

function concat(chunks: readonly Uint8Array[]): Uint8Array {
  let size = 0;
  for (const chunk of chunks) {
    size += chunk.byteLength;
  }
  const joined = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return joined;
}

function snippetOf(text: string): string {
  const snippet = text.slice(0, 200).trim();
  return snippet.length > 0 ? `: ${snippet}` : '';
}
