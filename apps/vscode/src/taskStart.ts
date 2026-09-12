// Pure task-start policy for the VS Code product surface.
//
// P0 Shadow default: the `faktor.mutationMode` setting defaults to ""
// (inherit-daemon). The client sends `mutation_mode` ONLY when the user
// configured an explicit value and NEVER fabricates `direct_compat` as a
// fallback: a daemon 409 (conflict) is a typed, actionable refusal that
// names the cause and the explicit opt-in, and it is attempted exactly
// ONCE. There is no 409 -> direct_compat downgrade path anywhere.
//
// Dependency-free (no vscode import) so scripts/selftest.mjs drives it
// with a fake client.

import { NativeApiError } from './nativeClient.ts';
import type { NativeTaskRunStarted, StartTaskRunRequest } from './nativeClient.ts';

/** The `faktor.mutationMode` setting vocabulary. `''` = inherit-daemon. */
export type MutationModeSetting = '' | 'shadow' | 'direct_compat';

export interface StartTaskSettings {
  readonly mutationMode: string;
  readonly maxTokens: number;
  readonly maxCostMicro: number;
}

/** The typed classification of a refused task start. */
export type StartFailureKind =
  | 'shadow_unregistered'
  | 'validation'
  | 'conflict'
  | 'auth'
  | 'server'
  | 'transport';

export interface StartFailure {
  readonly kind: StartFailureKind;
  readonly status: number | null;
  readonly code: string | null;
  /** User-facing, actionable, and explicit about the opt-in. */
  readonly message: string;
}

export interface StartTaskOutcome {
  readonly ok: boolean;
  readonly runId: string | null;
  readonly started: NativeTaskRunStarted | null;
  readonly failure: StartFailure | null;
}

export interface StartRunClient {
  startTaskRun(sessionId: string, request: StartTaskRunRequest): Promise<NativeTaskRunStarted>;
}

/**
 * Build the strict request body. The empty setting (inherit-daemon) OMITS
 * `mutation_mode` entirely; only an explicit user/policy value is sent.
 */
export function startTaskRequest(goal: string, settings: StartTaskSettings): StartTaskRunRequest {
  const request: StartTaskRunRequest = {
    goal,
    ...(settings.maxTokens > 0 ? { max_tokens: settings.maxTokens } : {}),
    ...(settings.maxCostMicro > 0 ? { max_cost_micro: settings.maxCostMicro } : {}),
  };
  if (settings.mutationMode === 'shadow' || settings.mutationMode === 'direct_compat') {
    return { ...request, mutation_mode: settings.mutationMode };
  }
  return request;
}

function failureOf(error: unknown, settings: StartTaskSettings): StartFailure {
  if (error instanceof NativeApiError) {
    if (error.status === 409) {
      const shadowed = settings.mutationMode !== 'direct_compat';
      if (shadowed) {
        return {
          kind: 'shadow_unregistered',
          status: error.status,
          code: error.code,
          message:
            'cannot start task: the daemon refused the shadowed run because the session is ' +
            'not registered in the daemon worktree registry (shadow mutation needs a ' +
            `registered workspace/worktree). Server said: ${error.message}. Restart the ` +
            'daemon so a registered session is created, or opt in explicitly with ' +
            '"faktor.mutationMode": "direct_compat".',
        };
      }
      return {
        kind: 'conflict',
        status: error.status,
        code: error.code,
        message:
          'cannot start task: the daemon refused the run even though direct_compat was ' +
          `configured. Server said: ${error.message}.`,
      };
    }
    const kind: StartFailureKind =
      error.status === 400
        ? 'validation'
        : error.status === 401 || error.status === 403
          ? 'auth'
          : 'server';
    return { kind, status: error.status, code: error.code, message: `cannot start task: ${error.message}` };
  }
  const message = error instanceof Error ? error.message : String(error);
  return {
    kind: 'transport',
    status: null,
    code: null,
    message: `cannot start task: ${message}`,
  };
}

/**
 * Start ONE task run: exactly one request, no downgrade retry. The outcome
 * is always returned (never thrown) so callers can ack the composer and
 * report the error deterministically.
 */
export async function startTaskRun(input: {
  readonly client: StartRunClient;
  readonly sessionId: string;
  readonly goal: string;
  readonly settings: StartTaskSettings;
  readonly onStarted: (started: NativeTaskRunStarted) => void;
  readonly onFailure: (failure: StartFailure) => void;
}): Promise<StartTaskOutcome> {
  const request = startTaskRequest(input.goal, input.settings);
  try {
    const started = await input.client.startTaskRun(input.sessionId, request);
    input.onStarted(started);
    return { ok: true, runId: started.run_id, started, failure: null };
  } catch (error) {
    const failure = failureOf(error, input.settings);
    input.onFailure(failure);
    return { ok: false, runId: null, started: null, failure };
  }
}
