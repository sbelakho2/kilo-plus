// The Task cockpit: ONE persistent panel assembled from the native state
// the extension already holds — acceptance criteria, plan/DAG steps,
// children, current phase, blockers (additive fields when the daemon
// serves them), verification status, evidence refs (retrievable) and
// durable spend. Pure and dependency-free so scripts/selftest.mjs renders
// every section from a mock native payload.
//
// The builder never invents data: a missing field yields an explicitly
// empty section (`present: false`, one "none" line), never a fabricated
// value. Every string is bounded; evidence ids parse only from explicit
// `evidence:<n>` refs.

import type { AgentSummary, TaskSummary, UsageSummary, VerificationSummary } from './state';

export interface CockpitEvidenceRef {
  readonly id: number | null;
  readonly label: string;
}

export interface CockpitStepView {
  readonly id: string;
  readonly summary: string;
  readonly status: string;
  readonly dependsOn: readonly string[];
  readonly childIds: readonly string[];
}

export interface CockpitChildView {
  readonly childId: string;
  readonly state: string;
  readonly itemId: string | null;
  readonly itemKind: string | null;
  readonly worktreeId: number | null;
  readonly ownership: string;
  readonly model: string | null;
  readonly provider: string | null;
  readonly reasoning: boolean | null;
  readonly thinking: boolean | null;
  readonly budget: number | null;
  readonly progress: string | null;
  readonly result: string | null;
  readonly presentation: 'foreground' | 'background';
}

export interface CockpitVerificationView {
  readonly status: string;
  readonly criteriaPassed: number;
  readonly criteriaTotal: number;
  readonly checksFailed: number;
  readonly owed: number;
  readonly failedChecks: number;
}

export interface CockpitSpendView {
  readonly spentTokens: number | null;
  readonly maxTokens: number | null;
  readonly spentCostMicro: number;
  readonly maxCostMicro: number | null;
  readonly openReservedMicro: number;
}

/** One tournament candidate (mirrors the durable native candidate rows). */
export interface CockpitTournamentCandidateView {
  readonly childId: string;
  readonly state: string;
  readonly verification: number | null;
  readonly verificationPass: boolean | null;
  readonly reviewRank: string | null;
  readonly reviewer: string | null;
  readonly costMicro: number;
  readonly wallMs: number;
  readonly winner: boolean;
}

/** The durable tournament as the cockpit renders it (never auto-integrates). */
export interface CockpitTournamentView {
  readonly id: string;
  readonly state: string;
  /** The engine can still act: open | deciding. */
  readonly open: boolean;
  /** Decide waits until every candidate settled (the engine's rule). */
  readonly canDecide: boolean;
  readonly winner: string | null;
  readonly criteria: readonly string[];
  readonly candidates: readonly CockpitTournamentCandidateView[];
}

export interface CockpitView {
  readonly goal: string;
  readonly state: string;
  readonly phase: string;
  readonly acceptanceCriteria: readonly string[];
  readonly steps: readonly CockpitStepView[];
  readonly children: readonly CockpitChildView[];
  readonly blockers: readonly string[];
  readonly verification: CockpitVerificationView;
  readonly evidence: readonly CockpitEvidenceRef[];
  readonly spend: CockpitSpendView | null;
  readonly tournament: CockpitTournamentView | null;
  /** The Task-mode completion contract + durable step statuses (null when
   * the run carries no contract). Provenance is explicit (`daemon`,
   * `derived` or `unavailable`) and never fabricated. */
  readonly completion: CockpitCompletionView | null;
}

/** The completion-contract block as the cockpit renders it. */
export interface CockpitCompletionView {
  readonly includeCommit: boolean;
  readonly includePush: boolean;
  readonly includePr: boolean;
  readonly steps: readonly {
    readonly step: string;
    readonly status: string;
    readonly detail: string | null;
  }[];
  readonly source: string;
  readonly reason: string | null;
}

/** One state-gated control a cockpit section renders. */
export interface CockpitAction {
  readonly key: 'decide' | 'abort';
  readonly label: string;
  readonly enabled: boolean;
}

export interface CockpitSection {
  readonly key: string;
  readonly title: string;
  readonly present: boolean;
  readonly lines: readonly string[];
  readonly evidence: readonly CockpitEvidenceRef[];
  /** State-gated controls (tournament decide/abort), when the section owns any. */
  readonly actions?: readonly CockpitAction[];
}

/** Minimal structural view of the durable native tournament payload. */
export interface CockpitNativeTournament {
  readonly id: string;
  readonly state: string;
  readonly winner: string | null;
  readonly criteria: readonly { readonly id: string; readonly spec: string }[];
  readonly candidates: readonly {
    readonly childId: string;
    readonly state: string;
    readonly verification: number | null;
    readonly verificationPass: boolean | null;
    readonly reviewRank: string | null;
    readonly reviewer: string | null;
    readonly costMicro: number;
    readonly wallMs: number;
  }[];
}

/**
 * Pure tournament projection: `open` while the engine can still act and
 * `canDecide` only once every candidate has settled (the engine's own rule).
 * Integration is never automatic — a winner is a proposal.
 */
export function tournamentViewOf(
  tournament: CockpitNativeTournament | null,
): CockpitTournamentView | null {
  if (tournament === null) {
    return null;
  }
  const open = tournament.state === 'open' || tournament.state === 'deciding';
  const candidates = tournament.candidates.map((candidate) => ({
    childId: candidate.childId,
    state: candidate.state,
    verification: candidate.verification,
    verificationPass: candidate.verificationPass,
    reviewRank: candidate.reviewRank,
    reviewer: candidate.reviewer,
    costMicro: candidate.costMicro,
    wallMs: candidate.wallMs,
    winner: tournament.winner !== null && candidate.childId === tournament.winner,
  }));
  return {
    id: tournament.id,
    state: tournament.state,
    open,
    canDecide: open && candidates.length > 0 && candidates.every((c) => c.state !== 'running'),
    winner: tournament.winner,
    criteria: tournament.criteria.map((criterion) => criterion.spec),
    candidates,
  };
}

/** Minimal structural view of the task-verification wire payload. */
export interface CockpitTaskVerification {
  readonly records: readonly {
    readonly status: string;
    readonly criteria: readonly { readonly criterionKey: string; readonly passed: boolean; readonly evidence: string | null }[];
    readonly checks: readonly { readonly check: string; readonly status: string; readonly required: boolean }[];
  }[];
}

export interface CockpitInput {
  readonly task: TaskSummary | null;
  readonly agents: readonly AgentSummary[];
  readonly verification: VerificationSummary | null;
  readonly usage: UsageSummary | null;
  readonly taskVerification: CockpitTaskVerification | null;
  /** The durable tournament of the session, when one exists. */
  readonly tournament?: CockpitTournamentView | null;
}

const MAX_LINES = 64;
const MAX_TEXT = 240;

function clamp(value: string): string {
  return value.length > MAX_TEXT ? `${value.slice(0, MAX_TEXT)}…` : value;
}

function textOf(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? clamp(value.trim()) : null;
}

/** Extract a current-phase label from a native progress record, if any. */
export function phaseOf(progress: unknown): string | null {
  if (typeof progress !== 'object' || progress === null || Array.isArray(progress)) {
    return null;
  }
  const record = progress as Record<string, unknown>;
  for (const key of ['phase', 'stage', 'current_phase', 'currentPhase', 'label']) {
    const value = textOf(record[key]);
    if (value !== null) {
      return value;
    }
  }
  return null;
}

function blockersOf(progress: unknown): string[] {
  if (typeof progress !== 'object' || progress === null || Array.isArray(progress)) {
    return [];
  }
  const record = progress as Record<string, unknown>;
  for (const key of ['blockers', 'blocked_on', 'blockedOn', 'blocker']) {
    const value = record[key];
    if (typeof value === 'string' && value.trim().length > 0) {
      return [clamp(value.trim())];
    }
    if (Array.isArray(value)) {
      return value
        .map((entry) => {
          if (typeof entry === 'string') {
            return textOf(entry);
          }
          if (typeof entry === 'object' && entry !== null) {
            const object = entry as Record<string, unknown>;
            return (
              textOf(object.detail) ??
              textOf(object.message) ??
              textOf(object.summary) ??
              textOf(object.id)
            );
          }
          return null;
        })
        .filter((entry): entry is string => entry !== null)
        .slice(0, MAX_LINES);
    }
  }
  return [];
}

/** Parse `evidence:41`, `evidence/41` or `#41` refs; free text is kept. */
export function evidenceRefOf(raw: string): CockpitEvidenceRef {
  const match = /(?:^|[^\w])evidence[:#/](\d+)(?![\d])/.exec(raw);
  return { id: match ? Number(match[1]) : null, label: clamp(raw) };
}

function boundedJson(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  try {
    const encoded = JSON.stringify(value);
    return encoded === undefined ? null : clamp(encoded);
  } catch {
    return null;
  }
}

export function buildCockpit(input: CockpitInput): CockpitView | null {
  const { task, agents, verification, usage, taskVerification } = input;
  const tournament = input.tournament ?? null;
  // Background children are TUCKED: they render after the foreground ones
  // (the durable presentation field decides; never the state heuristics).
  const allChildren = agents.filter((agent) => agent.kind === 'child');
  const children = allChildren.filter((child) => child.presentation !== 'background').concat(
    allChildren.filter((child) => child.presentation === 'background'),
  );
  if (task === null && children.length === 0 && verification === null && tournament === null) {
    return null;
  }

  // Acceptance criteria: the task row's explicit list first; otherwise the
  // verification record's criterion keys (never invented).
  const explicitCriteria = (task?.acceptanceCriteria ?? []).filter(
    (entry) => entry.trim().length > 0,
  );
  const recordCriteria = (taskVerification?.records ?? []).flatMap((record) =>
    record.criteria.map((criterion) => criterion.criterionKey),
  );
  const acceptanceCriteria = (explicitCriteria.length > 0 ? explicitCriteria : recordCriteria).slice(
    0,
    MAX_LINES,
  );

  // Plan / DAG steps: explicit plan steps when served, else the milestone
  // lists; children link by item id.
  const planSteps = task?.plan ?? [];
  const steps: CockpitStepView[] = planSteps.map((step) => ({
    id: step.id,
    summary: step.summary,
    status: step.state,
    dependsOn: step.dependsOn,
    childIds: children.filter((child) => child.itemId === step.id).map((child) => child.agentId),
  }));
  const plannedIds = new Set(steps.map((step) => step.id));
  for (const completed of task?.completed ?? []) {
    if (!plannedIds.has(completed)) {
      steps.push({
        id: completed,
        summary: completed,
        status: 'done',
        dependsOn: [],
        childIds: children.filter((child) => child.itemId === completed).map((child) => child.agentId),
      });
    }
  }
  for (const open of task?.open ?? []) {
    if (!plannedIds.has(open)) {
      steps.push({
        id: open,
        summary: open,
        status: 'open',
        dependsOn: [],
        childIds: children.filter((child) => child.itemId === open).map((child) => child.agentId),
      });
    }
  }

  // Blockers: explicit additive fields on the task view first, then the
  // progress record, then visibly-blocked children.
  const blockers: string[] = [];
  for (const explicit of task?.blockers ?? []) {
    blockers.push(clamp(explicit));
  }
  for (const derived of blockersOf(task?.progress)) {
    blockers.push(derived);
  }
  for (const child of children) {
    if (child.state.trim().toLowerCase() === 'blocked') {
      blockers.push(`child ${child.agentId} is blocked on ${child.itemId ?? 'its work item'}`);
    }
  }

  // Verification: the durable record summary when available, else the
  // session verification view.
  let criteriaPassed = 0;
  let criteriaTotal = 0;
  let checksFailed = 0;
  let recordStatus: string | null = null;
  for (const record of taskVerification?.records ?? []) {
    recordStatus = record.status;
    for (const criterion of record.criteria) {
      criteriaTotal += 1;
      if (criterion.passed) {
        criteriaPassed += 1;
      }
    }
    for (const check of record.checks) {
      if (check.status.toLowerCase() === 'failed') {
        checksFailed += 1;
      }
    }
  }
  const owed = verification?.owed.length ?? 0;
  const failedChecks = verification?.failedChecks.length ?? 0;
  const verificationView: CockpitVerificationView = {
    status:
      recordStatus ??
      (criteriaTotal > 0 && criteriaPassed === criteriaTotal
        ? 'passed'
        : failedChecks > 0 || checksFailed > 0
          ? 'failed'
          : owed > 0
            ? 'pending'
            : task?.state ?? 'unknown'),
    criteriaPassed,
    criteriaTotal,
    checksFailed,
    owed,
    failedChecks,
  };

  // Evidence refs: verification criteria evidence + check summaries + any
  // explicit refs carried on the task view.
  const evidence: CockpitEvidenceRef[] = [];
  const seenEvidence = new Set<string>();
  const pushEvidence = (raw: string | null | undefined): void => {
    if (typeof raw !== 'string' || raw.trim().length === 0) {
      return;
    }
    const ref = evidenceRefOf(raw.trim());
    const dedupe = `${ref.id ?? 'text'}:${ref.label}`;
    if (!seenEvidence.has(dedupe) && evidence.length < MAX_LINES) {
      seenEvidence.add(dedupe);
      evidence.push(ref);
    }
  };
  for (const record of taskVerification?.records ?? []) {
    for (const criterion of record.criteria) {
      pushEvidence(criterion.evidence);
    }
  }
  for (const ref of task?.evidenceRefs ?? []) {
    pushEvidence(ref);
  }

  const spend: CockpitSpendView | null = usage
    ? {
        spentTokens: usage.tokens,
        maxTokens: task?.budget?.maxTokens ?? null,
        spentCostMicro: usage.spentMicro,
        maxCostMicro: usage.maxMicro,
        openReservedMicro: usage.openMicro,
      }
    : task?.budget
      ? {
          spentTokens: task.budget.spentTokens,
          maxTokens: task.budget.maxTokens,
          spentCostMicro: task.budget.spentCostMicro,
          maxCostMicro: task.budget.maxCostMicro,
          openReservedMicro: task.budget.openReservedMicro,
        }
      : null;

  return {
    goal: task?.goal ?? children[0]?.goal ?? '',
    state: task?.state ?? 'unknown',
    phase: task?.phase ?? phaseOf(task?.progress) ?? task?.state ?? 'unknown',
    acceptanceCriteria,
    steps,
    children: children.map((child) => ({
      childId: child.agentId,
      state: child.state,
      itemId: child.itemId,
      itemKind: child.itemKind,
      worktreeId: child.worktreeId,
      ownership: child.ownership,
      model: child.model,
      provider: child.provider,
      reasoning: child.reasoning,
      thinking: child.thinking,
      budget: child.budget,
      progress: boundedJson(child.progress),
      result: boundedJson(child.result),
      presentation: child.presentation === 'background' ? 'background' : 'foreground',
    })),
    blockers,
    verification: verificationView,
    evidence,
    spend,
    tournament,
    completion: task?.completion ?? null,
  };
}

/** The render plan of the persistent cockpit panel, in fixed order. */
export function cockpitSections(view: CockpitView): CockpitSection[] {
  const sections: CockpitSection[] = [];
  sections.push({
    key: 'acceptance',
    title: 'Acceptance criteria',
    present: view.acceptanceCriteria.length > 0,
    lines:
      view.acceptanceCriteria.length > 0
        ? view.acceptanceCriteria.map((criterion) => `• ${criterion}`)
        : ['none'],
    evidence: [],
  });
  sections.push({
    key: 'plan',
    title: 'Plan / DAG steps',
    present: view.steps.length > 0,
    lines:
      view.steps.length > 0
        ? view.steps.map((step) => {
            const deps = step.dependsOn.length > 0 ? ` (after ${step.dependsOn.join(', ')})` : '';
            const kids = step.childIds.length > 0 ? ` → ${step.childIds.join(', ')}` : '';
            return `[${step.status}] ${step.id}: ${step.summary}${deps}${kids}`;
          })
        : ['none'],
    evidence: [],
  });
  sections.push({
    key: 'completion',
    title: 'Completion contract',
    present: view.completion !== null,
    lines:
      view.completion === null
        ? ['none (plain task; no conditional commit/push/PR steps)']
        : [
            `requested: ${
              [
                view.completion.includeCommit ? 'commit' : null,
                view.completion.includePush ? 'push' : null,
                view.completion.includePr ? 'pr' : null,
              ]
                .filter((step): step is string => step !== null)
                .join(', ') || 'none'
            }`,
            ...view.completion.steps.map(
              (step) =>
                `[${step.status}] ${step.step}${
                  step.detail !== null && step.detail.length > 0 ? ` — ${step.detail}` : ''
                }`,
            ),
            `status source: ${view.completion.source}${
              view.completion.reason !== null ? ` (${view.completion.reason})` : ''
            }`,
          ],
    evidence: [],
  });
  sections.push({
    key: 'children',
    title: 'Children',
    present: view.children.length > 0,
    lines:
      view.children.length > 0
        ? view.children.map((child) => {
            const bits = [
              `${child.childId} [${child.state}]`,
              child.presentation === 'background' ? 'background (dimmed)' : null,
              child.itemId ? `item ${child.itemId}${child.itemKind ? ` (${child.itemKind})` : ''}` : null,
              child.worktreeId !== null ? `worktree ${child.worktreeId}` : null,
              `ownership ${child.ownership}`,
              child.model ? `model ${child.model}` : null,
              child.provider ? `provider ${child.provider}` : null,
              child.reasoning !== null ? `reasoning ${child.reasoning ? 'yes' : 'no'}` : null,
              child.thinking !== null ? `thinking ${child.thinking ? 'yes' : 'no'}` : null,
              child.budget !== null ? `budget ${child.budget}` : null,
            ].filter((bit): bit is string => bit !== null);
            return bits.join(' · ');
          })
        : ['none'],
    evidence: [],
  });
  sections.push({
    key: 'tournament',
    title: 'Tournament',
    present: view.tournament !== null,
    lines:
      view.tournament === null
        ? ['none']
        : [
            `tournament ${view.tournament.id} [${view.tournament.state}] winner ${
              view.tournament.winner ?? '—'
            }`,
            `criteria ${view.tournament.criteria.length}: ${
              view.tournament.criteria.join('; ') || '—'
            }`,
            ...view.tournament.candidates.map((candidate) => {
              const verdict =
                candidate.verification === null
                  ? 'unverified'
                  : candidate.verificationPass === true
                    ? 'pass'
                    : 'fail';
              const review =
                candidate.reviewRank === null
                  ? 'no review'
                  : `${candidate.reviewRank}${
                      candidate.reviewer !== null ? ` (${candidate.reviewer})` : ''
                    }`;
              return `${candidate.winner ? '* ' : ''}${candidate.childId} [${candidate.state}] ${verdict} ${review} cost ${candidate.costMicro} wall ${candidate.wallMs}ms`;
            }),
          ],
    evidence: [],
    actions:
      view.tournament === null
        ? []
        : [
            { key: 'decide', label: 'Decide winner', enabled: view.tournament.canDecide },
            { key: 'abort', label: 'Abort', enabled: view.tournament.open },
          ],
  });
  sections.push({
    key: 'phase',
    title: 'Current phase',
    present: view.phase.length > 0 && view.phase !== 'unknown',
    lines: [`${view.phase} — ${view.state}`],
    evidence: [],
  });
  sections.push({
    key: 'blockers',
    title: 'Blockers',
    present: view.blockers.length > 0,
    lines: view.blockers.length > 0 ? view.blockers.map((blocker) => `! ${blocker}`) : ['none'],
    evidence: [],
  });
  sections.push({
    key: 'verification',
    title: 'Verification',
    present: true,
    lines: [
      `status ${view.verification.status}`,
      `criteria ${view.verification.criteriaPassed}/${view.verification.criteriaTotal} passed`,
      `checks failed ${view.verification.checksFailed}`,
      `owed ${view.verification.owed} · failed checks ${view.verification.failedChecks}`,
    ],
    evidence: [],
  });
  sections.push({
    key: 'evidence',
    title: 'Evidence',
    present: view.evidence.length > 0,
    lines: view.evidence.length > 0 ? view.evidence.map((ref) => ref.label) : ['none'],
    evidence: view.evidence,
  });
  sections.push({
    key: 'spend',
    title: 'Spend',
    present: view.spend !== null,
    lines: view.spend
      ? [
          `tokens ${view.spend.spentTokens ?? '—'}${
            view.spend.maxTokens !== null ? ` / ${view.spend.maxTokens}` : ''
          }`,
          `cost ${(view.spend.spentCostMicro / 1_000_000).toFixed(4)}${
            view.spend.maxCostMicro !== null
              ? ` / ${(view.spend.maxCostMicro / 1_000_000).toFixed(4)}`
              : ''
          }`,
          `reserved ${(view.spend.openReservedMicro / 1_000_000).toFixed(4)}`,
        ]
      : ['none'],
    evidence: [],
  });
  return sections;
}
