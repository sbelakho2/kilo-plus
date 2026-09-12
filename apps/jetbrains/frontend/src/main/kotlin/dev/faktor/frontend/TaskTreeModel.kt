// Pure view-model construction for the rich task-tree panel. Everything
// here is display-independent (no Swing types except the deterministic
// pixel avatars) so the smoke can build and assert EVERY section from a
// canned native payload without a display, and the panel classes only map
// this model onto components.
//
// Sources are the native surfaces already served by the daemon:
//   GET /session/{id}/projection          -> goal/state/phase fallback
//   GET /native/session/{id}/tasks        -> task view (acceptance/plan/...)
//   GET /native/agents                    -> children + blockers + result
//   GET /native/orchestrator/graph        -> durable plan step/DAG order
//   GET /models                           -> provider/reasoning metadata
//   GET /native/session/{id}/verification -> owed/failed checks
//   GET .../tasks/{task_id}/verification  -> criteria + evidence refs
//   GET /native/session/{id}/usage        -> durable spend
//   GET /native/session/{id}/tournament/{tid} -> candidates/winner state
package dev.faktor.frontend

import dev.faktor.shared.NativeAgent
import dev.faktor.shared.NativeAgentProgress
import dev.faktor.shared.NativeBlocker
import dev.faktor.shared.NativeChildResult
import dev.faktor.shared.NativeModelInfo
import dev.faktor.shared.NativeOrchestratorGraph
import dev.faktor.shared.NativeProjection
import dev.faktor.shared.NativeSessionUsage
import dev.faktor.shared.NativeTaskVerification
import dev.faktor.shared.NativeTaskView
import dev.faktor.shared.NativeTournament
import dev.faktor.shared.NativeTournamentCandidate
import dev.faktor.shared.NativeTournamentCriterion
import dev.faktor.shared.NativeVerificationView

/** Max evidence refs surfaced by one task tree (bounded like the cockpit). */
const val MAX_TREE_EVIDENCE = 64

/** Max label chars of one evidence ref (free text is kept but bounded). */
const val MAX_EVIDENCE_LABEL = 300

/** A blocker action that applies to one blocked child. */
enum class BlockerAction(val label: String) {
    RESUME("Resume"),
    RETRY("Retry"),
    PERMISSION_ALLOW("Allow"),
    PERMISSION_DENY("Deny"),
    CANCEL("Cancel")
}

/** One evidence reference: `evidence:41` / `evidence/41` / `#41` or free text. */
data class EvidenceRef(val id: Long?, val label: String)

/** `evidence:<n>` ref parsing, mirroring the VS Code cockpit vocabulary. */
object EvidenceRefs {
    private val PATTERN = Regex("(?:^|[^A-Za-z0-9_])evidence[:#/]([0-9]+)(?![0-9])")

    fun parse(raw: String): EvidenceRef {
        val bounded = raw.trim().take(MAX_EVIDENCE_LABEL)
        val match = PATTERN.find(bounded)
        val id = if (match == null) null else match.groupValues[1].toLongOrNull()
        return EvidenceRef(id, bounded)
    }
}

/** One blocked child with the actions that apply to its blocker kind. */
data class BlockerRow(
    val childId: String,
    val kind: String,
    val reason: String,
    val dependency: String?,
    val resolution: String?,
    val lastProgressMs: Long?,
    val presence: PixelPresence,
    val actions: List<BlockerAction>
)

/** One plan/DAG step with the children currently driving it. */
data class StepNode(
    val id: String,
    val summary: String,
    val state: String,
    val dependsOn: List<String>,
    val childIds: List<String>
)

/** One child agent with every native field the listing serves. */
data class ChildNode(
    val childId: String,
    val runId: String,
    val sessionId: Long,
    val itemId: String?,
    val itemKind: String?,
    val goal: String,
    val state: String,
    val presence: PixelPresence,
    val blocker: NativeBlocker?,
    val model: String?,
    val provider: String?,
    val reasoning: Boolean?,
    val tools: Boolean?,
    val ownership: String,
    val worktreeId: Long,
    val budgetMaxTokens: Long?,
    val spentTokens: Long?,
    val spentCostMicro: Long?,
    val maxCostMicro: Long?,
    val remainingTokens: Long?,
    val remainingCostMicro: Long?,
    val progress: NativeAgentProgress?,
    val result: NativeChildResult?
)

/** Verification status of the task tree (criteria/checks/owed). */
data class VerificationSummary(
    val status: String,
    val criteriaPassed: Int,
    val criteriaTotal: Int,
    val failedChecks: Int,
    val owed: Int,
    val recordStatus: String?
)

/** Durable spend of the session task (tokens + microUSD, remaining computed). */
data class SpendSummary(
    val spentTokens: Long?,
    val maxTokens: Long?,
    val spentCostMicro: Long?,
    val maxCostMicro: Long?,
    val openReservedMicro: Long,
    val remainingTokens: Long?,
    val remainingCostMicro: Long?,
    val durable: Boolean
)

/** One tournament candidate (verification/review verdicts + measured axes). */
data class TournamentCandidateView(
    val childId: String,
    val state: String,
    val presence: PixelPresence,
    val worktree: String,
    val baseRevision: String,
    val verification: Long?,
    val verificationPass: Boolean?,
    val reviewRank: String?,
    val reviewer: String?,
    val costMicro: Long,
    val wallMs: Long,
    val winner: Boolean
)

/** The tournament view (candidates, winner state). */
data class TournamentView(
    val id: String,
    val runFamily: String,
    val goal: String,
    val criteria: List<NativeTournamentCriterion>,
    val candidates: List<TournamentCandidateView>,
    val winner: String?,
    val state: String
) {
    val decided: Boolean
        get() = winner != null
}

/** Everything the rich tree panel renders, built purely from native DTOs. */
data class TaskTreeModel(
    val goal: String,
    val state: String,
    val phase: String,
    val acceptanceCriteria: List<String>,
    val steps: List<StepNode>,
    val children: List<ChildNode>,
    val blockers: List<BlockerRow>,
    val taskBlockers: List<String>,
    val verification: VerificationSummary,
    val evidence: List<EvidenceRef>,
    val spend: SpendSummary?,
    val tournament: TournamentView?
)

/** Pure builder over the native DTOs. */
object TaskTree {

    fun build(
        projection: NativeProjection? = null,
        task: NativeTaskView? = null,
        agents: List<NativeAgent> = emptyList(),
        graph: NativeOrchestratorGraph? = null,
        catalog: List<NativeModelInfo> = emptyList(),
        verification: NativeVerificationView? = null,
        taskVerification: NativeTaskVerification? = null,
        usage: NativeSessionUsage? = null,
        childUsage: Map<String, NativeSessionUsage> = emptyMap(),
        tournament: NativeTournament? = null
    ): TaskTreeModel {
        val children = agents
            .filter { it.kind == "child" }
            .map { child ->
                childNode(child, catalog, childUsage[child.sessionId.toString()])
            }
        val goal = task?.goal?.takeIf { it.isNotEmpty() }
            ?: graph?.goal?.takeIf { it.isNotEmpty() }
            ?: children.firstOrNull()?.goal?.takeIf { it.isNotEmpty() }
            ?: ""
        val state = task?.state?.takeIf { it.isNotEmpty() }
            ?: projection?.machine
            ?: "unknown"
        val phase = task?.phase?.takeIf { it.isNotEmpty() }
            ?: projection?.label?.takeIf { it.isNotEmpty() }
            ?: state
        val criteria = acceptanceCriteria(task, taskVerification)
        return TaskTreeModel(
            goal = goal,
            state = state,
            phase = phase,
            acceptanceCriteria = criteria,
            steps = steps(task, graph, children),
            children = children,
            blockers = children.mapNotNull { blockerRow(it) },
            taskBlockers = task?.blockers ?: emptyList(),
            verification = verification(verification, taskVerification, task),
            evidence = evidence(task, taskVerification),
            spend = spend(usage, task),
            tournament = tournament?.let { tournamentView(it) }
        )
    }

    // ------------------------------------------------------------ children

    private fun childNode(
        agent: NativeAgent,
        catalog: List<NativeModelInfo>,
        usage: NativeSessionUsage?
    ): ChildNode {
        val info = if (agent.model == null) null else catalog.firstOrNull { it.model == agent.model }
        var spentTokens: Long? = null
        var spentCostMicro: Long? = null
        var maxCostMicro: Long? = null
        var openReserved: Long = 0
        if (usage != null) {
            for (taskUsage in usage.tasks) {
                val budget = taskUsage.budget
                budget.spentTokens?.let { spentTokens = (spentTokens ?: 0L) + it }
                spentCostMicro = (spentCostMicro ?: 0L) + budget.spentCostMicro
                budget.maxCostMicro?.let { maxCostMicro = (maxCostMicro ?: 0L) + it }
                openReserved += budget.openReservedMicro
            }
            if (usage.tokens > 0 && spentTokens == null) spentTokens = usage.tokens
        }
        val maxTokens = agent.budget
        // kotlinc 1.3 (CI fallback toolchain) cannot smart-cast captured
        // vars inside closures: bind the current values first.
        val spentTokensNow = spentTokens
        val spentCostNow = spentCostMicro
        val remainingTokens = if (maxTokens == null || spentTokensNow == null) {
            null
        } else {
            (maxTokens - spentTokensNow).coerceAtLeast(0L)
        }
        val remainingCost = if (maxCostMicro == null || spentCostNow == null) {
            null
        } else {
            (maxCostMicro - spentCostNow - openReserved).coerceAtLeast(0L)
        }
        return ChildNode(
            childId = agent.agentId,
            runId = agent.runId,
            sessionId = agent.sessionId,
            itemId = agent.itemId,
            itemKind = agent.itemKind,
            goal = agent.goal,
            state = agent.state,
            presence = PixelAgents.presence(agent.agentId, agent.state),
            blocker = agent.blocker,
            model = agent.model,
            provider = info?.provider,
            reasoning = info?.reasoning,
            tools = info?.tools,
            ownership = agent.ownership,
            worktreeId = agent.worktreeId,
            budgetMaxTokens = maxTokens,
            spentTokens = spentTokens,
            spentCostMicro = spentCostMicro,
            maxCostMicro = maxCostMicro,
            remainingTokens = remainingTokens,
            remainingCostMicro = remainingCost,
            progress = agent.progress,
            result = agent.result
        )
    }

    private fun blockerRow(child: ChildNode): BlockerRow? {
        val blocker = child.blocker ?: return null
        val actions = when (blocker.kind) {
            "permission" -> listOf(
                BlockerAction.RESUME,
                BlockerAction.PERMISSION_ALLOW,
                BlockerAction.PERMISSION_DENY,
                BlockerAction.RETRY
            )
            "dependency" -> listOf(BlockerAction.RESUME, BlockerAction.RETRY)
            "budget" -> listOf(BlockerAction.RESUME, BlockerAction.RETRY)
            else -> listOf(BlockerAction.RESUME, BlockerAction.RETRY)
        }
        return BlockerRow(
            childId = child.childId,
            kind = blocker.kind,
            reason = blocker.reason,
            dependency = blocker.dependency,
            resolution = blocker.resolution,
            lastProgressMs = blocker.lastProgressMs,
            presence = child.presence,
            actions = actions
        )
    }

    // --------------------------------------------------------------- steps

    private fun steps(
        task: NativeTaskView?,
        graph: NativeOrchestratorGraph?,
        children: List<ChildNode>
    ): List<StepNode> {
        val plan = task?.plan ?: emptyList()
        if (plan.isNotEmpty()) {
            return plan.map { step ->
                StepNode(
                    id = step.id,
                    summary = step.summary,
                    state = step.state,
                    dependsOn = step.dependsOn,
                    childIds = children.filter { it.itemId == step.id }.map { it.childId }
                )
            }
        }
        if (graph != null && graph.workItems.isNotEmpty()) {
            return graph.workItems.mapIndexed { index, item ->
                StepNode(
                    id = item.itemId,
                    summary = item.kind,
                    state = item.state,
                    dependsOn = emptyList(),
                    childIds = graph.children
                        .filter { it.planStepIndex == index }
                        .map { it.childId }
                )
            }
        }
        if (task != null) {
            return task.milestones.completed.map { text ->
                StepNode(text, text, "done", emptyList(), children.filter { it.itemId == text }.map { it.childId })
            } + task.milestones.open.map { text ->
                StepNode(text, text, "open", emptyList(), children.filter { it.itemId == text }.map { it.childId })
            }
        }
        return emptyList()
    }

    // ----------------------------------------------------- criteria/evidence

    private fun acceptanceCriteria(
        task: NativeTaskView?,
        taskVerification: NativeTaskVerification?
    ): List<String> {
        val explicit = (task?.acceptanceCriteria ?: emptyList()).filter { it.trim().isNotEmpty() }
        if (explicit.isNotEmpty()) return explicit
        val fromRecords = ArrayList<String>()
        for (record in taskVerification?.records ?: emptyList()) {
            for (criterion in record.criteria) {
                if (!fromRecords.contains(criterion.criterionKey)) {
                    fromRecords.add(criterion.criterionKey)
                }
                if (fromRecords.size >= MAX_TREE_EVIDENCE) return fromRecords
            }
        }
        return fromRecords
    }

    private fun evidence(
        task: NativeTaskView?,
        taskVerification: NativeTaskVerification?
    ): List<EvidenceRef> {
        val out = ArrayList<EvidenceRef>()
        val seen = HashSet<String>()
        fun push(raw: String?) {
            if (raw == null || raw.trim().isEmpty()) return
            if (out.size >= MAX_TREE_EVIDENCE) return
            val ref = EvidenceRefs.parse(raw)
            val dedupe = (ref.id?.toString() ?: "text") + ":" + ref.label
            if (seen.add(dedupe)) out.add(ref)
        }
        for (record in taskVerification?.records ?: emptyList()) {
            for (criterion in record.criteria) push(criterion.evidence)
            for (summary in record.checkSummaries) push(summary)
        }
        for (ref in task?.evidenceRefs ?: emptyList()) push(ref)
        return out
    }

    // --------------------------------------------------------- verification

    private fun verification(
        verification: NativeVerificationView?,
        taskVerification: NativeTaskVerification?,
        task: NativeTaskView?
    ): VerificationSummary {
        var passed = 0
        var total = 0
        var recordStatus: String? = null
        for (record in taskVerification?.records ?: emptyList()) {
            recordStatus = record.status
            passed += record.criteriaPassed
            total += record.criteriaTotal
        }
        val failed = verification?.failedChecks?.size ?: 0
        val owed = verification?.owed?.size ?: 0
        val status = recordStatus ?: when {
            total > 0 && passed == total -> "passed"
            failed > 0 -> "failed"
            owed > 0 -> "pending"
            else -> task?.state ?: "unknown"
        }
        return VerificationSummary(status, passed, total, failed, owed, recordStatus)
    }

    // ---------------------------------------------------------------- spend

    private fun spend(usage: NativeSessionUsage?, task: NativeTaskView?): SpendSummary? {
        if (usage != null) {
            var spentTokens: Long? = null
            var maxTokens: Long? = null
            var spentCost: Long? = null
            var maxCost: Long? = null
            var open = 0L
            for (taskUsage in usage.tasks) {
                val budget = taskUsage.budget
                budget.spentTokens?.let { spentTokens = (spentTokens ?: 0L) + it }
                budget.maxTokens?.let { maxTokens = (maxTokens ?: 0L) + it }
                spentCost = (spentCost ?: 0L) + budget.spentCostMicro
                budget.maxCostMicro?.let { maxCost = (maxCost ?: 0L) + it }
                open += budget.openReservedMicro
            }
            if (spentTokens == null && usage.tokens > 0) spentTokens = usage.tokens
            return SpendSummary(
                spentTokens = spentTokens,
                maxTokens = maxTokens,
                spentCostMicro = spentCost,
                maxCostMicro = maxCost,
                openReservedMicro = open,
                remainingTokens = remaining(maxTokens, spentTokens, 0L),
                remainingCostMicro = remaining(maxCost, spentCost, open),
                durable = true
            )
        }
        val budget = task?.budget ?: return null
        return SpendSummary(
            spentTokens = budget.spentTokens,
            maxTokens = budget.maxTokens,
            spentCostMicro = budget.spentCostMicro,
            maxCostMicro = budget.maxCostMicro,
            openReservedMicro = budget.openReservedMicro,
            remainingTokens = remaining(budget.maxTokens, budget.spentTokens, 0L),
            remainingCostMicro = remaining(budget.maxCostMicro, budget.spentCostMicro, budget.openReservedMicro),
            durable = true
        )
    }

    private fun remaining(max: Long?, spent: Long?, open: Long): Long? {
        if (max == null || spent == null) return null
        return (max - spent - open).coerceAtLeast(0L)
    }

    // ----------------------------------------------------------- tournament

    fun tournamentView(tournament: NativeTournament): TournamentView =
        TournamentView(
            id = tournament.id,
            runFamily = tournament.runFamily,
            goal = tournament.goal,
            criteria = tournament.criteria,
            candidates = tournament.candidates.map { candidate ->
                candidateView(candidate, tournament.winner)
            },
            winner = tournament.winner,
            state = tournament.state
        )

    private fun candidateView(
        candidate: NativeTournamentCandidate,
        winner: String?
    ): TournamentCandidateView = TournamentCandidateView(
        childId = candidate.childId,
        state = candidate.state,
        presence = PixelAgents.presence(candidate.childId, candidate.state),
        worktree = candidate.worktree,
        baseRevision = candidate.baseRevision,
        verification = candidate.verification,
        verificationPass = candidate.verificationPass,
        reviewRank = candidate.reviewRank,
        reviewer = candidate.reviewer,
        costMicro = candidate.costMicro,
        wallMs = candidate.wallMs,
        winner = winner != null && winner == candidate.childId
    )
}
