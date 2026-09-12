// Rich task tree: goal, acceptance criteria, plan/DAG steps with the
// children driving them, full child identity (pixel avatar, state, blocker,
// model/provider/reasoning, ownership/worktree, budget spend/remaining,
// progress, latest result), current phase, verification status, evidence
// list (selection = one-click retrieval) and durable spend. The blockers
// section and tournament view embed below the tree. Pure presentation: the
// model is built in TaskTreeModel.kt and all actions go to a Listener.
package dev.faktor.frontend

import java.awt.BorderLayout
import java.awt.Component
import java.awt.Font
import java.awt.GridLayout
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JSplitPane
import javax.swing.JTree
import javax.swing.event.TreeSelectionEvent
import javax.swing.tree.DefaultMutableTreeNode
import javax.swing.tree.DefaultTreeCellRenderer
import javax.swing.tree.DefaultTreeModel

/** Typed payloads the tree renders and routes on selection. */
sealed class TaskTreeNode(val label: String) {
    class Goal(label: String) : TaskTreeNode(label)
    class Child(val child: ChildNode) : TaskTreeNode("")
    class Evidence(val ref: EvidenceRef) : TaskTreeNode("")
    class Plain(label: String) : TaskTreeNode(label)
}

class TaskTreePanel : JPanel(BorderLayout()) {

    interface Listener {
        fun onEvidenceSelected(ref: EvidenceRef)
        fun onChildSelected(child: ChildNode)
    }

    private val stateLabel = JLabel("state: -")

    private val phaseLabel = JLabel("phase: -")

    private val spendLabel = JLabel("spend: -")

    private val verificationLabel = JLabel("verification: -")

    private val treeModel = DefaultTreeModel(DefaultMutableTreeNode(TaskTreeNode.Plain("no task data")))

    private val tree = JTree(treeModel)

    private var listener: Listener? = null

    private var currentModel: TaskTreeModel? = null

    init {
        tree.isRootVisible = true
        tree.showsRootHandles = true
        tree.cellRenderer = TaskTreeRenderer()
        tree.addTreeSelectionListener { event: TreeSelectionEvent ->
            val node = event.path.lastPathComponent as? DefaultMutableTreeNode ?: return@addTreeSelectionListener
            when (val payload = node.userObject) {
                is TaskTreeNode.Child -> listener?.onChildSelected(payload.child)
                is TaskTreeNode.Evidence -> listener?.onEvidenceSelected(payload.ref)
                else -> {}
            }
        }

        val summary = JPanel(GridLayout(0, 1, 2, 2))
        summary.add(stateLabel)
        summary.add(phaseLabel)
        summary.add(verificationLabel)
        summary.add(spendLabel)

        val top = JPanel(BorderLayout(0, 4))
        top.add(summary, BorderLayout.NORTH)
        top.add(JScrollPane(tree), BorderLayout.CENTER)

        val split = JSplitPane(JSplitPane.VERTICAL_SPLIT, top, scroll(compactArea(4)))
        split.resizeWeight = 0.7
        split.isContinuousLayout = true
        add(split, BorderLayout.CENTER)
    }

    fun setListener(value: Listener?) {
        listener = value
    }

    fun model(): TaskTreeModel? = currentModel

    /** Rebuilds every section from one model; all values are real DTO fields. */
    fun update(model: TaskTreeModel) {
        currentModel = model
        stateLabel.text = "state: ${model.state}"
        phaseLabel.text = "phase: ${model.phase}"
        verificationLabel.text = "verification: " + model.verification.status +
            " (criteria ${model.verification.criteriaPassed}/${model.verification.criteriaTotal}" +
            ", failed ${model.verification.failedChecks}, owed ${model.verification.owed})"
        spendLabel.text = spendText(model.spend)

        val root = DefaultMutableTreeNode(
            TaskTreeNode.Goal("goal: ${model.goal.ifEmpty { "(none)" }} [${model.state}]")
        )
        root.add(criteriaNode(model))
        root.add(planNode(model))
        root.add(childrenNode(model))
        root.add(verificationNode(model))
        root.add(evidenceNode(model))
        root.add(DefaultMutableTreeNode(TaskTreeNode.Plain(spendText(model.spend))))
        if (model.tournament != null) root.add(tournamentNode(model.tournament))
        treeModel.setRoot(root)
        for (row in 0 until tree.rowCount) tree.expandRow(row)
        tree.selectionModel.clearSelection()
    }

    private fun criteriaNode(model: TaskTreeModel): DefaultMutableTreeNode {
        val node = DefaultMutableTreeNode(
            TaskTreeNode.Plain("acceptance criteria (${model.acceptanceCriteria.size})")
        )
        for (criterion in model.acceptanceCriteria) {
            node.add(DefaultMutableTreeNode(TaskTreeNode.Plain(bound(criterion, 240))))
        }
        if (model.acceptanceCriteria.isEmpty()) {
            node.add(DefaultMutableTreeNode(TaskTreeNode.Plain("(none served)")))
        }
        return node
    }

    private fun planNode(model: TaskTreeModel): DefaultMutableTreeNode {
        val node = DefaultMutableTreeNode(TaskTreeNode.Plain("plan / DAG steps (${model.steps.size})"))
        for (step in model.steps) {
            val label = StringBuilder()
            label.append("[").append(step.state).append("] ").append(step.id)
            if (step.summary.isNotEmpty() && step.summary != step.id) {
                label.append(" - ").append(step.summary)
            }
            if (step.dependsOn.isNotEmpty()) {
                label.append(" (depends on ").append(step.dependsOn.joinToString(",")).append(")")
            }
            if (step.childIds.isNotEmpty()) {
                label.append(" children=").append(step.childIds.joinToString(","))
            }
            val stepNode = DefaultMutableTreeNode(TaskTreeNode.Plain(bound(label.toString(), 240)))
            node.add(stepNode)
            for (childId in step.childIds) {
                val child = model.children.firstOrNull { it.childId == childId } ?: continue
                stepNode.add(DefaultMutableTreeNode(TaskTreeNode.Child(child)))
            }
        }
        if (model.steps.isEmpty()) {
            node.add(DefaultMutableTreeNode(TaskTreeNode.Plain("(none served)")))
        }
        return node
    }

    private fun childrenNode(model: TaskTreeModel): DefaultMutableTreeNode {
        val node = DefaultMutableTreeNode(TaskTreeNode.Plain("children (${model.children.size})"))
        for (child in model.children) {
            node.add(DefaultMutableTreeNode(TaskTreeNode.Child(child)))
        }
        if (model.children.isEmpty()) {
            node.add(DefaultMutableTreeNode(TaskTreeNode.Plain("(no child agents)")))
        }
        return node
    }

    private fun verificationNode(model: TaskTreeModel): DefaultMutableTreeNode {
        val summary = model.verification
        val node = DefaultMutableTreeNode(
            TaskTreeNode.Plain(
                "verification: ${summary.status} criteria=${summary.criteriaPassed}/${summary.criteriaTotal}" +
                    " failedChecks=${summary.failedChecks} owed=${summary.owed}" +
                    (if (summary.recordStatus == null) "" else " record=${summary.recordStatus}")
            )
        )
        if (summary.recordStatus != null) {
            node.add(DefaultMutableTreeNode(TaskTreeNode.Plain("record status: ${summary.recordStatus}")))
        }
        return node
    }

    private fun evidenceNode(model: TaskTreeModel): DefaultMutableTreeNode {
        val node = DefaultMutableTreeNode(
            TaskTreeNode.Plain("evidence (${model.evidence.size}) - select to retrieve")
        )
        for (ref in model.evidence) {
            val label = if (ref.id == null) {
                bound(ref.label, 200)
            } else {
                "evidence:${ref.id} ${bound(ref.label, 160)}"
            }
            val child = DefaultMutableTreeNode(TaskTreeNode.Evidence(ref))
            node.add(child)
        }
        if (model.evidence.isEmpty()) {
            node.add(DefaultMutableTreeNode(TaskTreeNode.Plain("(no evidence refs)")))
        }
        return node
    }

    private fun tournamentNode(tournament: TournamentView): DefaultMutableTreeNode {
        val node = DefaultMutableTreeNode(
            TaskTreeNode.Plain(
                "tournament ${tournament.id} [${tournament.state}] winner=${tournament.winner ?: "-"}"
            )
        )
        for (candidate in tournament.candidates) {
            val label = "${candidate.childId} [${candidate.state}]" +
                " verification=" + (candidate.verification?.let { "#$it" } ?: "-") +
                " review=" + (candidate.reviewRank ?: "-") +
                " cost=${candidate.costMicro}micro wall=${candidate.wallMs}ms" +
                (if (candidate.winner) " WINNER" else "")
            node.add(DefaultMutableTreeNode(TaskTreeNode.Plain(label)))
        }
        return node
    }

    private fun spendText(spend: SpendSummary?): String {
        if (spend == null) return "spend: -"
        val tokens = "${spend.spentTokens ?: 0}/${spend.maxTokens ?: "unlimited"}" +
            (spend.remainingTokens?.let { " remaining $it" } ?: "")
        val cost = "${spend.spentCostMicro ?: 0}/${spend.maxCostMicro ?: "unlimited"} micro" +
            (spend.remainingCostMicro?.let { " remaining $it" } ?: "")
        val open = if (spend.openReservedMicro > 0) " openReserved=${spend.openReservedMicro}" else ""
        return "spend (${if (spend.durable) "durable" else "estimate"}): tokens $tokens; cost $cost$open"
    }

    /** The label of one child node, surfacing every native field. */
    fun childLabel(child: ChildNode): String {
        val text = StringBuilder()
        text.append(child.childId).append(" [").append(child.state).append("]")
        if (child.itemId != null) text.append(" item=").append(child.itemId)
        if (child.itemKind != null) text.append(" kind=").append(child.itemKind)
        text.append(" model=").append(child.model ?: "-")
        text.append(" provider=").append(child.provider ?: "-")
        text.append(" reasoning=").append(
            if (child.reasoning == null) "-" else if (child.reasoning == true) "yes" else "no"
        )
        text.append(" ownership=").append(child.ownership)
        text.append(" worktree=").append(child.worktreeId)
        text.append(" tokens=").append(child.spentTokens ?: 0).append("/").append(child.budgetMaxTokens ?: "unlimited")
        if (child.remainingTokens != null) text.append(" remaining=").append(child.remainingTokens)
        text.append(" cost=").append(child.spentCostMicro ?: 0).append("/").append(child.maxCostMicro ?: "unlimited")
        child.blocker?.let { blocker ->
            text.append(" blocker=").append(blocker.kind).append(": ").append(bound(blocker.reason, 80))
        }
        child.progress?.let { progress ->
            text.append(" progress=").append(if (progress.stalled) "STALLED" else "live")
            if (progress.silenceMs != null) text.append("(").append(progress.silenceMs).append("ms)")
        }
        return text.toString()
    }

    /** The second line of one child node: latest result / merge envelope. */
    fun childResultLabel(child: ChildNode): String {
        val result = child.result ?: return "result: -"
        val text = StringBuilder("result: ").append(bound(result.summary, 160))
        result.merge?.let { merge ->
            text.append(" | merge ")
            if (merge.changeSetId != null) text.append(merge.changeSetId)
            text.append(" merged=").append(merge.merged ?: 0)
            text.append(" rejected=").append(merge.rejected ?: 0)
            text.append(" conflicts=").append(merge.conflicts ?: 0)
        }
        return text.toString()
    }

    private inner class TaskTreeRenderer : DefaultTreeCellRenderer() {
        private var spriteId: String? = null
        private var sprite: PixelSprite? = null
        private val panel = JPanel(BorderLayout(4, 0))
        private val label = JLabel()

        override fun getTreeCellRendererComponent(
            tree: JTree?,
            value: Any?,
            selected: Boolean,
            expanded: Boolean,
            leaf: Boolean,
            row: Int,
            hasFocus: Boolean
        ): Component {
            val payload = ((value as? DefaultMutableTreeNode)?.userObject)
            if (payload is TaskTreeNode.Child) {
                val child = payload.child
                if (spriteId != child.childId) {
                    spriteId = child.childId
                    sprite = PixelSprite(child.childId, child.state)
                    panel.removeAll()
                    paneAdd(sprite!!)
                    panel.add(label, BorderLayout.CENTER)
                }
                sprite?.setState(child.state)
                panel.background = if (selected) backgroundSelectionColor else backgroundNonSelectionColor
                panel.isOpaque = true
                label.foreground = if (selected) textSelectionColor else textNonSelectionColor
                label.font = label.font.deriveFont(Font.PLAIN)
                label.text = bound(childLabel(child) + " || " + childResultLabel(child), 420)
                return panel
            }
            return super.getTreeCellRendererComponent(
                tree, value, selected, expanded, leaf, row, hasFocus
            )
        }

        private fun paneAdd(component: Component) {
            panel.add(component, BorderLayout.WEST)
        }
    }
}
