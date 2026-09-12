// Native Swing tool-window panel for the Faktor bridge. Every interaction
// routes to a native endpoint through FaktorFrontendService:
//
//   chat input  -> POST /session/prompt (abort -> POST /native/session/{id}/abort)
//   task goal   -> POST /native/session/{id}/task-runs (cancel -> .../cancel)
//   agents      -> GET /native/agents + pause/resume/cancel/retry/steer/model/budget
//   status      -> GET /session/{id}/projection, /native/usage,
//                  /native/session/{id}/usage, /native/session/{id}/verification
//   evidence    -> GET /native/evidence/{id} + POST /native/evidence/{id}/retrieve
//   streaming   -> SSE /api/session/{id}/events with cursor resume
//
// Swing/JCEF decision: JCEF hosting needs the IntelliJ Platform/JCEF runtime
// which the repository's offline build does not carry, so the panel is a
// native Swing component (zero third-party deps, compiles with plain
// kotlinc like the rest of the split-mode tree) that a future
// ToolWindowFactory can embed directly.
package dev.faktor.frontend

import dev.faktor.backend.NativeSseEvent
import dev.faktor.shared.JsonValue
import dev.faktor.shared.NativeAgent
import dev.faktor.shared.NativeMessage
import dev.faktor.shared.NativeModelInfo
import dev.faktor.shared.NativePermissionEntry
import dev.faktor.shared.NativeProjection
import dev.faktor.shared.NativeTaskRun
import dev.faktor.shared.NativeTournament
import java.awt.BorderLayout
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.GridLayout
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import javax.swing.BorderFactory
import javax.swing.DefaultComboBoxModel
import javax.swing.JButton
import javax.swing.JComboBox
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.JOptionPane
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JSplitPane
import javax.swing.JTabbedPane
import javax.swing.JTextArea
import javax.swing.JTextField
import javax.swing.SwingUtilities
import javax.swing.WindowConstants

private const val MAX_TRANSCRIPT_LINES = 4000
private const val MAX_TRANSCRIPT_CHARS = 400_000
private const val MAX_EVIDENCE_PREVIEW_CHARS = 4000
private const val MAX_CHILD_USAGE_FETCHES = 12

class FaktorChatPanel(private val service: FaktorFrontendService) :
    JPanel(BorderLayout()), FaktorFrontendService.Listener {

    private val worker = Executors.newSingleThreadExecutor { runnable ->
        val thread = Thread(runnable, "faktor-ui")
        thread.isDaemon = true
        thread
    }

    private val refreshQueued = AtomicBoolean(false)

    private val transcript = JTextArea()

    private val input = JTextArea(3, 60)

    private val providerField = JTextField("default", 10)

    private val modelField = JTextField("default", 10)

    private val startButton = JButton("Start daemon")

    private val stopButton = JButton("Stop daemon")

    private val newSessionButton = JButton("New session")

    private val refreshButton = JButton("Refresh")

    private val sendButton = JButton("Send")

    private val abortButton = JButton("Abort")

    private val daemonLabel = JLabel("daemon: stopped")

    private val streamLabel = JLabel("stream: off")

    private val stateLabel = JLabel("state: -")

    private val modelLabel = JLabel("model: -")

    private val toolLabel = JLabel("active tool: -")

    private val queuedLabel = JLabel("queued: 0")

    private val usageLabel = JLabel("usage: -")

    private val verifyLabel = JLabel("verification: -")

    private val filesLabel = JLabel("files changed: 0")

    private val goalField = JTextField(24)

    private val criteriaField = JTextField(24)

    private val attachments = AttachmentsPanel()

    private val startTaskButton = JButton("Start task")

    private val runsModel = DefaultComboBoxModel<NativeTaskRun>()

    private val runsCombo = JComboBox(runsModel)

    private val cancelRunButton = JButton("Cancel run")

    private val taskArea = JTextArea(5, 32)

    private val agentsModel = DefaultComboBoxModel<NativeAgent>()

    private val agentsCombo = JComboBox(agentsModel)

    private val agentsArea = JTextArea(5, 32)

    private val treePanel = TaskTreePanel()

    private val blockersPanel = BlockersPanel()

    private val tournamentPanel = TournamentPanel()

    private val navigator = EvidenceNavigatorPanel()

    private val tabs = JTabbedPane()

    private var renderedSeq: Long = 0

    private var currentTree: TaskTreeModel? = null

    private var trackedTournamentId: String? = null

    private var pendingPermissions: List<NativePermissionEntry> = emptyList()

    private val transcriptOffsets = LinkedHashMap<Long, Pair<Int, Int>>()

    init {
        service.setListener(this)
        buildLayout()
        wireActions()
        setControlsEnabled(false)
    }

    // ------------------------------------------------------------------ UI

    private fun buildLayout() {
        transcript.isEditable = false
        transcript.lineWrap = true
        transcript.wrapStyleWord = true
        transcript.font = transcript.font.deriveFont(13f)

        input.lineWrap = true
        input.wrapStyleWord = true

        val toolbar = JPanel(FlowLayout(FlowLayout.LEFT))
        toolbar.add(startButton)
        toolbar.add(stopButton)
        toolbar.add(newSessionButton)
        toolbar.add(JLabel("provider"))
        toolbar.add(providerField)
        toolbar.add(JLabel("model"))
        toolbar.add(modelField)
        toolbar.add(refreshButton)

        val chat = JPanel(BorderLayout())
        chat.add(JScrollPane(transcript), BorderLayout.CENTER)
        val inputRow = JPanel(BorderLayout())
        inputRow.add(JScrollPane(input), BorderLayout.CENTER)
        val buttons = JPanel(FlowLayout(FlowLayout.RIGHT))
        buttons.add(abortButton)
        buttons.add(sendButton)
        inputRow.add(buttons, BorderLayout.SOUTH)
        chat.add(inputRow, BorderLayout.SOUTH)

        val tabs = this.tabs
        tabs.removeAll()
        tabs.addTab("Status", buildStatusTab())
        tabs.addTab("Task", buildTaskTab())
        tabs.addTab("Task Tree", buildTaskTreeTab())
        tabs.addTab("Agents", buildAgentsTab())
        tabs.addTab("Tournament", tournamentPanel)
        tabs.addTab("Evidence", navigator)
        tabs.preferredSize = Dimension(430, 600)

        add(toolbar, BorderLayout.NORTH)
        add(chat, BorderLayout.CENTER)
        add(tabs, BorderLayout.EAST)
    }

    private fun buildStatusTab(): JPanel {
        val panel = JPanel(GridLayout(0, 1, 4, 4))
        panel.border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        panel.add(daemonLabel)
        panel.add(streamLabel)
        panel.add(stateLabel)
        panel.add(modelLabel)
        panel.add(toolLabel)
        panel.add(queuedLabel)
        panel.add(usageLabel)
        panel.add(verifyLabel)
        panel.add(filesLabel)
        return panel
    }

    private fun buildTaskTab(): JPanel {
        val panel = JPanel(BorderLayout())
        panel.border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        val form = JPanel(GridLayout(0, 1, 4, 4))
        form.add(JLabel("goal"))
        form.add(goalField)
        form.add(JLabel("criteria (comma separated, optional)"))
        form.add(criteriaField)
        form.add(titledSection("attachments (submitted as files)", attachments))
        val startRow = JPanel(FlowLayout(FlowLayout.LEFT))
        startRow.add(startTaskButton)
        form.add(startRow)
        form.add(JLabel("task runs"))
        form.add(runsCombo)
        val cancelRow = JPanel(FlowLayout(FlowLayout.LEFT))
        cancelRow.add(cancelRunButton)
        form.add(cancelRow)
        taskArea.isEditable = false
        form.add(JScrollPane(taskArea))
        panel.add(form, BorderLayout.NORTH)
        return panel
    }

    private fun buildTaskTreeTab(): JPanel {
        val split = JSplitPane(JSplitPane.VERTICAL_SPLIT, treePanel, JScrollPane(blockersPanel))
        split.resizeWeight = 0.62
        split.isContinuousLayout = true
        val panel = JPanel(BorderLayout())
        panel.add(split, BorderLayout.CENTER)
        return panel
    }

    private fun buildAgentsTab(): JPanel {
        val panel = JPanel(BorderLayout())
        panel.border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        val form = JPanel(GridLayout(0, 1, 4, 4))
        val refreshAgents = JButton("Refresh agents")
        refreshAgents.addActionListener {
            runAsync("refresh agents") {
                refreshAgentsBlocking()
            }
        }
        form.add(refreshAgents)
        form.add(agentsCombo)
        val controls = JPanel(GridLayout(0, 2, 4, 4))
        controls.add(agentButton("Pause") { agent -> service.pauseAgent(agent.agentId) })
        controls.add(agentButton("Resume") { agent -> service.resumeAgent(agent.agentId) })
        controls.add(agentButton("Cancel") { agent -> service.cancelAgent(agent.agentId) })
        controls.add(agentButton("Retry") { agent -> service.retryAgent(agent.agentId) })
        controls.add(agentButton("Steer") { agent ->
            val text = JOptionPane.showInputDialog(this, "Steer note for ${agent.agentId}")
            if (text != null && text.isNotEmpty()) service.steerAgent(agent.agentId, text)
        })
        controls.add(agentButton("Model") { agent ->
            val model = JOptionPane.showInputDialog(this, "Model for ${agent.agentId}", agent.model ?: "")
            if (model != null && model.isNotEmpty()) service.setAgentModel(agent.agentId, model)
        })
        controls.add(agentButton("Token budget") { agent ->
            val raw = JOptionPane.showInputDialog(this, "max_tokens for ${agent.agentId}", "10000")
            val tokens = raw?.trim()?.toLongOrNull()
            if (tokens != null && tokens > 0) service.setAgentBudget(agent.agentId, maxTokens = tokens)
        })
        controls.add(agentButton("Cost budget") { agent ->
            val raw = JOptionPane.showInputDialog(this, "max_cost_micro for ${agent.agentId}", "1000000")
            val micro = raw?.trim()?.toLongOrNull()
            if (micro != null && micro > 0) service.setAgentBudget(agent.agentId, maxCostMicro = micro)
        })
        form.add(controls)
        agentsArea.isEditable = false
        form.add(JScrollPane(agentsArea))
        panel.add(form, BorderLayout.NORTH)
        return panel
    }

    private fun agentButton(
        label: String,
        control: (NativeAgent) -> Any?
    ): JButton {
        val button = JButton(label)
        button.addActionListener {
            val agent = agentsCombo.selectedItem as? NativeAgent
            if (agent == null) {
                appendSystem("select an agent first")
                return@addActionListener
            }
            runAsync("agent $label") {
                control(agent)
                refreshAgentsBlocking()
            }
        }
        return button
    }

    private fun wireActions() {
        startButton.addActionListener { startDaemon() }
        stopButton.addActionListener {
            runAsync("stop daemon") {
                service.stop()
                onEdt {
                    setControlsEnabled(false)
                    appendSystem("daemon stopped")
                }
            }
        }
        newSessionButton.addActionListener {
            runAsync("new session") {
                val provider = providerField.text.trim().ifEmpty { "default" }
                val model = modelField.text.trim().ifEmpty { "default" }
                val created = service.createSession(provider, model, title = "JetBrains session")
                onEdt { appendSystem("session ${created.id} created (${created.title})") }
                service.watchSession(created.id, 0)
                refreshAllBlocking()
            }
        }
        refreshButton.addActionListener {
            runAsync("refresh") { refreshAllBlocking() }
        }
        sendButton.addActionListener { sendPrompt() }
        abortButton.addActionListener {
            runAsync("abort") {
                val ack = service.abort()
                onEdt { appendSystem("abort requested: ${ack.aborted}") }
            }
        }
        startTaskButton.addActionListener {
            val goal = goalField.text.trim()
            if (goal.isEmpty()) {
                appendSystem("task goal must not be empty")
                return@addActionListener
            }
            val criteria = criteriaField.text.split(',')
                .map { it.trim() }
                .filter { it.isNotEmpty() }
            val files = attachments.files()
            runAsync("start task") {
                val started = service.startTaskRun(
                    goal,
                    if (criteria.isEmpty()) null else criteria,
                    files = if (files.isEmpty()) null else files
                )
                onEdt {
                    appendSystem(
                        "task run ${started.runId} started (${started.state})" +
                            if (files.isEmpty()) "" else " with ${files.size} attachment(s)"
                    )
                }
                refreshTaskRunsBlocking()
            }
        }
        cancelRunButton.addActionListener {
            val run = runsCombo.selectedItem as? NativeTaskRun
            if (run == null) {
                appendSystem("select a task run first")
                return@addActionListener
            }
            runAsync("cancel task run") {
                val ack = service.cancelTaskRun(run.runId)
                onEdt { appendSystem("cancel ${ack.runId}: cancelled=${ack.cancelled}") }
                refreshTaskRunsBlocking()
            }
        }
        treePanel.setListener(object : TaskTreePanel.Listener {
            override fun onEvidenceSelected(ref: EvidenceRef) {
                onEdt { tabs.selectedComponent = navigator }
                runAsync("evidence ${ref.id}") {
                    navigator.selectEvidence(ref)
                    if (ref.id != null) retrieveEvidenceIntoNavigator(ref.id, navigator.selectorJson())
                }
            }

            override fun onChildSelected(child: ChildNode) {
                loadChildTranscript(child)
            }
        })
        blockersPanel.setListener(object : BlockersPanel.Listener {
            override fun onBlockerAction(row: BlockerRow, action: BlockerAction) {
                runAsync("blocker ${action.label} ${row.childId}") {
                    when (action) {
                        BlockerAction.RESUME -> service.resumeAgent(row.childId)
                        BlockerAction.RETRY -> service.retryAgent(row.childId)
                        BlockerAction.CANCEL -> service.cancelAgent(row.childId)
                        BlockerAction.PERMISSION_ALLOW,
                        BlockerAction.PERMISSION_DENY -> {
                            val decision = if (action == BlockerAction.PERMISSION_ALLOW) "allow" else "deny"
                            val permission = permissionForChild(row.childId)
                            if (permission == null) {
                                onEdt {
                                    appendSystem(
                                        "no pending permission request for ${row.childId}; " +
                                            "resume the child or reply from the pending permissions list"
                                    )
                                }
                            } else {
                                service.replyPermission(permission.id, decision)
                                onEdt { appendSystem("permission ${permission.id}: $decision") }
                            }
                        }
                    }
                    refreshTaskTreeBlocking()
                    refreshAgentsBlocking()
                }
            }

            override fun onPermissionReply(permission: NativePermissionEntry, decision: String) {
                runAsync("permission ${permission.id} $decision") {
                    service.replyPermission(permission.id, decision)
                    onEdt { appendSystem("permission ${permission.id}: $decision") }
                    refreshTaskTreeBlocking()
                }
            }
        })
        tournamentPanel.setListener(object : TournamentPanel.Listener {
            override fun onLoadTournament(tournamentId: String) {
                runAsync("load tournament") {
                    val tournament = service.tournamentState(tournamentId)
                    trackedTournamentId = tournament.id
                    onEdt { applyTournament(tournament) }
                }
            }

            override fun onStartTournament(goal: String, criteria: List<String>, n: Int) {
                runAsync("start tournament") {
                    val started = service.startTournament(goal, criteria, n)
                    trackedTournamentId = started.tournamentId
                    onEdt {
                        appendSystem(
                            "tournament ${started.tournamentId} started (${started.state}) " +
                                "candidates=${started.candidates.size}"
                        )
                    }
                    refreshTaskTreeBlocking()
                }
            }
        })
        navigator.setListener(object : EvidenceNavigatorPanel.Listener {
            override fun onRetrieve(evidenceId: Long, selectorJson: String) {
                runAsync("evidence $evidenceId") {
                    retrieveEvidenceIntoNavigator(evidenceId, selectorJson)
                }
            }

            override fun onMessageSelected(seq: Long) {
                onEdt { jumpToTranscript(seq) }
            }
        })
    }

    /** Starts the daemon off the EDT (public so the app entry point can call it). */
    fun startDaemon() {
        runAsync("start daemon") {
            val health = service.start()
            onEdt {
                setControlsEnabled(true)
                appendSystem("daemon ready: version ${health.version}")
            }
            runAsync("new session") {
                val provider = providerField.text.trim().ifEmpty { "default" }
                val model = modelField.text.trim().ifEmpty { "default" }
                val created = service.createSession(provider, model, title = "JetBrains session")
                onEdt { appendSystem("session ${created.id} created") }
                service.watchSession(created.id, 0)
                refreshAllBlocking()
            }
        }
    }

    private fun sendPrompt() {
        if (!service.isRunning()) {
            appendSystem("start the daemon first")
            return
        }
        if (service.currentSessionId() == null) {
            appendSystem("create a session first")
            return
        }
        val text = input.text.trim()
        if (text.isEmpty()) return
        input.text = ""
        appendUser(text)
        runAsync("prompt") {
            val receipt = service.prompt(text)
            onEdt { appendSystem("turn ${receipt.opId} accepted (queued=${receipt.queued})") }
            refreshAllBlocking()
        }
    }

    // ----------------------------------------------------------- refreshers

    private fun refreshAllBlocking() {
        if (!service.isRunning() || service.currentSessionId() == null) return
        refreshStatusBlocking()
        refreshMessagesBlocking()
        refreshTaskRunsBlocking()
        refreshAgentsBlocking()
        refreshUsageBlocking()
        refreshVerificationBlocking()
        refreshTaskTreeBlocking()
    }

    private fun refreshStatusBlocking() {
        val projection = service.projection()
        onEdt { applyProjection(projection) }
    }

    private fun refreshMessagesBlocking() {
        val page = service.messages(limit = 50)
        onEdt {
            navigator.setMessages(page.messages)
            val fresh = page.messages.filter { it.seq > renderedSeq }.sortedBy { it.seq }
            for (message in fresh) {
                renderedSeq = maxOf(renderedSeq, message.seq)
                appendMessage(message)
            }
        }
    }

    private fun refreshTaskRunsBlocking() {
        val runs = service.taskRuns()
        val views = try {
            service.taskViews()
        } catch (e: Exception) {
            emptyList<dev.faktor.shared.NativeTaskView>()
        }
        onEdt {
            runsModel.removeAllElements()
            for (run in runs) runsModel.addElement(run)
            if (runs.isEmpty()) {
                taskArea.text = "no task runs"
            } else {
                val sb = StringBuilder()
                for (run in runs) {
                    sb.append(run.runId).append(" [").append(run.mode).append("] ")
                        .append(run.state).append(" goal=").append(run.goal ?: "-")
                        .append('\n')
                }
                for (view in views) {
                    sb.append("task: ").append(view.state).append(" goal=").append(view.goal)
                        .append(" changed=").append(view.changedFiles.size)
                        .append(" tests=").append(view.testsRun.size)
                        .append(" failed=").append(view.testsFailed.size)
                    val budget = view.budget
                    if (budget != null) {
                        sb.append(" tokens=").append(budget.spentTokens ?: 0)
                            .append('/').append(budget.maxTokens ?: 0)
                            .append(" costMicro=").append(budget.spentCostMicro)
                            .append('/').append(budget.maxCostMicro ?: 0)
                    }
                    sb.append('\n')
                }
                taskArea.text = sb.toString()
            }
        }
    }

    private fun refreshAgentsBlocking() {
        val agents = service.agents()
        onEdt {
            agentsModel.removeAllElements()
            for (agent in agents) agentsModel.addElement(agent)
            if (agents.isEmpty()) {
                agentsArea.text = "no background agents"
            } else {
                val sb = StringBuilder()
                for (agent in agents) {
                    sb.append(agent.agentId).append(" [").append(agent.kind).append("] ")
                        .append(agent.state).append(" ownership=").append(agent.ownership)
                        .append(" model=").append(agent.model ?: "-")
                        .append(" budget=").append(agent.budget ?: "-")
                        .append('\n')
                }
                agentsArea.text = sb.toString()
            }
        }
    }

    private fun refreshUsageBlocking() {
        val usage = service.usage()
        val sessionUsage = service.sessionUsage()
        onEdt {
            usageLabel.text =
                "usage: sessions=${usage.sessions} tokens=${sessionUsage.tokens} " +
                    "taskCostMicro=${usage.settledCostMicro}"
        }
    }

    private fun refreshVerificationBlocking() {
        val verification = service.verification()
        onEdt {
            verifyLabel.text =
                "verification: owed=${verification.owed.size} " +
                    "failed=${verification.failedChecks.size}"
        }
    }

    // ---------------------------------------------------------------- tree

    private fun refreshTaskTreeBlocking() {
        val projection = service.projection()
        val task = try {
            service.taskViews().firstOrNull()
        } catch (e: Exception) {
            null
        }
        val agents = service.agents()
        val graph = service.orchestratorGraph()
        val catalogNow = try {
            service.modelCatalog()
        } catch (e: Exception) {
            emptyList<NativeModelInfo>()
        }
        val verification = try {
            service.verification()
        } catch (e: Exception) {
            null
        }
        val runs = try {
            service.taskRuns()
        } catch (e: Exception) {
            emptyList<NativeTaskRun>()
        }
        val taskVerification = if (runs.isEmpty()) {
            null
        } else {
            try {
                service.taskVerification(runs[0].taskId.toString())
            } catch (e: Exception) {
                null
            }
        }
        val sessionUsage = try {
            service.sessionUsage()
        } catch (e: Exception) {
            null
        }
        val childUsage = HashMap<String, dev.faktor.shared.NativeSessionUsage>()
        for (child in agents.filter { it.kind == "child" }.take(MAX_CHILD_USAGE_FETCHES)) {
            try {
                childUsage[child.sessionId.toString()] = service.sessionUsageFor(child.sessionId.toString())
            } catch (e: Exception) {
                // A child session without usage simply has no spend envelope.
            }
        }
        val tournament = try {
            trackedTournamentId?.let { service.tournamentState(it) }
        } catch (e: Exception) {
            null
        }
        val permissions = try {
            service.permissions()
        } catch (e: Exception) {
            emptyList<NativePermissionEntry>()
        }
        pendingPermissions = permissions
        val model = TaskTree.build(
            projection = projection,
            task = task,
            agents = agents,
            graph = graph,
            catalog = catalogNow,
            verification = verification,
            taskVerification = taskVerification,
            usage = sessionUsage,
            childUsage = childUsage,
            tournament = tournament
        )
        onEdt {
            currentTree = model
            treePanel.update(model)
            blockersPanel.update(model.blockers, permissions, model.taskBlockers)
            tournamentPanel.setTournament(model.tournament)
            navigator.setEvidence(model.evidence)
        }
    }

    private fun applyTournament(tournament: NativeTournament) {
        tournamentPanel.setTournament(TaskTree.tournamentView(tournament))
        appendSystem("tournament ${tournament.id} [${tournament.state}] winner=${tournament.winner ?: "-"}")
    }

    private fun retrieveEvidenceIntoNavigator(evidenceId: Long, selectorJson: String?) {
        val selector = selectorJson
            ?: dev.faktor.shared.NativeRequests.evidenceSelectorAll()
        try {
            val meta = service.evidence(evidenceId)
            val retrieval = service.retrieveEvidence(evidenceId, selector)
            val preview = String(retrieval.bytes, Charsets.UTF_8).take(MAX_EVIDENCE_PREVIEW_CHARS)
            onEdt {
                navigator.showRetrieval(
                    evidenceId,
                    "id=${meta.id} retained=${meta.backingRetained} " +
                        "backingLen=${meta.backingLen ?: 0} " +
                        "selector=${retrieval.selectorJson} " +
                        "truncated=${retrieval.truncatedByPolicy}\n" + preview,
                    retrieval.byteLen,
                    retrieval.truncatedByPolicy
                )
                appendSystem("evidence ${meta.id}: ${retrieval.byteLen} bytes retrieved")
            }
        } catch (e: Exception) {
            val message = e.message ?: e.javaClass.simpleName
            onEdt {
                navigator.showError(evidenceId, message)
                appendSystem("evidence $evidenceId error: $message")
            }
        }
    }

    private fun permissionForChild(childId: String): NativePermissionEntry? {
        val child = currentTree?.children?.firstOrNull { it.childId == childId } ?: return null
        return pendingPermissions.firstOrNull { it.sessionId == child.sessionId.toString() }
    }

    private fun loadChildTranscript(child: ChildNode) {
        runAsync("child transcript ${child.childId}") {
            val page = service.messagesFor(child.sessionId.toString(), limit = 50)
            val text = page.messages.joinToString("\n") {
                "#${it.seq} ${it.role}: ${it.text}"
            }
            onEdt {
                navigator.showTranscriptSlice(
                    "child ${child.childId} (session ${child.sessionId})",
                    text.ifEmpty { "(no messages in the child transcript window)" }
                )
                tabs.selectedComponent = navigator
            }
        }
    }

    private fun appendMessage(message: NativeMessage) {
        val start = transcript.text.length
        transcript.append("${message.role}: ${message.text}\n")
        transcriptOffsets[message.seq] = Pair(start, transcript.text.length)
        trimTranscript()
    }

    private fun jumpToTranscript(seq: Long) {
        val range = transcriptOffsets[seq]
        if (range == null) {
            appendSystem("message #$seq is outside the retained transcript window")
            return
        }
        try {
            transcript.requestFocusInWindow()
            transcript.select(range.first, range.second)
            transcript.caretPosition = range.second
            val view = transcript.modelToView(range.first)
            if (view != null) transcript.scrollRectToVisible(view)
        } catch (e: Exception) {
            appendSystem("message #$seq cannot be focused: ${e.message}")
        }
    }

    private fun applyProjection(projection: NativeProjection) {
        stateLabel.text = "state: ${projection.machine} (${projection.label})"
        val active = projection.activeModel
        modelLabel.text = if (active == null) {
            "model: ${projection.provider}/${projection.model}"
        } else {
            "model: ${active.provider}/${active.model}" +
                (if (active.variant == null) "" else " (${active.variant})")
        }
        val tool = projection.activeTool
        toolLabel.text = if (tool == null) "active tool: -" else "active tool: ${tool.tool} [${tool.status}]"
        queuedLabel.text = "queued: ${projection.queued}"
        filesLabel.text = "files changed: ${projection.filesChanged.size}"
    }

    // -------------------------------------------------------------- listener

    override fun onDaemonStatus(status: String, detail: String?) {
        onEdt {
            daemonLabel.text = "daemon: $status" + (if (detail == null) "" else " ($detail)")
        }
    }

    override fun onStreamStatus(status: String, detail: String?) {
        onEdt {
            streamLabel.text = "stream: $status" + (if (detail == null) "" else " ($detail)")
        }
    }

    override fun onEvent(event: NativeSseEvent) {
        onEdt {
            appendSystem("[${event.event}] id=${event.id} ${eventSummary(event.data)}")
            scheduleStatusRefresh()
        }
    }

    override fun onError(message: String) {
        onEdt { appendSystem("error: $message") }
    }

    private fun eventSummary(data: JsonValue): String {
        val obj = data as? JsonValue.Obj ?: return ""
        val state = (obj.fields["state"] as? JsonValue.Str)?.value
        val kind = (obj.fields["kind"] as? JsonValue.Str)?.value
        val parts = ArrayList<String>()
        if (kind != null) parts.add("kind=$kind")
        if (state != null) parts.add("state=$state")
        return parts.joinToString(" ")
    }

    private fun scheduleStatusRefresh() {
        if (!refreshQueued.compareAndSet(false, true)) return
        worker.execute {
            try {
                if (service.isRunning() && service.currentSessionId() != null) {
                    refreshStatusBlocking()
                    refreshMessagesBlocking()
                    refreshTaskTreeBlocking()
                }
            } catch (e: Exception) {
                // The stream may race a shutdown; errors surface via onError.
            } finally {
                refreshQueued.set(false)
            }
        }
    }

    // ------------------------------------------------------------- plumbing

    fun shutdown() {
        service.stopStream()
        worker.shutdownNow()
    }

    private fun setControlsEnabled(running: Boolean) {
        startButton.isEnabled = !running
        stopButton.isEnabled = running
        newSessionButton.isEnabled = running
        refreshButton.isEnabled = running
        sendButton.isEnabled = running
        abortButton.isEnabled = running
        startTaskButton.isEnabled = running
        cancelRunButton.isEnabled = running
    }

    private fun runAsync(label: String, work: () -> Unit) {
        worker.execute {
            try {
                work()
            } catch (e: Throwable) {
                val message = e.message ?: e.javaClass.simpleName
                onEdt { appendSystem("error: $label: $message") }
            }
        }
    }

    private fun appendSystem(text: String) {
        transcript.append("[faktor] " + text + "\n")
        trimTranscript()
    }

    private fun appendUser(text: String) {
        transcript.append("you: " + text + "\n")
        trimTranscript()
    }

    private fun trimTranscript() {
        val text = transcript.text
        if (text.length <= MAX_TRANSCRIPT_CHARS) {
            val lineCount = text.count { it == '\n' }
            if (lineCount <= MAX_TRANSCRIPT_LINES) return
        }
        val kept = text.split('\n').takeLast(MAX_TRANSCRIPT_LINES).joinToString("\n")
        transcript.text = kept.takeLast(MAX_TRANSCRIPT_CHARS)
        transcript.caretPosition = transcript.text.length
        // Offsets no longer map onto the rebuilt text; message navigation
        // falls back to a loud "outside the retained window" notice.
        transcriptOffsets.clear()
    }

    private fun onEdt(block: () -> Unit) {
        if (SwingUtilities.isEventDispatchThread()) {
            block()
        } else {
            SwingUtilities.invokeLater { block() }
        }
    }
}

/** Standalone launcher for the Swing panel (the tool-window host entry). */
object FaktorFrontendApp {
    @JvmStatic
    fun main(args: Array<String>) {
        val binary = if (args.isNotEmpty()) {
            args[0]
        } else {
            val env = System.getenv("FAKTOR_BIN")
            if (env != null && env.isNotEmpty()) env else "target/debug/faktor-cli"
        }
        val dataDir = if (args.size > 1) {
            args[1]
        } else {
            java.nio.file.Paths.get(
                System.getProperty("java.io.tmpdir"), "faktor-jetbrains"
            ).toString()
        }
        java.nio.file.Files.createDirectories(java.nio.file.Paths.get(dataDir))
        val service = FaktorFrontendService(
            java.nio.file.Paths.get(binary),
            java.nio.file.Paths.get(dataDir)
        )
        SwingUtilities.invokeLater {
            val frame = JFrame("Faktor")
            val panel = FaktorChatPanel(service)
            frame.contentPane.add(panel)
            frame.defaultCloseOperation = WindowConstants.DO_NOTHING_ON_CLOSE
            frame.addWindowListener(object : java.awt.event.WindowAdapter() {
                override fun windowClosing(e: java.awt.event.WindowEvent) {
                    panel.shutdown()
                    val stopper = Thread {
                        service.stop()
                        SwingUtilities.invokeLater {
                            frame.isVisible = false
                            frame.dispose()
                            System.exit(0)
                        }
                    }
                    stopper.isDaemon = true
                    stopper.start()
                }
            })
            frame.pack()
            frame.setSize(1280, 820)
            frame.setLocationRelativeTo(null)
            frame.isVisible = true
            panel.startDaemon()
        }
    }
}
