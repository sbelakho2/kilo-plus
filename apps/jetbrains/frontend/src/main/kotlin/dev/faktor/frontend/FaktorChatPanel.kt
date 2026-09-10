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
import dev.faktor.shared.NativeProjection
import dev.faktor.shared.NativeTaskRun
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
import javax.swing.JTabbedPane
import javax.swing.JTextArea
import javax.swing.JTextField
import javax.swing.SwingUtilities
import javax.swing.WindowConstants

private const val MAX_TRANSCRIPT_LINES = 4000
private const val MAX_TRANSCRIPT_CHARS = 400_000
private const val MAX_EVIDENCE_PREVIEW_CHARS = 4000

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

    private val startTaskButton = JButton("Start task")

    private val runsModel = DefaultComboBoxModel<NativeTaskRun>()

    private val runsCombo = JComboBox(runsModel)

    private val cancelRunButton = JButton("Cancel run")

    private val taskArea = JTextArea(5, 32)

    private val agentsModel = DefaultComboBoxModel<NativeAgent>()

    private val agentsCombo = JComboBox(agentsModel)

    private val agentsArea = JTextArea(5, 32)

    private val evidenceIdField = JTextField(8)

    private val retrieveEvidenceButton = JButton("Retrieve")

    private val evidenceArea = JTextArea(8, 32)

    private var renderedSeq: Long = 0

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

        val tabs = JTabbedPane()
        tabs.addTab("Status", buildStatusTab())
        tabs.addTab("Task", buildTaskTab())
        tabs.addTab("Agents", buildAgentsTab())
        tabs.addTab("Evidence", buildEvidenceTab())
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

    private fun buildEvidenceTab(): JPanel {
        val panel = JPanel(GridLayout(0, 1, 4, 4))
        panel.border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        val row = JPanel(FlowLayout(FlowLayout.LEFT))
        row.add(JLabel("evidence id"))
        row.add(evidenceIdField)
        row.add(retrieveEvidenceButton)
        panel.add(row)
        evidenceArea.isEditable = false
        panel.add(JScrollPane(evidenceArea))
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
            runAsync("start task") {
                val started = service.startTaskRun(
                    goal,
                    if (criteria.isEmpty()) null else criteria
                )
                onEdt { appendSystem("task run ${started.runId} started (${started.state})") }
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
        retrieveEvidenceButton.addActionListener {
            val id = evidenceIdField.text.trim().toLongOrNull()
            if (id == null) {
                appendSystem("evidence id must be an integer")
                return@addActionListener
            }
            runAsync("evidence $id") {
                val meta = service.evidence(id)
                val retrieval = service.retrieveEvidence(
                    id, dev.faktor.shared.NativeRequests.evidenceSelectorAll()
                )
                val preview = String(retrieval.bytes, Charsets.UTF_8)
                    .take(MAX_EVIDENCE_PREVIEW_CHARS)
                onEdt {
                    evidenceArea.text =
                        "id=${meta.id} retained=${meta.backingRetained} " +
                            "backingLen=${meta.backingLen ?: 0} bytes=${retrieval.byteLen} " +
                            "truncated=${retrieval.truncatedByPolicy}\n" + preview
                    appendSystem("evidence ${meta.id}: ${retrieval.byteLen} bytes retrieved")
                }
            }
        }
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
    }

    private fun refreshStatusBlocking() {
        val projection = service.projection()
        onEdt { applyProjection(projection) }
    }

    private fun refreshMessagesBlocking() {
        val page = service.messages(limit = 50)
        onEdt {
            val fresh = page.messages.filter { it.seq > renderedSeq }.sortedBy { it.seq }
            for (message in fresh) {
                renderedSeq = maxOf(renderedSeq, message.seq)
                appendSystem("${message.role}: ${message.text}")
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
        retrieveEvidenceButton.isEnabled = running
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
