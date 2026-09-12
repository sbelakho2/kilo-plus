// The real JetBrains-side bridge: owns the daemon lifecycle (through the
// existing BackendProcessManager), authenticates with the ephemeral
// password the manager generated and passed to the child via the
// environment (protected channel), and speaks Faktor Native Protocol v1
// (HTTP + SSE) through dev.faktor.backend.NativeClient /
// NativeEventStream.
//
// This class is UI-free on purpose: the Swing panel (FaktorChatPanel) and
// any future IntelliJ tool-window adapter drive the exact same surface, so
// bridge behavior is unit-testable without a display.
package dev.faktor.frontend

import dev.faktor.backend.BackendConnection
import dev.faktor.backend.BackendException
import dev.faktor.backend.BackendProcessManager
import dev.faktor.backend.NativeClient
import dev.faktor.backend.NativeEventStream
import dev.faktor.backend.NativeSseEvent
import dev.faktor.shared.NativeAbortAck
import dev.faktor.shared.NativeAgent
import dev.faktor.shared.NativeAgentControlAck
import dev.faktor.shared.NativeEvidence
import dev.faktor.shared.NativeEvidenceRetrieval
import dev.faktor.shared.NativeHealth
import dev.faktor.shared.NativeModelInfo
import dev.faktor.shared.NativeOrchestratorGraph
import dev.faktor.shared.NativePermissionAck
import dev.faktor.shared.NativePermissionEntry
import dev.faktor.shared.NativeProjection
import dev.faktor.shared.NativePromptReceipt
import dev.faktor.shared.NativeSessionCreated
import dev.faktor.shared.NativeSessionSummary
import dev.faktor.shared.NativeSessionUsage
import dev.faktor.shared.NativeTaskRun
import dev.faktor.shared.NativeTaskRunCancelled
import dev.faktor.shared.NativeTaskRunStarted
import dev.faktor.shared.NativeTaskVerification
import dev.faktor.shared.NativeTaskView
import dev.faktor.shared.NativeTournament
import dev.faktor.shared.NativeTournamentStarted
import dev.faktor.shared.NativeUsageTotals
import dev.faktor.shared.NativeVerificationView
import java.nio.file.Path

/**
 * Bridge facade over one daemon. All mutating calls are synchronized on the
 * lifecycle lock; a stopped service refuses every other call loudly.
 */
class FaktorFrontendService(
    private val binaryPath: Path,
    private val dataDir: Path
) {
    interface Listener {
        fun onDaemonStatus(status: String, detail: String?) {}
        fun onStreamStatus(status: String, detail: String?) {}
        fun onEvent(event: NativeSseEvent) {}
        fun onError(message: String) {}
    }

    private val lifecycleLock = Any()

    private var manager: BackendProcessManager? = null

    private var connection: BackendConnection? = null

    private var client: NativeClient? = null

    private var stream: NativeEventStream? = null

    @Volatile
    private var listener: Listener? = null

    @Volatile
    private var sessionId: String? = null

    fun setListener(value: Listener?) {
        listener = value
    }

    fun isRunning(): Boolean = synchronized(lifecycleLock) { client != null }

    fun currentSessionId(): String? = sessionId

    fun daemonDescription(): String = synchronized(lifecycleLock) {
        val c = client ?: return "stopped"
        return "connected: ${c.baseUrl}"
    }

    // ------------------------------------------------------------- lifecycle

    /** Starts the daemon, authenticates, and waits until it reports ready. */
    fun start(): NativeHealth {
        synchronized(lifecycleLock) {
            if (client != null) return client!!.health()
        }
        val mgr = BackendProcessManager(binaryPath, dataDir)
        listener?.onDaemonStatus("starting", binaryPath.toString())
        val backendConnection = try {
            mgr.start()
        } catch (e: BackendException) {
            listener?.onDaemonStatus("failed", e.message)
            throw e
        }
        val cl = NativeClient.forConnection(backendConnection)
        try {
            val health = cl.health()
            val ready = cl.awaitReady()
            if (!ready.ready) {
                throw BackendException("daemon never reported ready")
            }
            synchronized(lifecycleLock) {
                manager = mgr
                connection = backendConnection
                client = cl
            }
            listener?.onDaemonStatus(
                "running",
                "pid ${backendConnection.pid()} port ${backendConnection.port} version ${health.version}"
            )
            return health
        } catch (e: Exception) {
            mgr.stop(backendConnection)
            listener?.onDaemonStatus("failed", e.message)
            throw e
        }
    }

    /** Stops the SSE stream and the daemon; idempotent. */
    fun stop() {
        var toStop: Pair<BackendProcessManager, BackendConnection>? = null
        synchronized(lifecycleLock) {
            stream?.stop()
            stream = null
            sessionId = null
            val mgr = manager
            val conn = connection
            manager = null
            connection = null
            client = null
            if (mgr != null && conn != null) {
                toStop = Pair(mgr, conn)
            }
        }
        val stopPair = toStop
        if (stopPair != null) {
            stopPair.first.stop(stopPair.second)
            listener?.onDaemonStatus("stopped", null)
        }
    }

    // ------------------------------------------------------------- sessions

    fun createSession(
        provider: String,
        model: String,
        workspace: String? = null,
        title: String? = null
    ): NativeSessionCreated {
        val created = clientOrThrow().createSession(provider, model, workspace, title)
        sessionId = created.id
        return created
    }

    fun listSessions(): List<NativeSessionSummary> = clientOrThrow().listSessions()

    fun modelCatalog(): List<NativeModelInfo> = clientOrThrow().modelCatalog()

    fun useSession(id: String) {
        sessionId = id
    }

    fun prompt(text: String, files: List<String>? = null): NativePromptReceipt {
        val sid = requireSession()
        return clientOrThrow().prompt(sid, text, files)
    }

    fun abort(opId: String? = null): NativeAbortAck {
        val sid = requireSession()
        return clientOrThrow().abortSession(sid, opId)
    }

    // ---------------------------------------------------------- projections

    fun projection(): NativeProjection = clientOrThrow().projection(requireSession())

    fun messages(limit: Long = 50): dev.faktor.shared.NativeMessagePage =
        clientOrThrow().messages(requireSession(), null, limit)

    /** The durable message page of ANY session id (child transcript slices). */
    fun messagesFor(sessionId: String, limit: Long = 50): dev.faktor.shared.NativeMessagePage =
        clientOrThrow().messages(sessionId, null, limit)

    fun events(after: Long, limit: Long = 256): dev.faktor.shared.NativeEventPage =
        clientOrThrow().events(requireSession(), after, limit)

    fun taskViews(): List<NativeTaskView> = clientOrThrow().taskViews(requireSession())

    fun orchestratorGraph(): NativeOrchestratorGraph? = try {
        clientOrThrow().orchestratorGraph(requireSession())
    } catch (e: Exception) {
        // The graph is best-effort: a session without a single orchestration
        // run (404) or holding several (409) has no unambiguous graph; the
        // tree degrades to task views and the agent listing.
        null
    }

    // ------------------------------------------------------------ task runs

    fun taskRuns(): List<NativeTaskRun> = clientOrThrow().taskRuns(requireSession())

    fun startTaskRun(
        goal: String,
        criteria: List<String>? = null,
        model: String? = null,
        maxTokens: Long? = null,
        maxCostMicro: Long? = null,
        mutationMode: String? = null,
        files: List<String>? = null
    ): NativeTaskRunStarted = clientOrThrow().startTaskRun(
        requireSession(), goal, criteria, model, maxTokens, maxCostMicro, mutationMode, files
    )

    fun cancelTaskRun(runId: String): NativeTaskRunCancelled =
        clientOrThrow().cancelTaskRun(requireSession(), runId)

    // ------------------------------------------------------------ tournaments

    fun startTournament(
        goal: String,
        criteria: List<String>,
        n: Int,
        model: String? = null,
        maxTokens: Long? = null,
        maxCostMicro: Long? = null,
        files: List<String>? = null
    ): NativeTournamentStarted = clientOrThrow().startTournament(
        requireSession(), goal, criteria, n, model, maxTokens, maxCostMicro, null, files
    )

    fun tournamentState(tournamentId: String): NativeTournament =
        clientOrThrow().tournamentState(requireSession(), tournamentId)

    // -------------------------------------------------------------- permissions

    fun permissions(): List<NativePermissionEntry> =
        clientOrThrow().permissions(requireSession())

    fun replyPermission(permissionId: String, decision: String): NativePermissionAck =
        clientOrThrow().replyPermission(permissionId, decision)

    // --------------------------------------------------------------- agents

    fun agents(): List<NativeAgent> = clientOrThrow().agents(requireSession())

    fun pauseAgent(childId: String): NativeAgentControlAck =
        clientOrThrow().pauseAgent(childId)

    fun resumeAgent(childId: String): NativeAgentControlAck =
        clientOrThrow().resumeAgent(childId)

    fun cancelAgent(childId: String): NativeAgentControlAck =
        clientOrThrow().cancelAgent(childId)

    fun retryAgent(childId: String): NativeAgentControlAck =
        clientOrThrow().retryAgent(childId)

    fun steerAgent(childId: String, text: String): NativeAgentControlAck =
        clientOrThrow().steerAgent(childId, text)

    fun setAgentModel(childId: String, model: String): NativeAgentControlAck =
        clientOrThrow().setAgentModel(childId, model)

    fun setAgentBudget(
        childId: String,
        maxTokens: Long? = null,
        maxCostMicro: Long? = null
    ): NativeAgentControlAck = clientOrThrow().setAgentBudget(childId, maxTokens, maxCostMicro)

    // ------------------------------------------------------ usage/verification

    fun usage(): NativeUsageTotals = clientOrThrow().usage()

    fun sessionUsage(): NativeSessionUsage = clientOrThrow().sessionUsage(requireSession())

    /** The durable usage of ANY session id (child session spend envelopes). */
    fun sessionUsageFor(sessionId: String): NativeSessionUsage =
        clientOrThrow().sessionUsage(sessionId)

    fun verification(): NativeVerificationView =
        clientOrThrow().verification(requireSession())

    fun taskVerification(taskId: String): NativeTaskVerification =
        clientOrThrow().taskVerification(requireSession(), taskId)

    // ------------------------------------------------------------- evidence

    fun evidence(evidenceId: Long): NativeEvidence =
        clientOrThrow().evidence(requireSession(), evidenceId)

    fun retrieveEvidence(evidenceId: Long, selectorJson: String): NativeEvidenceRetrieval =
        clientOrThrow().retrieveEvidence(requireSession(), evidenceId, selectorJson)

    // --------------------------------------------------------------- stream

    /**
     * (Re)starts the SSE journal stream for [sessionId] at [cursor] (0 =
     * replay from the beginning). Events are delivered to the listener.
     */
    fun watchSession(id: String, cursor: Long = 0) {
        val backendConnection = synchronized(lifecycleLock) {
            sessionId = id
            connection
        } ?: throw BackendException("daemon is not running")
        stream?.stop()
        val s = NativeEventStream.forConnection(
            backendConnection, id, cursor,
            onEvent = { listener?.onEvent(it) },
            onStatus = { status, detail -> listener?.onStreamStatus(status, detail) },
            onError = { e -> listener?.onError(e.message ?: e.javaClass.simpleName) }
        )
        synchronized(lifecycleLock) { stream = s }
        s.start()
    }

    fun stopStream() {
        synchronized(lifecycleLock) {
            stream?.stop()
            stream = null
        }
    }

    fun streamCursor(): Long = synchronized(lifecycleLock) { stream?.cursor ?: 0L }

    fun streamStatus(): String = synchronized(lifecycleLock) { stream?.status ?: "off" }

    // ------------------------------------------------------------- internals

    private fun clientOrThrow(): NativeClient =
        synchronized(lifecycleLock) { client } ?: throw BackendException("daemon is not running")

    private fun requireSession(): String =
        sessionId ?: throw BackendException("no session selected")
}
