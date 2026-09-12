// Frontend smoke: the new model parsing (blocker, tournament,
// provider/reasoning, attachments) against canned native JSON, a real
// daemon where feasible (task-run with `files`, permissions reply route,
// unknown-tournament typed 404, best-effort graph), and construction of
// EVERY new UI section from a mock payload (no display needed: Swing
// components are created but never shown).
package dev.faktor.frontend

import dev.faktor.backend.BackendConnection
import dev.faktor.backend.BackendProcessManager
import dev.faktor.backend.assertEquals
import dev.faktor.backend.assertTrue
import dev.faktor.backend.fail
import dev.faktor.backend.NativeClient
import dev.faktor.shared.NativeApiException
import dev.faktor.shared.NativeMessage
import dev.faktor.shared.NativeRequests
import dev.faktor.shared.parseNativeAgents
import dev.faktor.shared.parseNativeModelCatalog
import dev.faktor.shared.parseNativeOrchestratorGraph
import dev.faktor.shared.parseNativePermissionList
import dev.faktor.shared.parseNativePresentationAck
import dev.faktor.shared.parseNativeSessionUsage
import dev.faktor.shared.parseNativeTaskVerification
import dev.faktor.shared.parseNativeTaskViews
import dev.faktor.shared.parseNativeTournament
import dev.faktor.shared.parseNativeTournamentDecision
import dev.faktor.shared.parseNativeTournamentStarted
import dev.faktor.shared.parseNativeTournamentSummaries
import dev.faktor.shared.parseNativeVerificationView
import java.awt.image.BufferedImage
import java.nio.file.Files
import java.nio.file.Paths

// ----------------------------------------------------------------- fixtures

private const val AGENTS_JSON = "[" +
    "{\"agent_id\":\"self-1\",\"kind\":\"self\",\"run_id\":\"run-1\"," +
    "\"session_id\":1,\"worktree_id\":1,\"goal\":\"ship it\",\"state\":\"Running\"," +
    "\"model\":\"m\",\"budget\":null,\"ownership\":\"self\",\"item_ids\":[\"main\"]," +
    "\"progress\":null,\"result\":null}," +
    "{\"agent_id\":\"child-1\",\"kind\":\"child\",\"run_id\":\"run-1\"," +
    "\"session_id\":9,\"worktree_id\":2,\"goal\":\"drive main step\",\"state\":\"Blocked\"," +
    "\"model\":\"m\",\"budget\":1000,\"ownership\":\"Mutating\",\"item_id\":\"main\"," +
    "\"item_kind\":\"Implementation\"," +
    "\"blocker\":{\"kind\":\"permission\",\"reason\":\"shell call needs approval\"," +
    "\"dependency\":null,\"resolution\":\"allow the shell tool\"," +
    "\"last_progress_ms\":42}," +
    "\"capabilities\":[{\"cap\":\"ReadWorkspace\"}]," +
    "\"progress\":{\"lastOutputAt\":1,\"lastProgressAt\":2,\"lastOpCompletedAt\":3," +
    "\"inFlightOp\":null,\"silenceMs\":500,\"stallThresholdMs\":1000,\"stalled\":true}," +
    "\"result\":{\"summary\":\"main step output\",\"merge\":null}," +
    "\"presentation\":\"background\"}" +
    "]"

private const val TASK_JSON = "[" +
    "{\"goal\":\"ship it\",\"constraints\":[\"no network\"],\"state\":\"in_progress\"," +
    "\"milestones\":{\"completed\":[\"m1\"],\"open\":[\"m2\"]}," +
    "\"decisions\":[],\"failures\":[],\"changedFiles\":[\"a.rs\"]," +
    "\"tests\":{\"run\":[\"cargo test\"],\"failed\":[]}," +
    "\"preferences\":[],\"verification\":[],\"progress\":null," +
    "\"budget\":{\"maxTokens\":100,\"maxTurns\":5,\"spentTokens\":7,\"spentTurns\":1," +
    "\"maxCostMicro\":50,\"spentCostMicro\":2,\"openReservedMicro\":0}," +
    "\"acceptanceCriteria\":[\"criterion A\"]," +
    "\"plan\":[" +
    "{\"id\":\"step-1\",\"summary\":\"first\",\"state\":\"done\",\"depends_on\":[]}," +
    "{\"id\":\"step-2\",\"summary\":\"second\",\"state\":\"running\",\"depends_on\":[\"step-1\"]}" +
    "]," +
    "\"blockers\":[\"waiting on review\"]," +
    "\"evidenceRefs\":[\"evidence:41\"],\"phase\":\"building\"}" +
    "]"

private const val SESSION_USAGE_JSON = "{" +
    "\"sessionId\":\"1\",\"providerCalls\":{\"tokens\":7,\"prefixObservations\":[]}," +
    "\"prefixStability\":null,\"tasks\":[{\"taskId\":\"3\"," +
    "\"budget\":{\"maxTokens\":100,\"maxTurns\":5,\"spentTokens\":7,\"spentTurns\":1," +
    "\"maxCostMicro\":50,\"spentCostMicro\":2,\"openReservedMicro\":0}}]}"

private const val VERIFICATION_JSON = "{" +
    "\"owed\":[{\"opId\":\"op-3\",\"tool\":\"shell\",\"startedMs\":6," +
    "\"status\":\"running\",\"effectStatus\":\"unknown\"}]," +
    "\"failedChecks\":[]}"

private const val TASK_VERIFICATION_JSON = "{" +
    "\"sessionId\":\"1\",\"taskId\":\"3\",\"records\":[{" +
    "\"recordId\":\"r1\",\"revision\":\"rev\",\"workspaceId\":\"1\",\"worktreeId\":\"1\"," +
    "\"treeHash\":null," +
    "\"criteria\":[{\"criterionKey\":\"c1\",\"passed\":true," +
    "\"evidence\":\"evidence:42 tool output\"}]," +
    "\"checks\":[{\"check\":\"unit\",\"program\":\"cargo\",\"args\":[\"test\"]," +
    "\"category\":\"test\",\"required\":true,\"status\":\"passed\",\"startedMs\":1," +
    "\"finishedMs\":2,\"exit\":0,\"summary\":\"see evidence:43\"}]," +
    "\"changedFiles\":[{\"path\":\"a.rs\",\"digestHex\":\"aa\",\"size\":3}]," +
    "\"unrelatedChanges\":[],\"reviewer\":null,\"status\":\"passed\"," +
    "\"startedMs\":1,\"completedMs\":2}]}"

private const val MODELS_JSON = "[" +
    "{\"provider\":\"fake\",\"model\":\"m\",\"context\":1000,\"maxOutput\":100," +
    "\"tools\":true,\"parallelTools\":false,\"reasoning\":true,\"thinking\":true," +
    "\"vision\":false,\"structuredOutput\":false,\"embeddings\":false," +
    "\"streaming\":true,\"source\":\"conservativeDefault\"}" +
    "]"

private const val GRAPH_JSON = "{" +
    "\"plan_id\":\"run-1\",\"goal\":\"graph goal\",\"state\":\"Running\"," +
    "\"work_items\":[" +
    "{\"item_id\":\"step-1\",\"kind\":\"Analysis\",\"state\":\"Done\"}," +
    "{\"item_id\":\"step-2\",\"kind\":\"Implementation\",\"state\":\"Running\"}]," +
    "\"children\":[{\"child_id\":\"child-1\",\"session_id\":9,\"operation_id\":1," +
    "\"worktree_id\":2,\"ownership\":\"Mutating\",\"state\":\"Blocked\"," +
    "\"blocker\":{\"kind\":\"dependency\",\"reason\":\"step-1 pending\"," +
    "\"dependency\":\"step-1\",\"resolution\":\"wait for step-1\",\"last_progress_ms\":5}," +
    "\"budget\":1000,\"capabilities\":[],\"plan_step_index\":1," +
    "\"steer_events\":[],\"merge\":null}]}"

private const val TOURNAMENT_JSON = "{" +
    "\"id\":\"t-1\",\"run_family\":\"run-7\",\"goal\":\"pick winner\"," +
    "\"criteria\":[{\"id\":\"c-1\",\"spec\":\"tests pass\"}]," +
    "\"candidates\":[" +
    "{\"child_id\":\"child-0\",\"worktree\":\"/tmp/w0\",\"base_revision\":\"abc\"," +
    "\"state\":\"done\",\"verification\":12,\"verification_pass\":true," +
    "\"review\":{\"rank\":\"clean\",\"reviewer\":\"rev-1\"}," +
    "\"cost_micro\":100,\"wall_ms\":1000}," +
    "{\"child_id\":\"child-1\",\"worktree\":\"/tmp/w1\",\"base_revision\":\"abc\"," +
    "\"state\":\"discarded\",\"verification\":null,\"verification_pass\":null," +
    "\"review\":null,\"cost_micro\":50,\"wall_ms\":900}]," +
    "\"winner\":\"child-0\",\"state\":\"decided\"}"

private const val TOURNAMENT_STARTED_JSON = "{" +
    "\"tournament_id\":\"t-1\",\"run_id\":\"run-7\"," +
    "\"candidates\":[\"child-0\",\"child-1\"],\"state\":\"open\",\"winner\":null}"

private const val TOURNAMENTS_LIST_JSON = "[" +
    "{\"id\":\"t-1\",\"state\":\"decided\",\"candidate_count\":2," +
    "\"winner\":\"child-0\",\"decided_ms\":123}," +
    "{\"id\":\"t-2\",\"state\":\"open\",\"candidate_count\":2," +
    "\"winner\":null,\"decided_ms\":null}]"

private const val TOURNAMENT_OPEN_JSON = "{" +
    "\"id\":\"t-2\",\"run_family\":\"run-8\",\"goal\":\"pick the open winner\"," +
    "\"criteria\":[{\"id\":\"c-1\",\"spec\":\"tests pass\"}]," +
    "\"candidates\":[" +
    "{\"child_id\":\"child-0\",\"worktree\":\"/tmp/w0\",\"base_revision\":\"abc\"," +
    "\"state\":\"done\",\"verification\":12,\"verification_pass\":true," +
    "\"review\":{\"rank\":\"clean\",\"reviewer\":\"rev-1\"}," +
    "\"cost_micro\":100,\"wall_ms\":1000}," +
    "{\"child_id\":\"child-1\",\"worktree\":\"/tmp/w1\",\"base_revision\":\"abc\"," +
    "\"state\":\"running\",\"verification\":null,\"verification_pass\":null," +
    "\"review\":null,\"cost_micro\":0,\"wall_ms\":0}]," +
    "\"winner\":null,\"state\":\"open\"}"

private const val TOURNAMENT_DECISION_JSON = "{" +
    "\"tournament_id\":\"t-2\",\"winner\":\"child-0\"," +
    "\"rationale\":\"winner child-0 (verification=pass)\"," +
    "\"discarded\":[{\"child_id\":\"child-1\",\"reason\":\"candidate ended failed\"}]}"

private const val PRESENTATION_ACK_JSON = "{" +
    "\"child_id\":\"child-1\",\"presentation\":\"background\",\"changed\":true}"

private const val PERMISSION_LIST_JSON = "{" +
    "\"permissions\":[{\"id\":\"7\",\"session_id\":\"9\",\"capability\":\"shell\"," +
    "\"detail\":{\"tool\":\"bash\"}}]}"

// -------------------------------------------------------------------- smoke

object FrontendSmoke {

    private var failures = 0

    @JvmStatic
    fun main(args: Array<String>) {
        step("canned native parsing: blocker/progress/result/capabilities") {
            val agents = parseNativeAgents(AGENTS_JSON)
            assertEquals(2, agents.size)
            val child = agents[1]
            assertEquals("child-1", child.agentId)
            assertEquals("main", child.itemId)
            assertEquals("Implementation", child.itemKind)
            assertEquals(1, child.capabilities.size)
            val blocker = child.blocker ?: fail("blocker must parse")
            assertEquals("permission", blocker.kind)
            assertEquals("shell call needs approval", blocker.reason)
            assertEquals("allow the shell tool", blocker.resolution)
            assertEquals(42L, blocker.lastProgressMs)
            val progress = child.progress ?: fail("progress must parse")
            assertEquals(true, progress.stalled)
            assertEquals(500L, progress.silenceMs)
            assertEquals("main step output", child.result?.summary)
            assertEquals(null, child.result?.merge)
        }

        step("canned native parsing: task view additive fields") {
            val task = parseNativeTaskViews(TASK_JSON)[0]
            assertEquals(listOf("criterion A"), task.acceptanceCriteria)
            assertEquals(2, task.plan.size)
            assertEquals(listOf("step-1"), task.plan[1].dependsOn)
            assertEquals(listOf("waiting on review"), task.blockers)
            assertEquals(listOf("evidence:41"), task.evidenceRefs)
            assertEquals("building", task.phase)
        }

        step("canned native parsing: verification criteria/check summaries") {
            val verification = parseNativeTaskVerification(TASK_VERIFICATION_JSON)
            val record = verification.records[0]
            assertEquals("passed", record.status)
            assertEquals(1, record.criteriaPassed)
            assertEquals(1, record.criteriaTotal)
            assertEquals(1, record.criteria.size)
            assertEquals("evidence:42 tool output", record.criteria[0].evidence)
            assertEquals(listOf("see evidence:43"), record.checkSummaries)
        }

        step("canned native parsing: tournament + tournament start receipt") {
            val tournament = parseNativeTournament(TOURNAMENT_JSON)
            assertEquals("t-1", tournament.id)
            assertEquals("run-7", tournament.runFamily)
            assertEquals("decided", tournament.state)
            assertEquals("child-0", tournament.winner)
            assertEquals(2, tournament.candidates.size)
            assertEquals("clean", tournament.candidates[0].reviewRank)
            assertEquals("rev-1", tournament.candidates[0].reviewer)
            assertEquals(12L, tournament.candidates[0].verification)
            assertEquals(true, tournament.candidates[0].verificationPass)
            val started = parseNativeTournamentStarted(TOURNAMENT_STARTED_JSON)
            assertEquals("t-1", started.tournamentId)
            assertEquals(listOf("child-0", "child-1"), started.candidates)
            assertEquals("open", started.state)
            assertEquals(null, started.winner)

            // The durable listing + decision ack + presentation ack parse.
            val summaries = parseNativeTournamentSummaries(TOURNAMENTS_LIST_JSON)
            assertEquals(2, summaries.size)
            assertEquals("t-1", summaries[0].id)
            assertEquals(2L, summaries[0].candidateCount)
            assertEquals("child-0", summaries[0].winner)
            assertEquals(123L, summaries[0].decidedMs)
            assertEquals(null, summaries[1].winner)
            assertEquals(null, summaries[1].decidedMs)
            assertEquals("t-2", latestTournamentId(summaries))
            val decision = parseNativeTournamentDecision(TOURNAMENT_DECISION_JSON)
            assertEquals("t-2", decision.tournamentId)
            assertEquals("child-0", decision.winner)
            assertEquals(1, decision.discarded.size)
            assertEquals("child-1", decision.discarded[0].childId)
            assertTrue(decision.rationale.contains("child-0"), "rationale must surface")
            val presentation = parseNativePresentationAck(PRESENTATION_ACK_JSON)
            assertEquals("child-1", presentation.childId)
            assertEquals("background", presentation.presentation)
            assertEquals(true, presentation.changed)
        }

        step("tournament panel gates decide on all candidates settled") {
            val runningOpen = TaskTree.tournamentView(parseNativeTournament(TOURNAMENT_OPEN_JSON))
            val panel = TournamentPanel()
            panel.setSummaries(parseNativeTournamentSummaries(TOURNAMENTS_LIST_JSON))
            assertTrue(panel.summariesText().contains("t-2[open]"), panel.summariesText())
            panel.setTournament(runningOpen)
            assertEquals(false, panel.decideEnabled())
            assertEquals(true, panel.abortEnabled())
            // All candidates settled: decide becomes available.
            val settled = runningOpen.copy(
                candidates = runningOpen.candidates.map { it.copy(state = "done") }
            )
            panel.setTournament(settled)
            assertEquals(true, panel.decideEnabled())
            assertEquals(true, panel.abortEnabled())
            // A decided tournament exposes no controls.
            panel.setTournament(TaskTree.tournamentView(parseNativeTournament(TOURNAMENT_JSON)))
            assertEquals(false, panel.decideEnabled())
            assertEquals(false, panel.abortEnabled())
            assertEquals(null, panel.current()?.takeIf { it.open })
        }

        step("canned native parsing: provider/reasoning catalog join + permissions") {
            val catalog = parseNativeModelCatalog(MODELS_JSON)
            assertEquals("fake", catalog[0].provider)
            assertEquals(true, catalog[0].reasoning)
            val permissions = parseNativePermissionList(PERMISSION_LIST_JSON)
            assertEquals("7", permissions[0].id)
            assertEquals("9", permissions[0].sessionId)
            assertEquals("shell", permissions[0].capability)
            assertTrue(permissions[0].detail.contains("bash"), "detail JSON must be kept")
        }

        step("attachments ride the task-run/tournament requests") {
            assertEquals(
                "{\"goal\":\"g\",\"criteria\":[\"c\"],\"files\":[\"/tmp/a.rs\",\"/tmp/b.rs\"]}",
                NativeRequests.startTaskRun("g", listOf("c"), files = listOf("/tmp/a.rs", "/tmp/b.rs"))
            )
            assertEquals(
                "{\"goal\":\"g\"}",
                NativeRequests.startTaskRun("g")
            )
            assertEquals(
                "{\"goal\":\"g\",\"criteria\":[\"c\"],\"n\":3,\"files\":[\"/tmp/a.rs\"]}",
                NativeRequests.startTournament("g", listOf("c"), 3, files = listOf("/tmp/a.rs"))
            )
            assertEquals(
                "{\"permission_id\":\"7\",\"decision\":\"allow\"}",
                NativeRequests.permissionReply("7", "allow")
            )
            assertEquals(
                "{\"selector\":\"line_range\",\"start\":2,\"end\":5}",
                NativeRequests.evidenceSelectorLines(2, 5)
            )
        }

        step("evidence ref parsing mirrors the cockpit vocabulary") {
            assertEquals(41L, EvidenceRefs.parse("evidence:41").id)
            assertEquals(42L, EvidenceRefs.parse("see evidence/42 here").id)
            assertEquals(43L, EvidenceRefs.parse("evidence#43").id)
            assertEquals(null, EvidenceRefs.parse("free text").id)
        }

        step("pixel identity is deterministic and every state animates") {
            val a = PixelAgents.avatar("child-1")
            val b = PixelAgents.avatar("child-1")
            assertEquals(a, b)
            assertEquals(25, a.pixels.size)
            for (y in 0 until 5) {
                for (x in 0 until 3) {
                    assertEquals(
                        a.pixels[y * 5 + x],
                        a.pixels[y * 5 + (4 - x)],
                        "sprite must be symmetric"
                    )
                }
            }
            assertTrue(a.hue in 0..359, "hue must be in range")
            val states = mapOf(
                "Running" to PixelState.RUNNING,
                "Paused" to PixelState.PAUSED,
                "Waiting" to PixelState.WAITING,
                "Blocked" to PixelState.BLOCKED,
                "Done" to PixelState.DONE,
                "FailedRecoverable" to PixelState.FAILED,
                "Cancelled" to PixelState.CANCELLED,
                "mystery" to PixelState.WAITING
            )
            for ((tag, expected) in states) {
                assertEquals(expected, PixelAgents.stateOf(tag), "state $tag")
            }
            assertEquals("pixel-blocked", PixelState.BLOCKED.animation)
            assertTrue(
                PixelAgents.frame(PixelState.RUNNING, 0) != PixelAgents.frame(PixelState.RUNNING, 2),
                "running must bob"
            )
            assertEquals(true, PixelAgents.frame(PixelState.BLOCKED, 0).alert)
            assertTrue(
                PixelAgents.frame(PixelState.FAILED, 0).alpha != PixelAgents.frame(PixelState.FAILED, 2).alpha,
                "failed must flicker"
            )
            assertEquals(
                PixelState.CANCELLED,
                PixelAgents.fold(
                    mapOf("child-1" to PixelAgents.presence("child-1", "Done")),
                    listOf(Pair("child-1", "Cancelled"))
                )["child-1"]?.state
            )
            val retained = PixelAgents.fold(
                mapOf("child-1" to PixelAgents.presence("child-1", "Done")),
                emptyList()
            )
            assertEquals(1, retained.size)
            assertEquals(PixelState.DONE, retained["child-1"]?.state)
            val image = BufferedImage(22, 22, BufferedImage.TYPE_INT_ARGB)
            val graphics = image.createGraphics()
            try {
                PixelSprite.paintSprite(graphics, PixelAgents.presence("child-1", "Blocked"), 1)
            } finally {
                graphics.dispose()
            }
        }

        step("task tree model surfaces every section from mock payloads") {
            val task = parseNativeTaskViews(TASK_JSON)[0]
            val agents = parseNativeAgents(AGENTS_JSON)
            val catalog = parseNativeModelCatalog(MODELS_JSON)
            val usage = parseNativeSessionUsage(SESSION_USAGE_JSON)
            val model = TaskTree.build(
                task = task,
                agents = agents,
                graph = parseNativeOrchestratorGraph(GRAPH_JSON),
                catalog = catalog,
                verification = parseNativeVerificationView(VERIFICATION_JSON),
                taskVerification = parseNativeTaskVerification(TASK_VERIFICATION_JSON),
                usage = usage,
                childUsage = mapOf("9" to usage),
                tournament = parseNativeTournament(TOURNAMENT_JSON)
            )
            assertEquals("ship it", model.goal)
            assertEquals("in_progress", model.state)
            assertEquals("building", model.phase)
            assertEquals(listOf("criterion A"), model.acceptanceCriteria)
            assertEquals(2, model.steps.size)
            assertEquals(listOf("step-1"), model.steps[1].dependsOn)
            assertEquals(1, model.children.size)
            val child = model.children[0]
            assertEquals("permission", child.blocker?.kind)
            assertEquals("background", child.presentation)
            assertEquals(true, child.background)
            assertEquals("fake", child.provider)
            assertEquals(true, child.reasoning)
            assertEquals(1000L, child.budgetMaxTokens)
            assertEquals(7L, child.spentTokens)
            assertEquals(993L, child.remainingTokens)
            assertEquals(true, child.progress?.stalled)
            assertEquals("main step output", child.result?.summary)
            assertEquals(1, model.blockers.size)
            assertTrue(
                model.blockers[0].actions.contains(BlockerAction.PERMISSION_ALLOW) &&
                    model.blockers[0].actions.contains(BlockerAction.RESUME) &&
                    model.blockers[0].actions.contains(BlockerAction.RETRY),
                "permission blockers must offer resume/allow/retry"
            )
            assertEquals("passed", model.verification.status)
            assertEquals(1, model.verification.owed)
            val evidenceIds = model.evidence.mapNotNull { it.id }.sorted()
            assertEquals(listOf(41L, 42L, 43L), evidenceIds)
            assertEquals(93L, model.spend?.remainingTokens)
            assertEquals(true, model.spend?.durable)
            // Background children are tucked after foreground children.
            val foregroundChild = agents[1].copy(agentId = "child-2", presentation = "foreground")
            val ordered = TaskTree.build(agents = listOf(agents[1], foregroundChild)).children
            assertEquals(listOf("child-2", "child-1"), ordered.map { it.childId })
            assertEquals("child-0", model.tournament?.winner)
            assertEquals(true, model.tournament?.candidates?.get(0)?.winner)
            assertEquals(false, model.tournament?.candidates?.get(1)?.winner)
        }

        step("every new UI section is constructible from the mock payload") {
            val model = TaskTree.build(
                task = parseNativeTaskViews(TASK_JSON)[0],
                agents = parseNativeAgents(AGENTS_JSON),
                catalog = parseNativeModelCatalog(MODELS_JSON),
                verification = parseNativeVerificationView(VERIFICATION_JSON),
                taskVerification = parseNativeTaskVerification(TASK_VERIFICATION_JSON),
                usage = parseNativeSessionUsage(SESSION_USAGE_JSON),
                childUsage = mapOf("9" to parseNativeSessionUsage(SESSION_USAGE_JSON)),
                tournament = parseNativeTournament(TOURNAMENT_JSON)
            )
            val tree = TaskTreePanel()
            tree.update(model)
            assertEquals(model, tree.model())
            assertTrue(
                tree.childLabel(model.children[0]).contains("child-1"),
                "child label must surface the child id"
            )
            assertTrue(
                tree.childLabel(model.children[0]).contains("(background)"),
                "background children must be dimmed/marked in the tree"
            )
            val blockers = BlockersPanel()
            blockers.update(model.blockers, parseNativePermissionList(PERMISSION_LIST_JSON), model.taskBlockers)
            assertEquals(1, blockers.blockerCount())
            assertEquals(1, blockers.permissionCount())
            assertTrue(
                blockers.applicableActions().contains(BlockerAction.PERMISSION_DENY),
                "selected blocker actions must be visible"
            )
            val tournament = TournamentPanel()
            tournament.setTournament(
                TaskTree.tournamentView(parseNativeTournament(TOURNAMENT_JSON))
            )
            assertEquals(2, tournament.candidateCount())
            val navigator = EvidenceNavigatorPanel()
            navigator.setEvidence(model.evidence)
            navigator.setMessages(
                listOf(
                    NativeMessage(seq = 1, id = 1, role = "user", createdMs = 1, text = "hi"),
                    NativeMessage(seq = 2, id = 2, role = "assistant", createdMs = 2, text = "hello")
                )
            )
            assertEquals(3, navigator.evidenceCount())
            assertEquals(2, navigator.messageCount())
            assertEquals("{\"selector\":\"all\"}", navigator.selectorJson())
            val attachments = AttachmentsPanel()
            val dir = Files.createTempDirectory("faktor-attach-smoke-")
            val file = Files.createFile(Paths.get(dir.toString(), "note.txt"))
            attachments.addFiles(listOf(file.toString(), file.toString()))
            assertEquals(1, attachments.count())
            assertEquals(1, attachments.files().size)
            assertTrue(!attachments.files()[0].startsWith(".."), "paths must be absolute")
            attachments.clear()
            assertEquals(0, attachments.count())
            val panel = FaktorChatPanel(
                FaktorFrontendService(
                    Paths.get("target/debug/faktor-cli"),
                    Paths.get(System.getProperty("java.io.tmpdir"), "faktor-frontend-smoke")
                )
            )
            panel.shutdown()
        }

        if (args.isEmpty()) {
            println("FRONTEND SMOKE PASS (canned only: no daemon binary argument)")
            kotlin.system.exitProcess(if (failures == 0) 0 else 1)
        }
        runAgainstRealDaemon(args[0])

        println(if (failures == 0) "FRONTEND SMOKE PASS" else "FRONTEND SMOKE FAIL ($failures)")
        kotlin.system.exitProcess(if (failures == 0) 0 else 1)
    }

    // ------------------------------------------------------- real daemon

    private fun runAgainstRealDaemon(binaryPath: String) {
        val binary = Paths.get(binaryPath)
        val dataDir = Files.createTempDirectory("faktor-frontend-smoke-")
        // The shadow mutation default copies the session's WORKSPACE on a
        // task start; a tiny dedicated workspace keeps the smoke hermetic
        // and fast instead of duplicating the whole checkout.
        val workspace = Files.createTempDirectory("faktor-frontend-workspace-")
        Files.write(Paths.get(workspace.toString(), "seed.txt"), "smoke".toByteArray())
        val manager = BackendProcessManager(binary, dataDir)
        var connection: BackendConnection? = null
        try {
            var sessionId: String? = null
            step("start daemon for frontend smoke") {
                connection = manager.start()
                println("  port=${connection!!.port} pid=${connection!!.pid()}")
            }
            val conn = connection
            if (conn != null) {
                val client = NativeClient.forConnection(conn)
                step("native health + readiness") {
                    if (!client.health().ok) fail("health ok=false")
                    if (!client.awaitReady(10_000L).ready) fail("daemon never ready")
                }
                step("create session") {
                    sessionId = client.createSession(
                        "default", "default", workspace.toString(), "frontend smoke"
                    ).id
                }
                val sid = sessionId
                if (sid != null) {
                    step("task-run accepts `files` (attachments path)") {
                        val started = client.startTaskRun(
                            sid,
                            "frontend smoke attachment",
                            files = listOf("/tmp/faktor-frontend-smoke-attachment.txt")
                        )
                        if (started.runId.isEmpty()) fail("no run id")
                        println("  run=${started.runId} state=${started.state}")
                    }
                    step("permission list is served (reply route reachable)") {
                        val permissions = client.permissions(sid)
                        println("  pending permissions=${permissions.size}")
                    }
                    step("unknown tournament is a typed 404") {
                        try {
                            client.tournamentState(sid, "does-not-exist")
                            fail("unknown tournament must not answer 200")
                        } catch (e: NativeApiException) {
                            if (e.status != 404) {
                                fail("unexpected tournament error ${e.status} ${e.code}")
                            }
                        }
                    }
                    step("tournament listing + decide/abort are typed") {
                        val summaries = client.tournaments(sid)
                        println("  tournaments=${summaries.size}")
                        try {
                            client.decideTournament(sid, "does-not-exist")
                            fail("unknown decide must not answer 200")
                        } catch (e: NativeApiException) {
                            if (e.status != 404) {
                                fail("unexpected decide error ${e.status} ${e.code}")
                            }
                        }
                        try {
                            client.abortTournament(sid, "does-not-exist", "smoke")
                            fail("unknown abort must not answer 200")
                        } catch (e: NativeApiException) {
                            if (e.status != 404) {
                                fail("unexpected abort error ${e.status} ${e.code}")
                            }
                        }
                    }
                    step("presentation transition on an unknown child is a typed 404") {
                        try {
                            client.setAgentPresentation(sid, "no-such-child", "background")
                            fail("unknown child presentation must not answer 200")
                        } catch (e: NativeApiException) {
                            if (e.status != 404) {
                                fail("unexpected presentation error ${e.status} ${e.code}")
                            }
                        }
                    }
                    step("orchestrator graph is typed (graph | 404 | 409)") {
                        try {
                            val graph = client.orchestratorGraph(sid)
                            println("  graph plan=${graph.planId} children=${graph.children.size}")
                        } catch (e: NativeApiException) {
                            if (e.status != 404 && e.status != 409) {
                                fail("unexpected graph error ${e.status} ${e.code}")
                            }
                        }
                    }
                    step("evidence access stays typed (404 unknown / 503 unwired)") {
                        try {
                            client.evidence(sid, 1L)
                            fail("unknown evidence id must not answer 200")
                        } catch (e: NativeApiException) {
                            if (e.status != 404 && e.status != 503) {
                                fail("unexpected evidence error ${e.status} ${e.code}")
                            }
                        }
                    }
                }
            }
        } finally {
            if (connection != null) {
                val c = connection!!
                try {
                    manager.stop(c)
                    println("PASS stop daemon")
                } catch (e: Throwable) {
                    failures++
                    println("FAIL stop daemon: ${e.message}")
                    c.process.destroyForcibly()
                }
            }
            dataDir.toFile().deleteRecursively()
            workspace.toFile().deleteRecursively()
        }
    }

    private fun step(name: String, body: () -> Unit) {
        try {
            body()
            println("PASS $name")
        } catch (e: Throwable) {
            failures++
            println("FAIL $name: ${e.message}")
        }
    }
}
