// Deterministic pixel-avatar presence for child agents, mirroring the VS
// Code pixel system (apps/vscode/src/pixelAgents.ts) state-for-state.
//
// Every ChildId maps to a STABLE avatar (FNV-1a hash -> hue + symmetric 5x5
// sprite) and a typed presence state with its own animation. The mapping is
// a pure function of the id, so the same child keeps the same avatar across
// frames, reconnects and daemon restarts — no randomness, no external
// assets, no image files. [PixelSprite] paints the sprite with plain Swing
// paint code (Timer-driven, no third-party dependency) and renders overlays
// per state, so the tree/inspector animations stay keyed to the same state
// tags the native agent listing serves.
package dev.faktor.frontend

import java.awt.BasicStroke
import java.awt.Color
import java.awt.Dimension
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.RenderingHints
import javax.swing.JComponent
import javax.swing.Timer

/** The seven presence states of the VS Code pixel system. */
enum class PixelState(val tag: String) {
    RUNNING("running"),
    PAUSED("paused"),
    WAITING("waiting"),
    BLOCKED("blocked"),
    DONE("done"),
    FAILED("failed"),
    CANCELLED("cancelled");

    /** The animation class name (same vocabulary as the webview CSS). */
    val animation: String
        get() = "pixel-$tag"
}

/** The deterministic avatar of one ChildId. */
data class PixelAvatar(
    val childId: String,
    val hash: Long,
    val hue: Int,
    /** 5x5 row-major sprite bits (1 = filled) — symmetric by construction. */
    val pixels: List<Int>,
    val version: Int = PixelAgents.PIXEL_AVATAR_VERSION
) {
    /** Primary sprite color (deterministic hue from the hash). */
    val color: Color
        get() = Color.getHSBColor(hue / 360f, 0.72f, 0.85f)

    /** Highlight color used for the sprite's eyes/outline. */
    val accent: Color
        get() = Color.getHSBColor(((hue + 42) % 360) / 360f, 0.55f, 1.0f)
}

/** One child's identity + state + animation name. */
data class PixelPresence(
    val childId: String,
    val state: PixelState,
    val avatar: PixelAvatar,
    val animation: String
)

/** One animation frame: bounded offsets/alpha/pulse for the paint code. */
data class SpriteFrame(val dx: Int, val dy: Int, val alpha: Float, val alert: Boolean, val blink: Boolean)

/**
 * System animation policy: reduced motion when the IDE platform says so.
 *
 * The standalone kotlinc smoke compiles without platform jars, so the
 * documented platform switches are consulted REFLECTIVELY — missing classes
 * (standalone launcher, smoke) are the safe default (animations on), and a
 * platform exception never propagates. The probes are:
 *   - `com.intellij.ui.UISettings` presentation mode (animations off);
 *   - `com.intellij.openapi.util.registry.Registry` animation-disable keys.
 */
object PixelMotion {

    /** Test/explicit override; null consults the platform once. */
    @Volatile
    var override: Boolean? = null

    private val platformReduced: Boolean by lazy { detectPlatformReducedMotion() }

    /** True when motion should be suppressed (reduced motion). */
    fun reducedMotion(): Boolean = override ?: platformReduced

    private fun detectPlatformReducedMotion(): Boolean =
        probeUISettings() || probeRegistry()

    private fun probeUISettings(): Boolean {
        return try {
            val cls = Class.forName("com.intellij.ui.UISettings")
            val instance = cls.getMethod("getInstance").invoke(null) ?: return false
            val presentation = invokeBoolean(instance, "getPresentationMode")
            if (presentation != null) return presentation
            invokeBoolean(instance, "isPresentationMode") == true
        } catch (e: Throwable) {
            false
        }
    }

    private fun probeRegistry(): Boolean {
        return try {
            val cls = Class.forName("com.intellij.openapi.util.registry.Registry")
            val isMethod = cls.getMethod("is", String::class.java)
            var disabled = false
            for (key in REGISTRY_ANIMATION_KEYS) {
                if (isMethod.invoke(null, key) == true) {
                    disabled = true
                    break
                }
            }
            disabled
        } catch (e: Throwable) {
            false
        }
    }

    private fun invokeBoolean(target: Any, method: String): Boolean? {
        return try {
            val value = target.javaClass.getMethod(method).invoke(target)
            value as? Boolean
        } catch (e: Throwable) {
            null
        }
    }

    private val REGISTRY_ANIMATION_KEYS = listOf(
        "ide.animations.disabled",
        "idea.animations.disabled",
        "animations.disabled"
    )
}

/** Pure, dependency-free identity/animation authority. */
object PixelAgents {

    /** Avatar version so a future sprite change can be detected. */
    const val PIXEL_AVATAR_VERSION = 1

    /**
     * Bounded terminal transition: a terminal sprite may animate for this
     * many 200ms ticks, after which its frame is static forever and the
     * owning timer stops. Done gets one settle bounce, Failed one flicker;
     * Cancelled is static from its first frame.
     */
    const val TERMINAL_TRANSITION_TICKS = 3

    /** FNV-1a 32-bit over the UTF-16 code units — byte-identical to the VS Code system. */
    fun hash(childId: String): Long {
        var hash = 0x811c9dc5L
        for (c in childId) {
            hash = hash xor c.toInt().toLong()
            hash = (hash * 0x01000193L) and 0xffffffffL
        }
        return hash and 0xffffffffL
    }

    /** Deterministic 5x5 symmetric sprite + hue for one ChildId. */
    fun avatar(childId: String): PixelAvatar {
        val hash = hash(childId)
        val pixels = IntArray(25)
        for (y in 0 until 5) {
            for (x in 0 until 3) {
                val bit = ((hash shr ((y * 3 + x) % 15)) and 1L).toInt()
                pixels[y * 5 + x] = bit
                pixels[y * 5 + (4 - x)] = bit
            }
        }
        return PixelAvatar(
            childId = childId,
            hash = hash,
            hue = (hash % 360L).toInt(),
            pixels = pixels.toList()
        )
    }

    /**
     * Native state tag -> presence state. Terminal states are Done / Failed /
     * Cancelled; Waiting/Blocked/Paused keep their own animations; unknown or
     * idle tags read as `waiting` (never as running) so the pixel never lies
     * about activity. Mirrors `pixelStateOf` in the VS Code system.
     */
    fun stateOf(state: String): PixelState {
        val tag = asciiLower(state.trim())
        if (tag == "done" || tag == "completed" || tag == "verifiedcomplete") return PixelState.DONE
        if (tag == "failed" || tag == "failedrecoverable" || tag == "failedpermanent") {
            return PixelState.FAILED
        }
        if (tag == "cancelled" || tag == "canceled") return PixelState.CANCELLED
        if (tag == "blocked" || tag == "needsuserinput") return PixelState.BLOCKED
        if (tag == "paused" || tag == "suspended") return PixelState.PAUSED
        if (tag == "running" || tag == "preparing" || tag == "buildingcontext" ||
            tag == "waitingformodel" || tag == "streaming" || tag == "toolrequested" ||
            tag == "executingtool" || tag == "validating"
        ) {
            return PixelState.RUNNING
        }
        return PixelState.WAITING
    }

    fun presence(childId: String, state: String): PixelPresence {
        val pixelState = stateOf(state)
        return PixelPresence(childId, pixelState, avatar(childId), pixelState.animation)
    }

    /**
     * Fold one native agent frame into a persistent presence map: existing
     * children keep their identity and order, missing ids are retained
     * (transient page gaps), current frames update state, unknown ids are
     * appended. Avatars are always recomputed from the id, never mutated.
     */
    fun fold(
        previous: Map<String, PixelPresence>,
        frame: List<Pair<String, String>>
    ): Map<String, PixelPresence> {
        val next = LinkedHashMap<String, PixelPresence>()
        for ((childId, presence) in previous) next[childId] = presence
        for ((childId, state) in frame) {
            if (childId.isEmpty()) continue
            next[childId] = presence(childId, state)
        }
        return next
    }

    /**
     * The deterministic animation frame of one state at [tick] (200ms steps):
     * running bobs, waiting blinks, blocked shakes with a red alert. Terminal
     * states are BOUNDED: Done bounces once and settles, Failed flickers once
     * and settles, Cancelled is dim from its first frame; their X/slash marks
     * are painted statically, never as animation. Pure so the smoke can assert
     * every state's frames without a display.
     */
    fun frame(state: PixelState, tick: Int): SpriteFrame = when (state) {
        PixelState.RUNNING -> SpriteFrame(0, if (tick % 4 < 2) -1 else 0, 1f, false, false)
        PixelState.PAUSED -> SpriteFrame(0, 0, 0.55f, false, false)
        PixelState.WAITING -> SpriteFrame(0, 0, if (tick % 4 == 3) 0.55f else 0.85f, false, tick % 4 == 0)
        PixelState.BLOCKED -> SpriteFrame(if (tick % 2 == 0) -1 else 1, 0, 1f, true, false)
        PixelState.DONE -> if (tick < TERMINAL_TRANSITION_TICKS) {
            SpriteFrame(0, if (tick % 2 == 0) -1 else 0, 1f, false, false)
        } else {
            SpriteFrame(0, 0, 1f, false, false)
        }
        PixelState.FAILED -> if (tick < TERMINAL_TRANSITION_TICKS) {
            SpriteFrame(0, 0, if (tick % 2 == 0) 1f else 0.35f, false, false)
        } else {
            SpriteFrame(0, 0, 1f, false, false)
        }
        PixelState.CANCELLED -> SpriteFrame(0, 0, 0.35f, false, false)
    }

    /**
     * Whether one state's frame has fully settled at [tick]: Done/Failed are
     * static after their bounded transition; Cancelled is static (its dim
     * slash) from the first frame; live states never settle. The owning
     * sprite timer stops exactly at this boundary.
     */
    fun isStatic(state: PixelState, tick: Int): Boolean = when (state) {
        PixelState.RUNNING,
        PixelState.PAUSED,
        PixelState.WAITING,
        PixelState.BLOCKED -> false
        PixelState.CANCELLED -> true
        PixelState.DONE,
        PixelState.FAILED -> tick >= TERMINAL_TRANSITION_TICKS
    }

    private fun asciiLower(text: String): String {
        val sb = StringBuilder(text.length)
        for (c in text) sb.append(if (c in 'A'..'Z') (c + 32) else c)
        return sb.toString()
    }
}

/**
 * A live sprite component for one child. The timer runs only while the
 * component is displayed AND the state is still animating; terminal states
 * settle after their bounded transition (Done/Failed) or immediately
 * (Cancelled), then the timer stops. Under reduced motion no timer starts and
 * the settled/static frame renders directly.
 */
class PixelSprite(childId: String, state: String = "waiting") : JComponent() {

    var presence: PixelPresence = PixelAgents.presence(childId, state)
        private set

    private var tick = 0

    private val timer = Timer(200) { advance() }

    init {
        preferredSize = Dimension(22, 22)
        minimumSize = Dimension(22, 22)
        toolTipText = "${presence.childId}: ${presence.state.tag} (${presence.animation})"
        isOpaque = false
    }

    fun setState(state: String) {
        val updated = PixelAgents.presence(presence.childId, state)
        if (updated.state == presence.state) return
        presence = updated
        tick = 0
        toolTipText = "${presence.childId}: ${presence.state.tag} (${presence.animation})"
        syncTimer()
        repaint()
    }

    /** One animation step (the Timer body; visible to the smoke). */
    internal fun advance() {
        tick++
        repaint()
        if (PixelAgents.isStatic(presence.state, tick)) {
            timer.stop()
        }
    }

    /** Whether the sprite timer is currently running. */
    fun animationRunning(): Boolean = timer.isRunning

    /** The current frame tick. */
    fun tick(): Int = tick

    /** True when the state's frame is settled at the current tick. */
    fun settled(): Boolean = PixelAgents.isStatic(presence.state, tick)

    /** Applies the motion/timer policy at the current state and tick. */
    internal fun syncTimer() {
        val shouldRun = !PixelMotion.reducedMotion() && !PixelAgents.isStatic(presence.state, tick)
        if (shouldRun && !timer.isRunning) {
            timer.start()
        } else if (!shouldRun && timer.isRunning) {
            timer.stop()
        }
    }

    override fun addNotify() {
        super.addNotify()
        syncTimer()
    }

    override fun removeNotify() {
        timer.stop()
        super.removeNotify()
    }

    override fun paintComponent(g: Graphics) {
        val g2 = g.create() as Graphics2D
        try {
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            paintSprite(g2, presence, tick)
        } finally {
            g2.dispose()
        }
    }

    companion object {
        /** Paints one presence at [tick] — the render path the smoke can drive directly. */
        fun paintSprite(g2: Graphics2D, presence: PixelPresence, tick: Int) {
            val frame = PixelAgents.frame(presence.state, tick)
            val size = 22
            val cell = 3
            val grid = 5 * cell
            val originX = (size - grid) / 2 + frame.dx
            val originY = (size - grid) / 2 + frame.dy
            val avatar = presence.avatar
            val previousComposite = g2.composite
            g2.composite = java.awt.AlphaComposite.getInstance(
                java.awt.AlphaComposite.SRC_OVER, frame.alpha
            )
            for (y in 0 until 5) {
                for (x in 0 until 5) {
                    if (avatar.pixels[y * 5 + x] == 0) continue
                    val eyes = y == 1 && (x == 1 || x == 3)
                    g2.color = if (eyes || frame.blink) avatar.accent else avatar.color
                    g2.fillRect(originX + x * cell, originY + y * cell, cell, cell)
                }
            }
            if (frame.alert) {
                g2.color = Color(220, 40, 40)
                g2.stroke = BasicStroke(1.5f)
                g2.drawOval(originX - 2, originY - 2, grid + 4, grid + 4)
            }
            when (presence.state) {
                PixelState.DONE -> {
                    g2.color = Color(40, 170, 80)
                    g2.stroke = BasicStroke(2f)
                    g2.drawLine(originX + 1, originY + grid / 2, originX + 4, originY + grid - 2)
                    g2.drawLine(originX + 4, originY + grid - 2, originX + grid - 1, originY + 1)
                }
                PixelState.FAILED -> {
                    g2.color = Color(200, 30, 30)
                    g2.stroke = BasicStroke(2f)
                    g2.drawLine(originX, originY, originX + grid, originY + grid)
                    g2.drawLine(originX + grid, originY, originX, originY + grid)
                }
                PixelState.PAUSED -> {
                    g2.color = Color(90, 90, 90)
                    g2.fillRect(originX + 3, originY + 2, 2, grid - 4)
                    g2.fillRect(originX + grid - 5, originY + 2, 2, grid - 4)
                }
                PixelState.CANCELLED -> {
                    g2.color = Color(120, 120, 120)
                    g2.stroke = BasicStroke(2f)
                    g2.drawLine(originX, originY + grid - 1, originX + grid, originY + 1)
                }
                else -> {}
            }
            g2.composite = previousComposite
        }
    }
}
