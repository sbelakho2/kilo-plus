// Coordination-board panel: the durable run-family board of the current
// session as served by the additive native route
// (`GET /native/session/{id}/board`), plus one bounded post composer
// (`POST .../board`). When the serving daemon exposes no board route the
// panel records an explicit unavailable state with the typed reason — it
// never fabricates posts or unread counts. Pure presentation: reads and
// posts are delegated to a Listener.
package dev.faktor.frontend

import dev.faktor.shared.NativeBoardPage
import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.GridLayout
import javax.swing.JButton
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JTextArea
import javax.swing.JTextField

private const val MAX_BOARD_SUBJECT_CHARS = 512
private const val MAX_BOARD_BODY_CHARS = 16 * 1024
private const val MAX_BOARD_LINE_CHARS = 240

class BoardPanel : JPanel(BorderLayout()) {

    interface Listener {
        /** Explicit read: acknowledges the current top of the board. */
        fun onRead()

        /** One bounded post; engine refusals stay typed and visible. */
        fun onPost(subject: String, body: String)
    }

    private val header = JLabel("board: unavailable (not read yet)")

    private val postsArea = compactArea(10)

    private val subjectField = JTextField(18)

    private val bodyArea = JTextArea(2, 18)

    private val postButton = JButton("Post")

    private val readButton = JButton("Refresh")

    private val composer = JPanel(GridLayout(0, 1, 2, 2))

    private var listener: Listener? = null

    private var available = false

    private var seenRevision: Long = 0L

    init {
        bodyArea.lineWrap = true
        bodyArea.wrapStyleWord = true
        subjectField.maximumSize = java.awt.Dimension(Int.MAX_VALUE, 24)
        postButton.isEnabled = false
        readButton.isEnabled = false
        postButton.addActionListener { submitComposer() }
        readButton.addActionListener { if (readButton.isEnabled) listener?.onRead() }

        val row1 = JPanel(FlowLayout(FlowLayout.LEFT, 4, 0))
        row1.add(JLabel("subject"))
        row1.add(subjectField)
        row1.add(readButton)
        val row2 = JPanel(BorderLayout(4, 0))
        row2.add(JLabel("body"), BorderLayout.WEST)
        row2.add(JScrollPane(bodyArea), BorderLayout.CENTER)
        val row3 = JPanel(FlowLayout(FlowLayout.RIGHT, 4, 0))
        row3.add(postButton)
        composer.add(row1)
        composer.add(row2)
        composer.add(row3)

        add(header, BorderLayout.NORTH)
        add(JScrollPane(postsArea), BorderLayout.CENTER)
        add(composer, BorderLayout.SOUTH)
        setAvailable(false)
    }

    fun setListener(value: Listener?) {
        listener = value
    }

    /**
     * Renders one validated page. `acknowledge` is true only for an explicit
     * read: it moves the read watermark so the next render reports 0 unread.
     * Automatic refreshes never mark posts read.
     */
    fun setBoard(page: NativeBoardPage, acknowledge: Boolean = false) {
        setAvailable(true)
        if (acknowledge) {
            seenRevision = maxOf(seenRevision, page.revision)
        }
        val unread = page.posts.count { it.revision > seenRevision }
        header.text = "board: rev=${page.revision} unread=$unread posts=${page.posts.size}" +
            if (page.hasMore) " (older pages exist)" else ""
        val text = StringBuilder()
        for (post in page.posts) {
            val author = post.authorChild?.let { "child:$it" } ?: "root"
            text.append('#').append(post.revision).append(" [").append(author).append("] ")
                .append(bound(post.subject, MAX_BOARD_LINE_CHARS))
                .append(" - ")
                .append(bound(post.body, MAX_BOARD_LINE_CHARS))
            if (post.refs.isNotEmpty()) {
                text.append(" refs=").append(post.refs.joinToString(",", limit = 5))
            }
            text.append('\n')
        }
        postsArea.text = if (text.isEmpty()) {
            "no posts on this run-family board"
        } else {
            text.toString()
        }
    }

    /**
     * Explicit unavailable state: the serving daemon has no board read (or
     * the read failed). The typed reason is recorded verbatim; no posts are
     * ever fabricated and the composer is disabled.
     */
    fun setUnavailable(reason: String) {
        setAvailable(false)
        header.text = "board: unavailable (" + bound(reason, MAX_BOARD_LINE_CHARS) + ")"
        postsArea.text = ""
    }

    /** Clears to the pre-read state (daemon stopped / session switched). */
    fun reset() {
        seenRevision = 0L
        setUnavailable("not read yet")
    }

    fun clearComposer() {
        subjectField.text = ""
        bodyArea.text = ""
    }

    fun available(): Boolean = available

    fun headerText(): String = header.text

    fun postsText(): String = postsArea.text

    fun postEnabled(): Boolean = postButton.isEnabled

    fun readEnabled(): Boolean = readButton.isEnabled

    fun subject(): String = subjectField.text.trim()

    fun body(): String = bodyArea.text

    fun composerVisible(): Boolean = composer.parent != null

    /** Test/composer hook: fills the composer without a display. */
    fun setComposerFields(subject: String, body: String) {
        subjectField.text = subject
        bodyArea.text = body
    }

    /** The Post action exactly as the button runs it (gated on availability). */
    fun submitComposer() {
        if (!postButton.isEnabled) return
        listener?.onPost(
            subjectField.text.trim().take(MAX_BOARD_SUBJECT_CHARS),
            bodyArea.text.take(MAX_BOARD_BODY_CHARS)
        )
    }

    private fun setAvailable(value: Boolean) {
        available = value
        postButton.isEnabled = value
        readButton.isEnabled = value
    }
}
