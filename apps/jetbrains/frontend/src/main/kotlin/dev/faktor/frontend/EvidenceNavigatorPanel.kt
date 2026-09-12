// Direct transcript/evidence navigation: the evidence list with one-click
// retrieval through the native evidence retrieve endpoint, typed selectors
// (all / byte range / line range / search), a compact output pane, the
// durable message rows (selection jumps the main transcript to the message
// slice) and a child-transcript pane for child rows. Pure presentation:
// retrieval and transcript jumps are delegated to a Listener.
package dev.faktor.frontend

import dev.faktor.shared.NativeMessage
import dev.faktor.shared.NativeRequests
import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.GridLayout
import javax.swing.DefaultListModel
import javax.swing.JButton
import javax.swing.JComboBox
import javax.swing.JLabel
import javax.swing.JList
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JTextField
import javax.swing.ListSelectionModel

class EvidenceNavigatorPanel : JPanel(BorderLayout()) {

    interface Listener {
        fun onRetrieve(evidenceId: Long, selectorJson: String)
        fun onMessageSelected(seq: Long)
    }

    private val evidenceModel = DefaultListModel<EvidenceRef>()

    private val evidenceList = JList(evidenceModel)

    private val modeBox = JComboBox(arrayOf("all", "byte_range", "line_range", "search"))

    private val startField = JTextField(6)

    private val endField = JTextField(6)

    private val queryField = JTextField(12)

    private val maxHitsField = JTextField(4)

    private val retrieveButton = JButton("Retrieve")

    private val output = compactArea(9)

    private val messagesModel = DefaultListModel<NativeMessage>()

    private val messagesList = JList(messagesModel)

    private val transcriptArea = compactArea(8)

    private var listener: Listener? = null

    init {
        maxHitsField.text = "50"
        evidenceList.selectionMode = ListSelectionModel.SINGLE_SELECTION
        evidenceList.cellRenderer = EvidenceCellRenderer()
        evidenceList.addListSelectionListener {
            val ref = evidenceList.selectedValue ?: return@addListSelectionListener
            output.text = "selected ${describe(ref)} - press Retrieve (evidence refs from the tree retrieve instantly)"
        }
        messagesList.selectionMode = ListSelectionModel.SINGLE_SELECTION
        messagesList.cellRenderer = MessageCellRenderer()
        messagesList.addListSelectionListener {
            val message = messagesList.selectedValue ?: return@addListSelectionListener
            listener?.onMessageSelected(message.seq)
        }
        modeBox.addActionListener { updateSelectorFields() }
        retrieveButton.addActionListener { retrieve() }

        val controls = JPanel(GridLayout(0, 1, 2, 2))
        val modeRow = JPanel(FlowLayout(FlowLayout.LEFT, 4, 0))
        modeRow.add(JLabel("selector"))
        modeRow.add(modeBox)
        controls.add(modeRow)
        val rangeRow = JPanel(FlowLayout(FlowLayout.LEFT, 4, 0))
        rangeRow.add(JLabel("start"))
        rangeRow.add(startField)
        rangeRow.add(JLabel("end"))
        rangeRow.add(endField)
        rangeRow.add(JLabel("query"))
        rangeRow.add(queryField)
        rangeRow.add(JLabel("max hits"))
        rangeRow.add(maxHitsField)
        rangeRow.add(retrieveButton)
        controls.add(rangeRow)

        val evidence = JPanel(BorderLayout(0, 2))
        evidence.add(JScrollPane(evidenceList), BorderLayout.CENTER)
        evidence.add(controls, BorderLayout.SOUTH)

        val body = JPanel(GridLayout(0, 1, 0, 4))
        body.add(titledSection("evidence", evidence))
        body.add(titledSection("compact representation", JScrollPane(output)))
        body.add(
            titledSection(
                "messages (select to jump the transcript)",
                JScrollPane(messagesList)
            )
        )
        body.add(titledSection("transcript slice", JScrollPane(transcriptArea)))
        add(body, BorderLayout.NORTH)
        updateSelectorFields()
    }

    fun setListener(value: Listener?) {
        listener = value
    }

    fun setEvidence(refs: List<EvidenceRef>) {
        val selectedId = evidenceList.selectedValue?.id
        evidenceModel.clear()
        for (ref in refs) evidenceModel.addElement(ref)
        if (selectedId != null) {
            for (i in 0 until evidenceModel.size()) {
                if (evidenceModel.getElementAt(i).id == selectedId) {
                    evidenceList.selectedIndex = i
                    break
                }
            }
        }
    }

    fun evidenceCount(): Int = evidenceModel.size()

    fun messageCount(): Int = messagesModel.size()

    fun setMessages(messages: List<NativeMessage>) {
        messagesModel.clear()
        for (message in messages) messagesModel.addElement(message)
    }

    /** Selects one ref and immediately retrieves it (one-click from the tree). */
    fun selectEvidence(ref: EvidenceRef) {
        for (i in 0 until evidenceModel.size()) {
            if (evidenceModel.getElementAt(i) == ref) {
                evidenceList.selectedIndex = i
                break
            }
        }
        if (ref.id == null) {
            output.text = "evidence ref carries no numeric id: ${ref.label}"
            return
        }
        modeBox.selectedItem = "all"
        retrieve()
    }

    fun showRetrieval(evidenceId: Long, text: String, byteLen: Long, truncated: Boolean) {
        output.text = "evidence $evidenceId: $byteLen bytes retrieved" +
            (if (truncated) " (truncated by policy)" else "") + "\n" + text
    }

    fun showError(evidenceId: Long, message: String) {
        output.text = "evidence $evidenceId: $message"
    }

    fun showTranscriptSlice(title: String, text: String) {
        transcriptArea.text = "== $title ==\n$text"
    }

    fun selectedEvidence(): EvidenceRef? = evidenceList.selectedValue

    /** The selector body the Retrieve button would send (null when invalid). */
    fun selectorJson(): String? {
        val mode = modeBox.selectedItem as? String ?: "all"
        return when (mode) {
            "all" -> NativeRequests.evidenceSelectorAll()
            "byte_range" -> {
                val start = startField.text.trim().toLongOrNull()
                val end = endField.text.trim().toLongOrNull()
                if (start == null || end == null || start < 0 || end < start) {
                    output.text = "byte range needs 0 <= start <= end"
                    null
                } else {
                    NativeRequests.evidenceSelectorRange(start, end)
                }
            }
            "line_range" -> {
                val start = startField.text.trim().toLongOrNull()
                val end = endField.text.trim().toLongOrNull()
                if (start == null || end == null || start < 1 || end < start) {
                    output.text = "line range needs 1 <= start <= end"
                    null
                } else {
                    NativeRequests.evidenceSelectorLines(start, end)
                }
            }
            else -> {
                val query = queryField.text
                val maxHits = maxHitsField.text.trim().toLongOrNull() ?: 50L
                if (query.isEmpty() || maxHits < 1) {
                    output.text = "search needs a query and max_hits >= 1"
                    null
                } else {
                    NativeRequests.evidenceSelectorSearch(query, maxHits)
                }
            }
        }
    }

    private fun retrieve() {
        val ref = evidenceList.selectedValue
        if (ref == null) {
            output.text = "select an evidence ref first"
            return
        }
        val id = ref.id
        if (id == null) {
            output.text = "evidence ref carries no numeric id: ${ref.label}"
            return
        }
        val selector = selectorJson() ?: return
        listener?.onRetrieve(id, selector)
    }

    private fun updateSelectorFields() {
        val mode = modeBox.selectedItem as? String ?: "all"
        val range = mode == "byte_range" || mode == "line_range"
        startField.isEnabled = range
        endField.isEnabled = range
        queryField.isEnabled = mode == "search"
        maxHitsField.isEnabled = mode == "search"
    }

    private fun describe(ref: EvidenceRef): String =
        if (ref.id == null) bound(ref.label, 160) else "evidence:${ref.id}"

    private class EvidenceCellRenderer : javax.swing.DefaultListCellRenderer() {
        override fun getListCellRendererComponent(
            list: JList<*>?,
            value: Any?,
            index: Int,
            selected: Boolean,
            focus: Boolean
        ): java.awt.Component {
            val ref = value as? EvidenceRef
            val text = if (ref == null) {
                ""
            } else if (ref.id == null) {
                "[ref] ${bound(ref.label, 100)}"
            } else {
                "evidence:${ref.id} ${bound(ref.label, 100)}"
            }
            return super.getListCellRendererComponent(list, text, index, selected, focus)
        }
    }

    private class MessageCellRenderer : javax.swing.DefaultListCellRenderer() {
        override fun getListCellRendererComponent(
            list: JList<*>?,
            value: Any?,
            index: Int,
            selected: Boolean,
            focus: Boolean
        ): java.awt.Component {
            val message = value as? NativeMessage
            val text = if (message == null) {
                ""
            } else {
                "#${message.seq} ${message.role}: ${bound(message.text, 100)}"
            }
            return super.getListCellRendererComponent(list, text, index, selected, focus)
        }
    }
}
