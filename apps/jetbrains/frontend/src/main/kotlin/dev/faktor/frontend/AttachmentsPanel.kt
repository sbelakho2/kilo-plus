// File attachments for the Task entry: a file chooser plus a drop list
// (java.awt.dnd via Swing's TransferHandler) whose paths are submitted as
// the `files` array of the native task-run request. Pure presentation: the
// panel only owns the path list; callers read [files] when starting a run.
package dev.faktor.frontend

import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.datatransfer.DataFlavor
import java.io.File
import java.nio.file.Paths
import javax.swing.DefaultListModel
import javax.swing.JButton
import javax.swing.JFileChooser
import javax.swing.JList
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.ListSelectionModel
import javax.swing.TransferHandler

class AttachmentsPanel : JPanel(BorderLayout()) {

    private val paths = DefaultListModel<String>()

    private val list = JList(paths)

    private val addButton = JButton("Add files...")

    private val removeButton = JButton("Remove")

    private val clearButton = JButton("Clear")

    init {
        list.selectionMode = ListSelectionModel.MULTIPLE_INTERVAL_SELECTION
        list.toolTipText = "drop files here to attach them to the task"
        list.transferHandler = object : TransferHandler() {
            override fun canImport(support: TransferSupport): Boolean =
                support.isDataFlavorSupported(DataFlavor.javaFileListFlavor)

            override fun importData(support: TransferSupport): Boolean {
                if (!canImport(support)) return false
                @Suppress("UNCHECKED_CAST")
                val files = support.transferable
                    .getTransferData(DataFlavor.javaFileListFlavor) as List<File>
                addFiles(files.map { it.absolutePath })
                return true
            }
        }

        val buttons = JPanel(FlowLayout(FlowLayout.LEFT, 4, 0))
        buttons.add(addButton)
        buttons.add(removeButton)
        buttons.add(clearButton)

        addButton.addActionListener { chooseFiles() }
        removeButton.addActionListener {
            val selected = list.selectedValuesList
            for (path in selected) paths.removeElement(path)
        }
        clearButton.addActionListener { paths.clear() }

        val body = JPanel(BorderLayout(0, 2))
        body.add(JScrollPane(list), BorderLayout.CENTER)
        body.add(buttons, BorderLayout.SOUTH)
        add(body, BorderLayout.CENTER)
    }

    /** The attached absolute paths, in list order (the task-run `files` array). */
    fun files(): List<String> {
        val out = ArrayList<String>()
        for (i in 0 until paths.size()) out.add(paths.getElementAt(i))
        return out
    }

    fun addFiles(pathsToAdd: List<String>) {
        for (path in pathsToAdd) {
            val normalized = normalize(path) ?: continue
            if (!contains(normalized)) paths.addElement(normalized)
        }
    }

    fun count(): Int = paths.size()

    fun clear() {
        paths.clear()
    }

    private fun chooseFiles() {
        val chooser = JFileChooser()
        chooser.isMultiSelectionEnabled = true
        chooser.fileSelectionMode = JFileChooser.FILES_ONLY
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            val selected = chooser.selectedFiles ?: emptyArray()
            addFiles(selected.map { it.absolutePath })
        }
    }

    private fun normalize(path: String): String? {
        val trimmed = path.trim()
        if (trimmed.isEmpty()) return null
        return try {
            Paths.get(trimmed).toAbsolutePath().normalize().toString()
        } catch (e: Exception) {
            null
        }
    }

    private fun contains(path: String): Boolean {
        for (i in 0 until paths.size()) {
            if (paths.getElementAt(i) == path) return true
        }
        return false
    }
}
