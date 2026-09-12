// Dedicated blockers section: every blocked child with its blocker kind,
// reason, dependency, suggested resolution and the action buttons that
// apply (resume / permission reply / retry), plus the session's pending
// permission requests with allow/deny replies over the daemon's permission
// reply route. Pure presentation: actions are delivered to a Listener.
package dev.faktor.frontend

import dev.faktor.shared.NativePermissionEntry
import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.GridLayout
import javax.swing.DefaultListModel
import javax.swing.JButton
import javax.swing.JLabel
import javax.swing.JList
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JTextArea
import javax.swing.ListSelectionModel

class BlockersPanel : JPanel(BorderLayout()) {

    interface Listener {
        fun onBlockerAction(row: BlockerRow, action: BlockerAction)
        fun onPermissionReply(permission: NativePermissionEntry, decision: String)
    }

    private val blockersModel = DefaultListModel<BlockerRow>()

    private val blockersList = JList(blockersModel)

    private val permissionsModel = DefaultListModel<NativePermissionEntry>()

    private val permissionsList = JList(permissionsModel)

    private val taskBlockersArea = compactArea(2)

    private val detail = compactArea(4)

    private val resumeButton = JButton("Resume")

    private val retryButton = JButton("Retry")

    private val allowButton = JButton("Allow")

    private val denyButton = JButton("Deny")

    private var listener: Listener? = null

    init {
        blockersList.selectionMode = ListSelectionModel.SINGLE_SELECTION
        blockersList.cellRenderer = BlockerCellRenderer()
        blockersList.addListSelectionListener { applySelection() }
        permissionsList.selectionMode = ListSelectionModel.SINGLE_SELECTION
        permissionsList.cellRenderer = PermissionCellRenderer()
        permissionsList.addListSelectionListener { updatePermissionButtons() }

        val body = JPanel(GridLayout(0, 1, 0, 4))
        body.add(titledSection("blocked children", JScrollPane(blockersList)))
        body.add(titledSection("blocker detail", JScrollPane(detail)))

        val actions = JPanel(FlowLayout(FlowLayout.LEFT, 4, 0))
        actions.add(resumeButton)
        actions.add(retryButton)
        actions.add(allowButton)
        actions.add(denyButton)
        body.add(titledSection("actions that apply", actions))

        body.add(titledSection("task blockers", JScrollPane(taskBlockersArea)))
        body.add(titledSection("pending permissions", JScrollPane(permissionsList)))

        add(body, BorderLayout.NORTH)

        resumeButton.addActionListener { withSelected { row -> listener?.onBlockerAction(row, BlockerAction.RESUME) } }
        retryButton.addActionListener { withSelected { row -> listener?.onBlockerAction(row, BlockerAction.RETRY) } }
        allowButton.addActionListener {
            val permission = permissionsList.selectedValue
            if (permission != null) {
                listener?.onPermissionReply(permission, "allow")
            } else {
                withSelected { row -> listener?.onBlockerAction(row, BlockerAction.PERMISSION_ALLOW) }
            }
        }
        denyButton.addActionListener {
            val permission = permissionsList.selectedValue
            if (permission != null) {
                listener?.onPermissionReply(permission, "deny")
            } else {
                withSelected { row -> listener?.onBlockerAction(row, BlockerAction.PERMISSION_DENY) }
            }
        }
        updateButtons(null)
    }

    fun setListener(value: Listener?) {
        listener = value
    }

    /** Replaces the section contents from the task tree model. */
    fun update(model: TaskTreeModel) {
        update(model.blockers, emptyList(), model.taskBlockers)
    }

    fun update(
        blockers: List<BlockerRow>,
        permissions: List<NativePermissionEntry>,
        taskBlockers: List<String>
    ) {
        val selectedId = blockersList.selectedValue?.childId
        blockersModel.clear()
        for (blocker in blockers) blockersModel.addElement(blocker)
        taskBlockersArea.text = if (taskBlockers.isEmpty()) {
            "no task-level blockers"
        } else {
            taskBlockers.joinToString("\n")
        }
        permissionsModel.clear()
        for (permission in permissions) permissionsModel.addElement(permission)
        if (selectedId != null) {
            for (i in 0 until blockersModel.size()) {
                if (blockersModel.getElementAt(i).childId == selectedId) {
                    blockersList.selectedIndex = i
                    break
                }
            }
        } else if (blockersModel.size() > 0) {
            blockersList.selectedIndex = 0
        } else {
            detail.text = "no blocked children"
        }
        updatePermissionButtons()
    }

    fun blockerCount(): Int = blockersModel.size()

    fun permissionCount(): Int = permissionsModel.size()

    /** The action buttons the selected blocker applies (for the smoke). */
    fun applicableActions(): List<BlockerAction> =
        blockersList.selectedValue?.actions ?: emptyList()

    private fun withSelected(block: (BlockerRow) -> Unit) {
        val row = blockersList.selectedValue ?: return
        block(row)
    }

    private fun applySelection() {
        val row = blockersList.selectedValue
        updateButtons(row)
        if (row == null) {
            detail.text = if (blockersModel.size() == 0) "no blocked children" else "select a child"
            return
        }
        val text = StringBuilder()
        text.append("child: ").append(row.childId)
        text.append("\nstate: ").append(row.presence.state.tag)
        text.append("\nblocker kind: ").append(row.kind)
        text.append("\nreason: ").append(row.reason)
        if (row.dependency != null) text.append("\ndependency: ").append(row.dependency)
        if (row.resolution != null) text.append("\nsuggested resolution: ").append(row.resolution)
        if (row.lastProgressMs != null) text.append("\nlast progress ms: ").append(row.lastProgressMs)
        detail.text = text.toString()
    }

    private fun updateButtons(row: BlockerRow?) {
        val actions = row?.actions ?: emptyList()
        resumeButton.isEnabled = actions.contains(BlockerAction.RESUME)
        retryButton.isEnabled = actions.contains(BlockerAction.RETRY)
        allowButton.isEnabled = actions.contains(BlockerAction.PERMISSION_ALLOW) || permissionsList.selectedValue != null
        denyButton.isEnabled = actions.contains(BlockerAction.PERMISSION_DENY) || permissionsList.selectedValue != null
    }

    private fun updatePermissionButtons() {
        val permissionSelected = permissionsList.selectedValue != null
        if (permissionSelected) {
            allowButton.isEnabled = true
            denyButton.isEnabled = true
        } else {
            val actions = blockersList.selectedValue?.actions ?: emptyList()
            allowButton.isEnabled = actions.contains(BlockerAction.PERMISSION_ALLOW)
            denyButton.isEnabled = actions.contains(BlockerAction.PERMISSION_DENY)
        }
    }

    private class BlockerCellRenderer : javax.swing.DefaultListCellRenderer() {
        private val panel = JPanel(BorderLayout(4, 0))
        private val label = JLabel()
        private var spriteId: String? = null
        private var sprite: PixelSprite? = null

        override fun getListCellRendererComponent(
            list: JList<*>?,
            value: Any?,
            index: Int,
            selected: Boolean,
            focus: Boolean
        ): java.awt.Component {
            val row = value as? BlockerRow
                ?: return super.getListCellRendererComponent(list, value, index, selected, focus)
            if (spriteId != row.childId) {
                spriteId = row.childId
                sprite = PixelSprite(row.childId, row.presence.state.tag)
                panel.removeAll()
                panel.add(sprite, BorderLayout.WEST)
                panel.add(label, BorderLayout.CENTER)
            }
            sprite?.setState(row.presence.state.tag)
            label.text = "${row.childId} [${row.presence.state.tag}/${row.kind}] " +
                bound(row.reason, 120)
            panel.background = if (selected) list?.selectionBackground else list?.background
            panel.isOpaque = true
            label.foreground = if (selected) list?.selectionForeground else list?.foreground
            return panel
        }
    }

    private class PermissionCellRenderer : javax.swing.DefaultListCellRenderer() {
        override fun getListCellRendererComponent(
            list: JList<*>?,
            value: Any?,
            index: Int,
            selected: Boolean,
            focus: Boolean
        ): java.awt.Component {
            val permission = value as? NativePermissionEntry
            val text = if (permission == null) {
                ""
            } else {
                "#${permission.id} ${permission.capability} ${bound(permission.detail, 80)}"
            }
            return super.getListCellRendererComponent(list, text, index, selected, focus)
        }
    }
}
