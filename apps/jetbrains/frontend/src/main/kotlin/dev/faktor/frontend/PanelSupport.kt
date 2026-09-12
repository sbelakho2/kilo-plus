// Small shared Swing helpers for the Faktor panels (no third-party deps).
package dev.faktor.frontend

import java.awt.BorderLayout
import java.awt.Component
import java.awt.Font
import javax.swing.BorderFactory
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JTextArea

/** A read-only, wrapped, monospace text area sized for compact payloads. */
internal fun compactArea(rows: Int, cols: Int = 36): JTextArea {
    val area = JTextArea(rows, cols)
    area.isEditable = false
    area.lineWrap = true
    area.wrapStyleWord = true
    area.font = Font(Font.MONOSPACED, Font.PLAIN, 12)
    return area
}

/** A titled vertical section with a bordered body. */
internal fun titledSection(title: String, body: Component): JPanel {
    val panel = JPanel(BorderLayout(0, 2))
    panel.border = BorderFactory.createEmptyBorder(4, 6, 4, 6)
    panel.add(JLabel(title).apply { font = font.deriveFont(Font.BOLD) }, BorderLayout.NORTH)
    panel.add(body, BorderLayout.CENTER)
    return panel
}

/** Wraps a component in a scroll pane with a vertical-only policy default. */
internal fun scroll(component: Component): JScrollPane = JScrollPane(component)

/** Bounds a display string to [max] chars with an ellipsis marker. */
internal fun bound(text: String?, max: Int): String {
    val value = text ?: ""
    if (value.length <= max) return value
    return value.substring(0, max) + "..."
}
