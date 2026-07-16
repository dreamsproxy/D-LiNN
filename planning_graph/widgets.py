"""Palette and inspector widgets for the planning graph editor."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .models import EDGE_TYPES, NODE_KINDS, NODE_STATUSES, PlanningEdge, PlanningNode
from .scene import MIME_NODE_KIND


class NodePalette(QListWidget):
    """Drag source for semantic planning node types."""

    def __init__(self) -> None:
        super().__init__()
        self.setDragEnabled(True)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setSpacing(3)
        for kind in NODE_KINDS:
            item = QListWidgetItem(kind)
            item.setData(Qt.ItemDataRole.UserRole, kind)
            item.setToolTip(f"Drag a {kind} node onto the canvas")
            self.addItem(item)

    def startDrag(self, supported_actions) -> None:
        item = self.currentItem()
        if item is None:
            return
        kind = str(item.data(Qt.ItemDataRole.UserRole))
        mime = QMimeData()
        mime.setData(MIME_NODE_KIND, kind.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class InspectorPanel(QWidget):
    node_applied = Signal(str, dict)
    edge_applied = Signal(str, dict)

    def __init__(self) -> None:
        super().__init__()
        self.current_node_id: str | None = None
        self.current_edge_id: str | None = None

        title = QLabel("Inspector")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")

        self.stack = QStackedWidget()
        self.empty_page = self._build_empty_page()
        self.node_page = self._build_node_page()
        self.edge_page = self._build_edge_page()
        self.stack.addWidget(self.empty_page)
        self.stack.addWidget(self.node_page)
        self.stack.addWidget(self.edge_page)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.stack, 1)
        self.show_empty()

    def _build_empty_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        message = QLabel(
            "Select a node or connection to inspect it.\n\n"
            "Double-click an item to focus this panel."
        )
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(message)
        layout.addStretch(1)
        return page

    def _build_node_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()

        self.node_kind = QComboBox()
        self.node_kind.addItems(NODE_KINDS)
        self.node_title = QLineEdit()
        self.node_status = QComboBox()
        self.node_status.addItems(NODE_STATUSES)
        self.node_priority = QSpinBox()
        self.node_priority.setRange(0, 5)
        self.node_tags = QLineEdit()
        self.node_tags.setPlaceholderText("comma, separated, tags")
        self.node_body = QTextEdit()
        self.node_body.setPlaceholderText("Reasoning, details, assumptions, or notes...")
        self.node_body.setMinimumHeight(180)

        form.addRow("Kind", self.node_kind)
        form.addRow("Title", self.node_title)
        form.addRow("Status", self.node_status)
        form.addRow("Priority", self.node_priority)
        form.addRow("Tags", self.node_tags)
        form.addRow("Body", self.node_body)
        layout.addLayout(form)

        apply_button = QPushButton("Apply Node Changes")
        apply_button.clicked.connect(self._apply_node)
        layout.addWidget(apply_button)
        layout.addStretch(1)
        return page

    def _build_edge_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()

        self.edge_relation = QComboBox()
        self.edge_relation.addItems(EDGE_TYPES)
        self.edge_label = QLineEdit()
        self.edge_label.setPlaceholderText("Optional custom edge label")
        self.edge_weight = QDoubleSpinBox()
        self.edge_weight.setRange(-1_000_000.0, 1_000_000.0)
        self.edge_weight.setDecimals(4)
        self.edge_weight.setValue(1.0)

        form.addRow("Relation", self.edge_relation)
        form.addRow("Label", self.edge_label)
        form.addRow("Weight", self.edge_weight)
        layout.addLayout(form)

        apply_button = QPushButton("Apply Connection Changes")
        apply_button.clicked.connect(self._apply_edge)
        layout.addWidget(apply_button)
        layout.addStretch(1)
        return page

    def show_empty(self) -> None:
        self.current_node_id = None
        self.current_edge_id = None
        self.stack.setCurrentWidget(self.empty_page)

    def show_node(self, node: PlanningNode) -> None:
        self.current_node_id = node.node_id
        self.current_edge_id = None
        self.node_kind.setCurrentText(node.kind)
        self.node_title.setText(node.title)
        self.node_status.setCurrentText(node.status)
        self.node_priority.setValue(node.priority)
        self.node_tags.setText(", ".join(node.tags))
        self.node_body.setPlainText(node.body)
        self.stack.setCurrentWidget(self.node_page)

    def show_edge(self, edge: PlanningEdge) -> None:
        self.current_edge_id = edge.edge_id
        self.current_node_id = None
        self.edge_relation.setCurrentText(edge.relation)
        self.edge_label.setText(edge.label)
        self.edge_weight.setValue(edge.weight)
        self.stack.setCurrentWidget(self.edge_page)

    def focus_title(self) -> None:
        if self.stack.currentWidget() is self.node_page:
            self.node_title.setFocus()
            self.node_title.selectAll()
        elif self.stack.currentWidget() is self.edge_page:
            self.edge_label.setFocus()
            self.edge_label.selectAll()

    def _apply_node(self) -> None:
        if self.current_node_id is None:
            return
        payload: dict[str, Any] = {
            "kind": self.node_kind.currentText(),
            "title": self.node_title.text().strip(),
            "status": self.node_status.currentText(),
            "priority": self.node_priority.value(),
            "tags": [tag.strip() for tag in self.node_tags.text().split(",") if tag.strip()],
            "body": self.node_body.toPlainText(),
        }
        self.node_applied.emit(self.current_node_id, payload)

    def _apply_edge(self) -> None:
        if self.current_edge_id is None:
            return
        payload = {
            "relation": self.edge_relation.currentText(),
            "label": self.edge_label.text().strip(),
            "weight": self.edge_weight.value(),
        }
        self.edge_applied.emit(self.current_edge_id, payload)
