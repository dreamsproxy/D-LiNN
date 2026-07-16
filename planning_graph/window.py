"""Main window for the LiSNN U1 planning graph editor."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .items import GraphEdgeItem, GraphGroupItem, GraphNodeItem
from .models import EDGE_TYPES, PlanningDocument
from .scene import PlanningScene, PlanningView
from .serialization import load_document, save_document
from .widgets import InspectorPanel, NodePalette


class PlanningGraphWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LiSNN Planning Graph - U1")
        self.resize(1540, 940)
        self.current_path: Path | None = None
        self.dirty = False

        self.scene = PlanningScene()
        self.view = PlanningView(self.scene)
        self.palette = NodePalette()
        self.inspector = InspectorPanel()

        self._build_central_layout()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._connect_signals()
        self.statusBar().showMessage(
            "Drag blocks from the left. Right-click for delete, group, invert, and backdrop controls."
        )
        self._update_title()

    def _build_central_layout(self) -> None:
        palette_panel = QWidget()
        palette_layout = QVBoxLayout(palette_panel)
        palette_title = QLabel("Planning Blocks")
        palette_title.setStyleSheet("font-size: 15px; font-weight: 700;")
        help_label = QLabel("Drag a semantic block onto the canvas.")
        help_label.setWordWrap(True)
        palette_layout.addWidget(palette_title)
        palette_layout.addWidget(help_label)
        palette_layout.addWidget(self.palette, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(palette_panel)
        splitter.addWidget(self.view)
        splitter.addWidget(self.inspector)
        splitter.setSizes([210, 1030, 320])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _build_actions(self) -> None:
        self.new_action = QAction("New", self)
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.triggered.connect(self.new_document)

        self.open_action = QAction("Open...", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_document)

        self.save_action = QAction("Save", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_document)

        self.save_as_action = QAction("Save As...", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(self.save_document_as)

        self.load_example_action = QAction("Load LiSNN Roadmap Example", self)
        self.load_example_action.triggered.connect(self.load_lisnn_example)

        self.delete_action = QAction("Delete Selected", self)
        self.delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        self.delete_action.triggered.connect(self.scene.delete_selected)

        self.duplicate_action = QAction("Duplicate Block", self)
        self.duplicate_action.setShortcut("Ctrl+D")
        self.duplicate_action.triggered.connect(self.scene.duplicate_selected_node)

        self.group_action = QAction("Group Selected Blocks", self)
        self.group_action.setShortcut("Ctrl+G")
        self.group_action.triggered.connect(self.scene.group_selected_nodes)

        self.ungroup_action = QAction("Ungroup Selected", self)
        self.ungroup_action.setShortcut("Ctrl+Shift+G")
        self.ungroup_action.triggered.connect(self.scene.ungroup_selected)

        self.invert_action = QAction("Invert Selection", self)
        self.invert_action.setShortcut("Ctrl+Shift+I")
        self.invert_action.triggered.connect(self.scene.invert_selection)

        self.fit_action = QAction("Fit Graph", self)
        self.fit_action.setShortcut("F")
        self.fit_action.triggered.connect(self.view.fit_graph)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.load_example_action)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self.duplicate_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.group_action)
        edit_menu.addAction(self.ungroup_action)
        edit_menu.addAction(self.invert_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.delete_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.fit_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Planning Graph")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(self.new_action)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.duplicate_action)
        toolbar.addAction(self.group_action)
        toolbar.addAction(self.ungroup_action)
        toolbar.addAction(self.delete_action)
        toolbar.addAction(self.fit_action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("New connection: "))
        self.relation_combo = QComboBox()
        self.relation_combo.addItems(EDGE_TYPES)
        self.relation_combo.currentTextChanged.connect(self._set_default_relation)
        toolbar.addWidget(self.relation_combo)

    def _connect_signals(self) -> None:
        self.scene.document_changed.connect(self._mark_dirty)
        self.scene.selectionChanged.connect(self._selection_changed)
        self.scene.edit_node_requested.connect(self._focus_node)
        self.scene.edit_edge_requested.connect(self._focus_edge)
        self.scene.edit_group_requested.connect(self._focus_group)
        self.scene.status_message.connect(lambda message: self.statusBar().showMessage(message, 5000))
        self.inspector.node_applied.connect(self._apply_node_changes)
        self.inspector.edge_applied.connect(self._apply_edge_changes)
        self.inspector.group_applied.connect(self._apply_group_changes)

    def _set_default_relation(self, relation: str) -> None:
        self.scene.default_relation = relation

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = self.current_path.name if self.current_path else self.scene.document.title
        marker = " *" if self.dirty else ""
        self.setWindowTitle(f"LiSNN Planning Graph - {name}{marker}")

    def _selection_changed(self) -> None:
        selected = self.scene.selectedItems()
        if len(selected) != 1:
            self.inspector.show_empty()
            return
        item = selected[0]
        if isinstance(item, GraphNodeItem):
            self.inspector.show_node(item.model)
        elif isinstance(item, GraphEdgeItem):
            self.inspector.show_edge(item.model)
        elif isinstance(item, GraphGroupItem):
            titles = [
                self.scene.document.nodes[node_id].title
                for node_id in item.model.node_ids
                if node_id in self.scene.document.nodes
            ]
            self.inspector.show_group(item.model, titles)
        else:
            self.inspector.show_empty()

    def _focus_node(self, node_id: str) -> None:
        item = self.scene.node_items.get(node_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        self.inspector.show_node(item.model)
        self.inspector.focus_title()

    def _focus_edge(self, edge_id: str) -> None:
        item = self.scene.edge_items.get(edge_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        self.inspector.show_edge(item.model)
        self.inspector.focus_title()

    def _focus_group(self, group_id: str) -> None:
        item = self.scene.group_items.get(group_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        titles = [
            self.scene.document.nodes[node_id].title
            for node_id in item.model.node_ids
            if node_id in self.scene.document.nodes
        ]
        self.inspector.show_group(item.model, titles)
        self.inspector.focus_title()

    def _apply_node_changes(self, node_id: str, payload: dict) -> None:
        node = self.scene.document.nodes[node_id]
        title = str(payload["title"]).strip()
        if not title:
            QMessageBox.warning(self, "Invalid block", "Block title must not be empty.")
            return
        node.kind = str(payload["kind"])
        node.title = title
        node.status = str(payload["status"])
        node.priority = int(payload["priority"])
        node.tags = list(payload["tags"])
        node.body = str(payload["body"])
        node.__post_init__()
        self.scene.node_items[node_id].update()
        self.scene.document.touch()
        self._mark_dirty()
        self.statusBar().showMessage(f"Updated block: {node.title}", 3000)

    def _apply_edge_changes(self, edge_id: str, payload: dict) -> None:
        edge = self.scene.document.edges[edge_id]
        edge.relation = str(payload["relation"])
        edge.label = str(payload["label"])
        edge.weight = float(payload["weight"])
        edge.__post_init__()
        self.scene.edge_items[edge_id].update_geometry()
        self.scene.document.touch()
        self._mark_dirty()
        self.statusBar().showMessage(f"Updated connection: {edge.relation}", 3000)

    def _apply_group_changes(self, group_id: str, payload: dict) -> None:
        group = self.scene.document.groups[group_id]
        title = str(payload["title"]).strip()
        if not title:
            QMessageBox.warning(self, "Invalid group", "Group title must not be empty.")
            return
        group.title = title
        group.backdrop = bool(payload["backdrop"])
        group.color = str(payload["color"])
        group.layer = int(payload["layer"])
        group.width = float(payload["width"])
        group.height = float(payload["height"])
        try:
            group.__post_init__()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid group", str(exc))
            return
        item = self.scene.group_items[group_id]
        item.set_scene_rect(group.x, group.y, group.width, group.height)
        item.refresh_layer()
        item.update()
        self.scene.document.touch()
        self._mark_dirty()
        self.statusBar().showMessage(f"Updated group: {group.title}", 3000)

    def new_document(self) -> None:
        if not self._confirm_discard_changes():
            return
        self.current_path = None
        self.dirty = False
        self.scene.load_document(PlanningDocument())
        self.inspector.show_empty()
        self._update_title()

    def open_document(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Planning Graph",
            "",
            "LiSNN Planning Graph (*.json);;JSON Files (*.json)",
        )
        if not path:
            return
        self._load_path(Path(path))

    def save_document(self) -> bool:
        if self.current_path is None:
            return self.save_document_as()
        try:
            save_document(self.scene.document, self.current_path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.current_path}", 4000)
        return True

    def save_document_as(self) -> bool:
        suggested = self.current_path.name if self.current_path else "planning_graph.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Planning Graph",
            suggested,
            "LiSNN Planning Graph (*.json);;JSON Files (*.json)",
        )
        if not path:
            return False
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        self.current_path = destination
        return self.save_document()

    def load_lisnn_example(self) -> None:
        if not self._confirm_discard_changes():
            return
        example = Path(__file__).resolve().parent / "examples" / "lisnn_roadmap.json"
        self._load_path(example, treat_as_unsaved=True)

    def _load_path(self, path: Path, *, treat_as_unsaved: bool = False) -> None:
        try:
            document = load_document(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.scene.load_document(document)
        self.current_path = None if treat_as_unsaved else path
        self.dirty = treat_as_unsaved
        self.inspector.show_empty()
        self._update_title()
        self.view.fit_graph()
        self.statusBar().showMessage(f"Loaded {path.name}", 4000)

    def _confirm_discard_changes(self) -> bool:
        if not self.dirty:
            return True
        choice = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save changes before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            return self.save_document()
        return choice == QMessageBox.StandardButton.Discard

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._confirm_discard_changes():
            event.accept()
        else:
            event.ignore()
