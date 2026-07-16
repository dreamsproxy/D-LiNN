"""Canvas scene and view behavior for the U1 planning graph."""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QColorDialog,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSceneContextMenuEvent,
    QGraphicsSceneDragDropEvent,
    QGraphicsView,
    QMenu,
)

from .grid import GRID_SIZE, snap_rect_outward, snap_xy
from .items import GraphEdgeItem, GraphGroupItem, GraphNodeItem
from .models import (
    PlanningDocument,
    PlanningEdge,
    PlanningGroup,
    PlanningNode,
    normalize_hex_color,
)
from .theme import CANVAS_BACKGROUND, GRID_DOT, GROUP_COLOR_PRESETS, MUTED_TEXT


MIME_NODE_KIND = "application/x-lisnn-planning-node-kind"
GROUP_MARGIN = 2.0 * GRID_SIZE


class PlanningScene(QGraphicsScene):
    document_changed = Signal()
    edit_node_requested = Signal(str)
    edit_edge_requested = Signal(str)
    edit_group_requested = Signal(str)
    status_message = Signal(str)

    def __init__(self, document: PlanningDocument | None = None) -> None:
        super().__init__(-5000.0, -5000.0, 10000.0, 10000.0)
        self.document = PlanningDocument() if document is None else document
        self.node_items: dict[str, GraphNodeItem] = {}
        self.edge_items: dict[str, GraphEdgeItem] = {}
        self.group_items: dict[str, GraphGroupItem] = {}
        self.default_relation = "Related"
        self.connection_active = False
        self._connection_source: GraphNodeItem | None = None
        self._moving_group = False
        self._temp_edge = QGraphicsPathItem()
        self._temp_edge.setPen(QPen(QColor(MUTED_TEXT), 2.0, Qt.PenStyle.DashLine))
        self._temp_edge.setZValue(-2.0)
        self.addItem(self._temp_edge)
        self._temp_edge.hide()
        self.setBackgroundBrush(QColor(CANVAS_BACKGROUND))
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.BspTreeIndex)
        self.load_document(self.document)

    def load_document(self, document: PlanningDocument) -> None:
        self.clear()
        self.document = document
        self.node_items = {}
        self.edge_items = {}
        self.group_items = {}
        self._temp_edge = QGraphicsPathItem()
        self._temp_edge.setPen(QPen(QColor(MUTED_TEXT), 2.0, Qt.PenStyle.DashLine))
        self._temp_edge.setZValue(-2.0)
        self.addItem(self._temp_edge)
        self._temp_edge.hide()

        for group in document.groups.values():
            self._add_group_item(group)
        for node in document.nodes.values():
            self._add_node_item(node)
        for edge in document.edges.values():
            self._add_edge_item(edge)

    def create_node(self, kind: str, pos: QPointF, title: str | None = None) -> PlanningNode:
        x, y = snap_xy(pos.x(), pos.y())
        node = PlanningNode.create(
            kind=kind,
            title=title or f"New {kind}",
            x=x,
            y=y,
        )
        self.document.add_node(node)
        item = self._add_node_item(node)
        self.clearSelection()
        item.setSelected(True)
        self.document_changed.emit()
        return node

    def create_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str | None = None,
    ) -> PlanningEdge:
        edge = PlanningEdge.create(
            source_id=source_id,
            target_id=target_id,
            relation=relation or self.default_relation,
        )
        self.document.add_edge(edge)
        item = self._add_edge_item(edge)
        self.clearSelection()
        item.setSelected(True)
        self.document_changed.emit()
        return edge

    def _add_node_item(self, node: PlanningNode) -> GraphNodeItem:
        item = GraphNodeItem(node)
        self.node_items[node.node_id] = item
        self.addItem(item)
        return item

    def _add_edge_item(self, edge: PlanningEdge) -> GraphEdgeItem:
        item = GraphEdgeItem(
            edge,
            self.node_items[edge.source_id],
            self.node_items[edge.target_id],
        )
        self.edge_items[edge.edge_id] = item
        self.addItem(item)
        return item

    def _add_group_item(self, group: PlanningGroup) -> GraphGroupItem:
        item = GraphGroupItem(group)
        self.group_items[group.group_id] = item
        self.addItem(item)
        return item

    def request_edit_node(self, node_id: str) -> None:
        self.edit_node_requested.emit(node_id)

    def request_edit_edge(self, edge_id: str) -> None:
        self.edit_edge_requested.emit(edge_id)

    def request_edit_group(self, group_id: str) -> None:
        self.edit_group_requested.emit(group_id)

    def node_moved(self, node_id: str) -> None:
        for edge_item in self.edge_items.values():
            if node_id in (edge_item.model.source_id, edge_item.model.target_id):
                edge_item.update_geometry()
        self.document.touch()
        self.document_changed.emit()

    def group_geometry_changed(self, group_id: str) -> None:
        if group_id in self.document.groups:
            self.document.touch()
            self.document_changed.emit()

    def move_group_nodes(self, group_id: str, dx: float, dy: float) -> None:
        group = self.document.groups.get(group_id)
        if group is None or self._moving_group:
            return
        self._moving_group = True
        try:
            for node_id in group.node_ids:
                item = self.node_items.get(node_id)
                if item is not None:
                    item.setPos(item.pos().x() + dx, item.pos().y() + dy)
        finally:
            self._moving_group = False
        self.document.touch()
        self.document_changed.emit()

    def delete_selected(self) -> None:
        selected = list(self.selectedItems())
        group_ids = [
            item.model.group_id for item in selected if isinstance(item, GraphGroupItem)
        ]
        edge_ids = [item.model.edge_id for item in selected if isinstance(item, GraphEdgeItem)]
        node_ids = [item.model.node_id for item in selected if isinstance(item, GraphNodeItem)]

        for group_id in group_ids:
            self.remove_group(group_id)
        for edge_id in edge_ids:
            self.remove_edge(edge_id)
        for node_id in node_ids:
            self.remove_node(node_id)
        if group_ids or edge_ids or node_ids:
            self.document_changed.emit()

    def remove_edge(self, edge_id: str) -> None:
        item = self.edge_items.pop(edge_id, None)
        if item is not None:
            self.removeItem(item)
        self.document.remove_edge(edge_id)

    def remove_group(self, group_id: str) -> None:
        item = self.group_items.pop(group_id, None)
        if item is not None:
            self.removeItem(item)
        self.document.remove_group(group_id)

    def remove_node(self, node_id: str) -> None:
        groups_before = set(self.document.groups)
        removed_edges = self.document.remove_node(node_id)
        for edge_id in removed_edges:
            item = self.edge_items.pop(edge_id, None)
            if item is not None:
                self.removeItem(item)
        item = self.node_items.pop(node_id, None)
        if item is not None:
            self.removeItem(item)

        removed_groups = groups_before.difference(self.document.groups)
        for group_id in removed_groups:
            group_item = self.group_items.pop(group_id, None)
            if group_item is not None:
                self.removeItem(group_item)
        for group_id, group_item in self.group_items.items():
            if group_id in self.document.groups:
                group_item.update()

    def duplicate_selected_node(self) -> None:
        selected = [item for item in self.selectedItems() if isinstance(item, GraphNodeItem)]
        if len(selected) != 1:
            return
        source = selected[0].model
        x, y = snap_xy(
            source.x + 2.0 * GRID_SIZE,
            source.y + 2.0 * GRID_SIZE,
        )
        node = PlanningNode.create(
            kind=source.kind,
            title=f"{source.title} (copy)",
            body=source.body,
            status=source.status,
            priority=source.priority,
            tags=source.tags.copy(),
            x=x,
            y=y,
        )
        self.document.add_node(node)
        item = self._add_node_item(node)
        self.clearSelection()
        item.setSelected(True)
        self.document_changed.emit()

    def selected_node_items(self) -> list[GraphNodeItem]:
        return [item for item in self.selectedItems() if isinstance(item, GraphNodeItem)]

    def selected_group_items(self) -> list[GraphGroupItem]:
        return [item for item in self.selectedItems() if isinstance(item, GraphGroupItem)]

    def can_group_selection(self) -> bool:
        nodes = self.selected_node_items()
        return len(nodes) >= 2 and all(
            self.document.group_for_node(item.model.node_id) is None for item in nodes
        )

    def group_selected_nodes(self) -> PlanningGroup | None:
        nodes = self.selected_node_items()
        if len(nodes) < 2:
            self.status_message.emit("Select at least two blocks before grouping.")
            return None
        occupied = [
            item.model.title
            for item in nodes
            if self.document.group_for_node(item.model.node_id) is not None
        ]
        if occupied:
            self.status_message.emit(
                "Ungroup existing members before regrouping: " + ", ".join(occupied)
            )
            return None

        bounds = nodes[0].sceneBoundingRect()
        for item in nodes[1:]:
            bounds = bounds.united(item.sceneBoundingRect())
        left, top, right, bottom = snap_rect_outward(
            bounds.left() - GROUP_MARGIN,
            bounds.top() - GROUP_MARGIN,
            bounds.right() + GROUP_MARGIN,
            bounds.bottom() + GROUP_MARGIN,
        )
        layer = max((group.layer for group in self.document.groups.values()), default=-1) + 1
        color = GROUP_COLOR_PRESETS[len(self.document.groups) % len(GROUP_COLOR_PRESETS)]
        group = PlanningGroup.create(
            title=f"Group {len(self.document.groups) + 1}",
            node_ids=[item.model.node_id for item in nodes],
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
            backdrop=True,
            color=color,
            layer=layer,
        )
        self.document.add_group(group)
        group_item = self._add_group_item(group)
        self.clearSelection()
        group_item.setSelected(True)
        self.document_changed.emit()
        self.status_message.emit(f"Created {group.title} with {len(group.node_ids)} blocks.")
        return group

    def ungroup_selected(self) -> None:
        group_ids = [item.model.group_id for item in self.selected_group_items()]
        for group_id in group_ids:
            self.remove_group(group_id)
        if group_ids:
            self.document_changed.emit()
            self.status_message.emit(f"Ungrouped {len(group_ids)} group(s).")

    def invert_selection(self) -> None:
        selectable = [
            *self.node_items.values(),
            *self.edge_items.values(),
            *self.group_items.values(),
        ]
        states = {item: item.isSelected() for item in selectable}
        for item, selected in states.items():
            item.setSelected(not selected)

    def toggle_group_backdrop(self, group_id: str) -> None:
        group = self.document.groups.get(group_id)
        if group is None:
            return
        group.backdrop = not group.backdrop
        self.group_items[group_id].update()
        self.document.touch()
        self.document_changed.emit()

    def set_group_color(self, group_id: str, color: str) -> None:
        group = self.document.groups.get(group_id)
        if group is None:
            return
        group.color = normalize_hex_color(color)
        self.group_items[group_id].update()
        self.document.touch()
        self.document_changed.emit()

    def set_group_layer(self, group_id: str, layer: int) -> None:
        group = self.document.groups.get(group_id)
        if group is None:
            return
        group.layer = int(layer)
        self.group_items[group_id].refresh_layer()
        self.document.touch()
        self.document_changed.emit()

    def bring_group_to_front(self, group_id: str) -> None:
        layer = max((group.layer for group in self.document.groups.values()), default=0) + 1
        self.set_group_layer(group_id, layer)

    def send_group_to_back(self, group_id: str) -> None:
        layer = min((group.layer for group in self.document.groups.values()), default=0) - 1
        self.set_group_layer(group_id, layer)

    def move_group_layer(self, group_id: str, delta: int) -> None:
        group = self.document.groups.get(group_id)
        if group is not None:
            self.set_group_layer(group_id, group.layer + int(delta))

    def start_connection(self, source: GraphNodeItem, start: QPointF) -> None:
        self.connection_active = True
        self._connection_source = source
        self._temp_edge.setPath(QPainterPath(start))
        self._temp_edge.show()

    def update_connection(self, end: QPointF) -> None:
        if not self.connection_active or self._connection_source is None:
            return
        start = self._connection_source.best_port_scene(end)
        path = QPainterPath(start)
        midpoint = (start.x() + end.x()) / 2.0
        path.cubicTo(QPointF(midpoint, start.y()), QPointF(midpoint, end.y()), end)
        self._temp_edge.setPath(path)

    def finish_connection(self, end: QPointF) -> None:
        source = self._connection_source
        self.connection_active = False
        self._connection_source = None
        self._temp_edge.hide()
        self._temp_edge.setPath(QPainterPath())
        if source is None:
            return

        target = None
        for item in self.items(end):
            if isinstance(item, GraphNodeItem):
                target = item
                break
        if target is None or target is source:
            return
        self.create_edge(source.model.node_id, target.model.node_id)

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent) -> None:
        clicked = next(
            (
                item
                for item in self.items(event.scenePos())
                if isinstance(item, (GraphNodeItem, GraphEdgeItem, GraphGroupItem))
            ),
            None,
        )
        if clicked is not None and not clicked.isSelected():
            self.clearSelection()
            clicked.setSelected(True)

        menu = QMenu()
        delete_action = menu.addAction("Delete")
        delete_action.setEnabled(bool(self.selectedItems()))
        group_action = menu.addAction("Group")
        group_action.setEnabled(self.can_group_selection())
        invert_action = menu.addAction("Invert Selection")

        group_item = clicked if isinstance(clicked, GraphGroupItem) else None
        color_actions: dict[object, str] = {}
        if group_item is not None:
            menu.addSeparator()
            ungroup_action = menu.addAction("Ungroup")
            backdrop_text = "Remove Backdrop" if group_item.model.backdrop else "Add Backdrop"
            backdrop_action = menu.addAction(backdrop_text)

            color_menu = menu.addMenu("Backdrop Color")
            for color in GROUP_COLOR_PRESETS:
                action = color_menu.addAction(color)
                color_actions[action] = color
            custom_color_action = color_menu.addAction("Custom...")

            layer_menu = menu.addMenu("Layer")
            front_action = layer_menu.addAction("Bring to Front")
            forward_action = layer_menu.addAction("Bring Forward")
            backward_action = layer_menu.addAction("Send Backward")
            back_action = layer_menu.addAction("Send to Back")
        else:
            ungroup_action = None
            backdrop_action = None
            custom_color_action = None
            front_action = forward_action = backward_action = back_action = None

        chosen = menu.exec(event.screenPos())
        if chosen is None:
            return
        if chosen is delete_action:
            self.delete_selected()
        elif chosen is group_action:
            self.group_selected_nodes()
        elif chosen is invert_action:
            self.invert_selection()
        elif chosen is ungroup_action and group_item is not None:
            self.remove_group(group_item.model.group_id)
            self.document_changed.emit()
        elif chosen is backdrop_action and group_item is not None:
            self.toggle_group_backdrop(group_item.model.group_id)
        elif chosen in color_actions and group_item is not None:
            self.set_group_color(group_item.model.group_id, color_actions[chosen])
        elif chosen is custom_color_action and group_item is not None:
            parent = self.views()[0] if self.views() else None
            color = QColorDialog.getColor(QColor(group_item.model.color), parent, "Backdrop Color")
            if color.isValid():
                self.set_group_color(group_item.model.group_id, color.name())
        elif chosen is front_action and group_item is not None:
            self.bring_group_to_front(group_item.model.group_id)
        elif chosen is forward_action and group_item is not None:
            self.move_group_layer(group_item.model.group_id, 1)
        elif chosen is backward_action and group_item is not None:
            self.move_group_layer(group_item.model.group_id, -1)
        elif chosen is back_action and group_item is not None:
            self.send_group_to_back(group_item.model.group_id)

    def dragEnterEvent(self, event: QGraphicsSceneDragDropEvent) -> None:
        if event.mimeData().hasFormat(MIME_NODE_KIND):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QGraphicsSceneDragDropEvent) -> None:
        if event.mimeData().hasFormat(MIME_NODE_KIND):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QGraphicsSceneDragDropEvent) -> None:
        if event.mimeData().hasFormat(MIME_NODE_KIND):
            kind = bytes(event.mimeData().data(MIME_NODE_KIND)).decode("utf-8")
            self.create_node(kind, event.scenePos())
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)

        left = math.floor(rect.left() / GRID_SIZE) * GRID_SIZE
        top = math.floor(rect.top() / GRID_SIZE) * GRID_SIZE
        points = QPolygonF()

        x = left
        while x <= rect.right():
            y = top
            while y <= rect.bottom():
                points.append(QPointF(x, y))
                y += GRID_SIZE
            x += GRID_SIZE

        dot_pen = QPen(QColor(*GRID_DOT))
        dot_pen.setWidthF(2.0)
        dot_pen.setCosmetic(True)
        dot_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(dot_pen)
        painter.drawPoints(points)


class PlanningView(QGraphicsView):
    """Zoomable and pannable planning canvas."""

    def __init__(self, scene: PlanningScene) -> None:
        super().__init__(scene)
        self.setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._panning = False
        self._pan_start = QPoint()

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        current = self.transform().m11()
        target = current * factor
        if 0.15 <= target <= 4.5:
            self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def fit_graph(self) -> None:
        rect = self.scene().itemsBoundingRect()
        if rect.isValid() and not rect.isEmpty():
            self.fitInView(rect.adjusted(-80, -80, 80, 80), Qt.AspectRatioMode.KeepAspectRatio)
