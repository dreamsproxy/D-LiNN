"""Canvas scene and view behavior for the U1 planning graph."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSceneDragDropEvent,
    QGraphicsView,
)

from .items import GraphEdgeItem, GraphNodeItem
from .models import PlanningDocument, PlanningEdge, PlanningNode


MIME_NODE_KIND = "application/x-lisnn-planning-node-kind"


class PlanningScene(QGraphicsScene):
    document_changed = pyqtSignal()
    edit_node_requested = pyqtSignal(str)
    edit_edge_requested = pyqtSignal(str)

    def __init__(self, document: PlanningDocument | None = None) -> None:
        super().__init__(-5000.0, -5000.0, 10000.0, 10000.0)
        self.document = PlanningDocument() if document is None else document
        self.node_items: dict[str, GraphNodeItem] = {}
        self.edge_items: dict[str, GraphEdgeItem] = {}
        self.default_relation = "Related"
        self.connection_active = False
        self._connection_source: GraphNodeItem | None = None
        self._temp_edge = QGraphicsPathItem()
        self._temp_edge.setPen(QPen(QColor("#6e7781"), 2.0, Qt.PenStyle.DashLine))
        self._temp_edge.setZValue(-2.0)
        self.addItem(self._temp_edge)
        self._temp_edge.hide()
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.BspTreeIndex)
        self.load_document(self.document)

    def load_document(self, document: PlanningDocument) -> None:
        self.clear()
        self.document = document
        self.node_items = {}
        self.edge_items = {}
        self._temp_edge = QGraphicsPathItem()
        self._temp_edge.setPen(QPen(QColor("#6e7781"), 2.0, Qt.PenStyle.DashLine))
        self._temp_edge.setZValue(-2.0)
        self.addItem(self._temp_edge)
        self._temp_edge.hide()

        for node in document.nodes.values():
            self._add_node_item(node)
        for edge in document.edges.values():
            self._add_edge_item(edge)

    def create_node(self, kind: str, pos: QPointF, title: str | None = None) -> PlanningNode:
        node = PlanningNode.create(
            kind=kind,
            title=title or f"New {kind}",
            x=pos.x(),
            y=pos.y(),
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

    def request_edit_node(self, node_id: str) -> None:
        self.edit_node_requested.emit(node_id)

    def request_edit_edge(self, edge_id: str) -> None:
        self.edit_edge_requested.emit(edge_id)

    def node_moved(self, node_id: str) -> None:
        for edge_item in self.edge_items.values():
            if node_id in (edge_item.model.source_id, edge_item.model.target_id):
                edge_item.update_geometry()
        self.document.touch()
        self.document_changed.emit()

    def delete_selected(self) -> None:
        selected = list(self.selectedItems())
        edge_ids = [item.model.edge_id for item in selected if isinstance(item, GraphEdgeItem)]
        node_ids = [item.model.node_id for item in selected if isinstance(item, GraphNodeItem)]

        for edge_id in edge_ids:
            self.remove_edge(edge_id)
        for node_id in node_ids:
            self.remove_node(node_id)
        if edge_ids or node_ids:
            self.document_changed.emit()

    def remove_edge(self, edge_id: str) -> None:
        item = self.edge_items.pop(edge_id, None)
        if item is not None:
            self.removeItem(item)
        self.document.remove_edge(edge_id)

    def remove_node(self, node_id: str) -> None:
        removed_edges = self.document.remove_node(node_id)
        for edge_id in removed_edges:
            item = self.edge_items.pop(edge_id, None)
            if item is not None:
                self.removeItem(item)
        item = self.node_items.pop(node_id, None)
        if item is not None:
            self.removeItem(item)

    def duplicate_selected_node(self) -> None:
        selected = [item for item in self.selectedItems() if isinstance(item, GraphNodeItem)]
        if len(selected) != 1:
            return
        source = selected[0].model
        node = PlanningNode.create(
            kind=source.kind,
            title=f"{source.title} (copy)",
            body=source.body,
            status=source.status,
            priority=source.priority,
            tags=source.tags.copy(),
            x=source.x + 36.0,
            y=source.y + 36.0,
        )
        self.document.add_node(node)
        item = self._add_node_item(node)
        self.clearSelection()
        item.setSelected(True)
        self.document_changed.emit()

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
        minor = 25
        major = 100
        left = int(rect.left()) - (int(rect.left()) % minor)
        top = int(rect.top()) - (int(rect.top()) % minor)

        minor_pen = QPen(QColor("#edf0f3"), 1.0)
        major_pen = QPen(QColor("#d8dde3"), 1.0)
        x = left
        while x < rect.right():
            painter.setPen(major_pen if x % major == 0 else minor_pen)
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += minor
        y = top
        while y < rect.bottom():
            painter.setPen(major_pen if y % major == 0 else minor_pen)
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += minor


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
