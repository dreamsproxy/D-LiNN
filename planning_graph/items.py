"""QGraphics items used by the planning graph canvas."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from .models import PlanningEdge, PlanningNode

if TYPE_CHECKING:
    from .scene import PlanningScene


NODE_WIDTH = 230.0
NODE_HEIGHT = 142.0
HEADER_HEIGHT = 34.0
PORT_RADIUS = 5.0
PORT_HIT_RADIUS = 13.0

NODE_COLORS = {
    "Idea": QColor("#d9e8ff"),
    "Goal": QColor("#d8f5df"),
    "Question": QColor("#efe0ff"),
    "Task": QColor("#fff0c9"),
    "Evidence": QColor("#d7f3f5"),
    "Decision": QColor("#ffd9c9"),
    "Constraint": QColor("#ffd8df"),
    "Result": QColor("#dff3d5"),
    "Note": QColor("#ececec"),
}

EDGE_COLORS = {
    "Related": QColor("#65727e"),
    "Supports": QColor("#27864b"),
    "Contradicts": QColor("#c33d3d"),
    "Depends On": QColor("#5d4fb2"),
    "Leads To": QColor("#2878b5"),
    "Part Of": QColor("#8b5a2b"),
    "Refines": QColor("#b26a18"),
    "Blocks": QColor("#8f3030"),
}


class GraphNodeItem(QGraphicsObject):
    """Visual node backed directly by a :class:`PlanningNode`."""

    def __init__(self, model: PlanningNode) -> None:
        super().__init__()
        self.model = model
        self.hovered = False
        self.setPos(model.x, model.y)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def boundingRect(self) -> QRectF:
        return QRectF(-NODE_WIDTH / 2, -NODE_HEIGHT / 2, NODE_WIDTH, NODE_HEIGHT)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.boundingRect()
        base = NODE_COLORS.get(self.model.kind, QColor("#eeeeee"))
        outline = QColor("#2d333b")
        if self.isSelected():
            outline = QColor("#0969da")

        painter.setPen(QPen(outline, 2.5 if self.isSelected() else 1.5))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        header = QRectF(rect.left(), rect.top(), rect.width(), HEADER_HEIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(base))
        painter.drawRoundedRect(header, 10.0, 10.0)
        painter.drawRect(
            QRectF(header.left(), header.bottom() - 10.0, header.width(), 10.0)
        )

        painter.setPen(QPen(QColor("#1f2328"), 1.0))
        kind_font = QFont()
        kind_font.setPointSize(8)
        kind_font.setBold(True)
        painter.setFont(kind_font)
        painter.drawText(
            QRectF(header.left() + 10.0, header.top(), 82.0, header.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.model.kind.upper(),
        )
        painter.drawText(
            QRectF(header.right() - 98.0, header.top(), 88.0, header.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            self.model.status,
        )

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor("#111111"), 1.0))
        painter.drawText(
            QRectF(rect.left() + 12.0, header.bottom() + 8.0, rect.width() - 24.0, 42.0),
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
            | Qt.TextFlag.TextWordWrap,
            self.model.title,
        )

        body_font = QFont()
        body_font.setPointSize(8)
        painter.setFont(body_font)
        snippet = self.model.body.strip().replace("\n", " ")
        if len(snippet) > 96:
            snippet = snippet[:93] + "..."
        painter.setPen(QPen(QColor("#4a4f55"), 1.0))
        painter.drawText(
            QRectF(rect.left() + 12.0, header.bottom() + 50.0, rect.width() - 24.0, 40.0),
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
            | Qt.TextFlag.TextWordWrap,
            snippet,
        )

        footer = []
        if self.model.priority:
            footer.append(f"P{self.model.priority}")
        if self.model.tags:
            footer.append("#" + " #".join(self.model.tags[:3]))
        painter.setPen(QPen(QColor("#6e7781"), 1.0))
        painter.drawText(
            QRectF(rect.left() + 12.0, rect.bottom() - 22.0, rect.width() - 24.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "  ".join(footer),
        )

        if self.hovered or self.isSelected():
            painter.setPen(QPen(QColor("#24292f"), 1.0))
            painter.setBrush(QBrush(QColor("#ffffff")))
            for point in self.port_positions().values():
                painter.drawEllipse(point, PORT_RADIUS, PORT_RADIUS)

    def port_positions(self) -> dict[str, QPointF]:
        rect = self.boundingRect()
        return {
            "top": QPointF(0.0, rect.top()),
            "right": QPointF(rect.right(), 0.0),
            "bottom": QPointF(0.0, rect.bottom()),
            "left": QPointF(rect.left(), 0.0),
        }

    def port_at(self, local_pos: QPointF) -> str | None:
        for name, point in self.port_positions().items():
            if math.hypot(local_pos.x() - point.x(), local_pos.y() - point.y()) <= PORT_HIT_RADIUS:
                return name
        return None

    def best_port_scene(self, toward: QPointF) -> QPointF:
        candidates = [self.mapToScene(point) for point in self.port_positions().values()]
        return min(
            candidates,
            key=lambda point: (point.x() - toward.x()) ** 2 + (point.y() - toward.y()) ** 2,
        )

    def hoverEnterEvent(self, event) -> None:
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            port = self.port_at(event.pos())
            scene = self.scene()
            if port is not None and hasattr(scene, "start_connection"):
                scene.start_connection(self, self.mapToScene(self.port_positions()[port]))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "connection_active") and scene.connection_active:
            scene.update_connection(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "connection_active") and scene.connection_active:
            scene.finish_connection(event.scenePos())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "request_edit_node"):
            scene.request_edit_node(self.model.node_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.model.x = float(self.pos().x())
            self.model.y = float(self.pos().y())
            scene = self.scene()
            if hasattr(scene, "node_moved"):
                scene.node_moved(self.model.node_id)
        return result


class GraphEdgeItem(QGraphicsObject):
    """Directed, typed connection between two graph nodes."""

    def __init__(
        self,
        model: PlanningEdge,
        source: GraphNodeItem,
        target: GraphNodeItem,
    ) -> None:
        super().__init__()
        self.model = model
        self.source = source
        self.target = target
        self.path = QPainterPath()
        self.arrow = QPolygonF()
        self.label_point = QPointF()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(-1.0)
        self.update_geometry()

    def update_geometry(self) -> None:
        self.prepareGeometryChange()
        source_center = self.source.scenePos()
        target_center = self.target.scenePos()
        start = self.source.best_port_scene(target_center)
        end = self.target.best_port_scene(source_center)

        dx = end.x() - start.x()
        control = max(abs(dx) * 0.45, 70.0)
        c1 = QPointF(start.x() + (control if dx >= 0 else -control), start.y())
        c2 = QPointF(end.x() - (control if dx >= 0 else -control), end.y())

        path = QPainterPath(start)
        path.cubicTo(c1, c2, end)
        self.path = path
        self.label_point = path.pointAtPercent(0.5)

        tangent_a = path.pointAtPercent(0.97)
        angle = math.atan2(end.y() - tangent_a.y(), end.x() - tangent_a.x())
        arrow_size = 10.0
        left = QPointF(
            end.x() - arrow_size * math.cos(angle - math.pi / 6),
            end.y() - arrow_size * math.sin(angle - math.pi / 6),
        )
        right = QPointF(
            end.x() - arrow_size * math.cos(angle + math.pi / 6),
            end.y() - arrow_size * math.sin(angle + math.pi / 6),
        )
        self.arrow = QPolygonF([end, left, right])
        self.update()

    def boundingRect(self) -> QRectF:
        return self.path.boundingRect().adjusted(-28.0, -28.0, 28.0, 28.0)

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(14.0)
        return stroker.createStroke(self.path)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = EDGE_COLORS.get(self.model.relation, QColor("#65727e"))
        width = 3.0 if self.isSelected() else 2.0
        painter.setPen(QPen(color, width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(self.arrow)

        label = self.model.label.strip() or self.model.relation
        font = QFont()
        font.setPointSize(8)
        font.setBold(self.isSelected())
        painter.setFont(font)
        metrics = painter.fontMetrics()
        label_rect = metrics.boundingRect(label).adjusted(-5, -3, 5, 3)
        label_rect.moveCenter(self.label_point.toPoint())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 225)))
        painter.drawRoundedRect(QRectF(label_rect), 4.0, 4.0)
        painter.setPen(QPen(color, 1.0))
        painter.drawText(QRectF(label_rect), Qt.AlignmentFlag.AlignCenter, label)

    def mouseDoubleClickEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "request_edit_edge"):
            scene.request_edit_edge(self.model.edge_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
