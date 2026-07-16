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

from .grid import (
    GRID_SIZE,
    GROUP_MIN_HEIGHT,
    GROUP_MIN_WIDTH,
    NODE_HEIGHT,
    NODE_WIDTH,
    snap_value,
    snap_xy,
)
from .models import PlanningEdge, PlanningGroup, PlanningNode
from .theme import (
    EDGE_COLORS,
    MUTED_TEXT,
    NODE_ACCENTS,
    NODE_BODY,
    NODE_BORDER,
    NODE_HEADER,
    SELECTION,
    TEXT,
)

if TYPE_CHECKING:
    from .scene import PlanningScene


HEADER_HEIGHT = 34.0
PORT_RADIUS = 5.0
PORT_HIT_RADIUS = 13.0
GROUP_Z_BASE = -1000.0
GROUP_Z_STEP = 0.1
GROUP_HANDLE_SIZE = 12.0
GROUP_LABEL_HEIGHT = 26.0


class GraphNodeItem(QGraphicsObject):
    """Visual node backed directly by a :class:`PlanningNode`."""

    def __init__(self, model: PlanningNode) -> None:
        super().__init__()
        self.model = model
        self.hovered = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

        snapped_x, snapped_y = snap_xy(model.x, model.y)
        self.model.x = snapped_x
        self.model.y = snapped_y
        self.setPos(snapped_x, snapped_y)

    def boundingRect(self) -> QRectF:
        return QRectF(-NODE_WIDTH / 2, -NODE_HEIGHT / 2, NODE_WIDTH, NODE_HEIGHT)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.boundingRect()
        outline = QColor(SELECTION if self.isSelected() else NODE_BORDER)
        accent = QColor(NODE_ACCENTS.get(self.model.kind, MUTED_TEXT))

        painter.setPen(QPen(outline, 2.5 if self.isSelected() else 1.4))
        painter.setBrush(QBrush(QColor(NODE_BODY)))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        header = QRectF(rect.left(), rect.top(), rect.width(), HEADER_HEIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(NODE_HEADER)))
        painter.drawRoundedRect(header, 10.0, 10.0)
        painter.drawRect(
            QRectF(header.left(), header.bottom() - 10.0, header.width(), 10.0)
        )
        painter.setBrush(QBrush(accent))
        painter.drawRoundedRect(
            QRectF(rect.left(), rect.top(), 6.0, rect.height()),
            3.0,
            3.0,
        )

        painter.setPen(QPen(QColor(TEXT), 1.0))
        kind_font = QFont()
        kind_font.setPointSize(8)
        kind_font.setBold(True)
        painter.setFont(kind_font)
        painter.drawText(
            QRectF(header.left() + 14.0, header.top(), 92.0, header.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.model.kind.upper(),
        )
        painter.drawText(
            QRectF(header.right() - 106.0, header.top(), 94.0, header.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            self.model.status,
        )

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(TEXT), 1.0))
        painter.drawText(
            QRectF(rect.left() + 14.0, header.bottom() + 8.0, rect.width() - 28.0, 42.0),
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
        painter.setPen(QPen(QColor(MUTED_TEXT), 1.0))
        painter.drawText(
            QRectF(rect.left() + 14.0, header.bottom() + 50.0, rect.width() - 28.0, 40.0),
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
        painter.setPen(QPen(QColor(MUTED_TEXT), 1.0))
        painter.drawText(
            QRectF(rect.left() + 14.0, rect.bottom() - 22.0, rect.width() - 28.0, 16.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "  ".join(footer),
        )

        if self.hovered or self.isSelected():
            painter.setPen(QPen(QColor(SELECTION if self.isSelected() else TEXT), 1.0))
            painter.setBrush(QBrush(QColor(NODE_BODY)))
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
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            value = QPointF(snap_value(value.x()), snap_value(value.y()))

        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.model.x = float(self.pos().x())
            self.model.y = float(self.pos().y())
            scene = self.scene()
            if hasattr(scene, "node_moved"):
                scene.node_moved(self.model.node_id)
        return result


class GraphGroupItem(QGraphicsObject):
    """Selectable, movable, grid-resizable visual container for a group."""

    def __init__(self, model: PlanningGroup) -> None:
        super().__init__()
        self.model = model
        self._resizing = False
        self._resize_handle: str | None = None
        self._resize_origin: tuple[float, float, float, float] | None = None
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPos(snap_value(model.x), snap_value(model.y))
        self.model.x = float(self.pos().x())
        self.model.y = float(self.pos().y())
        self.model.width = max(GROUP_MIN_WIDTH, snap_value(model.width))
        self.model.height = max(GROUP_MIN_HEIGHT, snap_value(model.height))
        self.refresh_layer()

    def boundingRect(self) -> QRectF:
        padding = GROUP_HANDLE_SIZE / 2.0 + 2.0
        return QRectF(
            -padding,
            -padding,
            self.model.width + 2 * padding,
            self.model.height + 2 * padding,
        )

    def content_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.model.width, self.model.height)

    def refresh_layer(self) -> None:
        self.setZValue(GROUP_Z_BASE + float(self.model.layer) * GROUP_Z_STEP)
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.content_rect()
        color = QColor(self.model.color)
        selected = self.isSelected()

        fill = QColor(color)
        fill.setAlpha(58 if self.model.backdrop else 0)
        outline = QColor(SELECTION if selected else color)
        outline.setAlpha(255 if selected else (205 if self.model.backdrop else 130))
        pen = QPen(outline, 2.2 if selected else 1.5)
        if not self.model.backdrop:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(rect, 12.0, 12.0)

        label_font = QFont()
        label_font.setPointSize(9)
        label_font.setBold(True)
        painter.setFont(label_font)
        metrics = painter.fontMetrics()
        label_width = min(
            max(110.0, float(metrics.horizontalAdvance(self.model.title) + 24)),
            max(110.0, self.model.width - 16.0),
        )
        label_rect = QRectF(8.0, 8.0, label_width, GROUP_LABEL_HEIGHT)
        label_fill = QColor(color)
        label_fill.setAlpha(225)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(label_fill))
        painter.drawRoundedRect(label_rect, 6.0, 6.0)
        painter.setPen(QPen(QColor(TEXT), 1.0))
        painter.drawText(
            label_rect.adjusted(9.0, 0.0, -9.0, 0.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.model.title,
        )

        if selected:
            painter.setPen(QPen(QColor(SELECTION), 1.0))
            painter.setBrush(QBrush(QColor(NODE_BODY)))
            for handle_rect in self._handle_rects().values():
                painter.drawRect(handle_rect)

    def _handle_rects(self) -> dict[str, QRectF]:
        half = GROUP_HANDLE_SIZE / 2.0
        width = self.model.width
        height = self.model.height
        return {
            "top_left": QRectF(-half, -half, GROUP_HANDLE_SIZE, GROUP_HANDLE_SIZE),
            "top_right": QRectF(width - half, -half, GROUP_HANDLE_SIZE, GROUP_HANDLE_SIZE),
            "bottom_left": QRectF(-half, height - half, GROUP_HANDLE_SIZE, GROUP_HANDLE_SIZE),
            "bottom_right": QRectF(
                width - half,
                height - half,
                GROUP_HANDLE_SIZE,
                GROUP_HANDLE_SIZE,
            ),
        }

    def _handle_at(self, point: QPointF) -> str | None:
        for name, rect in self._handle_rects().items():
            if rect.contains(point):
                return name
        return None

    def hoverMoveEvent(self, event) -> None:
        handle = self._handle_at(event.pos()) if self.isSelected() else None
        if handle in ("top_left", "bottom_right"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in ("top_right", "bottom_left"):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isSelected():
            handle = self._handle_at(event.pos())
            if handle is not None:
                self._resizing = True
                self._resize_handle = handle
                self._resize_origin = (
                    self.scenePos().x(),
                    self.scenePos().y(),
                    self.model.width,
                    self.model.height,
                )
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._resizing and self._resize_origin and self._resize_handle:
            left, top, width, height = self._resize_origin
            right = left + width
            bottom = top + height
            x = snap_value(event.scenePos().x())
            y = snap_value(event.scenePos().y())

            if "left" in self._resize_handle:
                left = min(x, right - GROUP_MIN_WIDTH)
            else:
                right = max(x, left + GROUP_MIN_WIDTH)
            if "top" in self._resize_handle:
                top = min(y, bottom - GROUP_MIN_HEIGHT)
            else:
                bottom = max(y, top + GROUP_MIN_HEIGHT)

            self.set_scene_rect(left, top, right - left, bottom - top)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._resizing:
            self._resizing = False
            self._resize_handle = None
            self._resize_origin = None
            scene = self.scene()
            if hasattr(scene, "group_geometry_changed"):
                scene.group_geometry_changed(self.model.group_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        scene = self.scene()
        if hasattr(scene, "request_edit_group"):
            scene.request_edit_group(self.model.group_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_scene_rect(self, x: float, y: float, width: float, height: float) -> None:
        x = snap_value(x)
        y = snap_value(y)
        width = max(GROUP_MIN_WIDTH, snap_value(width))
        height = max(GROUP_MIN_HEIGHT, snap_value(height))
        self.prepareGeometryChange()
        was_resizing = self._resizing
        self._resizing = True
        self.setPos(x, y)
        self.model.x = x
        self.model.y = y
        self.model.width = width
        self.model.height = height
        self._resizing = was_resizing
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            value = QPointF(snap_value(value.x()), snap_value(value.y()))

        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            new_x = float(self.pos().x())
            new_y = float(self.pos().y())
            dx = new_x - self.model.x
            dy = new_y - self.model.y
            self.model.x = new_x
            self.model.y = new_y
            scene = self.scene()
            if (
                not self._resizing
                and (dx or dy)
                and hasattr(scene, "move_group_nodes")
            ):
                scene.move_group_nodes(self.model.group_id, dx, dy)
            if hasattr(scene, "group_geometry_changed"):
                scene.group_geometry_changed(self.model.group_id)
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
        color = QColor(EDGE_COLORS.get(self.model.relation, MUTED_TEXT))
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
        label_rect = metrics.boundingRect(label).adjusted(-6, -4, 6, 4)
        label_rect.moveCenter(self.label_point.toPoint())
        painter.setPen(Qt.PenStyle.NoPen)
        label_fill = QColor(NODE_BODY)
        label_fill.setAlpha(235)
        painter.setBrush(QBrush(label_fill))
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
