"""Offscreen smoke tests for the PySide6 planning graph alpha."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from planning_graph.definitions import DefinitionRegistry
from planning_graph.scene import PlanningScene
from planning_graph.window import PlanningGraphWindow


_APP = QApplication.instance() or QApplication([])


class PlanningGraphGuiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        path = Path(self.temporary_directory.name) / "custom_blocks.json"
        self.registry = DefinitionRegistry.defaults(path)
        self.registry.save()
        self.scene = PlanningScene(self.registry)

    def tearDown(self) -> None:
        self.scene.clear()
        self.temporary_directory.cleanup()

    def test_scene_defaults_to_leads_to(self) -> None:
        self.assertEqual(self.scene.default_relation, "Leads To")

    def test_create_undo_redo_round_trip(self) -> None:
        self.scene.create_node("Idea", QPointF(12.0, 13.0))
        self.assertEqual(len(self.scene.document.nodes), 1)
        self.scene.undo_stack.undo()
        self.assertEqual(len(self.scene.document.nodes), 0)
        self.scene.undo_stack.redo()
        self.assertEqual(len(self.scene.document.nodes), 1)

    def test_live_property_edits_merge_into_one_undo(self) -> None:
        node = self.scene.create_node("Idea", QPointF())
        self.scene.update_node_properties(node.node_id, {"title": "A"})
        self.scene.update_node_properties(node.node_id, {"title": "AB"})
        self.scene.update_node_properties(node.node_id, {"title": "ABC"})
        self.assertEqual(self.scene.document.nodes[node.node_id].title, "ABC")
        self.scene.undo_stack.undo()
        self.assertEqual(
            self.scene.document.nodes[node.node_id].title,
            "New Idea",
        )

    def test_live_block_dimensions_update_item_geometry(self) -> None:
        node = self.scene.create_node("Idea", QPointF())
        self.scene.update_node_properties(
            node.node_id,
            {"width": 275.0, "height": 225.0},
        )
        current = self.scene.document.nodes[node.node_id]
        item = self.scene.node_items[node.node_id]
        self.assertEqual(current.width, 300.0)
        self.assertEqual(current.height, 250.0)
        self.assertEqual(item.content_rect().width(), current.width)
        self.assertEqual(item.content_rect().height(), current.height)

    def test_parallel_departures_receive_distinct_lanes(self) -> None:
        source = self.scene.create_node("Idea", QPointF(0.0, 0.0))
        first = self.scene.create_node("Task", QPointF(400.0, -150.0))
        second = self.scene.create_node("Task", QPointF(400.0, 150.0))
        edge_a = self.scene.create_edge(source.node_id, first.node_id)
        edge_b = self.scene.create_edge(source.node_id, second.node_id)
        offsets = {
            self.scene.edge_items[edge_a.edge_id].lane_offset,
            self.scene.edge_items[edge_b.edge_id].lane_offset,
        }
        self.assertEqual(len(offsets), 2)
        self.assertNotEqual(offsets, {0.0})

    def test_manual_route_can_reset_to_automatic(self) -> None:
        source = self.scene.create_node("Idea", QPointF())
        target = self.scene.create_node("Task", QPointF(400.0, 0.0))
        edge = self.scene.create_edge(source.node_id, target.node_id)
        edge.route_x = 100.0
        edge.route_y = 100.0
        edge.__post_init__()
        self.scene.edge_items[edge.edge_id].update_geometry()
        self.assertTrue(edge.has_manual_route)
        self.scene.reset_edge_route(edge.edge_id)
        self.assertFalse(self.scene.document.edges[edge.edge_id].has_manual_route)

    def test_window_constructs_with_dark_alpha_components(self) -> None:
        window = PlanningGraphWindow()
        try:
            self.assertEqual(window.scene.default_relation, "Leads To")
            self.assertEqual(window.relation_combo.currentText(), "Leads To")
            self.assertGreater(window.palette.count(), 0)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
