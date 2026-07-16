"""Tests for the GUI-independent U1 planning graph model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from planning_graph.grid import (
    GRID_SIZE,
    NODE_HEIGHT,
    NODE_WIDTH,
    is_grid_aligned,
    snap_value,
    snap_xy,
)
from planning_graph.models import PlanningDocument, PlanningEdge, PlanningNode
from planning_graph.serialization import load_document, save_document


class PlanningGraphModelTests(unittest.TestCase):
    def test_round_trip_preserves_graph(self) -> None:
        document = PlanningDocument(title="Test Plan")
        idea = PlanningNode.create(kind="Idea", title="Candidate mechanism")
        task = PlanningNode.create(kind="Task", title="Run the ablation")
        document.add_node(idea)
        document.add_node(task)
        document.add_edge(
            PlanningEdge.create(
                source_id=idea.node_id,
                target_id=task.node_id,
                relation="Leads To",
                label="test with",
            )
        )

        decoded = PlanningDocument.from_dict(json.loads(json.dumps(document.to_dict())))
        self.assertEqual(decoded.to_dict(), document.to_dict())

    def test_rejects_dangling_edge(self) -> None:
        payload = {
            "version": 1,
            "title": "Invalid",
            "nodes": [
                {
                    "node_id": "N1",
                    "kind": "Idea",
                    "title": "Known",
                }
            ],
            "edges": [
                {
                    "edge_id": "E1",
                    "source_id": "N1",
                    "target_id": "N2",
                    "relation": "Related",
                }
            ],
        }
        with self.assertRaises(ValueError):
            PlanningDocument.from_dict(payload)

    def test_remove_node_cascades_connections(self) -> None:
        document = PlanningDocument()
        first = PlanningNode.create(kind="Goal", title="First")
        second = PlanningNode.create(kind="Task", title="Second")
        document.add_node(first)
        document.add_node(second)
        edge = PlanningEdge.create(
            source_id=first.node_id,
            target_id=second.node_id,
            relation="Depends On",
        )
        document.add_edge(edge)

        removed = document.remove_node(first.node_id)
        self.assertEqual(removed, [edge.edge_id])
        self.assertNotIn(first.node_id, document.nodes)
        self.assertFalse(document.edges)

    def test_file_save_and_load(self) -> None:
        document = PlanningDocument(title="Persistent")
        document.add_node(PlanningNode.create(kind="Note", title="Remember this"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            save_document(document, path)
            restored = load_document(path)
        self.assertEqual(restored.title, "Persistent")
        self.assertEqual(len(restored.nodes), 1)

    def test_self_edge_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PlanningEdge.create(
                source_id="N1",
                target_id="N1",
                relation="Related",
            )

    def test_grid_snapping_rounds_half_cells_away_from_zero(self) -> None:
        self.assertEqual(snap_value(12.4), 0.0)
        self.assertEqual(snap_value(12.5), GRID_SIZE)
        self.assertEqual(snap_value(-12.4), 0.0)
        self.assertEqual(snap_value(-12.5), -GRID_SIZE)
        self.assertEqual(snap_value(37.5), 2.0 * GRID_SIZE)

    def test_snap_xy_aligns_both_coordinates(self) -> None:
        x, y = snap_xy(63.0, -88.0)
        self.assertEqual((x, y), (75.0, -100.0))
        self.assertTrue(is_grid_aligned(x))
        self.assertTrue(is_grid_aligned(y))

    def test_node_dimensions_and_ports_align_to_grid(self) -> None:
        self.assertEqual(NODE_WIDTH, 10.0 * GRID_SIZE)
        self.assertEqual(NODE_HEIGHT, 6.0 * GRID_SIZE)
        self.assertTrue(is_grid_aligned(NODE_WIDTH / 2.0))
        self.assertTrue(is_grid_aligned(NODE_HEIGHT / 2.0))


if __name__ == "__main__":
    unittest.main()
