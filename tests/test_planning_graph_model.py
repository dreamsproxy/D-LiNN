"""Tests for the GUI-independent U1 planning graph model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from planning_graph.grid import (
    GRID_SIZE,
    NODE_HEIGHT,
    NODE_HEIGHT_CELLS,
    NODE_WIDTH,
    NODE_WIDTH_CELLS,
    is_grid_aligned,
    snap_rect_outward,
    snap_value,
)
from planning_graph.models import (
    PlanningDocument,
    PlanningEdge,
    PlanningGroup,
    PlanningNode,
)
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
                {"node_id": "N1", "kind": "Idea", "title": "Known"}
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

    def test_snap_value_uses_grid(self) -> None:
        self.assertEqual(snap_value(36.0), 25.0)
        self.assertEqual(snap_value(38.0), 50.0)
        self.assertEqual(snap_value(-38.0), -50.0)

    def test_node_dimensions_are_exact_grid_multiples(self) -> None:
        self.assertEqual(NODE_WIDTH, GRID_SIZE * NODE_WIDTH_CELLS)
        self.assertEqual(NODE_HEIGHT, GRID_SIZE * NODE_HEIGHT_CELLS)
        self.assertTrue(is_grid_aligned(NODE_WIDTH / 2.0))
        self.assertTrue(is_grid_aligned(NODE_HEIGHT / 2.0))

    def test_snap_rect_expands_outward(self) -> None:
        left, top, right, bottom = snap_rect_outward(1.0, 24.0, 51.0, 76.0)
        self.assertEqual((left, top, right, bottom), (0.0, 0.0, 75.0, 100.0))

    def test_group_round_trip_preserves_membership_and_backdrop(self) -> None:
        document = PlanningDocument(title="Grouped")
        first = PlanningNode.create(kind="Idea", title="First")
        second = PlanningNode.create(kind="Task", title="Second")
        document.add_node(first)
        document.add_node(second)
        group = PlanningGroup.create(
            title="Investigation",
            node_ids=[first.node_id, second.node_id],
            x=-50.0,
            y=-25.0,
            width=400.0,
            height=250.0,
            color="#3B5B78",
            layer=4,
        )
        document.add_group(group)

        restored = PlanningDocument.from_dict(document.to_dict())
        self.assertEqual(restored.groups[group.group_id].node_ids, group.node_ids)
        self.assertEqual(restored.groups[group.group_id].color, "#3B5B78")
        self.assertEqual(restored.groups[group.group_id].layer, 4)

    def test_group_rejects_unknown_members(self) -> None:
        document = PlanningDocument()
        first = PlanningNode.create(kind="Idea", title="First")
        document.add_node(first)
        group = PlanningGroup.create(
            title="Invalid",
            node_ids=[first.node_id, "missing"],
            x=0.0,
            y=0.0,
            width=250.0,
            height=150.0,
        )
        with self.assertRaises(ValueError):
            document.add_group(group)

    def test_node_cannot_belong_to_multiple_groups(self) -> None:
        document = PlanningDocument()
        nodes = [PlanningNode.create(kind="Idea", title=f"N{i}") for i in range(3)]
        for node in nodes:
            document.add_node(node)
        document.add_group(
            PlanningGroup.create(
                title="One",
                node_ids=[nodes[0].node_id, nodes[1].node_id],
                x=0.0,
                y=0.0,
                width=250.0,
                height=150.0,
            )
        )
        with self.assertRaises(ValueError):
            document.add_group(
                PlanningGroup.create(
                    title="Two",
                    node_ids=[nodes[1].node_id, nodes[2].node_id],
                    x=0.0,
                    y=0.0,
                    width=250.0,
                    height=150.0,
                )
            )

    def test_removing_member_removes_group_when_fewer_than_two_remain(self) -> None:
        document = PlanningDocument()
        first = PlanningNode.create(kind="Idea", title="First")
        second = PlanningNode.create(kind="Task", title="Second")
        document.add_node(first)
        document.add_node(second)
        group = PlanningGroup.create(
            title="Temporary",
            node_ids=[first.node_id, second.node_id],
            x=0.0,
            y=0.0,
            width=250.0,
            height=150.0,
        )
        document.add_group(group)
        document.remove_node(first.node_id)
        self.assertNotIn(group.group_id, document.groups)

    def test_group_color_requires_hex_rgb(self) -> None:
        with self.assertRaises(ValueError):
            PlanningGroup.create(
                title="Invalid color",
                node_ids=["N1", "N2"],
                x=0.0,
                y=0.0,
                width=250.0,
                height=150.0,
                color="blue",
            )

    def test_legacy_document_without_groups_still_loads(self) -> None:
        payload = {
            "version": 1,
            "title": "Legacy",
            "nodes": [],
            "edges": [],
        }
        document = PlanningDocument.from_dict(payload)
        self.assertEqual(document.groups, {})


if __name__ == "__main__":
    unittest.main()
