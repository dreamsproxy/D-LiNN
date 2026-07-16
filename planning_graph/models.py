"""Serializable data model for the LiSNN planning graph.

The model is intentionally independent from PyQt6 so it can be unit tested,
versioned, and reused by future non-GUI planning and LiSNN visualization tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from uuid import uuid4


DOCUMENT_VERSION = 1

NODE_KINDS = (
    "Idea",
    "Goal",
    "Question",
    "Task",
    "Evidence",
    "Decision",
    "Constraint",
    "Result",
    "Note",
)

NODE_STATUSES = (
    "Backlog",
    "Active",
    "Blocked",
    "Complete",
    "Rejected",
)

EDGE_TYPES = (
    "Related",
    "Supports",
    "Contradicts",
    "Depends On",
    "Leads To",
    "Part Of",
    "Refines",
    "Blocks",
)


def _nonempty(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PlanningNode:
    """One semantic object in a planning graph."""

    node_id: str
    kind: str
    title: str
    body: str = ""
    status: str = "Backlog"
    priority: int = 0
    tags: list[str] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0

    def __post_init__(self) -> None:
        self.node_id = _nonempty(self.node_id, "node_id")
        self.title = _nonempty(self.title, "title")
        if self.kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind: {self.kind!r}")
        if self.status not in NODE_STATUSES:
            raise ValueError(f"unknown node status: {self.status!r}")
        self.priority = int(self.priority)
        if not 0 <= self.priority <= 5:
            raise ValueError("priority must be within [0, 5]")
        self.tags = [tag.strip() for tag in self.tags if tag.strip()]
        self.x = float(self.x)
        self.y = float(self.y)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        title: str,
        x: float = 0.0,
        y: float = 0.0,
        **kwargs: Any,
    ) -> "PlanningNode":
        return cls(
            node_id=str(uuid4()),
            kind=kind,
            title=title,
            x=x,
            y=y,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningNode":
        return cls(
            node_id=str(payload["node_id"]),
            kind=str(payload["kind"]),
            title=str(payload["title"]),
            body=str(payload.get("body", "")),
            status=str(payload.get("status", "Backlog")),
            priority=int(payload.get("priority", 0)),
            tags=[str(tag) for tag in payload.get("tags", [])],
            x=float(payload.get("x", 0.0)),
            y=float(payload.get("y", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanningEdge:
    """One directed, typed relationship between planning nodes."""

    edge_id: str
    source_id: str
    target_id: str
    relation: str = "Related"
    label: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.edge_id = _nonempty(self.edge_id, "edge_id")
        self.source_id = _nonempty(self.source_id, "source_id")
        self.target_id = _nonempty(self.target_id, "target_id")
        if self.source_id == self.target_id:
            raise ValueError("self-edges are not supported in U1 v0.1")
        if self.relation not in EDGE_TYPES:
            raise ValueError(f"unknown edge relation: {self.relation!r}")
        self.weight = float(self.weight)

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        target_id: str,
        relation: str = "Related",
        label: str = "",
        weight: float = 1.0,
    ) -> "PlanningEdge":
        return cls(
            edge_id=str(uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            label=label,
            weight=weight,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningEdge":
        return cls(
            edge_id=str(payload["edge_id"]),
            source_id=str(payload["source_id"]),
            target_id=str(payload["target_id"]),
            relation=str(payload.get("relation", "Related")),
            label=str(payload.get("label", "")),
            weight=float(payload.get("weight", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanningDocument:
    """A complete planning graph document."""

    title: str = "Untitled Planning Graph"
    nodes: dict[str, PlanningNode] = field(default_factory=dict)
    edges: dict[str, PlanningEdge] = field(default_factory=dict)
    version: int = DOCUMENT_VERSION
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.title = _nonempty(self.title, "document title")
        self.version = int(self.version)
        if self.version != DOCUMENT_VERSION:
            raise ValueError(
                f"unsupported planning document version {self.version}; "
                f"expected {DOCUMENT_VERSION}"
            )
        self.validate()

    def validate(self) -> None:
        if any(node_id != node.node_id for node_id, node in self.nodes.items()):
            raise ValueError("node dictionary keys must match node_id values")
        if any(edge_id != edge.edge_id for edge_id, edge in self.edges.items()):
            raise ValueError("edge dictionary keys must match edge_id values")

        known = set(self.nodes)
        dangling = [
            edge.edge_id
            for edge in self.edges.values()
            if edge.source_id not in known or edge.target_id not in known
        ]
        if dangling:
            raise ValueError(f"edges reference missing nodes: {dangling}")

    def add_node(self, node: PlanningNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node.node_id}")
        self.nodes[node.node_id] = node
        self.touch()

    def add_edge(self, edge: PlanningEdge) -> None:
        if edge.edge_id in self.edges:
            raise ValueError(f"duplicate edge id: {edge.edge_id}")
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("edge endpoints must exist before the edge is added")
        self.edges[edge.edge_id] = edge
        self.touch()

    def remove_edge(self, edge_id: str) -> None:
        self.edges.pop(edge_id, None)
        self.touch()

    def remove_node(self, node_id: str) -> list[str]:
        if node_id not in self.nodes:
            return []
        del self.nodes[node_id]
        removed_edges = [
            edge_id
            for edge_id, edge in self.edges.items()
            if node_id in (edge.source_id, edge.target_id)
        ]
        for edge_id in removed_edges:
            del self.edges[edge_id]
        self.touch()
        return removed_edges

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningDocument":
        node_list = [PlanningNode.from_dict(item) for item in payload.get("nodes", [])]
        edge_list = [PlanningEdge.from_dict(item) for item in payload.get("edges", [])]

        nodes = _unique_map(node_list, key=lambda node: node.node_id, label="node")
        edges = _unique_map(edge_list, key=lambda edge: edge.edge_id, label="edge")

        return cls(
            title=str(payload.get("title", "Untitled Planning Graph")),
            nodes=nodes,
            edges=edges,
            version=int(payload.get("version", DOCUMENT_VERSION)),
            created_at=str(payload.get("created_at", _utc_now())),
            updated_at=str(payload.get("updated_at", _utc_now())),
        )


def _unique_map(
    values: Iterable[Any],
    *,
    key,
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        value_id = key(value)
        if value_id in result:
            raise ValueError(f"duplicate {label} id: {value_id}")
        result[value_id] = value
    return result
