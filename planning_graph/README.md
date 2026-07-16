# LiSNN Planning Graph - U1

U1 is a PySide6 graph editor for planning, reasoning, research decomposition, and project mapping. It is deliberately separate from the LiSNN runtime visualizer planned for U2.

## Launch

From the LiSNN repository root:

```powershell
python3 -m pip install PySide6
python3 main.py
```

Or install through the feature requirements file:

```powershell
python3 -m pip install -r planning_graph\requirements.txt
python3 main.py
```

Use `python3` rather than the Windows `py` launcher in this repository. On the current development system, `py` selects the free-threaded Python 3.14 runtime by default.

## Interaction

- Drag semantic node types from the left palette onto the canvas.
- Hover or select a node to reveal its four connection ports.
- Drag from a port to another node to create a directed connection.
- Select the connection type in the top toolbar before drawing.
- Edit a selected node or connection in the right inspector.
- Double-click a node or connection to focus its inspector fields.
- Mouse wheel zooms.
- Middle-mouse drag pans.
- Left-drag on empty canvas creates a rubber-band selection.
- `Delete` removes selected nodes or connections.
- `Ctrl+D` duplicates one selected node.
- `F` fits the complete graph into the view.

## Planning node types

- Idea
- Goal
- Question
- Task
- Evidence
- Decision
- Constraint
- Result
- Note

## Directed connection types

- Related
- Supports
- Contradicts
- Depends On
- Leads To
- Part Of
- Refines
- Blocks

## Persistence

Graphs are saved as human-readable, versioned JSON. Exact canvas positions, node metadata, and typed directed connections are preserved.

Use **File -> Load LiSNN Roadmap Example** to open the included project map as an unsaved working copy.

## Module layout

```text
main.py                         GUI launcher
planning_graph/models.py        GUI-independent graph data model
planning_graph/serialization.py JSON persistence
planning_graph/items.py         visual nodes and typed arrows
planning_graph/scene.py         graph canvas, connections, zoom/pan
planning_graph/widgets.py       node palette and inspector
planning_graph/window.py        menus, toolbar, file workflow
planning_graph/app.py           QApplication bootstrap
planning_graph/examples/        example planning graphs
```

## Scope boundary

U1 does not yet:

- execute a plan;
- infer connections automatically;
- visualize live LiSNN neuron or hypothesis activity;
- perform automatic graph layout;
- synchronize with external task systems.

Those remain later features. U1 establishes the visual planning surface and reusable graph document format first.
