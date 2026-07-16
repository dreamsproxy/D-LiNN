# LiSNN Planning Graph - U1

U1 is a PySide6 graph editor for planning, reasoning, research decomposition, and project mapping. It is deliberately separate from the LiSNN runtime visualizer planned for U2.

## Launch

From the LiSNN repository root:

```powershell
python3 -m pip install -r planning_graph\requirements.txt
python3 main.py
```

Use `python3` rather than the Windows `py` launcher in this repository. On the current development system, `py` selects the free-threaded Python 3.14 runtime by default.

## Visual language

- The entire application uses a dark theme.
- The canvas is near-black with half-alpha white dots at 25-unit intersections.
- Blocks snap to the grid and use fixed 250 x 150 dimensions.
- Block bodies are dark gray, headers are blue-gray, and text is white.
- A narrow accent strip color-codes each semantic block type.
- Connections are color-coded by relationship type.
- Groups can use low-alpha colored backdrops for rapid visual segmentation.

## Interaction

- Drag semantic block types from the left palette onto the canvas.
- Hover or select a block to reveal its four connection ports.
- Drag from a port to another block to create a directed connection.
- Select the connection type in the top toolbar before drawing.
- Edit one selected block, connection, or group in the right inspector.
- Double-click an item to focus its inspector fields.
- Mouse wheel zooms.
- Middle-mouse drag pans.
- Left-drag on empty canvas creates a rubber-band selection.
- `Delete` removes selected blocks, connections, or groups.
- `Ctrl+D` duplicates one selected block.
- `Ctrl+G` groups selected blocks.
- `Ctrl+Shift+G` ungroups selected groups.
- `Ctrl+Shift+I` inverts selection.
- `F` fits the complete graph into the view.

## Right-click menu

The base context menu provides:

- Delete
- Group
- Invert Selection

Right-clicking a group additionally provides:

- Ungroup
- Add Backdrop / Remove Backdrop
- Backdrop Color presets and a custom color picker
- Bring to Front
- Bring Forward
- Send Backward
- Send to Back

## Groups and backdrops

A group is persistent membership of two or more blocks. Ungrouping removes only the group metadata and backdrop; all member blocks and connections remain.

A backdrop is optional. Even with the fill removed, a faint dashed group frame and title remain so the group can still be selected and edited.

Selected group frames expose four corner handles. Drag a handle to resize the frame in whole grid cells. Drag the frame itself to move the group and all member blocks together.

The right inspector allows editing:

- group title;
- backdrop visibility;
- backdrop color;
- backdrop layer;
- width and height in grid cells.

## Planning block types

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

Graphs are saved as human-readable, versioned JSON. Exact canvas positions, node metadata, typed directed connections, group membership, backdrop geometry, colors, and layers are preserved.

Existing v1 JSON documents without a `groups` field remain valid.

Use **File -> Load LiSNN Roadmap Example** to open the included project map as an unsaved working copy.

## Module layout

```text
main.py                         GUI launcher
planning_graph/models.py        GUI-independent graph and group data model
planning_graph/grid.py          shared grid geometry and snapping
planning_graph/theme.py         dark theme and semantic colors
planning_graph/serialization.py JSON persistence
planning_graph/items.py         blocks, groups, backdrops, and typed arrows
planning_graph/scene.py         canvas, context menu, selection, grouping
planning_graph/widgets.py       block palette and inspector
planning_graph/window.py        menus, toolbar, file workflow
planning_graph/app.py           QApplication bootstrap
planning_graph/examples/        example planning graphs
```

## Scope boundary

U1 does not yet:

- execute a plan;
- infer blocks or relationships automatically;
- visualize live LiSNN neuron or hypothesis activity;
- perform automatic graph layout;
- synchronize with external task systems;
- support nested or overlapping membership groups.

Those remain later features. U1 establishes the visual planning surface and reusable graph document format first.
