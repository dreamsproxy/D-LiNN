"""Shared grid geometry for the planning graph.

The canvas grid is part of the planning document's spatial language. Nodes use
fixed dimensions measured in whole grid cells, and node centers snap to grid
intersections so their four connection ports also remain grid-aligned.
"""

from __future__ import annotations

import math


GRID_SIZE = 25.0
NODE_WIDTH_CELLS = 10
NODE_HEIGHT_CELLS = 6
NODE_WIDTH = GRID_SIZE * NODE_WIDTH_CELLS
NODE_HEIGHT = GRID_SIZE * NODE_HEIGHT_CELLS
GROUP_MIN_WIDTH_CELLS = 4
GROUP_MIN_HEIGHT_CELLS = 3
GROUP_MIN_WIDTH = GRID_SIZE * GROUP_MIN_WIDTH_CELLS
GROUP_MIN_HEIGHT = GRID_SIZE * GROUP_MIN_HEIGHT_CELLS


def snap_value(value: float, grid_size: float = GRID_SIZE) -> float:
    """Snap one coordinate to the nearest grid intersection.

    Exact half-cell positions round away from zero, avoiding Python's
    banker-rounding behavior during symmetric dragging around the origin.
    """

    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")

    scaled = float(value) / float(grid_size)
    if scaled >= 0.0:
        snapped_units = math.floor(scaled + 0.5)
    else:
        snapped_units = math.ceil(scaled - 0.5)
    return snapped_units * float(grid_size)


def snap_floor(value: float, grid_size: float = GRID_SIZE) -> float:
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")
    return math.floor(float(value) / grid_size) * grid_size


def snap_ceil(value: float, grid_size: float = GRID_SIZE) -> float:
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")
    return math.ceil(float(value) / grid_size) * grid_size


def snap_xy(x: float, y: float) -> tuple[float, float]:
    """Snap a two-dimensional position to the shared planning grid."""

    return snap_value(x), snap_value(y)


def snap_rect_outward(
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[float, float, float, float]:
    """Expand a rectangle outward so every edge lies on the grid."""

    return (
        snap_floor(left),
        snap_floor(top),
        snap_ceil(right),
        snap_ceil(bottom),
    )


def is_grid_aligned(value: float, grid_size: float = GRID_SIZE) -> bool:
    """Return whether a scalar coordinate lies on a grid intersection."""

    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")
    return math.isclose(value / grid_size, round(value / grid_size), abs_tol=1e-9)
