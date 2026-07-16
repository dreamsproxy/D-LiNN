"""Shared grid geometry for the planning graph."""

from __future__ import annotations

import math


GRID_SIZE = 25.0

DEFAULT_NODE_WIDTH_CELLS = 10
DEFAULT_NODE_HEIGHT_CELLS = 6
NODE_MIN_WIDTH_CELLS = 6
NODE_MIN_HEIGHT_CELLS = 4
NODE_WIDTH = GRID_SIZE * DEFAULT_NODE_WIDTH_CELLS
NODE_HEIGHT = GRID_SIZE * DEFAULT_NODE_HEIGHT_CELLS
NODE_MIN_WIDTH = GRID_SIZE * NODE_MIN_WIDTH_CELLS
NODE_MIN_HEIGHT = GRID_SIZE * NODE_MIN_HEIGHT_CELLS

GROUP_MIN_WIDTH_CELLS = 4
GROUP_MIN_HEIGHT_CELLS = 3
GROUP_MIN_WIDTH = GRID_SIZE * GROUP_MIN_WIDTH_CELLS
GROUP_MIN_HEIGHT = GRID_SIZE * GROUP_MIN_HEIGHT_CELLS


def snap_value(value: float, grid_size: float = GRID_SIZE) -> float:
    """Snap one coordinate to the nearest grid intersection."""

    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")
    scaled = float(value) / float(grid_size)
    if scaled >= 0.0:
        units = math.floor(scaled + 0.5)
    else:
        units = math.ceil(scaled - 0.5)
    return units * float(grid_size)


def snap_dimension(
    value: float,
    minimum: float,
    grid_size: float = GRID_SIZE,
) -> float:
    """Snap a positive dimension while enforcing a minimum."""

    return max(float(minimum), abs(snap_value(value, grid_size)))


def snap_even_dimension(
    value: float,
    minimum: float,
    grid_size: float = GRID_SIZE,
) -> float:
    """Snap a dimension to an even number of grid cells.

    Blocks are center-positioned. Even cell counts keep the center, all four
    sides, and all four connector ports on grid intersections simultaneously.
    """

    snapped = snap_dimension(value, minimum, grid_size)
    cells = max(2, int(round(snapped / grid_size)))
    if cells % 2:
        cells += 1
    minimum_cells = int(math.ceil(minimum / grid_size))
    if minimum_cells % 2:
        minimum_cells += 1
    return max(cells, minimum_cells) * grid_size


def snap_floor(value: float, grid_size: float = GRID_SIZE) -> float:
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")
    return math.floor(float(value) / grid_size) * grid_size


def snap_ceil(value: float, grid_size: float = GRID_SIZE) -> float:
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")
    return math.ceil(float(value) / grid_size) * grid_size


def snap_xy(x: float, y: float) -> tuple[float, float]:
    return snap_value(x), snap_value(y)


def snap_rect_outward(
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[float, float, float, float]:
    return (
        snap_floor(left),
        snap_floor(top),
        snap_ceil(right),
        snap_ceil(bottom),
    )


def is_grid_aligned(value: float, grid_size: float = GRID_SIZE) -> bool:
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")
    return math.isclose(value / grid_size, round(value / grid_size), abs_tol=1e-9)
