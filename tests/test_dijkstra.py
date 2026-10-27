"""Comprehensive test suite for DijkstraSolver in PATHFINDER.

Covers:
1. Open grid
2. Grid with walls
3. Weighted terrain where the cheapest route is not the shortest
4. Equal-cost paths
5. Start == destination
6. Unreachable destination (isolated by walls)
7. Cross-validation against an independent reference solver (Bellman-Ford)
   on 200+ randomized small grids.
"""

import math
import random
from typing import Dict, List, Optional, Tuple
import pytest

from src.dijkstra_solver import DijkstraSolver
from src.grid import CellType, Grid


def _reference_bellman_ford(
    grid: Grid,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> float:
    """Independent reference shortest-path solver using Bellman-Ford algorithm.

    Exhaustively relaxes all edges |V| - 1 times without using heapq or visited sets.
    """
    if not grid.is_valid(*start) or not grid.is_valid(*end):
        return math.inf

    start_cell = grid.get_cell(*start)
    end_cell = grid.get_cell(*end)
    if start_cell is None or start_cell.is_wall or end_cell is None or end_cell.is_wall:
        return math.inf

    if start == end:
        return 0.0

    open_nodes = grid.get_open_cells()
    if start not in open_nodes or end not in open_nodes:
        return math.inf

    # Collect all directed edges (u, v, weight_v)
    edges = []
    for u in open_nodes:
        for vx, vy, w in grid.get_neighbors(u[0], u[1]):
            edges.append((u, (vx, vy), w))

    dist: Dict[Tuple[int, int], float] = {node: math.inf for node in open_nodes}
    dist[start] = 0.0

    # Relax |V| - 1 times
    for _ in range(len(open_nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    return dist[end]


def test_open_grid() -> None:
    """Test Dijkstra on a completely open uniform grid."""
    grid = Grid(5, 5)
    for x in range(5):
        for y in range(5):
            grid.get_cell(x, y).set_type(CellType.OPEN, weight=1)

    start = (0, 0)
    end = (4, 4)
    path, cost, explored = DijkstraSolver.solve(grid, start, end)

    # Manhattan distance = 4 + 4 = 8 steps. Start free -> cost = 8.
    assert cost == 8
    assert len(path) == 9
    assert path[0] == start
    assert path[-1] == end
    assert len(explored) > 0


def test_grid_with_walls() -> None:
    """Test Dijkstra properly navigates around obstacles and walls."""
    # 5x3 grid with a vertical wall down x=2 except at y=2
    grid = Grid(5, 3)
    for x in range(5):
        for y in range(3):
            grid.get_cell(x, y).set_type(CellType.OPEN, weight=1)

    grid.get_cell(2, 0).set_type(CellType.WALL)
    grid.get_cell(2, 1).set_type(CellType.WALL)
    # Only (2, 2) is open as a bottleneck pass

    start = (0, 0)
    end = (4, 0)
    path, cost, _ = DijkstraSolver.solve(grid, start, end)

    assert (2, 0) not in path
    assert (2, 1) not in path
    assert (2, 2) in path  # Must route through the bottleneck
    assert path[0] == start
    assert path[-1] == end
    assert cost < math.inf


def test_cheapest_route_is_not_shortest() -> None:
    """Test terrain where fewest steps has high cost, and detour is cheaper."""
    grid = Grid(4, 2)
    grid.reset_all_to_walls()

    # Short top corridor with deep mire (swamp weight 5)
    # (0,0)[1] -> (1,0)[5] -> (2,0)[5] -> (3,0)[1] (3 steps entering, cost = 5 + 5 + 1 = 11)
    grid.get_cell(0, 0).set_type(CellType.OPEN, weight=1)
    grid.get_cell(1, 0).set_type(CellType.SWAMP, weight=5)
    grid.get_cell(2, 0).set_type(CellType.SWAMP, weight=5)
    grid.get_cell(3, 0).set_type(CellType.OPEN, weight=1)

    # Longer bottom corridor with open terrain (weight 1)
    # (0,1)[1] -> (1,1)[1] -> (2,1)[1] -> (3,1)[1]
    # Detour: (0,0) -> (0,1)[1] -> (1,1)[1] -> (2,1)[1] -> (3,1)[1] -> (3,0)[1] (5 steps entering, cost = 5)
    grid.get_cell(0, 1).set_type(CellType.OPEN, weight=1)
    grid.get_cell(1, 1).set_type(CellType.OPEN, weight=1)
    grid.get_cell(2, 1).set_type(CellType.OPEN, weight=1)
    grid.get_cell(3, 1).set_type(CellType.OPEN, weight=1)

    start = (0, 0)
    end = (3, 0)
    path, cost, _ = DijkstraSolver.solve(grid, start, end)

    assert cost == 5
    assert len(path) == 6
    assert (1, 0) not in path
    assert (2, 0) not in path
    assert path == [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 0)]


def test_equal_cost_paths() -> None:
    """Test deterministic resolution when multiple paths have identical costs."""
    grid = Grid(3, 3)
    grid.reset_all_to_walls()

    # Two parallel routes from (0,1) to (2,1)
    grid.get_cell(0, 1).set_type(CellType.OPEN, weight=1)
    grid.get_cell(2, 1).set_type(CellType.OPEN, weight=1)

    # Route A (top): via (0,0), (1,0), (2,0) with weight 2
    grid.get_cell(0, 0).set_type(CellType.SAND, weight=2)
    grid.get_cell(1, 0).set_type(CellType.SAND, weight=2)
    grid.get_cell(2, 0).set_type(CellType.SAND, weight=2)

    # Route B (bottom): via (0,2), (1,2), (2,2) with weight 2
    grid.get_cell(0, 2).set_type(CellType.SAND, weight=2)
    grid.get_cell(1, 2).set_type(CellType.SAND, weight=2)
    grid.get_cell(2, 2).set_type(CellType.SAND, weight=2)

    start = (0, 1)
    end = (2, 1)
    path, cost, explored = DijkstraSolver.solve(grid, start, end)

    # Both routes have cost: 2 + 2 + 2 + 1 = 7
    assert cost == 7
    assert len(path) == 5
    assert path[0] == start
    assert path[-1] == end


def test_start_equals_destination() -> None:
    """Test start == destination returns cost 0 and single-element path."""
    grid = Grid(3, 3)
    for x in range(3):
        for y in range(3):
            grid.get_cell(x, y).set_type(CellType.OPEN, weight=3)

    pos = (1, 1)
    path, cost, explored = DijkstraSolver.solve(grid, pos, pos)

    assert cost == 0
    assert path == [pos]
    assert explored == [pos]


def test_unreachable_destination() -> None:
    """Test clean handling when destination is walled off."""
    grid = Grid(4, 4)
    for x in range(4):
        for y in range(4):
            grid.get_cell(x, y).set_type(CellType.OPEN, weight=1)

    # Enclose destination (3, 3) completely with walls
    grid.get_cell(3, 3).set_type(CellType.OPEN, weight=1)
    grid.get_cell(2, 3).set_type(CellType.WALL)
    grid.get_cell(3, 2).set_type(CellType.WALL)

    start = (0, 0)
    end = (3, 3)
    path, cost, explored = DijkstraSolver.solve(grid, start, end)

    assert path == []
    assert math.isinf(cost)
    assert end not in explored


def test_cross_validation_against_reference_solver_200_grids() -> None:
    """Cross-validate Dijkstra against Bellman-Ford on 200+ random small grids."""
    rng = random.Random(42)
    terrain_types = [CellType.OPEN, CellType.SAND, CellType.FOREST, CellType.SWAMP]

    for seed_idx in range(210):
        w = rng.randint(3, 6)
        h = rng.randint(3, 6)
        grid = Grid(w, h)

        for x in range(w):
            for y in range(h):
                if rng.random() < 0.25:
                    grid.get_cell(x, y).set_type(CellType.WALL)
                else:
                    ttype = rng.choice(terrain_types)
                    grid.get_cell(x, y).set_type(ttype)

        open_cells = grid.get_open_cells()
        if len(open_cells) < 2:
            continue

        start = rng.choice(open_cells)
        end = rng.choice(open_cells)

        dijk_path, dijk_cost, _ = DijkstraSolver.solve(grid, start, end)
        ref_cost = _reference_bellman_ford(grid, start, end)

        if math.isinf(ref_cost):
            assert math.isinf(dijk_cost), f"Seed {seed_idx}: expected unreachable, got cost {dijk_cost}"
            assert dijk_path == []
        else:
            assert dijk_cost == ref_cost, (
                f"Seed {seed_idx}: Dijkstra cost {dijk_cost} != Reference cost {ref_cost} "
                f"for grid {w}x{h} from {start} to {end}"
            )
            # Verify path validity and cost summation
            assert dijk_path[0] == start
            assert dijk_path[-1] == end

            # Recalculate cost along path: start is free, subsequent paid
            calculated_cost = sum(grid.get_cell(px, py).weight for px, py in dijk_path[1:])
            assert calculated_cost == dijk_cost
