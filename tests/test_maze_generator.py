"""Comprehensive test suite for MazeGenerator in PATHFINDER.

Covers:
1. Solvability across 200+ random seeds
2. All four start corners work reliably
3. Loops and alternate routes exist (graph cyclomatic complexity > 0)
4. Destination is reachable and distinct from start
5. Room chamber sizes from 7 to 15 work without errors
"""

import math
import random
from typing import Set, Tuple
import pytest

from src.dijkstra_solver import DijkstraSolver
from src.grid import Grid
from src.maze_generator import MazeGenerator


def test_solvable_across_200_seeds() -> None:
    """Validate that mazes generated across 200+ diverse seeds are always solvable."""
    generator = MazeGenerator()

    for seed_val in range(1000, 1205):
        random.seed(seed_val)
        grid, start, end = generator.generate(rooms_x=7, rooms_y=7, loop_removals=8, max_attempts=50)

        assert grid is not None
        assert start != end

        path, cost, _ = DijkstraSolver.solve(grid, start, end)
        assert len(path) >= 2, f"Seed {seed_val} produced trivial/empty path: {path}"
        assert cost < math.inf, f"Seed {seed_val} destination {end} is unreachable from {start}"
        assert path[0] == start
        assert path[-1] == end


def test_all_four_start_corners_work() -> None:
    """Validate that starting in any of the four outer corners yields a solvable maze."""
    generator = MazeGenerator()
    rooms_x, rooms_y = 7, 7
    expected_corners = [
        (1, 1),
        (2 * rooms_x - 1, 1),
        (1, 2 * rooms_y - 1),
        (2 * rooms_x - 1, 2 * rooms_y - 1),
    ]

    for corner_idx, corner in enumerate(expected_corners):
        # Generate with specific start corner
        grid, start, end = generator.generate(
            rooms_x=rooms_x,
            rooms_y=rooms_y,
            start_corner=corner_idx,
        )

        assert start == corner, f"Expected start at corner {corner}, got {start}"
        assert end != start

        path, cost, _ = DijkstraSolver.solve(grid, start, end)
        assert len(path) >= 2
        assert cost < math.inf
        assert path[0] == corner
        assert path[-1] == end


def test_loops_and_alternate_routes_exist() -> None:
    """Verify that knocking down interior walls creates genuine loops (cycles)."""
    generator = MazeGenerator()
    grid, start, end = generator.generate(rooms_x=7, rooms_y=7, loop_removals=8)

    open_cells = grid.get_open_cells()
    V = len(open_cells)

    # Count unique undirected edges between open cells
    edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for x, y in open_cells:
        for nx, ny, _ in grid.get_neighbors(x, y):
            edge = tuple(sorted([(x, y), (nx, ny)]))
            edges.add(edge)

    E = len(edges)

    # In a simple spanning tree (no loops), E == V - 1.
    # Because loop_removals knocked down interior walls, E must be strictly greater than V - 1.
    cyclomatic_number = E - V + 1
    assert cyclomatic_number >= 1, (
        f"Expected cyclomatic complexity >= 1 (loops/cycles), but got E={E}, V={V}, E - V + 1 = {cyclomatic_number}"
    )

    # Confirm alternative route finder can identify a detour
    path, cost, _ = DijkstraSolver.solve(grid, start, end)
    alt = DijkstraSolver.find_alternative(grid, start, end, path)
    # With 8 loops on a 7x7 room maze, alternate routes frequently exist
    # (If middle edge was bridge, alt is None; if in cycle, alt exists)
    if alt is not None:
        alt_path, alt_cost = alt
        assert alt_path[0] == start
        assert alt_path[-1] == end
        assert alt_cost < math.inf


def test_destination_is_reachable_and_sufficiently_distant() -> None:
    """Verify destination selection guarantees reachability and distance."""
    generator = MazeGenerator()
    for _ in range(20):
        grid, start, end = generator.generate(rooms_x=7, rooms_y=7)
        manhattan_dist = abs(end[0] - start[0]) + abs(end[1] - start[1])

        # Destination must not be adjacent/trivial
        assert manhattan_dist >= 4, f"Destination {end} too close to start {start} (dist={manhattan_dist})"

        # Path must reach end
        path, cost, _ = DijkstraSolver.solve(grid, start, end)
        assert cost < math.inf
        assert path[-1] == end


def test_sizes_7_to_15_rooms_work() -> None:
    """Verify room chamber scaling from 7 up to 15 works smoothly."""
    generator = MazeGenerator()

    for rooms in range(7, 16):
        grid, start, end = generator.generate(rooms_x=rooms, rooms_y=rooms, loop_removals=8)

        expected_dim = 2 * rooms + 1
        assert grid.width == expected_dim, f"For {rooms} rooms, expected width {expected_dim}, got {grid.width}"
        assert grid.height == expected_dim, f"For {rooms} rooms, expected height {expected_dim}, got {grid.height}"

        path, cost, _ = DijkstraSolver.solve(grid, start, end)
        assert len(path) >= 2
        assert cost < math.inf
        assert path[0] == start
        assert path[-1] == end
