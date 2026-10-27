"""Automated test suite for PATHFINDER core requirements.

Covers:
1. Dijkstra optimal cost on a hand-built 3x3 grid
2. The cost rule (start free, entering paid) across Dijkstra and Player
3. Generator output is always solvable across diverse maze seeds
4. _validate rejects a grid where the weighted and BFS paths are identical
"""

import pytest
from typing import List, Tuple

from src.dijkstra_solver import DijkstraSolver
from src.grid import Cell, CellType, Grid
from src.maze_generator import MazeGenerator
from src.player import Player


def test_dijkstra_optimal_cost_hand_built_3x3() -> None:
    """Test Dijkstra optimal cost on a hand-built 3x3 grid where cheaper != shortest."""
    # Build 3x3 grid:
    # (0,0)[1] -- (1,0)[5:Swamp] -- (2,0)[1]
    #    |             |               |
    # (0,1)[1] -- (1,1)[1:Open]  -- (2,1)[1]
    #    |             |               |
    # (0,2)[WALL]-(1,2)[WALL]    -- (2,2)[WALL]
    grid = Grid(3, 3)
    grid.reset_all_to_walls()

    grid.get_cell(0, 0).set_type(CellType.OPEN, weight=1)
    grid.get_cell(1, 0).set_type(CellType.SWAMP, weight=5)
    grid.get_cell(2, 0).set_type(CellType.OPEN, weight=1)

    grid.get_cell(0, 1).set_type(CellType.OPEN, weight=1)
    grid.get_cell(1, 1).set_type(CellType.OPEN, weight=1)
    grid.get_cell(2, 1).set_type(CellType.OPEN, weight=1)

    start = (0, 0)
    end = (2, 0)

    # Top route: (0,0) -> (1,0)[+5] -> (2,0)[+1] = cost 6 (2 steps)
    # Bottom route: (0,0) -> (0,1)[+1] -> (1,1)[+1] -> (2,1)[+1] -> (2,0)[+1] = cost 4 (4 steps)
    path, cost, explored = DijkstraSolver.solve(grid, start, end)

    assert cost == 4, f"Expected optimal cost 4, got {cost}"
    assert path == [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)], f"Unexpected path: {path}"
    assert len(explored) > 0


def test_cost_rule_start_free_entering_paid() -> None:
    """Validate that the starting cell is free and cost is paid only upon entering."""
    grid = Grid(3, 1)
    grid.reset_all_to_walls()

    # Cell 0 has heavy swamp weight 5
    grid.get_cell(0, 0).set_type(CellType.SWAMP, weight=5)
    # Cell 1 has sand weight 2
    grid.get_cell(1, 0).set_type(CellType.SAND, weight=2)
    # Cell 2 has forest weight 3
    grid.get_cell(2, 0).set_type(CellType.FOREST, weight=3)

    start = (0, 0)
    end = (2, 0)

    # 1. Dijkstra validation:
    # Leaving start (0,0) -> cost 0
    # Entering (1,0) -> +2
    # Entering (2,0) -> +3
    # Total cost should be 5 (NOT 5+2+3 = 10)
    path, cost, _ = DijkstraSolver.solve(grid, start, end)
    assert cost == 5, f"Expected Dijkstra cost 5, got {cost}"
    assert path == [(0, 0), (1, 0), (2, 0)]

    # 2. Player validation:
    player = Player(start_pos=start)
    assert player.total_cost == 0, "Player starting cost must be 0"

    moved1 = player.move(1, 0, grid)
    assert moved1 is True
    assert player.pos == (1, 0)
    assert player.total_cost == 2, f"Player entered SAND (weight 2), expected cost 2, got {player.total_cost}"

    moved2 = player.move(1, 0, grid)
    assert moved2 is True
    assert player.pos == (2, 0)
    assert player.total_cost == 5, f"Player entered FOREST (weight 3), expected cost 5, got {player.total_cost}"


def test_generator_output_is_always_solvable() -> None:
    """Validate that generated mazes are always solvable across multiple configurations."""
    generator = MazeGenerator()

    # Test multiple rooms and seeds
    test_cases = [
        (5, 5),
        (7, 7),
        (9, 9),
    ]

    for rooms_x, rooms_y in test_cases:
        grid, start, end = generator.generate(rooms_x=rooms_x, rooms_y=rooms_y, loop_removals=6)

        assert grid is not None
        assert start != end

        # Run Dijkstra solver to confirm path existence and validity
        path, cost, _ = DijkstraSolver.solve(grid, start, end)
        assert len(path) >= 2, f"Expected non-trivial path for {rooms_x}x{rooms_y}, got {path}"
        assert cost < float("inf"), f"Destination was unreachable in generated maze ({rooms_x}x{rooms_y})"
        assert path[0] == start
        assert path[-1] == end


def test_validate_rejects_grid_where_weighted_and_bfs_are_identical() -> None:
    """Validate that _validate rejects grids where Dijkstra path and BFS path coincide."""
    generator = MazeGenerator()

    # Build a single straight corridor where all tiles have weight 1 (OPEN)
    # BFS and Dijkstra MUST pick the exact same single straight path
    grid = Grid(5, 3)
    grid.reset_all_to_walls()

    for x in range(5):
        grid.get_cell(x, 1).set_type(CellType.OPEN, weight=1)

    start = (0, 1)
    end = (4, 1)

    # In this uniform corridor, BFS path and Dijkstra path are identical:
    # [(0,1), (1,1), (2,1), (3,1), (4,1)]
    is_valid = generator._validate(grid, start, end)
    assert is_valid is False, "Validation must reject a grid where BFS and Dijkstra paths are identical"

    # Now add an alternative route with different costs so BFS and Dijkstra diverge:
    # Add parallel corridor along y=0 with high cost, but fewer or equal steps,
    # or create a detour scenario:
    # (0,1) -> (1,1)[w=5] -> (2,1)[w=5] -> (3,1)[w=5] -> (4,1)  (BFS will take this: 4 steps)
    # y=2: (0,2)[w=1] -> (1,2)[w=1] -> (2,2)[w=1] -> (3,2)[w=1] -> (4,2)[w=1] -> (4,1)
    # (Dijkstra will take the detour along y=2: cheaper cost!)
    grid.get_cell(1, 1).set_type(CellType.SWAMP, weight=5)
    grid.get_cell(2, 1).set_type(CellType.SWAMP, weight=5)
    grid.get_cell(3, 1).set_type(CellType.SWAMP, weight=5)

    for x in range(5):
        grid.get_cell(x, 2).set_type(CellType.OPEN, weight=1)

    # Connect corners
    grid.get_cell(0, 2).set_type(CellType.OPEN, weight=1)
    grid.get_cell(4, 2).set_type(CellType.OPEN, weight=1)

    is_valid_diverged = generator._validate(grid, start, end)
    assert is_valid_diverged is True, "Validation must accept a grid where Dijkstra optimal diverges from BFS path"
