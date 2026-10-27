"""Test suite for gameplay mechanics, state transitions, and efficiency calculations.

Covers:
1. Movement blocked by walls and boundaries
2. Cost accumulation correctness per entry
3. Efficiency calculation for normal, zero-cost, and unreachable/infinite cases
4. Restart resets all state (player position, breadcrumb trail, cost, path visibility)
"""

import math
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pytest
import pygame

from src.game import Game, GameState
from src.grid import CellType, Grid
from src.player import Player


def test_movement_blocked_by_walls() -> None:
    """Test player cannot step into wall tiles or cross grid boundaries."""
    grid = Grid(3, 3)
    grid.reset_all_to_walls()

    # Center (1,1) is open, all surrounding cells are walls
    grid.get_cell(1, 1).set_type(CellType.OPEN, weight=1)

    player = Player(start_pos=(1, 1))

    # Try moving in all 4 cardinal directions into walls
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        moved = player.move(dx, dy, grid)
        assert moved is False
        assert player.pos == (1, 1)
        assert player.total_cost == 0
        assert player.path == [(1, 1)]

    # Try moving outside boundary directly
    corner_player = Player(start_pos=(0, 0))
    grid.get_cell(0, 0).set_type(CellType.OPEN, weight=1)
    moved_left = corner_player.move(-1, 0, grid)
    moved_up = corner_player.move(0, -1, grid)
    assert moved_left is False
    assert moved_up is False


def test_cost_accumulation_per_entry() -> None:
    """Test traversal cost accumulates strictly based on entered cell weights."""
    grid = Grid(4, 1)
    grid.reset_all_to_walls()

    # (0,0)[OPEN:1] -> (1,0)[SAND:2] -> (2,0)[FOREST:3] -> (3,0)[SWAMP:5]
    grid.get_cell(0, 0).set_type(CellType.OPEN, weight=1)
    grid.get_cell(1, 0).set_type(CellType.SAND, weight=2)
    grid.get_cell(2, 0).set_type(CellType.FOREST, weight=3)
    grid.get_cell(3, 0).set_type(CellType.SWAMP, weight=5)

    player = Player(start_pos=(0, 0))
    assert player.total_cost == 0

    # Step 1: Enter SAND (+2)
    assert player.move(1, 0, grid) is True
    assert player.total_cost == 2

    # Step 2: Enter FOREST (+3)
    assert player.move(1, 0, grid) is True
    assert player.total_cost == 2 + 3

    # Step 3: Enter SWAMP (+5)
    assert player.move(1, 0, grid) is True
    assert player.total_cost == 2 + 3 + 5

    # Step 4: Backtrack into FOREST (+3)
    assert player.move(-1, 0, grid) is True
    assert player.total_cost == 2 + 3 + 5 + 3
    assert len(player.path) == 5


def test_efficiency_zero_cost_and_unreachable_cases() -> None:
    """Verify efficiency behaves accurately across edge conditions."""
    player = Player(start_pos=(0, 0))

    # Case 1: At starting position, player has spent 0 cost
    assert player.total_cost == 0
    assert player.calculate_efficiency(optimal_cost=15) == 100.0

    # Case 2: Start == Destination, both player cost and optimal are 0
    assert player.calculate_efficiency(optimal_cost=0) == 100.0

    # Case 3: Player moves and spends cost > 0
    player.total_cost = 20

    # Optimal matched player: 20 / 20 = 100%
    assert player.calculate_efficiency(optimal_cost=20) == 100.0

    # Optimal is less: 10 / 20 = 50%
    assert player.calculate_efficiency(optimal_cost=10) == 50.0

    # Player wanders when optimal was 0: 0 / 20 = 0%
    assert player.calculate_efficiency(optimal_cost=0) == 0.0

    # Destination is unreachable: optimal cost is math.inf
    assert player.calculate_efficiency(optimal_cost=math.inf) == 0.0
    assert player.calculate_efficiency(optimal_cost=float("inf")) == 0.0


def test_restart_resets_all_state() -> None:
    """Verify restarting clears player, trail, costs, and path visualization."""
    game = Game(config_path="config/game_config.json")
    game.state = GameState.PLAYING

    # Make player take moves
    neighbors = game.grid.get_neighbors(*game.player.pos)
    if neighbors:
        nx, ny, _ = neighbors[0]
        game.player.move(nx - game.player.x, ny - game.player.y, game.grid)

    assert len(game.player.path) > 1
    assert game.player.total_cost > 0

    # Toggle path visibility to True
    game.visualizer.toggle_path()
    assert game.visualizer.show_path is True

    # Complete level state
    game.level_completed = True

    # Execute restart
    game.restart_level()

    # Confirm complete state reset
    assert game.player.pos == game.start_pos
    assert game.player.path == [game.start_pos]
    assert game.player.total_cost == 0
    assert game.visualizer.show_path is False
    assert game.visualizer.is_animating is False
    assert len(game.visualizer.revealed_explored) == 0
    assert game.level_completed is False

    pygame.quit()
