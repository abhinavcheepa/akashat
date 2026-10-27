"""Integration test exercising Game initialization, state transitions, and gameplay actions."""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # Headless mode for headless environments

import pygame
from src.game import Game, GameState


def test_game_initialization_and_states() -> None:
    """Test game boots, initializes components, and transitions states properly."""
    game = Game(config_path="config/game_config.json")
    assert game.grid is not None
    assert game.grid.width == 15
    assert game.grid.height == 15
    assert game.player is not None
    assert game.optimal_cost > 0
    assert len(game.optimal_path) > 0
    assert game.state == GameState.MENU

    # Switch to PLAYING
    game.state = GameState.PLAYING
    assert game.state == GameState.PLAYING

    # Test toggling path visualizer
    game.visualizer.toggle_path()
    assert game.visualizer.show_path is True
    game.visualizer.update(0.1)

    # Test re-generation
    game._generate_level()
    assert game.grid is not None
    assert game.player.pos == game.start_pos

    # Test player movement
    # Move towards one valid open neighbor
    neighbors = game.grid.get_neighbors(*game.player.pos)
    if neighbors:
        nx, ny, w = neighbors[0]
        dx = nx - game.player.x
        dy = ny - game.player.y
        moved = game.player.move(dx, dy, game.grid)
        assert moved is True
        assert game.player.total_cost == w

    # Update and render one frame to confirm drawing logic works without errors
    game.update(0.016)
    game.render()

    # Switch to HOW_TO_PLAY and render
    game.state = GameState.HOW_TO_PLAY
    game.render()

    pygame.quit()
