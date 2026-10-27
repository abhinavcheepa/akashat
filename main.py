"""Entry point for PATHFINDER: Think. Navigate. Conquer.

An educational tactical maze game demonstrating Dijkstra's shortest path
algorithm through intuitive, hands-on gameplay.
"""

import sys
from src.game import Game


def main() -> None:
    """Launch PATHFINDER application."""
    game = Game(config_path="config/game_config.json")
    game.run()


if __name__ == "__main__":
    main()
