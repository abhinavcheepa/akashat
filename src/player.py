"""Player entity and movement mechanics for PATHFINDER.

Tracks player position, breadcrumb navigation trail, and running route cost.
Strictly adheres to the rule that the start tile is free (0 cost) and terrain
cost is paid strictly upon entering each subsequent cell.
"""

import math
from typing import List, Tuple, Union

from src.grid import Grid


class Player:
    """Represents the user's navigator on the grid."""

    def __init__(self, start_pos: Tuple[int, int]) -> None:
        """Initialize the player at the given starting coordinate."""
        self.x: int = start_pos[0]
        self.y: int = start_pos[1]
        self.path: List[Tuple[int, int]] = [start_pos]
        self.total_cost: int = 0

    @property
    def pos(self) -> Tuple[int, int]:
        """Return current coordinate as (x, y)."""
        return (self.x, self.y)

    def reset(self, start_pos: Tuple[int, int]) -> None:
        """Reset player state to a new or initial starting position."""
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.path = [start_pos]
        self.total_cost = 0

    def move(self, dx: int, dy: int, grid: Grid) -> bool:
        """Attempt to move the player by delta offset (dx, dy).

        Movement is validated against walls and grid boundaries. If valid,
        traversal cost of the entered cell is added to total_cost.

        Args:
            dx: Horizontal movement (-1, 0, or 1).
            dy: Vertical movement (-1, 0, or 1).
            grid: The active Grid.

        Returns:
            True if the move succeeded, False if blocked.
        """
        target_x = self.x + dx
        target_y = self.y + dy

        if not grid.is_valid(target_x, target_y):
            return False

        target_cell = grid.get_cell(target_x, target_y)
        if target_cell is None or target_cell.is_wall:
            return False

        # Valid move: enter target cell and pay weight cost
        self.x = target_x
        self.y = target_y
        self.path.append((target_x, target_y))
        self.total_cost += target_cell.weight
        return True

    def calculate_efficiency(self, optimal_cost: Union[int, float]) -> float:
        """Calculate navigation efficiency percentage: (optimal / player_cost) * 100.

        Capped at 100.0%. Safely handles 0 route cost at start, 0 optimal cost,
        and unreachable destinations (math.inf).

        Args:
            optimal_cost: The theoretical minimum cost found by Dijkstra.

        Returns:
            Efficiency float value between 0.0 and 100.0.
        """
        # Unreachable destination has 0.0% efficiency
        if math.isinf(optimal_cost) or optimal_cost is None:
            return 0.0

        # At start of navigation, before player has made moves
        if self.total_cost <= 0:
            return 100.0

        # If optimal cost is 0 (already at goal) but player took costly moves
        if optimal_cost <= 0:
            return 0.0

        ratio = (optimal_cost / self.total_cost) * 100.0
        return min(100.0, max(0.0, ratio))
