"""Grid and Cell data structures for PATHFINDER navigation system.

Manages the discrete 2D tile grid, cell terrain properties, wall states,
and neighbor adjacency calculations.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple


class CellType(Enum):
    """Enumeration of cell types and terrain classifications."""
    WALL = "WALL"
    OPEN = "OPEN"
    SAND = "SAND"
    FOREST = "FOREST"
    SWAMP = "SWAMP"


TERRAIN_WEIGHTS: Dict[CellType, int] = {
    CellType.WALL: 0,
    CellType.OPEN: 1,
    CellType.SAND: 2,
    CellType.FOREST: 3,
    CellType.SWAMP: 5,
}

TERRAIN_NAMES: Dict[CellType, str] = {
    CellType.WALL: "Dense Barrier",
    CellType.OPEN: "Clear Path (1)",
    CellType.SAND: "Sand Dune (2)",
    CellType.FOREST: "Dense Woods (3)",
    CellType.SWAMP: "Deep Mire (5)",
}


class Cell:
    """Represents an individual node/tile within the navigation grid.

    Attributes:
        x: Column index (0 to width - 1).
        y: Row index (0 to height - 1).
        cell_type: Classification (WALL, OPEN, SAND, FOREST, SWAMP).
        weight: Movement traversal cost incurred upon entering this cell.
        is_wall: Whether this cell blocks movement entirely.
    """

    def __init__(
        self,
        x: int,
        y: int,
        cell_type: CellType = CellType.WALL,
        weight: Optional[int] = None,
        is_wall: Optional[bool] = None,
    ) -> None:
        self.x: int = x
        self.y: int = y
        self.cell_type: CellType = cell_type
        self.is_wall: bool = is_wall if is_wall is not None else (cell_type == CellType.WALL)
        self.weight: int = weight if weight is not None else TERRAIN_WEIGHTS.get(cell_type, 1)

    def set_type(self, cell_type: CellType, weight: Optional[int] = None) -> None:
        """Update the cell's terrain type and associated traversal cost."""
        self.cell_type = cell_type
        self.is_wall = (cell_type == CellType.WALL)
        self.weight = weight if weight is not None else TERRAIN_WEIGHTS.get(cell_type, 1)

    @property
    def pos(self) -> Tuple[int, int]:
        """Return the coordinate tuple (x, y)."""
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Cell({self.x}, {self.y}, type={self.cell_type.name}, weight={self.weight})"


class Grid:
    """Represents the complete 2D maze tile grid.

    Folds graph adjacency generation directly into get_neighbors,
    producing weighted edges based on the entering cell's traversal cost.
    """

    def __init__(self, width: int, height: int) -> None:
        """Initialize a grid of given dimensions, default filled with walls."""
        self.width: int = width
        self.height: int = height
        self.cells: List[List[Cell]] = [
            [Cell(x=x, y=y, cell_type=CellType.WALL) for y in range(height)]
            for x in range(width)
        ]

    def is_valid(self, x: int, y: int) -> bool:
        """Check if coordinates fall within grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> Optional[Cell]:
        """Retrieve cell at given coordinates, or None if out of bounds."""
        if not self.is_valid(x, y):
            return None
        return self.cells[x][y]

    def get_neighbors(
        self,
        x: int,
        y: int,
        blocked_edge: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int, int]]:
        """Return accessible 4-directional neighbors for navigation.

        Each neighbor is returned as a tuple of (neighbor_x, neighbor_y, step_cost).
        Per the Pathfinder rule: cost is paid upon ENTERING the neighbor cell.

        Args:
            x: Source cell x-coordinate.
            y: Source cell y-coordinate.
            blocked_edge: Optional pair of coordinates representing an edge to prohibit.

        Returns:
            List of (nx, ny, weight) for all accessible, non-wall neighbors.
        """
        neighbors: List[Tuple[int, int, int]] = []
        curr = (x, y)

        # 4-connected directions: Right, Left, Down, Up
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not self.is_valid(nx, ny):
                continue

            neighbor_cell = self.cells[nx][ny]
            if neighbor_cell.is_wall:
                continue

            # Check if this edge is blocked
            if blocked_edge is not None:
                u, v = blocked_edge
                if (curr == u and (nx, ny) == v) or (curr == v and (nx, ny) == u):
                    continue

            # Entering cost is the neighbor cell's weight
            neighbors.append((nx, ny, neighbor_cell.weight))

        return neighbors

    def get_open_cells(self) -> List[Tuple[int, int]]:
        """Return list of (x, y) coordinates for all non-wall cells."""
        open_cells: List[Tuple[int, int]] = []
        for x in range(self.width):
            for y in range(self.height):
                if not self.cells[x][y].is_wall:
                    open_cells.append((x, y))
        return open_cells

    def reset_all_to_walls(self) -> None:
        """Reset every cell in the grid to a solid wall."""
        for x in range(self.width):
            for y in range(self.height):
                self.cells[x][y].set_type(CellType.WALL)
