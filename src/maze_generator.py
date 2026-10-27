"""Maze generator for PATHFINDER.

Generates guaranteed solvable mazes with loops and multi-weight terrain.
Uses iterative DFS to carve a spanning tree, removes interior walls to create
alternate paths, distributes weighted terrain, and validates that the weighted
optimal path differs from the unweighted (BFS) fewest-steps route.
"""

from collections import deque
import math
import random
from typing import List, Optional, Set, Tuple

from src.dijkstra_solver import DijkstraSolver
from src.grid import CellType, Grid


class MazeGenerator:
    """Generates procedural weighted mazes with validated path divergence."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the generator with an optional reproducible random seed."""
        if seed is not None:
            random.seed(seed)

    def generate(
        self,
        rooms_x: int = 7,
        rooms_y: int = 7,
        loop_removals: int = 8,
        max_attempts: int = 50,
        start_corner: Optional[int] = None,
    ) -> Tuple[Grid, Tuple[int, int], Tuple[int, int]]:
        """Generate a complete, validated maze grid with start and destination.

        Dimensions of the tile grid are (2*rooms_x + 1) by (2*rooms_y + 1).

        Args:
            rooms_x: Number of horizontal room chambers (default 7).
            rooms_y: Number of vertical room chambers (default 7).
            loop_removals: Number of interior wall barriers to knock down for loops.
            max_attempts: Maximum regeneration attempts to achieve BFS != Dijkstra.
            start_corner: Optional corner index (0: Top-Left, 1: Top-Right,
                          2: Bottom-Left, 3: Bottom-Right). If None, chosen randomly.

        Returns:
            Tuple of (grid, start_pos, end_pos).
        """
        width = 2 * rooms_x + 1
        height = 2 * rooms_y + 1

        corners = [
            (1, 1),
            (2 * rooms_x - 1, 1),
            (1, 2 * rooms_y - 1),
            (2 * rooms_x - 1, 2 * rooms_y - 1),
        ]

        last_valid_result: Optional[Tuple[Grid, Tuple[int, int], Tuple[int, int]]] = None

        for _ in range(max_attempts):
            grid = Grid(width, height)

            # 1. Carve spanning tree using iterative DFS
            self._dfs_carve(grid, rooms_x, rooms_y)

            # 2. Knock out interior walls to generate multiple loops & alternate routes
            self._add_loops(grid, count=loop_removals)

            # 3. Select start corner
            if start_corner is not None and 0 <= start_corner < len(corners):
                start_pos = corners[start_corner]
            else:
                start_pos = random.choice(corners)

            # 4. Select destination in an open cell sufficiently distant
            end_pos = self._select_destination(grid, start_pos, min_manhattan=max(rooms_x, rooms_y))

            # 5. Distribute weighted terrain (1, 2, 3, 5)
            self._assign_terrain(grid, start_pos, end_pos)

            last_valid_result = (grid, start_pos, end_pos)

            # 6. Validate: connected AND weighted Dijkstra path differs from BFS fewest-steps path
            if self._validate(grid, start_pos, end_pos):
                return grid, start_pos, end_pos

        # Fallback to the last generated grid if validation criteria didn't diverge in 50 tries
        if last_valid_result is not None:
            return last_valid_result

        # Extreme fallback: create a basic open path grid
        fallback_grid = Grid(width, height)
        self._dfs_carve(fallback_grid, rooms_x, rooms_y)
        return fallback_grid, (1, 1), (width - 2, height - 2)

    def _dfs_carve(self, grid: Grid, rooms_x: int, rooms_y: int) -> None:
        """Carve maze passages iteratively using an explicit stack."""
        grid.reset_all_to_walls()

        start_rx, start_ry = random.randint(0, rooms_x - 1), random.randint(0, rooms_y - 1)
        stack: List[Tuple[int, int]] = [(start_rx, start_ry)]
        visited: Set[Tuple[int, int]] = {(start_rx, start_ry)}

        # Open the initial room tile
        init_cell = grid.get_cell(2 * start_rx + 1, 2 * start_ry + 1)
        if init_cell:
            init_cell.set_type(CellType.OPEN)

        while stack:
            rx, ry = stack[-1]

            # Discover unvisited room neighbors
            unvisited_neighbors: List[Tuple[int, int, int, int]] = []
            for drx, dry in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nrx, nry = rx + drx, ry + dry
                if 0 <= nrx < rooms_x and 0 <= nry < rooms_y and (nrx, nry) not in visited:
                    unvisited_neighbors.append((nrx, nry, drx, dry))

            if unvisited_neighbors:
                nrx, nry, drx, dry = random.choice(unvisited_neighbors)
                # Carve the wall between current room and neighbor
                wall_x = 2 * rx + 1 + drx
                wall_y = 2 * ry + 1 + dry
                # Carve the destination room
                room_x = 2 * nrx + 1
                room_y = 2 * nry + 1

                wall_cell = grid.get_cell(wall_x, wall_y)
                if wall_cell:
                    wall_cell.set_type(CellType.OPEN)

                room_cell = grid.get_cell(room_x, room_y)
                if room_cell:
                    room_cell.set_type(CellType.OPEN)

                visited.add((nrx, nry))
                stack.append((nrx, nry))
            else:
                stack.pop()

    def _add_loops(self, grid: Grid, count: int) -> None:
        """Knock down interior wall barriers to create genuine loops."""
        candidate_walls: List[Tuple[int, int]] = []

        for x in range(1, grid.width - 1):
            for y in range(1, grid.height - 1):
                cell = grid.get_cell(x, y)
                if cell is None or not cell.is_wall:
                    continue

                left = grid.get_cell(x - 1, y)
                right = grid.get_cell(x + 1, y)
                top = grid.get_cell(x, y - 1)
                bottom = grid.get_cell(x, y + 1)

                if left and right and top and bottom:
                    # Horizontal bridge between two corridors
                    h_bridge = (not left.is_wall) and (not right.is_wall) and top.is_wall and bottom.is_wall
                    # Vertical bridge between two corridors
                    v_bridge = (not top.is_wall) and (not bottom.is_wall) and left.is_wall and right.is_wall

                    if h_bridge or v_bridge:
                        candidate_walls.append((x, y))

        random.shuffle(candidate_walls)
        for wx, wy in candidate_walls[:count]:
            target_wall = grid.get_cell(wx, wy)
            if target_wall:
                target_wall.set_type(CellType.OPEN)

    def _select_destination(
        self,
        grid: Grid,
        start_pos: Tuple[int, int],
        min_manhattan: int,
    ) -> Tuple[int, int]:
        """Select a destination tile reachable and sufficiently distant from start."""
        open_cells = [p for p in grid.get_open_cells() if p != start_pos]
        distant_cells = [
            pos for pos in open_cells
            if (abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])) >= min_manhattan
        ]

        if distant_cells:
            return random.choice(distant_cells)

        if open_cells:
            return max(
                open_cells,
                key=lambda p: abs(p[0] - start_pos[0]) + abs(p[1] - start_pos[1]),
            )

        return (grid.width - 2, grid.height - 2)

    def _assign_terrain(
        self,
        grid: Grid,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
    ) -> None:
        """Assign terrain weights (1, 2, 3, 5) across open cells.

        Open terrain is the most prevalent, with pockets of sand, forest, and swamp.
        Start and end positions are always kept as clear Open Path (weight 1).
        """
        terrain_pool = [
            (CellType.OPEN, 0.58),
            (CellType.SAND, 0.22),
            (CellType.FOREST, 0.12),
            (CellType.SWAMP, 0.08),
        ]
        types, weights = zip(*terrain_pool)

        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get_cell(x, y)
                if cell is None or cell.is_wall:
                    continue

                if (x, y) == start_pos or (x, y) == end_pos:
                    cell.set_type(CellType.OPEN)
                else:
                    chosen_type = random.choices(types, weights=weights, k=1)[0]
                    cell.set_type(chosen_type)

    def _bfs_fewest_steps(
        self,
        grid: Grid,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Find path with the minimal number of steps regardless of terrain weight."""
        if start == end:
            return [start]

        queue: deque[Tuple[int, int]] = deque([start])
        visited: Set[Tuple[int, int]] = {start}
        parent: dict[Tuple[int, int], Tuple[int, int]] = {}

        while queue:
            curr = queue.popleft()
            if curr == end:
                break

            for nx, ny, _ in grid.get_neighbors(curr[0], curr[1]):
                nbr = (nx, ny)
                if nbr not in visited:
                    visited.add(nbr)
                    parent[nbr] = curr
                    queue.append(nbr)

        if end not in parent:
            return []

        # Reconstruct path
        path = []
        c: Optional[Tuple[int, int]] = end
        while c is not None and c in parent:
            path.append(c)
            c = parent[c]
        path.append(start)
        path.reverse()
        return path

    def _validate(
        self,
        grid: Grid,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
    ) -> bool:
        """Validate that start and end are connected and optimal path != BFS path."""
        dijkstra_path, dijkstra_cost, _ = DijkstraSolver.solve(grid, start_pos, end_pos)
        if not dijkstra_path or math.isinf(dijkstra_cost):
            return False

        bfs_path = self._bfs_fewest_steps(grid, start_pos, end_pos)
        if not bfs_path:
            return False

        # Pedagogical goal: ensure the weighted shortest path differs from the fewest-steps path
        return dijkstra_path != bfs_path
