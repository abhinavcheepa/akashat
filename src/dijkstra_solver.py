"""Dijkstra shortest path algorithm solver for PATHFINDER.

Pure algorithmic logic with no Pygame dependencies. Uses heapq for priority
queue operations and tracks the exact order of visited nodes for non-blocking
visualization.
"""

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple, Union

from src.grid import Grid


class DijkstraSolver:
    """Stateless solver implementing Dijkstra's algorithm on a weighted Grid."""

    @staticmethod
    def solve(
        grid: Grid,
        start: Tuple[int, int],
        end: Tuple[int, int],
        blocked_edge: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> Tuple[List[Tuple[int, int]], Union[int, float], List[Tuple[int, int]]]:
        """Find the lowest-cost path from start to end using Dijkstra's algorithm.

        The start cell cost is free (0); cost is incurred strictly upon entering
        each subsequent cell based on that cell's terrain weight.

        Args:
            grid: The navigation Grid instance.
            start: Source coordinate (x, y).
            end: Target coordinate (x, y).
            blocked_edge: Optional edge ((x1, y1), (x2, y2)) prohibited during search.

        Returns:
            Tuple of:
            - path: List of (x, y) coordinates from start to end inclusive (empty if unreachable).
            - cost: Total accumulated traversal cost (math.inf if unreachable).
            - explored_order: List of coordinates in the order they were finalized/explored.
        """
        if not grid.is_valid(*start) or not grid.is_valid(*end):
            return [], math.inf, []

        start_cell = grid.get_cell(*start)
        end_cell = grid.get_cell(*end)
        if start_cell is None or start_cell.is_wall or end_cell is None or end_cell.is_wall:
            return [], math.inf, []

        if start == end:
            return [start], 0, [start]

        # Priority queue entries: (cost, counter, (x, y))
        counter = 0
        heap: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(heap, (0, counter, start))

        dist: Dict[Tuple[int, int], int] = {start: 0}
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
        visited: Set[Tuple[int, int]] = set()
        explored_order: List[Tuple[int, int]] = []

        reached = False

        while heap:
            current_cost, _, u = heapq.heappop(heap)

            if u in visited:
                continue

            visited.add(u)
            explored_order.append(u)

            if u == end:
                reached = True
                break

            if current_cost > dist.get(u, math.inf):
                continue

            for nx, ny, weight in grid.get_neighbors(u[0], u[1], blocked_edge=blocked_edge):
                v = (nx, ny)
                new_cost = current_cost + weight

                if new_cost < dist.get(v, math.inf):
                    dist[v] = new_cost
                    prev[v] = u
                    counter += 1
                    heapq.heappush(heap, (new_cost, counter, v))

        if not reached or end not in dist:
            return [], math.inf, explored_order

        # Reconstruct path from end backwards to start
        path: List[Tuple[int, int]] = []
        curr: Optional[Tuple[int, int]] = end
        while curr is not None and curr in prev:
            path.append(curr)
            curr = prev[curr]
        path.append(start)
        path.reverse()

        return path, dist[end], explored_order

    @staticmethod
    def find_alternative(
        grid: Grid,
        start: Tuple[int, int],
        end: Tuple[int, int],
        optimal_path: List[Tuple[int, int]],
    ) -> Optional[Tuple[List[Tuple[int, int]], Union[int, float]]]:
        """Find an alternative path by blocking the midpoint edge of the optimal path.

        Args:
            grid: The navigation Grid instance.
            start: Source coordinate (x, y).
            end: Target coordinate (x, y).
            optimal_path: The previously computed shortest path.

        Returns:
            Tuple of (alternative_path, alternative_cost) if an alternate route exists,
            or None if no alternate route is possible.
        """
        if not optimal_path or len(optimal_path) < 2:
            return None

        # Identify the middle edge of the optimal path
        mid_idx = (len(optimal_path) - 1) // 2
        blocked_edge = (optimal_path[mid_idx], optimal_path[mid_idx + 1])

        alt_path, alt_cost, _ = DijkstraSolver.solve(
            grid=grid,
            start=start,
            end=end,
            blocked_edge=blocked_edge,
        )

        if alt_path and alt_cost < math.inf:
            return alt_path, alt_cost

        return None
