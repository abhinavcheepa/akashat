"""Path visualization and non-blocking exploration animation for PATHFINDER.

Animates Dijkstra's wavefront expansion step-by-step using delta-time ticks
(never blocking the main game loop with time.sleep), renders the optimal path
in the Primary accent color (#38D9FF), and displays alternative paths in the
Secondary accent color (#8B7CFF).
"""

from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import pygame


class PathVisualizer:
    """Controls non-blocking visual playback of Dijkstra exploration and path solutions."""

    def __init__(self) -> None:
        """Initialize visualizer state."""
        self.show_path: bool = False
        self.show_alternative: bool = True
        self.is_animating: bool = False

        self.optimal_path: List[Tuple[int, int]] = []
        self.optimal_cost: Union[int, float] = 0
        self.explored_order: List[Tuple[int, int]] = []

        self.alternative_path: Optional[List[Tuple[int, int]]] = None
        self.alternative_cost: Optional[Union[int, float]] = None

        # Non-blocking animation progress
        self.anim_cursor: int = 0
        self.time_accumulator: float = 0.0
        self.nodes_per_second: float = 80.0  # smooth wavefront expansion
        self.revealed_explored: List[Tuple[int, int]] = []
        self.revealed_explored_set: Set[Tuple[int, int]] = set()

        # Cached surfaces to eliminate per-frame allocations
        self._cached_tile_surf: Optional[pygame.Surface] = None
        self._cached_tile_size: Tuple[int, int] = (0, 0)
        self._cached_glow_surf: Optional[pygame.Surface] = None
        self._cached_glow_size: Tuple[int, int] = (0, 0)

    def set_solution(
        self,
        optimal_path: List[Tuple[int, int]],
        optimal_cost: Union[int, float],
        explored_order: List[Tuple[int, int]],
        alternative_path: Optional[List[Tuple[int, int]]] = None,
        alternative_cost: Optional[Union[int, float]] = None,
    ) -> None:
        """Assign solution data computed by DijkstraSolver."""
        self.optimal_path = optimal_path
        self.optimal_cost = optimal_cost
        self.explored_order = explored_order
        self.alternative_path = alternative_path
        self.alternative_cost = alternative_cost

        self.reset_state()

    def reset_state(self) -> None:
        """Reset animation cursor and hide paths."""
        self.show_path = False
        self.is_animating = False
        self.anim_cursor = 0
        self.time_accumulator = 0.0
        self.revealed_explored.clear()
        self.revealed_explored_set.clear()

    def toggle_path(self) -> None:
        """Toggle the visibility / animation of the optimal path."""
        self.show_path = not self.show_path
        if self.show_path:
            self.start_animation()
        else:
            self.is_animating = False

    def start_animation(self) -> None:
        """Begin animated wavefront reveal from step 0."""
        self.is_animating = True
        self.anim_cursor = 0
        self.time_accumulator = 0.0
        self.revealed_explored.clear()
        self.revealed_explored_set.clear()

    def update(self, dt: float) -> None:
        """Update animation state forward by dt seconds without blocking."""
        if not self.is_animating or not self.show_path:
            return

        self.time_accumulator += dt
        steps_to_reveal = int(self.time_accumulator * self.nodes_per_second)

        if steps_to_reveal > 0:
            self.time_accumulator -= steps_to_reveal / self.nodes_per_second
            end_cursor = min(self.anim_cursor + steps_to_reveal, len(self.explored_order))

            for i in range(self.anim_cursor, end_cursor):
                node = self.explored_order[i]
                if node not in self.revealed_explored_set:
                    self.revealed_explored.append(node)
                    self.revealed_explored_set.add(node)

            self.anim_cursor = end_cursor
            if self.anim_cursor >= len(self.explored_order):
                self.is_animating = False

    def draw(
        self,
        surface: pygame.Surface,
        tile_to_rect: Callable[[int, int], pygame.Rect],
        tile_to_center: Callable[[int, int], Tuple[int, int]],
        colors: Dict[str, pygame.Color],
    ) -> None:
        """Render explored search frontier, alternative path, and optimal path.

        Uses cached alpha surfaces to prevent frame-by-frame memory allocations.
        """
        if not self.show_path:
            return

        explored_color = colors.get("explored", pygame.Color("#1B3552"))
        primary_color = colors.get("primary", pygame.Color("#38D9FF"))
        secondary_color = colors.get("secondary", pygame.Color("#8B7CFF"))

        # 1. Draw explored wavefront
        nodes_to_draw = self.revealed_explored if self.is_animating else self.explored_order
        if nodes_to_draw:
            first_rect = tile_to_rect(*nodes_to_draw[0])
            tile_dim = (first_rect.width, first_rect.height)

            if self._cached_tile_surf is None or self._cached_tile_size != tile_dim:
                self._cached_tile_surf = pygame.Surface(tile_dim, pygame.SRCALPHA)
                self._cached_tile_surf.fill((*explored_color[:3], 90))
                self._cached_tile_size = tile_dim

            for x, y in nodes_to_draw:
                rect = tile_to_rect(x, y)
                surface.blit(self._cached_tile_surf, rect.topleft)

        # 2. Draw alternative path (if available and enabled)
        if self.alternative_path and len(self.alternative_path) > 1 and self.show_alternative:
            alt_points = [tile_to_center(x, y) for x, y in self.alternative_path]
            pygame.draw.lines(surface, secondary_color, False, alt_points, width=3)
            for pt in alt_points:
                pygame.draw.circle(surface, secondary_color, pt, radius=3)

        # 3. Draw optimal path
        if self.optimal_path and len(self.optimal_path) > 1:
            points = [tile_to_center(x, y) for x, y in self.optimal_path]
            surf_size = surface.get_size()

            if self._cached_glow_surf is None or self._cached_glow_size != surf_size:
                self._cached_glow_surf = pygame.Surface(surf_size, pygame.SRCALPHA)
                self._cached_glow_size = surf_size

            self._cached_glow_surf.fill((0, 0, 0, 0))
            pygame.draw.lines(self._cached_glow_surf, (*primary_color[:3], 60), False, points, width=9)
            pygame.draw.lines(self._cached_glow_surf, (*primary_color[:3], 130), False, points, width=6)
            surface.blit(self._cached_glow_surf, (0, 0))

            # Core path line
            pygame.draw.lines(surface, primary_color, False, points, width=3)
            for pt in points:
                pygame.draw.circle(surface, primary_color, pt, radius=3)
