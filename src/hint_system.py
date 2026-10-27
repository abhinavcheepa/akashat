"""Hint system mechanics for PATHFINDER.

Provides gameplay-tied hints:
1. Navigation Route Hint: highlights the immediate next optimal step toward the destination.
2. Quiz 50/50 Hint: eliminates an incorrect option during gate challenges.
Costs points, guarantees score never goes negative, and tracks hint usage for bonuses.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from src.level_manager import LevelManager
from src.question import Question


class HintManager:
    """Coordinates tactical hint purchases and game state assistance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize HintManager with hint cost configurations."""
        self.config: Dict[str, Any] = config or {}
        self.hints_used_this_level: int = 0
        self.active_path_hint: Optional[Tuple[int, int]] = None
        self.path_hint_timer: float = 0.0

    def set_config(self, config: Dict[str, Any]) -> None:
        """Update active hint configuration."""
        self.config = config

    def get_cost(self, hint_type: str) -> int:
        """Retrieve point cost for specified hint type."""
        hints_cfg = self.config.get("hints", {})
        if hint_type == "path":
            return int(hints_cfg.get("path_hint_cost", 40))
        elif hint_type == "quiz":
            return int(hints_cfg.get("quiz_50_50_cost", 30))
        return 50

    def can_afford(self, current_score: int, hint_type: str) -> bool:
        """Check whether current score covers hint cost."""
        return current_score >= self.get_cost(hint_type)

    def request_path_hint(
        self,
        level_manager: LevelManager,
        player_pos: Tuple[int, int],
        optimal_path: List[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Purchase and reveal the next tile on Dijkstra's optimal route.

        Returns target coordinate if affordable and available, None otherwise.
        """
        cost = self.get_cost("path")
        if level_manager.score < cost:
            return None

        # Find next tile along optimal path
        next_step: Optional[Tuple[int, int]] = None
        if player_pos in optimal_path:
            idx = optimal_path.index(player_pos)
            if idx + 1 < len(optimal_path):
                next_step = optimal_path[idx + 1]
        else:
            # If player wandered off optimal path, find nearest path node
            closest = min(
                optimal_path,
                key=lambda p: abs(p[0] - player_pos[0]) + abs(p[1] - player_pos[1]),
            )
            idx = optimal_path.index(closest)
            if idx + 1 < len(optimal_path):
                next_step = optimal_path[idx + 1]
            else:
                next_step = closest

        if next_step is not None:
            level_manager.deduct_score(cost)
            self.hints_used_this_level += 1
            self.active_path_hint = next_step
            self.path_hint_timer = 5.0  # glow for 5 seconds
            return next_step

        return None

    def request_quiz_hint(
        self,
        level_manager: LevelManager,
        question: Question,
        already_eliminated: Set[int],
    ) -> Optional[int]:
        """Purchase 50/50 hint to eliminate one incorrect option from the quiz overlay.

        Returns index of eliminated option, or None if unaffordable or all eliminated.
        """
        cost = self.get_cost("quiz")
        if level_manager.score < cost:
            return None

        # Identify wrong options that are not yet eliminated
        wrong_candidates = [
            i for i in range(len(question.options))
            if i != question.answer and i not in already_eliminated
        ]

        if not wrong_candidates:
            return None

        level_manager.deduct_score(cost)
        self.hints_used_this_level += 1
        eliminated_choice = random.choice(wrong_candidates)
        return eliminated_choice

    def update(self, dt: float) -> None:
        """Update active path hint timer."""
        if self.path_hint_timer > 0.0:
            self.path_hint_timer = max(0.0, self.path_hint_timer - dt)
            if self.path_hint_timer == 0.0:
                self.active_path_hint = None

    def reset_level(self) -> None:
        """Reset hint tracking for new level or restart."""
        self.hints_used_this_level = 0
        self.active_path_hint = None
        self.path_hint_timer = 0.0
