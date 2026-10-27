"""Reward and scoring calculations for PATHFINDER.

Calculates multi-category point rewards for routing efficiency, gate clearances,
quiz precision, and tactical purity (completing missions without hints).
Guarantees score points never drop below zero.
"""

from typing import Any, Dict, List, Optional, Tuple


class RewardManager:
    """Calculates granular reward breakdowns based on configured values."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RewardManager with scoring configuration."""
        self.config: Dict[str, Any] = config or {}

    def set_config(self, config: Dict[str, Any]) -> None:
        """Update active reward configuration."""
        self.config = config

    def get_reward_value(self, key: str, default: int) -> int:
        """Retrieve configured point value."""
        rewards_cfg = self.config.get("rewards", {})
        return int(rewards_cfg.get(key, default))

    def calculate_level_rewards(
        self,
        player_cost: int,
        optimal_cost: float,
        efficiency: float,
        checkpoints_cleared: int,
        total_checkpoints: int,
        quiz_correct: int,
        hints_used: int,
    ) -> Dict[str, Any]:
        """Compute complete reward breakdown for Level Complete screen.

        Args:
            player_cost: Total movement cost incurred by player.
            optimal_cost: Theoretical minimum cost from Dijkstra.
            efficiency: Navigation efficiency percentage (0.0 to 100.0).
            checkpoints_cleared: Number of checkpoints successfully unlocked.
            total_checkpoints: Total checkpoints in the sector.
            quiz_correct: Number of correct quiz submissions.
            hints_used: Number of hints consumed during the mission.

        Returns:
            Dictionary containing itemized rewards and total points.
        """
        breakdown_items: List[Tuple[str, int]] = []

        # 1. Base level completion points
        base_pts = self.get_reward_value("base_level_completion", 100)
        breakdown_items.append(("Sector Clearance", base_pts))

        # 2. Navigation Efficiency Bonus
        eff_bonus = 0
        if efficiency >= 99.9:
            eff_bonus = self.get_reward_value("efficiency_100_bonus", 150)
            breakdown_items.append(("Flawless Dijkstra Match (100%)", eff_bonus))
        elif efficiency >= 90.0:
            eff_bonus = self.get_reward_value("efficiency_90_bonus", 100)
            breakdown_items.append((f"High Efficiency Routing ({efficiency:.1f}%)", eff_bonus))
        elif efficiency >= 75.0:
            eff_bonus = self.get_reward_value("efficiency_75_bonus", 50)
            breakdown_items.append((f"Sub-optimal Efficiency ({efficiency:.1f}%)", eff_bonus))

        # 3. Checkpoint Gate Completion Bonus
        cp_bonus = 0
        if total_checkpoints > 0 and checkpoints_cleared >= total_checkpoints:
            cp_bonus = self.get_reward_value("checkpoint_all_cleared_bonus", 200)
            breakdown_items.append(("All Security Gates Unlocked", cp_bonus))

        # 4. Quiz Question Points
        q_pts_unit = self.get_reward_value("correct_answer_points", 100)
        quiz_pts = quiz_correct * q_pts_unit
        if quiz_pts > 0:
            breakdown_items.append((f"Knowledge Verification ({quiz_correct} correct)", quiz_pts))

        # 5. Tactical Purity (No Hints Used) Bonus
        no_hint_bonus = 0
        if hints_used == 0:
            no_hint_bonus = self.get_reward_value("no_hints_used_bonus", 100)
            breakdown_items.append(("Tactical Purity (No Hints Used)", no_hint_bonus))

        total_pts = max(0, base_pts + eff_bonus + cp_bonus + quiz_pts + no_hint_bonus)

        return {
            "base_points": base_pts,
            "efficiency_bonus": eff_bonus,
            "checkpoint_bonus": cp_bonus,
            "quiz_points": quiz_pts,
            "no_hint_bonus": no_hint_bonus,
            "total_points": total_pts,
            "breakdown": breakdown_items,
            "hints_used": hints_used,
        }
