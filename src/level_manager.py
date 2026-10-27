"""Level and progression management for PATHFINDER.

Calculates grid dimensions according to room count specifications:
tile_dimension = 2 * rooms + 1 (where walls are explicit tiles).
Fully driven by configuration with graceful fallbacks for missing or corrupted entries.
"""

from typing import Any, Dict, Optional, Tuple


class LevelManager:
    """Manages player progression, level difficulty sizing, and score."""

    def __init__(self, start_level: int = 1, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the level manager with level configurations."""
        self.current_level: int = max(1, start_level)
        self.score: int = 0
        self.config: Dict[str, Any] = config or {}

        # Default fallback definitions if config is empty or missing keys
        self.default_levels: Dict[int, Dict[str, Any]] = {
            1: {"rooms": 7, "checkpoints": 1, "difficulty": "Easy", "name": "Sector Alpha"},
            2: {"rooms": 9, "checkpoints": 2, "difficulty": "Easy", "name": "Sector Beta"},
            3: {"rooms": 11, "checkpoints": 2, "difficulty": "Medium", "name": "Sector Gamma"},
            4: {"rooms": 13, "checkpoints": 3, "difficulty": "Medium", "name": "Sector Delta"},
            5: {"rooms": 15, "checkpoints": 4, "difficulty": "Hard", "name": "Sector Omega"},
        }

    def set_config(self, config: Dict[str, Any]) -> None:
        """Update active configuration."""
        self.config = config

    def get_level_data(self, level: Optional[int] = None) -> Dict[str, Any]:
        """Retrieve dictionary configuration for specified or current level."""
        lvl = level if level is not None else self.current_level
        levels_cfg = self.config.get("levels", {})

        # Try string key, then integer key
        if str(lvl) in levels_cfg:
            return levels_cfg[str(lvl)]
        if lvl in levels_cfg:
            return levels_cfg[lvl]

        if lvl in self.default_levels:
            return self.default_levels[lvl]

        # Procedural fallback for levels beyond default
        rooms = 7 + (lvl - 1) * 2
        cps = min(5, 1 + (lvl - 1) // 2)
        diff = "Hard" if lvl >= 5 else ("Medium" if lvl >= 3 else "Easy")
        return {"rooms": rooms, "checkpoints": cps, "difficulty": diff, "name": f"Sector {lvl}"}

    def get_rooms(self) -> int:
        """Return the number of room chambers on each axis for the current level."""
        data = self.get_level_data()
        return data.get("rooms", 7 + (self.current_level - 1) * 2)

    def get_grid_size(self) -> Tuple[int, int]:
        """Return the tile grid dimensions as (width, height).

        Calculated as 2 * rooms + 1 per project specification.
        For Level 1 (7 rooms), this yields (15, 15).
        """
        rooms = self.get_rooms()
        dim = 2 * rooms + 1
        return (dim, dim)

    def get_checkpoints_count(self) -> int:
        """Return the required number of checkpoints for the current level."""
        data = self.get_level_data()
        return data.get("checkpoints", 1)

    def get_difficulty(self) -> str:
        """Return difficulty tier string ('Easy', 'Medium', 'Hard')."""
        data = self.get_level_data()
        return data.get("difficulty", "Easy")

    def get_name(self) -> str:
        """Return sector title string."""
        data = self.get_level_data()
        return data.get("name", f"Sector {self.current_level}")

    @property
    def max_level(self) -> int:
        """Return maximum configured level."""
        levels_cfg = self.config.get("levels", {})
        if levels_cfg:
            try:
                keys = [int(k) for k in levels_cfg.keys()]
                return max(keys)
            except Exception:
                pass
        return max(self.default_levels.keys())

    @property
    def is_max_level(self) -> bool:
        """Whether current level is the final sector."""
        return self.current_level >= self.max_level

    def add_score(self, points: int) -> None:
        """Add points to cumulative score. Score never drops below zero."""
        self.score = max(0, self.score + max(0, points))

    def deduct_score(self, points: int) -> None:
        """Deduct points from cumulative score without going negative."""
        self.score = max(0, self.score - max(0, points))

    def advance(self) -> bool:
        """Advance to next level if available. Returns True if advanced, False if completed max level."""
        if self.current_level < self.max_level:
            self.current_level += 1
            return True
        return False

    def reset(self) -> None:
        """Reset player progression back to Level 1 and score to 0."""
        self.current_level = 1
        self.score = 0
