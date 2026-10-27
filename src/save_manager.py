"""Persistent game save and progression management for PATHFINDER.

Saves progress, best scores, and achievements to JSON.
Guarantees resilience against missing, empty, or corrupted files by backing up
malformed files and restoring a clean save state without crashing.
"""

from datetime import datetime
import json
import os
import shutil
from typing import Any, Dict, List, Optional


class SaveManager:
    """Manages profile persistence, achievements, and corruption recovery."""

    def __init__(self, filepath: str = "data/save_game.json") -> None:
        """Initialize SaveManager with save file destination."""
        self.filepath: str = filepath
        self.data: Dict[str, Any] = self.load()

    def get_fresh_save(self) -> Dict[str, Any]:
        """Return initial default save dictionary."""
        return {
            "version": "1.0",
            "current_level": 1,
            "highest_unlocked_level": 1,
            "total_score": 0,
            "completed_levels": {},
            "achievements": [],
            "selected_topic": "All Topics",
            "selected_difficulty": "Level Default",
            "last_updated": datetime.now().isoformat(),
        }

    def load(self) -> Dict[str, Any]:
        """Load save data from disk with corrupted-file backup and recovery."""
        if not os.path.exists(self.filepath):
            fresh = self.get_fresh_save()
            return fresh

        # Check for empty file
        if os.path.getsize(self.filepath) == 0:
            self._backup_corrupted("empty")
            fresh = self.get_fresh_save()
            self.save(fresh)
            return fresh

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if not isinstance(loaded, dict) or "current_level" not in loaded:
                raise ValueError("Invalid save data structure")

            self.data = loaded
            return loaded

        except Exception as e:
            print(f"[Warning] Corrupted save file detected ({e}). Backing up and creating fresh profile.")
            self._backup_corrupted("corrupted")
            fresh = self.get_fresh_save()
            self.save(fresh)
            return fresh

    def _backup_corrupted(self, reason: str) -> None:
        """Create timestamped backup of unreadable save file instead of deleting."""
        if not os.path.exists(self.filepath):
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.filepath}.{reason}_{timestamp}.bak"
        try:
            shutil.copyfile(self.filepath, backup_path)
            print(f"[Info] Corrupted save backed up to: {backup_path}")
        except Exception as e:
            print(f"[Warning] Failed creating save backup: {e}")

    def save(self, data: Optional[Dict[str, Any]] = None) -> bool:
        """Write save data safely to disk."""
        if data is not None:
            self.data = data

        self.data["last_updated"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)

        try:
            # Write to temporary file first then replace for atomicity
            tmp_path = f"{self.filepath}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)

            if os.path.exists(self.filepath):
                os.replace(tmp_path, self.filepath)
            else:
                os.rename(tmp_path, self.filepath)
            return True

        except Exception as e:
            print(f"[Error] Failed writing save file: {e}")
            return False

    def has_saved_game(self) -> bool:
        """Return True if active progress or completed levels exist in save."""
        return (
            self.data.get("current_level", 1) > 1
            or self.data.get("total_score", 0) > 0
            or len(self.data.get("completed_levels", {})) > 0
        )

    def record_level_completion(self, level: int, score: int, efficiency: float) -> None:
        """Record completed level, best score, highest level, and achievements."""
        lvl_str = str(level)
        completed = self.data.setdefault("completed_levels", {})

        prev_best = completed.get(lvl_str, {}).get("best_score", 0)
        completed[lvl_str] = {
            "best_score": max(prev_best, score),
            "efficiency": efficiency,
            "completed_at": datetime.now().isoformat(),
        }

        # Update unlocked level
        next_lvl = level + 1
        curr_highest = self.data.get("highest_unlocked_level", 1)
        self.data["highest_unlocked_level"] = max(curr_highest, next_lvl)
        self.data["current_level"] = next_lvl
        self.data["total_score"] = self.data.get("total_score", 0) + score

        # Check and grant achievements
        achievements = set(self.data.setdefault("achievements", []))
        if efficiency >= 99.9:
            achievements.add("Dijkstra Disciple (100% Efficiency)")
        if level >= 1:
            achievements.add("Sector Pioneer (Cleared Level 1)")
        if level >= 5:
            achievements.add("Master Navigator (Conquered All 5 Sectors)")

        self.data["achievements"] = list(achievements)
        self.save()
