"""Comprehensive adversarial tests for Phase B:
- Level configuration loading and malformed config resilience
- Strict checkpoint count matching across all 5 levels over many seeds (L1: 1, L2: 2, L3: 2, L4: 3, L5: 4)
- Reward calculation, itemized bonuses, and score non-negative invariant
- Hint system mechanics (path hint, 50/50 quiz hint, point costs, no negative score)
- SaveManager: clean roundtrip, missing file, corrupted file backup, empty file recovery
- QuestionManager topic & difficulty fallback when questions are scarce
"""

import json
import math
import os
import shutil
import tempfile
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pytest
import pygame

from src.checkpoint import CheckpointManager
from src.dijkstra_solver import DijkstraSolver
from src.game import Game, GameState
from src.hint_system import HintManager
from src.level_manager import LevelManager
from src.maze_generator import MazeGenerator
from src.player import Player
from src.question import Question, QuestionManager
from src.reward_manager import RewardManager
from src.save_manager import SaveManager


def test_level_config_loading_and_bad_config_resilience() -> None:
    """Verify LevelManager correctly parses configuration and handles missing/bad keys."""
    # 1. Standard config loading
    with open("config/game_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lm = LevelManager(start_level=1, config=cfg)
    assert lm.get_rooms() == 7
    assert lm.get_grid_size() == (15, 15)
    assert lm.get_checkpoints_count() == 1
    assert lm.get_difficulty() == "Easy"
    assert lm.max_level == 5

    # Advance to Level 4
    lm.current_level = 4
    assert lm.get_rooms() == 13
    assert lm.get_grid_size() == (27, 27)
    assert lm.get_checkpoints_count() == 3
    assert lm.get_difficulty() == "Medium"

    # Advance to Level 5
    lm.current_level = 5
    assert lm.get_rooms() == 15
    assert lm.get_grid_size() == (31, 31)
    assert lm.get_checkpoints_count() == 4
    assert lm.get_difficulty() == "Hard"

    # 2. Malformed / Empty config resilience
    broken_lm = LevelManager(start_level=1, config={})
    assert broken_lm.get_rooms() == 7
    assert broken_lm.get_grid_size() == (15, 15)
    assert broken_lm.get_checkpoints_count() == 1
    assert broken_lm.max_level == 5

    broken_lm.current_level = 99
    # Beyond default: procedural fallback generates valid dimensions
    assert broken_lm.get_rooms() == 7 + (99 - 1) * 2
    assert broken_lm.get_grid_size() == (2 * broken_lm.get_rooms() + 1, 2 * broken_lm.get_rooms() + 1)


def test_checkpoint_count_matches_config_all_5_levels_over_many_seeds() -> None:
    """Test cut-vertex checkpoint placement yields exact counts for L1-L5 across 15+ random seeds."""
    qm = QuestionManager(data_path="data/questions.json")
    cm = CheckpointManager(qm)

    level_specs = [
        (1, 7, 1),
        (2, 9, 2),
        (3, 11, 2),
        (4, 13, 3),
        (5, 15, 4),
    ]

    for lvl, rooms, expected_count in level_specs:
        for seed_idx in range(3):  # 3 seeds per level = 15 procedural maze tests
            seed = seed_idx * 1337 + lvl * 17
            gen = MazeGenerator(seed=seed)
            grid, start, end = gen.generate(rooms_x=rooms, rooms_y=rooms)

            opt_path, opt_cost, _ = DijkstraSolver.solve(grid, start, end)
            assert opt_path, f"Path must exist for Level {lvl} seed {seed}"

            checkpoints = cm.generate_checkpoints(
                grid=grid,
                start_pos=start,
                end_pos=end,
                optimal_path=opt_path,
                count=expected_count,
            )

            assert len(checkpoints) == expected_count, (
                f"Level {lvl} expected exactly {expected_count} checkpoints, but got {len(checkpoints)}"
            )

            # Ensure all checkpoints are cut-vertices
            for cp in checkpoints:
                assert cm.validate_unbypassable(grid, start, end, cp.pos), (
                    f"Checkpoint {cp.checkpoint_num} at {cp.pos} must be unbypassable"
                )


def test_reward_manager_calculations_and_non_negative_score() -> None:
    """Verify itemized point awards and guarantee score never drops below zero."""
    with open("config/game_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    rm = RewardManager(config=cfg)

    # 1. Flawless run: 100% efficiency, all gates cleared, 2 correct, 0 hints
    rewards_flawless = rm.calculate_level_rewards(
        player_cost=20,
        optimal_cost=20,
        efficiency=100.0,
        checkpoints_cleared=2,
        total_checkpoints=2,
        quiz_correct=2,
        hints_used=0,
    )

    # Base 100 + Eff 150 + Gates 200 + Quiz (2*100) + NoHint 100 = 750
    assert rewards_flawless["base_points"] == 100
    assert rewards_flawless["efficiency_bonus"] == 150
    assert rewards_flawless["checkpoint_bonus"] == 200
    assert rewards_flawless["quiz_points"] == 200
    assert rewards_flawless["no_hint_bonus"] == 100
    assert rewards_flawless["total_points"] == 750

    # 2. Run with hints and lower efficiency: no hint bonus
    rewards_hints = rm.calculate_level_rewards(
        player_cost=40,
        optimal_cost=20,
        efficiency=50.0,
        checkpoints_cleared=1,
        total_checkpoints=2,
        quiz_correct=1,
        hints_used=2,
    )
    assert rewards_hints["efficiency_bonus"] == 0
    assert rewards_hints["no_hint_bonus"] == 0
    assert rewards_hints["checkpoint_bonus"] == 0
    assert rewards_hints["total_points"] == 200

    # 3. Non-negative score enforcement in LevelManager
    lm = LevelManager(start_level=1)
    lm.score = 50
    lm.deduct_score(150)
    assert lm.score == 0  # Clamped at 0, never negative!


def test_hint_system_costs_and_effects() -> None:
    """Verify path hints, 50/50 quiz hints, point deductions, and non-negative score guards."""
    with open("config/game_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    hm = HintManager(config=cfg)
    lm = LevelManager(start_level=1)
    lm.score = 100

    path_cost = hm.get_cost("path")
    assert path_cost == 40
    quiz_cost = hm.get_cost("quiz")
    assert quiz_cost == 30

    # 1. Path hint execution
    optimal_path = [(1, 1), (2, 1), (3, 1), (4, 1)]
    next_step = hm.request_path_hint(lm, player_pos=(1, 1), optimal_path=optimal_path)
    assert next_step == (2, 1)
    assert lm.score == 100 - path_cost
    assert hm.hints_used_this_level == 1
    assert hm.active_path_hint == (2, 1)

    # 2. 50/50 Quiz hint execution
    q = Question(
        id="t_q",
        topic="DSA",
        difficulty="Easy",
        question="What is 2+2?",
        options=["1", "2", "3", "4"],
        answer=3,
        explanation="4",
    )
    eliminated = set()
    elim1 = hm.request_quiz_hint(lm, q, eliminated)
    assert elim1 is not None
    assert elim1 != q.answer  # Never eliminates correct answer!
    assert lm.score == 100 - path_cost - quiz_cost
    assert hm.hints_used_this_level == 2

    # 3. Cannot afford hint when score is too low
    lm.score = 10
    assert not hm.can_afford(lm.score, "path")
    no_hint = hm.request_path_hint(lm, (1, 1), optimal_path)
    assert no_hint is None
    assert lm.score == 10  # Score remains intact


def test_save_manager_lifecycle_and_corruption_recovery() -> None:
    """Verify save/load roundtrip, missing file, corrupted file, and empty file recovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = os.path.join(tmpdir, "test_save.json")

        # 1. Missing file: returns fresh profile
        sm = SaveManager(filepath=save_file)
        assert sm.data["current_level"] == 1
        assert sm.data["total_score"] == 0
        assert not sm.has_saved_game()

        # 2. Save progress roundtrip
        sm.record_level_completion(level=1, score=500, efficiency=95.0)
        assert sm.data["current_level"] == 2
        assert sm.data["total_score"] == 500
        assert "1" in sm.data["completed_levels"]
        assert sm.has_saved_game()

        # Reload from disk
        sm2 = SaveManager(filepath=save_file)
        assert sm2.data["current_level"] == 2
        assert sm2.data["total_score"] == 500

        # 3. Empty file recovery
        with open(save_file, "w", encoding="utf-8") as f:
            f.write("")

        sm_empty = SaveManager(filepath=save_file)
        assert sm_empty.data["current_level"] == 1  # Gracefully restored fresh save!
        # Verify backup was created
        backups = [f for f in os.listdir(tmpdir) if ".bak" in f]
        assert len(backups) >= 1

        # 4. Corrupted / Malformed JSON recovery
        with open(save_file, "w", encoding="utf-8") as f:
            f.write("{this is not valid json ::: 123456}")

        sm_corrupt = SaveManager(filepath=save_file)
        assert sm_corrupt.data["current_level"] == 1  # Gracefully recovered!
        corrupt_backups = [f for f in os.listdir(tmpdir) if "corrupted" in f and ".bak" in f]
        assert len(corrupt_backups) >= 1


def test_question_manager_scarce_topic_and_difficulty_fallback() -> None:
    """Verify QuestionManager handles scarce topic/difficulty combinations without crashing."""
    qm = QuestionManager(data_path="data/questions.json")

    # 1. Non-existent topic falls back gracefully to bank
    q_nonexistent = qm.get_question(topic="Astrophysics", difficulty="Easy")
    assert q_nonexistent is not None

    # 2. Topic without 'Hard' difficulty falls back gracefully without crash
    q_diff_fallback = qm.get_question(topic="Networks", difficulty="UltraHard")
    assert q_diff_fallback is not None

    # 3. 'All Topics' and 'Level Default' requests
    q_all = qm.get_question(topic="All Topics", difficulty="Level Default")
    assert q_all is not None


def test_scoring_never_negative() -> None:
    """Verify scoring never drops below zero across deductions, penalties, and hints."""
    # 1. LevelManager direct deductions
    lm = LevelManager(start_level=1)
    assert lm.score == 0
    lm.deduct_score(100)
    assert lm.score == 0  # Clamped at 0

    lm.score = 30
    lm.deduct_score(50)
    assert lm.score == 0  # Clamped at 0

    lm.deduct_score(1)
    assert lm.score == 0

    # 2. Hint purchasing affordability guard
    with open("config/game_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    hm = HintManager(config=cfg)
    assert not hm.can_afford(0, "path")
    assert not hm.can_afford(10, "path")
    assert not hm.can_afford(20, "quiz")

    # If score is lower than cost, request_path_hint returns None and does not deduct
    lm.score = 20
    assert hm.request_path_hint(lm, player_pos=(1, 1), optimal_path=[(1, 1), (1, 2)]) is None
    assert lm.score == 20  # Score untouched

    # 3. Game integration: wrong answer penalty when score is low
    game = Game(config_path="config/game_config.json")
    game.state = GameState.PLAYING
    game.level_manager.score = 10  # Less than penalty of 25

    cp = game.checkpoint_manager.checkpoints[0]
    game._trigger_checkpoint(cp)
    wrong_idx = (cp.question.answer + 1) % 4
    game.selected_option_index = wrong_idx
    game._submit_checkpoint_answer()

    assert game.level_manager.score == 0  # Clamped at zero, never negative

    # 4. Timeout penalty when score is low
    game.restart_level()
    game.level_manager.score = 5
    cp = game.checkpoint_manager.checkpoints[0]
    game._trigger_checkpoint(cp)
    game.update(35.0)  # Trigger timeout
    assert game.level_manager.score == 0  # Clamped at zero, never negative

    # 5. RewardManager calculation with worst possible performance
    rm = RewardManager(config=cfg)
    rewards_bad = rm.calculate_level_rewards(
        player_cost=1000,
        optimal_cost=10,
        efficiency=1.0,
        checkpoints_cleared=0,
        total_checkpoints=4,
        quiz_correct=0,
        hints_used=10,
    )
    assert rewards_bad["total_points"] >= 0
    assert rewards_bad["efficiency_bonus"] == 0
    assert rewards_bad["checkpoint_bonus"] == 0
    assert rewards_bad["no_hint_bonus"] == 0

    pygame.quit()


def test_config_driven_levels() -> None:
    """Verify that level progression and dimensions are completely config-driven."""
    with open("config/game_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    levels_cfg = cfg["levels"]
    assert len(levels_cfg) == 5

    # Verify each level 1-5 has required keys and valid values
    expected_rooms = {1: 7, 2: 9, 3: 11, 4: 13, 5: 15}
    expected_cps = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}
    expected_diffs = {1: "Easy", 2: "Easy", 3: "Medium", 4: "Medium", 5: "Hard"}

    for lvl_idx in range(1, 6):
        lvl_str = str(lvl_idx)
        assert lvl_str in levels_cfg
        data = levels_cfg[lvl_str]
        assert data["rooms"] == expected_rooms[lvl_idx]
        assert data["checkpoints"] == expected_cps[lvl_idx]
        assert data["difficulty"] == expected_diffs[lvl_idx]
        assert "name" in data

        lm = LevelManager(start_level=lvl_idx, config=cfg)
        assert lm.get_rooms() == expected_rooms[lvl_idx]
        assert lm.get_checkpoints_count() == expected_cps[lvl_idx]
        assert lm.get_difficulty() == expected_diffs[lvl_idx]
        expected_size = (2 * expected_rooms[lvl_idx] + 1, 2 * expected_rooms[lvl_idx] + 1)
        assert lm.get_grid_size() == expected_size

    # Verify custom config dynamically modifies level parameters without code changes
    custom_cfg = {
        "levels": {
            "1": {"rooms": 19, "checkpoints": 5, "difficulty": "Hard", "name": "Custom Sector"},
            "2": {"rooms": 21, "checkpoints": 6, "difficulty": "Hard", "name": "Custom Sector 2"},
        }
    }
    custom_lm = LevelManager(start_level=1, config=custom_cfg)
    assert custom_lm.get_rooms() == 19
    assert custom_lm.get_checkpoints_count() == 5
    assert custom_lm.get_difficulty() == "Hard"
    assert custom_lm.get_grid_size() == (39, 39)
    assert custom_lm.max_level == 2

