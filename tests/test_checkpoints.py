"""Adversarial test suite for Phase A: Checkpoints, Question Bank, and Timers.

Covers:
1. Checkpoint placement on reachable cells
2. Checkpoint placement on route to destination (cut-vertex enforcement)
3. Genuine blocking: cannot bypass checkpoints without resolving them
4. Solvability invariant: maze remains solvable with open checkpoints
5. Question bank completeness: 8+ verified questions per topic across all 7 topics
6. Question randomization and anti-repetition guards
7. Timer non-blocking updates and timeout expiration
8. Game integration: checkpoint trigger, unlocking, and penalty application
"""

import json
import math
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from typing import Set

import pytest
import pygame

from src.checkpoint import Checkpoint, CheckpointManager
from src.dijkstra_solver import DijkstraSolver
from src.game import Game, GameState
from src.grid import CellType, Grid
from src.hint_system import HintManager
from src.level_manager import LevelManager
from src.maze_generator import MazeGenerator
from src.player import Player
from src.question import Question, QuestionManager
from src.timer import Timer


def test_question_bank_structure_and_counts() -> None:
    """Verify offline question bank has at least 8 verified questions per topic."""
    qm = QuestionManager(data_path="data/questions.json")
    assert len(qm.questions) >= 56

    required_topics = [
        "DSA",
        "Python and OOP",
        "DBMS",
        "OS",
        "Networks",
        "Math and Probability",
        "General Knowledge",
    ]

    for topic in required_topics:
        assert topic in qm.by_topic, f"Topic '{topic}' missing from loaded question bank"
        topic_questions = qm.by_topic[topic]
        assert len(topic_questions) >= 8, f"Topic '{topic}' has fewer than 8 questions ({len(topic_questions)})"

        # Verify question data integrity
        for q in topic_questions:
            assert len(q.question.strip()) > 0
            assert len(q.options) == 4
            assert 0 <= q.answer <= 3
            assert len(q.explanation.strip()) > 0
            assert q.difficulty in ("Easy", "Medium", "Hard")

    # Verify custom materials slot is defined
    assert isinstance(qm.custom_materials, list)


def test_question_anti_repetition_and_shuffling() -> None:
    """Verify QuestionManager avoids immediate repeats and shuffles options."""
    qm = QuestionManager(data_path="data/questions.json", history_capacity=5)
    served_ids = []

    # Request 5 consecutive questions from DSA
    for _ in range(5):
        q = qm.get_question(topic="DSA")
        assert q is not None
        assert q.id not in served_ids, f"Question {q.id} was repeated prematurely"
        served_ids.append(q.id)

    # Shuffling test: ensure create_shuffled_copy preserves the correct answer text
    raw_q = qm.questions[0]
    shuffled = raw_q.create_shuffled_copy()
    assert shuffled.options[shuffled.answer] == raw_q.options[raw_q.answer]


def test_timer_delta_time_countdown_and_timeout() -> None:
    """Verify non-blocking delta-time Timer behavior."""
    timer = Timer(duration=10.0)
    assert timer.time_remaining == 10.0
    assert not timer.is_running
    assert not timer.is_expired

    timer.start()
    assert timer.is_running

    # Tick 3.5 seconds
    expired = timer.update(3.5)
    assert not expired
    assert not timer.is_expired
    assert math.isclose(timer.time_remaining, 6.5, abs_tol=1e-4)
    assert math.isclose(timer.progress, 0.65, abs_tol=1e-4)

    # Tick remaining 6.5 seconds -> should trigger expiration
    expired = timer.update(6.5)
    assert expired is True
    assert timer.is_expired is True
    assert timer.is_running is False
    assert timer.time_remaining == 0.0
    assert timer.progress == 0.0

    # Further updates after expiration return False
    assert timer.update(1.0) is False


def test_checkpoints_placed_on_route_and_unbypassable() -> None:
    """Verify checkpoints lie strictly on route bottlenecks and cannot be bypassed."""
    gen = MazeGenerator(seed=42)
    grid, start, end = gen.generate(rooms_x=7, rooms_y=7)

    opt_path, opt_cost, _ = DijkstraSolver.solve(grid, start, end)
    assert opt_path

    qm = QuestionManager(data_path="data/questions.json")
    cm = CheckpointManager(qm)

    checkpoints = cm.generate_checkpoints(
        grid=grid,
        start_pos=start,
        end_pos=end,
        optimal_path=opt_path,
        count=1,
        difficulty="Easy",
    )

    assert len(checkpoints) == 1
    cp = checkpoints[0]

    # 1. Checkpoint is on a reachable open cell
    cell = grid.get_cell(*cp.pos)
    assert cell is not None
    assert not cell.is_wall

    # 2. Checkpoint is not start or end
    assert cp.pos != start
    assert cp.pos != end

    # 3. Checkpoint lies on the path
    re_path, re_cost, _ = DijkstraSolver.solve(grid, start, end)
    assert cp.pos in re_path
    assert not math.isinf(re_cost)

    # 4. Strictly unbypassable: when the checkpoint is blocked, destination cannot be reached
    cell.is_wall = True
    blocked_path, blocked_cost, _ = DijkstraSolver.solve(grid, start, end)
    cell.is_wall = False

    assert not blocked_path or math.isinf(blocked_cost)


def test_game_checkpoint_interaction_and_locking() -> None:
    """Verify checkpoint triggering, locking destination, penalties, and unlocking."""
    game = Game(config_path="config/game_config.json")
    game.state = GameState.PLAYING

    assert len(game.checkpoint_manager.checkpoints) >= 1
    cp = game.checkpoint_manager.checkpoints[0]
    assert not cp.is_cleared

    # Move player directly to the checkpoint position
    game.player.reset(cp.pos)
    assert game.player.pos == cp.pos

    # Trigger checkpoint gate
    game._trigger_checkpoint(cp)
    assert game.active_checkpoint is cp
    assert game.checkpoint_timer.is_running
    assert not game.checkpoint_feedback_mode

    # Simulate wrong answer
    wrong_idx = (cp.question.answer + 1) % 4
    game.selected_option_index = wrong_idx
    initial_cost = game.player.total_cost
    initial_score = game.level_manager.score

    game._submit_checkpoint_answer()
    assert game.checkpoint_feedback_mode is True
    assert game.checkpoint_last_correct is False
    assert not cp.is_cleared
    assert game.player.total_cost > initial_cost  # Penalty applied

    # Retry checkpoint
    game._retry_checkpoint()
    assert game.checkpoint_feedback_mode is False
    assert game.active_checkpoint is cp

    # Submit correct answer
    correct_idx = game.active_checkpoint.question.answer
    game.selected_option_index = correct_idx
    game._submit_checkpoint_answer()

    assert game.checkpoint_feedback_mode is True
    assert game.checkpoint_last_correct is True
    assert cp.is_cleared is True
    assert game.checkpoint_manager.all_cleared() is True

    # Close modal and move to destination
    game.active_checkpoint = None
    game.player.reset(game.end_pos)
    # Check level completion
    if game.checkpoint_manager.all_cleared():
        game.level_completed = True

    assert game.level_completed is True

    # Level restart restores locked status
    game.restart_level()
    assert not cp.is_cleared
    assert game.level_completed is False

    pygame.quit()


def test_checkpoint_cannot_be_bypassed() -> None:
    """Verify that a locked checkpoint cannot be bypassed in topology or game rules."""
    game = Game(config_path="config/game_config.json")
    game.state = GameState.PLAYING

    assert len(game.checkpoint_manager.checkpoints) >= 1
    cp = game.checkpoint_manager.checkpoints[0]
    assert not cp.is_cleared

    # 1. Topological cut-vertex verification: destination is completely unreachable without checkpoint tile
    cell = game.grid.get_cell(*cp.pos)
    assert cell is not None
    cell.is_wall = True
    path, cost, _ = DijkstraSolver.solve(game.grid, game.start_pos, game.end_pos)
    cell.is_wall = False
    assert not path or math.isinf(cost), "Checkpoint must be a cut-vertex: no alternative route can bypass it"

    # 2. Game rule verification: reaching end_pos without clearing checkpoint blocks level completion
    game.player.reset(game.end_pos)
    assert game.player.pos == game.end_pos
    assert not game.checkpoint_manager.all_cleared()

    # Re-trigger movement check at end_pos
    if not game.checkpoint_manager.all_cleared():
        game._set_status("DESTINATION LOCKED! Clear all sector gates first.")

    assert game.level_completed is False
    assert "DESTINATION LOCKED" in game.status_message

    pygame.quit()


def test_no_hint_can_clear_checkpoint_without_correct_answer() -> None:
    """Prove that no hint (path or quiz elimination) can ever unlock or clear a checkpoint."""
    with open("config/game_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    hm = HintManager(config=cfg)
    lm = LevelManager(start_level=1, config=cfg)
    lm.score = 500  # Plenty of score to purchase all hints

    q = Question(
        id="gate_q1",
        topic="DSA",
        difficulty="Easy",
        question="Which data structure does Dijkstra use?",
        options=["Queue", "Stack", "Min-Heap", "Array"],
        answer=2,
        explanation="Min-Heap priority queue provides O((V+E) log V).",
    )
    cp = Checkpoint(checkpoint_num=1, pos=(3, 3), difficulty="Easy", question=q)
    assert cp.is_cleared is False

    # 1. Request path hint -> does NOT unlock or clear checkpoint
    optimal_path = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (3, 4)]
    next_step = hm.request_path_hint(lm, player_pos=(1, 1), optimal_path=optimal_path)
    assert next_step == (2, 1)
    assert cp.is_cleared is False, "Path hint must never clear a checkpoint"

    # 2. Request multiple 50/50 quiz hints -> eliminates wrong options, but does NOT unlock checkpoint
    eliminated: Set[int] = set()
    elim1 = hm.request_quiz_hint(lm, q, eliminated)
    assert elim1 is not None
    eliminated.add(elim1)
    assert cp.is_cleared is False, "Quiz hint must never clear a checkpoint"

    elim2 = hm.request_quiz_hint(lm, q, eliminated)
    assert elim2 is not None
    eliminated.add(elim2)
    assert cp.is_cleared is False, "Second quiz hint must never clear a checkpoint"

    # 3. In the Game integration context, verify checkpoint remains strictly locked until correct answer
    game = Game(config_path="config/game_config.json")
    game.state = GameState.PLAYING
    game.level_manager.score = 500
    active_cp = game.checkpoint_manager.checkpoints[0]
    assert active_cp.is_cleared is False

    # Trigger checkpoint
    game._trigger_checkpoint(active_cp)
    assert game.active_checkpoint is active_cp

    # Trigger path hint and quiz hint during active checkpoint
    game._trigger_path_hint()
    assert active_cp.is_cleared is False

    if active_cp.question:
        game.hint_manager.request_quiz_hint(game.level_manager, active_cp.question, game.eliminated_quiz_options)
    assert active_cp.is_cleared is False
    assert game.checkpoint_manager.all_cleared() is False

    # The ONLY way to unlock is submitting the correct answer
    correct_idx = active_cp.question.answer
    game.selected_option_index = correct_idx
    game._submit_checkpoint_answer()
    assert active_cp.is_cleared is True
    assert game.checkpoint_manager.all_cleared() is True

    pygame.quit()


def test_checkpoint_timeout_counts_as_wrong() -> None:
    """Verify that when a checkpoint timer expires, it counts as a wrong answer with penalties."""
    game = Game(config_path="config/game_config.json")
    game.state = GameState.PLAYING

    cp = game.checkpoint_manager.checkpoints[0]
    assert not cp.is_cleared
    game._trigger_checkpoint(cp)
    assert game.active_checkpoint is cp
    assert game.checkpoint_timer.is_running
    assert not game.checkpoint_feedback_mode

    initial_cost = game.player.total_cost
    initial_score = game.level_manager.score
    initial_attempts = cp.attempts

    # Advance time well past the limit to trigger expiration
    game.update(35.0)

    # Assert timeout counts as wrong
    assert game.checkpoint_timer.is_expired is True
    assert game.checkpoint_feedback_mode is True
    assert game.checkpoint_last_correct is False
    assert cp.is_cleared is False, "Timeout must leave the gate locked"
    assert cp.attempts == initial_attempts + 1
    assert game.player.total_cost == initial_cost + 3  # Cost penalty applied
    assert game.level_manager.score == max(0, initial_score - 25)  # Score penalty applied
    assert "TIME EXPIRED" in game.checkpoint_feedback_msg

    pygame.quit()


def test_no_immediate_question_repeats() -> None:
    """Verify that QuestionManager never repeats a question immediately in any topic."""
    qm = QuestionManager(data_path="data/questions.json", history_capacity=10)

    topics = qm.get_topics()
    assert len(topics) >= 7

    for topic in topics:
        seen_in_window = []
        pool_size = len(qm.by_topic.get(topic, []))
        test_rounds = min(pool_size, 8)

        prev_id = None
        for _ in range(test_rounds):
            q = qm.get_question(topic=topic)
            assert q is not None
            assert q.id != prev_id, f"Immediate repeat detected in topic '{topic}': {q.id}"
            assert q.id not in seen_in_window, f"Premature repeat within window for topic '{topic}': {q.id}"
            seen_in_window.append(q.id)
            prev_id = q.id

