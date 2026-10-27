"""Headless Smoke Test for PATHFINDER.

Workflow:
1. Ingest a sample markdown document (data/sample_study_material.md).
2. Segment text into chunks and index with pure-Python BM25 keyword retriever.
3. Generate grounded fill-in-the-blank (cloze) questions with source references.
4. Inject generated questions into QuestionManager under topic 'Custom Materials'.
5. Launch Game in headless mode (dummy video/audio drivers).
6. Start Level 1 with topic set to 'Custom Materials'.
7. Verify checkpoint gate receives custom material question.
8. Simulate player movement, checkpoint trigger, correct answer submission, and level completion.
"""

import os
import sys

# Ensure headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
from src.game import Game, GameState
from src.rag.pipeline import RAGPipeline


def run_smoke_test() -> bool:
    print("=" * 70)
    print("PATHFINDER HEADLESS SMOKE TEST: RAG INGESTION & GAMEPLAY PIPELINE")
    print("=" * 70)

    # 1. Ingest sample Markdown document
    sample_path = "data/sample_study_material.md"
    print(f"\n[Step 1] Ingesting document: {sample_path}")
    assert os.path.exists(sample_path), f"Sample file {sample_path} not found!"

    # Use pure-Python keyword search for fast, deterministic, offline execution
    pipeline = RAGPipeline(force_keyword=True)
    success, msg, questions = pipeline.process_file(sample_path, target_question_count=4)

    print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"  Status: {msg}")
    print(f"  Total Chunks: {len(pipeline.chunks)}")
    print(f"  Active Retrieval Mode: {pipeline.active_retrieval_mode}")
    print(f"  Questions Generated: {len(questions)}")

    if not success or not questions:
        print("[FAIL] Question generation failed.")
        return False

    # 2. Display Generated Questions
    print("\n[Step 2] Inspecting Generated Questions & Grounding References:")
    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q.question}")
        print(f"      Options: {q.options}")
        print(f"      Answer: [{q.answer}] -> {q.options[q.answer]}")
        print(f"      Explanation: {q.explanation}")

    # 3. Initialize Headless Game Instance
    print("\n[Step 3] Initializing Headless Game Instance...")
    game = Game(config_path="config/game_config.json")
    print(f"  Game Initialized: Grid {game.grid.width}x{game.grid.height}, State={game.state.name}")

    # 4. Integrate Custom Materials
    print("\n[Step 4] Injecting Custom Questions into QuestionManager...")
    game.question_manager.add_custom_questions(questions)
    assert game.question_manager.has_custom_materials() is True
    assert "Custom Materials" in game.question_manager.get_topics()
    print("  'Custom Materials' topic registered in QuestionManager.")

    # Select Custom Materials topic
    game.selected_topic = "Custom Materials"
    game.selected_difficulty = "Level Default"
    print(f"  Mission configured with topic: '{game.selected_topic}'")

    # 5. Start Level 1
    print("\n[Step 5] Launching Level 1 Mission...")
    game.level_manager.current_level = 1
    game._generate_level()
    game.state = GameState.PLAYING
    print(f"  Level 1 generated: Start={game.start_pos}, Destination={game.end_pos}")
    print(f"  Optimal Dijkstra Route Cost: {game.optimal_cost} (Length: {len(game.optimal_path)} steps)")

    checkpoints = game.checkpoint_manager.checkpoints
    print(f"  Checkpoints placed: {len(checkpoints)}")
    assert len(checkpoints) >= 1, "Level 1 must have at least 1 checkpoint gate"

    cp = checkpoints[0]
    print(f"  Gate #1 Position: {cp.pos}, Topic='{cp.question.topic}', Difficulty='{cp.question.difficulty}'")
    print(f"  Gate #1 Question Stem: {cp.question.question}")
    assert cp.question.topic == "Custom Materials", "Checkpoint question must be from Custom Materials"

    # 6. Simulate Player Navigation & Checkpoint Interaction
    print("\n[Step 6] Simulating Navigation to Gate & Answering Question...")
    # Move player directly onto the checkpoint gate tile
    game.player.reset(cp.pos)
    game._trigger_checkpoint(cp)

    assert game.active_checkpoint is cp
    assert cp.is_cleared is False
    print("  Gate triggered! Security modal overlay active.")

    # Submit correct answer
    correct_idx = cp.question.answer
    game.selected_option_index = correct_idx
    game._submit_checkpoint_answer()

    assert game.checkpoint_feedback_mode is True
    assert game.checkpoint_last_correct is True
    assert cp.is_cleared is True
    print(f"  Correct Answer Submitted! Feedback: {game.checkpoint_feedback_msg}")
    print(f"  Gate status: {'UNLOCKED' if cp.is_cleared else 'LOCKED'}")

    # 7. Complete Mission
    print("\n[Step 7] Navigating to Destination & Completing Level...")
    game.active_checkpoint = None
    game.player.reset(game.end_pos)

    if game.checkpoint_manager.all_cleared():
        game._complete_level()

    assert game.level_completed is True
    print(f"  Level Complete: {game.level_completed}")
    print(f"  Mission Summary: Score={game.level_manager.score}, Player Cost={game.player.total_cost}")
    print(f"  Efficiency: {game.player.calculate_efficiency(game.optimal_cost):.1f}%")

    pygame.quit()
    print("\n" + "=" * 70)
    print("HEADLESS SMOKE TEST RESULT: ALL PIPELINE STAGES PASSED SUCCESSFULLY")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
