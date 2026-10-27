"""Phase D tests: SoundManager procedural audio, screen transitions, and final integration."""

import os
import math
import array

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import pytest


def test_sound_manager_synthesis_and_play():
    """Verify procedural sound synthesis creates valid Sound objects and play() doesn't crash."""
    pygame.init()
    from src.sound_manager import SoundManager

    settings = {"sound_enabled": True, "volume": 75}
    sm = SoundManager(config_settings=settings)

    # Even in headless mode, the Sound objects should be created if mixer initialized
    if sm.is_available:
        expected_sounds = {"step", "correct", "wrong", "checkpoint", "hint", "victory", "click"}
        assert expected_sounds.issubset(set(sm.sounds.keys())), (
            f"Missing sounds: {expected_sounds - set(sm.sounds.keys())}"
        )

        # All Sound objects should be valid pygame.mixer.Sound instances
        for name, sound in sm.sounds.items():
            assert isinstance(sound, pygame.mixer.Sound), f"{name} is not a Sound object"

        # play() should not raise even in dummy mode
        for name in expected_sounds:
            sm.play(name)

    # With sound_enabled=False, play() should be a no-op (no crash)
    settings["sound_enabled"] = False
    sm2 = SoundManager(config_settings=settings)
    sm2.play("step")
    sm2.play("nonexistent_sound")

    # With volume=0, play() should be a no-op
    settings["sound_enabled"] = True
    settings["volume"] = 0
    sm3 = SoundManager(config_settings=settings)
    sm3.play("correct")


def test_sound_manager_graceful_degradation():
    """SoundManager must not crash when mixer is unavailable."""
    from src.sound_manager import SoundManager

    sm = SoundManager(config_settings=None)
    # play() should be safe regardless
    sm.play("step")
    sm.play("correct")
    sm.play("wrong")
    sm.play("nonexistent")


def test_screen_transition_state_machine():
    """Verify transition overlay alpha ramp-up, state switch at peak, and ramp-down."""
    from src.game import Game, GameState

    g = Game()

    # Initially no transition active
    assert g.transition_alpha == 0
    assert g.transition_target == 0
    assert g.pending_state is None

    # Start a transition to SETTINGS
    g._start_transition(GameState.SETTINGS)
    assert g.pending_state == GameState.SETTINGS
    assert g.transition_target == 255

    # Simulate updates until alpha reaches 255 (fade out)
    for _ in range(100):
        g._update_transition(0.016)
        if g.transition_alpha >= 255:
            break

    # At peak alpha, state should have switched and target should be 0 (fade in)
    assert g.state == GameState.SETTINGS
    assert g.pending_state is None
    assert g.transition_target == 0

    # Continue updates to fade back in
    for _ in range(100):
        g._update_transition(0.016)
        if g.transition_alpha == 0:
            break

    assert g.transition_alpha == 0


def test_transition_blocks_double_trigger():
    """Cannot start a new transition while one is already pending."""
    from src.game import Game, GameState

    g = Game()
    g._start_transition(GameState.SETTINGS)
    original_pending = g.pending_state

    # Second transition attempt should be ignored
    g._start_transition(GameState.MENU)
    assert g.pending_state == original_pending  # Still SETTINGS


def test_game_launch_with_sound_integration():
    """Headless smoke test: game initializes with SoundManager, generates level, renders frame."""
    from src.game import Game, GameState

    g = Game()

    # SoundManager should be initialized
    assert hasattr(g, "sound_manager")

    # Generate level and render
    g._generate_level()
    g.state = GameState.PLAYING
    g.update(0.016)
    g.render()

    # Transition overlay should render without crash
    g._render_transition_overlay()

    # Sound calls during gameplay should not crash
    g.sound_manager.play("step")
    g.sound_manager.play("checkpoint")
    g.sound_manager.play("victory")


def test_readme_completeness():
    """Verify README.md exists, covers key sections, and has required content."""
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    assert os.path.exists(readme_path), "README.md must exist at project root"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Key sections must be present
    required_sections = [
        "Dijkstra's Algorithm",
        "O((V + E) log V)",
        "RAG Document Ingestion",
        "Architecture Diagram",
        "Installation",
        "Controls",
        "Project Structure",
        "Cut-vertex",
        "BM25",
        "classDiagram",
    ]

    for section in required_sections:
        assert section in content, f"README.md must contain '{section}'"

    # Must document all source files
    source_files = [
        "game.py", "grid.py", "dijkstra_solver.py", "maze_generator.py",
        "player.py", "ui_manager.py", "level_manager.py", "checkpoint.py",
        "question.py", "reward_manager.py", "hint_system.py", "save_manager.py",
        "sound_manager.py", "document_ingestor.py", "retriever.py",
        "question_generator.py", "pipeline.py",
    ]

    for fname in source_files:
        assert fname in content, f"README.md must document '{fname}'"
