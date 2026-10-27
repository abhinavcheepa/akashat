"""Main Game coordinator and state machine for PATHFINDER.

Orchestrates game loop, input processing, screen rendering for Menu, Topic Selection,
Gameplay, Level Complete, Settings, Learn the Algorithm, How to Play, and Study Material RAG.
Integrates cut-vertex checkpoints, question banks, reward scoring, hints,
JSON save persistence, and optional document ingestion with keyword/semantic search.
"""

from enum import Enum
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pygame

from src.checkpoint import Checkpoint, CheckpointManager
from src.dijkstra_solver import DijkstraSolver
from src.grid import CellType, Grid
from src.hint_system import HintManager
from src.level_manager import LevelManager
from src.maze_generator import MazeGenerator
from src.path_visualizer import PathVisualizer
from src.player import Player
from src.question import Question, QuestionManager
from src.rag.pipeline import RAGPipeline
from src.reward_manager import RewardManager
from src.save_manager import SaveManager
from src.sound_manager import SoundManager
from src.timer import Timer
from src.ui_manager import UIManager


class GameState(Enum):
    """Finite state machine states for navigation console."""
    MENU = "MENU"
    PLAYING = "PLAYING"
    TOPIC_SELECTION = "TOPIC_SELECTION"
    LEVEL_COMPLETE = "LEVEL_COMPLETE"
    SETTINGS = "SETTINGS"
    LEARN_ALGORITHM = "LEARN_ALGORITHM"
    HOW_TO_PLAY = "HOW_TO_PLAY"
    STUDY_MATERIAL = "STUDY_MATERIAL"


class Game:
    """Core game application class holding state, assets, and main loop."""

    def __init__(self, config_path: str = "config/game_config.json") -> None:
        """Initialize Pygame, load configuration, and bootstrap subsystems."""
        pygame.init()

        self.config = self._load_config(config_path)
        palette_dict = self.config.get("palette", self.config.get("colors", {}))
        self.colors = self._parse_colors(palette_dict)
        self.bg_surface = self._create_paper_grain_surface()

        win_cfg = self.config.get("window", {})
        self.width: int = win_cfg.get("width", 1120)
        self.height: int = win_cfg.get("height", 720)
        self.fps: int = win_cfg.get("fps", 60)
        title: str = win_cfg.get("title", "PATHFINDER - Think. Navigate. Conquer.")

        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        # Core Subsystems
        self.ui = UIManager(self.colors)
        self.level_manager = LevelManager(start_level=1, config=self.config)
        self.maze_generator = MazeGenerator()
        self.visualizer = PathVisualizer()
        self.question_manager = QuestionManager(data_path="data/questions.json")
        self.checkpoint_manager = CheckpointManager(self.question_manager)
        self.checkpoint_timer = Timer(20.0)
        self.reward_manager = RewardManager(config=self.config)
        self.hint_manager = HintManager(config=self.config)
        self.save_manager = SaveManager(filepath="data/save_game.json")
        self.sound_manager = SoundManager(config_settings=self.config.get("settings", {}))

        # RAG Subsystem
        rag_cfg = self.config.get("rag", {})
        self.rag_pipeline = RAGPipeline(
            chunk_size=rag_cfg.get("chunk_size", 500),
            chunk_overlap=rag_cfg.get("chunk_overlap", 100),
            max_file_size_mb=rag_cfg.get("max_file_size_mb", 5.0),
            force_keyword=rag_cfg.get("force_keyword_search", False),
        )
        self.typed_file_path: str = "data/sample_study_material.md"
        self.input_active: bool = False

        # Game State
        self.state: GameState = GameState.MENU
        self.is_running: bool = True
        self.status_message: str = ""
        self.status_timer: float = 0.0

        # Topic & Difficulty Preferences
        saved_topic = self.save_manager.data.get("selected_topic", "All Topics")
        saved_diff = self.save_manager.data.get("selected_difficulty", "Level Default")
        self.selected_topic: str = saved_topic
        self.selected_difficulty: str = saved_diff

        # Checkpoint Overlay State
        self.active_checkpoint: Optional[Checkpoint] = None
        self.selected_option_index: Optional[int] = None
        self.checkpoint_feedback_mode: bool = False
        self.checkpoint_last_correct: bool = False
        self.checkpoint_feedback_msg: str = ""
        self.eliminated_quiz_options: Set[int] = set()

        # Current level runtime entities
        self.grid: Optional[Grid] = None
        self.start_pos: Tuple[int, int] = (1, 1)
        self.end_pos: Tuple[int, int] = (13, 13)
        self.player: Optional[Player] = None
        self.optimal_path: List[Tuple[int, int]] = []
        self.optimal_cost: Union[int, float] = 0
        self.level_completed: bool = False
        self.pulse_phase: float = 0.0
        self.last_rewards_summary: Dict[str, Any] = {}

        # Screen transition state
        self.transition_alpha: int = 0
        self.transition_target: int = 0
        self.transition_speed: int = 600
        self.pending_state: Optional[GameState] = None

        # Cached maze surface for high-performance static rendering
        self.maze_surface: Optional[pygame.Surface] = None
        self.maze_offset_x: int = 0
        self.maze_offset_y: int = 0
        self.tile_size: int = 36

        # Learn the Algorithm state
        self.learn_step: int = 0
        self.learn_auto_play: bool = False
        self.learn_timer: float = 0.0
        self._init_learn_scenario()

        # Initialize Level 1
        self._generate_level()

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Safely load JSON configuration with robust fallback to defaults."""
        defaults: Dict[str, Any] = {
            "window": {"width": 1120, "height": 720, "fps": 60, "title": "PATHFINDER"},
            "palette": {},
            "colors": {},
            "checkpoints": {
                "time_limits": {"Easy": 25.0, "Medium": 20.0, "Hard": 15.0},
                "cost_penalty_on_failure": 3,
                "score_penalty_on_failure": 25,
                "score_reward_on_success": 100,
            },
            "rewards": {
                "correct_answer_points": 100,
                "efficiency_100_bonus": 150,
                "efficiency_90_bonus": 100,
                "efficiency_75_bonus": 50,
                "checkpoint_all_cleared_bonus": 200,
                "no_hints_used_bonus": 100,
                "base_level_completion": 100,
            },
            "hints": {
                "path_hint_cost": 40,
                "quiz_50_50_cost": 30,
            },
            "settings": {
                "sound_enabled": True,
                "volume": 75,
                "timer_modifier": 0.0,
                "selected_topic": "All Topics",
                "selected_difficulty": "Level Default",
            },
            "levels": {
                "1": {"rooms": 7, "checkpoints": 1, "difficulty": "Easy", "name": "Sector Alpha - Fundamentals"},
                "2": {"rooms": 9, "checkpoints": 2, "difficulty": "Easy", "name": "Sector Beta - Routing Gates"},
                "3": {"rooms": 11, "checkpoints": 2, "difficulty": "Medium", "name": "Sector Gamma - Dynamic Topology"},
                "4": {"rooms": 13, "checkpoints": 3, "difficulty": "Medium", "name": "Sector Delta - Strategic Labyrinth"},
                "5": {"rooms": 15, "checkpoints": 4, "difficulty": "Hard", "name": "Sector Omega - Master Dijkstra"},
            },
            "rag": {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "max_file_size_mb": 5.0,
                "force_keyword_search": False,
            },
        }

