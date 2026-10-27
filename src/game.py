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
        self.colors = self._parse_colors(self.config.get("colors", {}))

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
            "colors": {
                "background": "#0B1020",
                "panel": "#141C2F",
                "panel_border": "#222E4A",
                "primary": "#38D9FF",
                "secondary": "#8B7CFF",
                "success": "#35D39A",
                "warning": "#FFBF69",
                "danger": "#FF647C",
                "text": "#F3F6FC",
                "text_dim": "#8B9AB4",
                "wall": "#162035",
                "wall_border": "#222F4C",
                "cell_open": "#0E182A",
                "cell_sand": "#332A15",
                "cell_forest": "#123023",
                "cell_swamp": "#3A1B2D",
                "explored": "#1A3654",
                "trail": "#1F3D63",
                "player": "#38D9FF",
                "destination": "#35D39A",
                "checkpoint_locked": "#FFBF69",
                "checkpoint_cleared": "#35D39A",
                "hint_glow": "#FFD166",
            },
            "checkpoints": {
                "time_limits": {"Easy": 25.0, "Medium": 20.0, "Hard": 15.0},
                "cost_penalty_on_failure": 3,
                "score_penalty_on_failure": 25,
                "score_reward_on_success": 100,
                "allow_retries": True,
                "max_retries": 3,
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
            "rag": {
                "max_file_size_mb": 5.0,
                "chunk_size": 500,
                "chunk_overlap": 100,
                "force_keyword_search": False,
                "use_custom_materials": True,
            },
            "levels": {
                "1": {"rooms": 7, "checkpoints": 1, "difficulty": "Easy", "name": "Sector Alpha - Fundamentals"},
                "2": {"rooms": 9, "checkpoints": 2, "difficulty": "Easy", "name": "Sector Beta - Routing Gates"},
                "3": {"rooms": 11, "checkpoints": 2, "difficulty": "Medium", "name": "Sector Gamma - Dynamic Topology"},
                "4": {"rooms": 13, "checkpoints": 3, "difficulty": "Medium", "name": "Sector Delta - Strategic Labyrinth"},
                "5": {"rooms": 15, "checkpoints": 4, "difficulty": "Hard", "name": "Sector Omega - Master Dijkstra"},
            },
        }

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    for k, v in loaded.items():
                        if isinstance(v, dict) and k in defaults:
                            defaults[k].update(v)
                        else:
                            defaults[k] = v
            except Exception as e:
                print(f"[Warning] Failed loading config file ({e}). Utilizing defaults.")
        return defaults

    def _parse_colors(self, color_dict: Dict[str, str]) -> Dict[str, pygame.Color]:
        """Convert hexadecimal color strings into Pygame Color instances."""
        parsed: Dict[str, pygame.Color] = {}
        for name, hex_val in color_dict.items():
            try:
                parsed[name] = pygame.Color(hex_val)
            except Exception:
                parsed[name] = pygame.Color("#FFFFFF")
        return parsed

    def _set_status(self, msg: str, duration: float = 3.0) -> None:
        """Display an informational status banner for duration seconds."""
        self.status_message = msg
        self.status_timer = duration

    def _init_learn_scenario(self) -> None:
        """Initialize fixed demonstration scenario for 'Learn the Algorithm' screen."""
        self.learn_grid = Grid(9, 7)
        self.learn_grid.reset_all_to_walls()

        for x in range(1, 8):
            self.learn_grid.get_cell(x, 1).set_type(CellType.SWAMP, weight=5)
            self.learn_grid.get_cell(x, 5).set_type(CellType.OPEN, weight=1)

        for y in range(1, 6):
            self.learn_grid.get_cell(1, y).set_type(CellType.OPEN, weight=1)
            self.learn_grid.get_cell(7, y).set_type(CellType.OPEN, weight=1)

        self.learn_start = (1, 3)
        self.learn_end = (7, 3)
        self.learn_path, self.learn_cost, self.learn_explored = DijkstraSolver.solve(
            self.learn_grid, self.learn_start, self.learn_end
        )
        self.learn_step = 0
        self.learn_auto_play = False
        self.learn_timer = 0.0

    def _generate_level(self) -> None:
        """Generate procedural maze, cut-vertex checkpoints, and Dijkstra solutions."""
        rooms = self.level_manager.get_rooms()
        cp_count = self.level_manager.get_checkpoints_count()
        difficulty = (
            self.selected_difficulty
            if self.selected_difficulty != "Level Default"
            else self.level_manager.get_difficulty()
        )

        # Topic fallback safeguard
        effective_topic = self.selected_topic
        if effective_topic == "Custom Materials" and not self.question_manager.has_custom_materials():
            self._set_status("Custom Materials empty. Falling back to offline question bank.")
            effective_topic = "All Topics"

        self.grid, self.start_pos, self.end_pos = self.maze_generator.generate(
            rooms_x=rooms,
            rooms_y=rooms,
            loop_removals=8,
            max_attempts=50,
        )

        path, cost, explored = DijkstraSolver.solve(self.grid, self.start_pos, self.end_pos)
        self.optimal_path = path
        self.optimal_cost = cost

        self.checkpoint_manager.generate_checkpoints(
            grid=self.grid,
            start_pos=self.start_pos,
            end_pos=self.end_pos,
            optimal_path=self.optimal_path,
            count=cp_count,
            difficulty=difficulty,
            topic=effective_topic,
        )

        re_path, re_cost, re_explored = DijkstraSolver.solve(self.grid, self.start_pos, self.end_pos)
        self.optimal_path = re_path
        self.optimal_cost = re_cost

        alt_result = DijkstraSolver.find_alternative(self.grid, self.start_pos, self.end_pos, self.optimal_path)
        alt_path, alt_cost = alt_result if alt_result is not None else (None, None)

        self.visualizer.set_solution(
            optimal_path=self.optimal_path,
            optimal_cost=self.optimal_cost,
            explored_order=re_explored,
            alternative_path=alt_path,
            alternative_cost=alt_cost,
        )

        self._invalidate_maze_cache()
        self.restart_level()

    def restart_level(self) -> None:
        """Reset player state, breadcrumbs, checkpoints, hints, and visualizer for current sector."""
        if self.player is None:
            self.player = Player(self.start_pos)
        else:
            self.player.reset(self.start_pos)

        for cp in self.checkpoint_manager.checkpoints:
            cp.reset()

        self.active_checkpoint = None
        self.selected_option_index = None
        self.checkpoint_feedback_mode = False
        self.eliminated_quiz_options.clear()
        self.checkpoint_timer.stop()

        self.hint_manager.reset_level()
        self.visualizer.reset_state()
        self.level_completed = False

    def _invalidate_maze_cache(self) -> None:
        """Mark static maze surface dirty for re-baking."""
        self.maze_surface = None

    # -------------------------------------------------------------------------
    # Coordinate Transforms
    # -------------------------------------------------------------------------

    def grid_to_rect(self, tx: int, ty: int) -> pygame.Rect:
        return pygame.Rect(
            self.maze_offset_x + tx * self.tile_size,
            self.maze_offset_y + ty * self.tile_size,
            self.tile_size,
            self.tile_size,
        )

    def grid_to_center(self, tx: int, ty: int) -> Tuple[int, int]:
        return (
            self.maze_offset_x + tx * self.tile_size + self.tile_size // 2,
            self.maze_offset_y + ty * self.tile_size + self.tile_size // 2,
        )

    # -------------------------------------------------------------------------
    # Layout Rect Providers
    # -------------------------------------------------------------------------

    def get_side_panel_rect(self) -> pygame.Rect:
        panel_w = min(380, max(300, int(self.width * 0.34)))
        return pygame.Rect(self.width - panel_w - 18, 68, panel_w, self.height - 86)

    def get_maze_viewport_rect(self) -> pygame.Rect:
        side_rect = self.get_side_panel_rect()
        return pygame.Rect(18, 68, side_rect.x - 36, self.height - 86)

    def get_action_buttons(self) -> Dict[str, pygame.Rect]:
        panel = self.get_side_panel_rect()
        btn_w = (panel.width - 48) // 2
        return {
            "show_path": pygame.Rect(panel.x + 18, panel.y + 242, btn_w, 38),
            "hint_path": pygame.Rect(panel.x + 24 + btn_w, panel.y + 242, btn_w, 38),
            "new_maze": pygame.Rect(panel.x + 18, panel.y + 286, btn_w, 38),
            "restart": pygame.Rect(panel.x + 24 + btn_w, panel.y + 286, btn_w, 38),
            "menu": pygame.Rect(panel.x + 18, panel.bottom - 48, panel.width - 36, 38),
        }

    def get_checkpoint_modal_rect(self) -> pygame.Rect:
        modal_w = min(740, self.width - 40)
        modal_h = min(580, self.height - 60)
        return pygame.Rect(self.width // 2 - modal_w // 2, self.height // 2 - modal_h // 2, modal_w, modal_h)

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Execute the main application loop."""
        while self.is_running:
            dt = self.clock.tick(self.fps) / 1000.0
            dt = min(dt, 0.1)

            self.pulse_phase = (self.pulse_phase + dt * 3.0) % (2.0 * math.pi)

            if self.status_timer > 0:
                self.status_timer = max(0.0, self.status_timer - dt)

            # Update transition overlay
            self._update_transition(dt)

            self.handle_events()
            self.update(dt)
            self.render()

        pygame.quit()

    def handle_events(self) -> None:
        """Dispatch window and keyboard inputs based on current state."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            if event.type == pygame.VIDEORESIZE:
                self.width = max(800, event.w)
                self.height = max(600, event.h)
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                self._invalidate_maze_cache()
                continue

            if self.state == GameState.MENU:
                self._handle_menu_event(event)
            elif self.state == GameState.PLAYING:
                if self.active_checkpoint is not None:
                    self._handle_checkpoint_overlay_event(event)
                else:
                    self._handle_playing_event(event)
            elif self.state == GameState.LEVEL_COMPLETE:
                self._handle_level_complete_event(event)
            elif self.state == GameState.TOPIC_SELECTION:
                self._handle_topic_selection_event(event)
            elif self.state == GameState.STUDY_MATERIAL:
                self._handle_study_material_event(event)
            elif self.state == GameState.SETTINGS:
                self._handle_settings_event(event)
            elif self.state == GameState.LEARN_ALGORITHM:
                self._handle_learn_event(event)
            elif self.state == GameState.HOW_TO_PLAY:
                self._handle_how_to_play_event(event)

    # -------------------------------------------------------------------------
    # Screen Event Handlers
    # -------------------------------------------------------------------------

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
        """Handle interactions on Main Menu screen."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            has_save = self.save_manager.has_saved_game()

            # Continue Mission
            if has_save and pygame.Rect(self.width // 2 - 140, 220, 280, 42).collidepoint(mouse_pos):
                self.level_manager.current_level = self.save_manager.data.get("current_level", 1)
                self.level_manager.score = self.save_manager.data.get("total_score", 0)
                self._generate_level()
                self.state = GameState.PLAYING

            # Play / Select Mission
            elif pygame.Rect(self.width // 2 - 140, 270, 280, 42).collidepoint(mouse_pos):
                self.state = GameState.TOPIC_SELECTION

            # Upload Study Material / RAG
            elif pygame.Rect(self.width // 2 - 140, 320, 280, 42).collidepoint(mouse_pos):
                self.state = GameState.STUDY_MATERIAL

            # Learn Dijkstra
            elif pygame.Rect(self.width // 2 - 140, 370, 280, 42).collidepoint(mouse_pos):
                self._init_learn_scenario()
                self.state = GameState.LEARN_ALGORITHM

            # How to Play
            elif pygame.Rect(self.width // 2 - 140, 420, 280, 42).collidepoint(mouse_pos):
                self.state = GameState.HOW_TO_PLAY

            # Settings
            elif pygame.Rect(self.width // 2 - 140, 470, 280, 42).collidepoint(mouse_pos):
                self.state = GameState.SETTINGS

            # Quit
            elif pygame.Rect(self.width // 2 - 140, 520, 280, 42).collidepoint(mouse_pos):
                self.is_running = False

        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.state = GameState.TOPIC_SELECTION
            elif event.key == pygame.K_c and self.save_manager.has_saved_game():
                self.level_manager.current_level = self.save_manager.data.get("current_level", 1)
                self.level_manager.score = self.save_manager.data.get("total_score", 0)
                self._generate_level()
                self.state = GameState.PLAYING
            elif event.key == pygame.K_ESCAPE:
                self.is_running = False

    def _handle_study_material_event(self, event: pygame.event.Event) -> None:
        """Handle document ingestion, file picking, and question generation."""
        modal_w = min(840, self.width - 40)
        modal_h = min(600, self.height - 60)
        panel = pygame.Rect(self.width // 2 - modal_w // 2, self.height // 2 - modal_h // 2, modal_w, modal_h)

        input_box = pygame.Rect(panel.x + 50, panel.y + 130, panel.width - 240, 40)
        browse_btn = pygame.Rect(panel.right - 180, panel.y + 130, 130, 40)
        sample_btn = pygame.Rect(panel.x + 50, panel.y + 180, 200, 36)
        process_btn = pygame.Rect(panel.x + 260, panel.y + 180, 200, 36)
        toggle_btn = pygame.Rect(panel.right - 220, panel.y + 180, 170, 36)
        back_btn = pygame.Rect(self.width // 2 - 120, panel.bottom - 50, 240, 40)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            # Click inside input box
            if input_box.collidepoint(mouse_pos):
                self.input_active = True
            else:
                self.input_active = False

            # Click Browse Button (Tkinter with safe fallback)
            if browse_btn.collidepoint(mouse_pos):
                picked_file = self._open_file_dialog()
                if picked_file:
                    self.typed_file_path = picked_file

            # Click Load Sample Notes
            elif sample_btn.collidepoint(mouse_pos):
                self.typed_file_path = "data/sample_study_material.md"
                self._set_status("Sample study material path loaded.")

            # Click Process Document
            elif process_btn.collidepoint(mouse_pos):
                self._process_uploaded_document()

            # Click Toggle In-Game
            elif toggle_btn.collidepoint(mouse_pos):
                self.rag_pipeline.use_in_game = not self.rag_pipeline.use_in_game
                if self.rag_pipeline.use_in_game and self.rag_pipeline.generated_questions:
                    self.question_manager.add_custom_questions(self.rag_pipeline.generated_questions)
                    self._set_status("Custom material active in-game!")
                else:
                    self.question_manager.clear_custom_questions()
                    self._set_status("Custom material disabled for this session.")

            # Click Back to Menu
            elif back_btn.collidepoint(mouse_pos):
                self.state = GameState.MENU

        elif event.type == pygame.KEYDOWN:
            if self.input_active:
                if event.key == pygame.K_RETURN:
                    self._process_uploaded_document()
                elif event.key == pygame.K_BACKSPACE:
                    self.typed_file_path = self.typed_file_path[:-1]
                elif event.unicode and event.unicode.isprintable():
                    self.typed_file_path += event.unicode
            elif event.key == pygame.K_ESCAPE:
                self.state = GameState.MENU

    def _open_file_dialog(self) -> Optional[str]:
        """Safely attempt to open OS file dialog, falling back gracefully without crash."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            file_path = filedialog.askopenfilename(
                title="Select Study Material Document",
                filetypes=[("Documents", "*.txt *.md *.pdf"), ("All Files", "*.*")],
            )
            root.destroy()
            return file_path if file_path else None
        except Exception:
            self._set_status("GUI file picker unavailable. Type or paste file path directly.")
            return None

    def _process_uploaded_document(self) -> None:
        """Run document through RAG pipeline, ingest, chunk, and generate questions."""
        target_path = self.typed_file_path.strip()
        if not target_path:
            self._set_status("Please provide a valid document path.")
            return

        success, msg, questions = self.rag_pipeline.process_file(target_path)
        self._set_status(msg, duration=4.0)

        if success and questions:
            if self.rag_pipeline.use_in_game:
                self.question_manager.add_custom_questions(questions)
                self.selected_topic = "Custom Materials"

    def _handle_topic_selection_event(self, event: pygame.event.Event) -> None:
        """Handle topic and difficulty configuration selection."""
        topics = ["All Topics"] + self.question_manager.get_topics()
        difficulties = ["Level Default", "Easy", "Medium", "Hard"]

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            for i, topic in enumerate(topics):
                r = pygame.Rect(self.width // 2 - 280, 160 + i * 40, 260, 34)
                if r.collidepoint(mouse_pos):
                    self.selected_topic = topic
                    self.save_manager.data["selected_topic"] = topic
                    self.save_manager.save()
                    return

            for i, diff in enumerate(difficulties):
                r = pygame.Rect(self.width // 2 + 20, 160 + i * 40, 260, 34)
                if r.collidepoint(mouse_pos):
                    self.selected_difficulty = diff
                    self.save_manager.data["selected_difficulty"] = diff
                    self.save_manager.save()
                    return

            launch_rect = pygame.Rect(self.width // 2 - 140, self.height - 110, 280, 46)
            if launch_rect.collidepoint(mouse_pos):
                self._generate_level()
                self.state = GameState.PLAYING
                return

            back_rect = pygame.Rect(self.width // 2 - 140, self.height - 54, 280, 34)
            if back_rect.collidepoint(mouse_pos):
                self.state = GameState.MENU
                return

        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._generate_level()
                self.state = GameState.PLAYING
            elif event.key == pygame.K_ESCAPE:
                self.state = GameState.MENU

    def _handle_playing_event(self, event: pygame.event.Event) -> None:
        """Handle movement, visualizer, hints, and action buttons during gameplay."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.state = GameState.MENU
                return

            if event.key in (pygame.K_r,):
                self._generate_level()
                return

            if event.key in (pygame.K_p,):
                self.visualizer.toggle_path()
                return

            if event.key in (pygame.K_h,):
                self._trigger_path_hint()
                return

            if self.player and self.grid and not self.level_completed:
                dx, dy = 0, 0
                if event.key in (pygame.K_UP, pygame.K_w):
                    dy = -1
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    dy = 1
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    dx = -1
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    dx = 1

                if dx != 0 or dy != 0:
                    moved = self.player.move(dx, dy, self.grid)
                    if moved:
                        self.sound_manager.play("step")
                        cp = self.checkpoint_manager.get_at_pos(self.player.pos)
                        if cp and not cp.is_cleared:
                            self._trigger_checkpoint(cp)

                        if self.player.pos == self.end_pos:
                            if self.checkpoint_manager.all_cleared():
                                self._complete_level()
                            else:
                                self._set_status("DESTINATION LOCKED! Clear all sector gates first.")

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            buttons = self.get_action_buttons()

            if buttons["show_path"].collidepoint(mouse_pos):
                self.visualizer.toggle_path()
            elif buttons["hint_path"].collidepoint(mouse_pos):
                self._trigger_path_hint()
            elif buttons["new_maze"].collidepoint(mouse_pos):
                self._generate_level()
            elif buttons["restart"].collidepoint(mouse_pos):
                self.restart_level()
            elif buttons["menu"].collidepoint(mouse_pos):
                self.state = GameState.MENU

    def _trigger_path_hint(self) -> None:
        cost = self.hint_manager.get_cost("path")
        if not self.hint_manager.can_afford(self.level_manager.score, "path"):
            self._set_status(f"INSUFFICIENT SCORE FOR HINT (Requires {cost} pts)")
            return

        next_step = self.hint_manager.request_path_hint(
            self.level_manager, self.player.pos, self.optimal_path
        )
        if next_step:
            self.sound_manager.play("hint")
            self._set_status(f"TACTICAL HINT ACTIVE (-{cost} pts): Next step highlighted!")

    def _complete_level(self) -> None:
        self.level_completed = True
        self.sound_manager.play("victory")
        eff = self.player.calculate_efficiency(self.optimal_cost) if self.player else 100.0
        cps_cleared = sum(1 for cp in self.checkpoint_manager.checkpoints if cp.is_cleared)
        total_cps = len(self.checkpoint_manager.checkpoints)

        rewards = self.reward_manager.calculate_level_rewards(
            player_cost=self.player.total_cost if self.player else 0,
            optimal_cost=self.optimal_cost,
            efficiency=eff,
            checkpoints_cleared=cps_cleared,
            total_checkpoints=total_cps,
            quiz_correct=cps_cleared,
            hints_used=self.hint_manager.hints_used_this_level,
        )
        self.last_rewards_summary = rewards

        self.level_manager.add_score(rewards["total_points"])
        self.save_manager.record_level_completion(
            level=self.level_manager.current_level,
            score=self.level_manager.score,
            efficiency=eff,
        )

        self.state = GameState.LEVEL_COMPLETE

    def _handle_level_complete_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            next_rect = pygame.Rect(self.width // 2 - 140, self.height - 160, 280, 46)
            if next_rect.collidepoint(mouse_pos):
                if self.level_manager.is_max_level:
                    self.state = GameState.MENU
                else:
                    self.level_manager.advance()
                    self._generate_level()
                    self.state = GameState.PLAYING
                return

            retry_rect = pygame.Rect(self.width // 2 - 140, self.height - 106, 280, 42)
            if retry_rect.collidepoint(mouse_pos):
                self._generate_level()
                self.state = GameState.PLAYING
                return

            menu_rect = pygame.Rect(self.width // 2 - 140, self.height - 56, 280, 38)
            if menu_rect.collidepoint(mouse_pos):
                self.state = GameState.MENU
                return

        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                if not self.level_manager.is_max_level:
                    self.level_manager.advance()
                    self._generate_level()
                    self.state = GameState.PLAYING
                else:
                    self.state = GameState.MENU
            elif event.key == pygame.K_ESCAPE:
                self.state = GameState.MENU

    def _trigger_checkpoint(self, cp: Checkpoint) -> None:
        self.active_checkpoint = cp
        self.selected_option_index = None
        self.checkpoint_feedback_mode = False
        self.checkpoint_last_correct = False
        self.checkpoint_feedback_msg = ""
        self.eliminated_quiz_options.clear()
        self.sound_manager.play("checkpoint")

        time_limits = self.config.get("checkpoints", {}).get("time_limits", {})
        limit = time_limits.get(cp.difficulty, 20.0)
        self.checkpoint_timer.start(limit)

    def _handle_checkpoint_overlay_event(self, event: pygame.event.Event) -> None:
        if not self.active_checkpoint or not self.active_checkpoint.question:
            return

        modal = self.get_checkpoint_modal_rect()
        q = self.active_checkpoint.question

        opt_rects: List[pygame.Rect] = []
        opt_y = modal.y + 195
        for i in range(len(q.options)):
            opt_rects.append(pygame.Rect(modal.x + 40, opt_y + i * 54, modal.width - 80, 46))

        action_btn_rect = pygame.Rect(modal.centerx - 130, modal.bottom - 56, 260, 42)
        hint_50_50_rect = pygame.Rect(modal.right - 180, modal.y + 60, 140, 30)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            if not self.checkpoint_feedback_mode:
                if hint_50_50_rect.collidepoint(mouse_pos):
                    cost = self.hint_manager.get_cost("quiz")
                    if self.hint_manager.can_afford(self.level_manager.score, "quiz"):
                        elim = self.hint_manager.request_quiz_hint(
                            self.level_manager, q, self.eliminated_quiz_options
                        )
                        if elim is not None:
                            self.eliminated_quiz_options.add(elim)
                            self._set_status(f"50/50 HINT USED (-{cost} pts)")
                    else:
                        self._set_status(f"INSUFFICIENT SCORE FOR HINT (Requires {cost} pts)")
                    return

                for i, r in enumerate(opt_rects):
                    if i in self.eliminated_quiz_options:
                        continue
                    if r.collidepoint(mouse_pos):
                        self.selected_option_index = i
                        return

                if action_btn_rect.collidepoint(mouse_pos) and self.selected_option_index is not None:
                    self._submit_checkpoint_answer()
            else:
                if action_btn_rect.collidepoint(mouse_pos):
                    if self.checkpoint_last_correct:
                        self.active_checkpoint = None
                        self.checkpoint_timer.stop()
                    else:
                        self._retry_checkpoint()

        elif event.type == pygame.KEYDOWN:
            if not self.checkpoint_feedback_mode:
                if event.key in (pygame.K_1, pygame.K_a) and 0 not in self.eliminated_quiz_options:
                    self.selected_option_index = 0
                elif event.key in (pygame.K_2, pygame.K_b) and 1 not in self.eliminated_quiz_options:
                    self.selected_option_index = 1
                elif event.key in (pygame.K_3, pygame.K_c) and 2 not in self.eliminated_quiz_options:
                    self.selected_option_index = 2
                elif event.key in (pygame.K_4, pygame.K_d) and 3 not in self.eliminated_quiz_options:
                    self.selected_option_index = 3
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE) and self.selected_option_index is not None:
                    self._submit_checkpoint_answer()
            else:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    if self.checkpoint_last_correct:
                        self.active_checkpoint = None
                        self.checkpoint_timer.stop()
                    else:
                        self._retry_checkpoint()

    def _submit_checkpoint_answer(self) -> None:
        if not self.active_checkpoint or not self.active_checkpoint.question:
            return

        self.checkpoint_timer.pause()
        self.active_checkpoint.attempts += 1
        is_correct = self.active_checkpoint.question.is_correct(self.selected_option_index or 0)

        cp_cfg = self.config.get("checkpoints", {})
        cost_pen = cp_cfg.get("cost_penalty_on_failure", 3)
        score_pen = cp_cfg.get("score_penalty_on_failure", 25)
        score_rew = cp_cfg.get("score_reward_on_success", 100)

        self.checkpoint_feedback_mode = True
        self.checkpoint_last_correct = is_correct

        if is_correct:
            self.active_checkpoint.unlock()
            self.level_manager.add_score(score_rew)
            self.checkpoint_feedback_msg = f"ACCESS GRANTED! Gate unlocked. (+{score_rew} Score)"
            self.sound_manager.play("correct")
        else:
            if self.player:
                self.player.total_cost += cost_pen
            self.level_manager.deduct_score(score_pen)
            self.checkpoint_feedback_msg = (
                f"INCORRECT! Gate remains locked. (+{cost_pen} Route Cost penalty, -{score_pen} Score)"
            )
            self.sound_manager.play("wrong")

    def _retry_checkpoint(self) -> None:
        if not self.active_checkpoint:
            return

        q = self.question_manager.get_question(
            topic=self.selected_topic,
            difficulty=self.active_checkpoint.difficulty,
        )
        self.active_checkpoint.question = q
        self.selected_option_index = None
        self.checkpoint_feedback_mode = False
        self.checkpoint_last_correct = False
        self.checkpoint_feedback_msg = ""
        self.eliminated_quiz_options.clear()

        time_limits = self.config.get("checkpoints", {}).get("time_limits", {})
        limit = time_limits.get(self.active_checkpoint.difficulty, 20.0)
        self.checkpoint_timer.start(limit)

    def _handle_settings_event(self, event: pygame.event.Event) -> None:
        settings_cfg = self.config.setdefault("settings", {})

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            sound_rect = pygame.Rect(self.width // 2 + 60, 200, 160, 36)
            if sound_rect.collidepoint(mouse_pos):
                settings_cfg["sound_enabled"] = not settings_cfg.get("sound_enabled", True)

            vol_down = pygame.Rect(self.width // 2 + 60, 250, 40, 36)
            vol_up = pygame.Rect(self.width // 2 + 180, 250, 40, 36)
            if vol_down.collidepoint(mouse_pos):
                settings_cfg["volume"] = max(0, settings_cfg.get("volume", 75) - 10)
            elif vol_up.collidepoint(mouse_pos):
                settings_cfg["volume"] = min(100, settings_cfg.get("volume", 75) + 10)

            back_btn = pygame.Rect(self.width // 2 - 120, self.height - 90, 240, 44)
            if back_btn.collidepoint(mouse_pos):
                self.state = GameState.MENU

        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
            self.state = GameState.MENU

    def _handle_learn_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            step_btn = pygame.Rect(self.width // 2 - 260, self.height - 80, 120, 38)
            auto_btn = pygame.Rect(self.width // 2 - 130, self.height - 80, 120, 38)
            reset_btn = pygame.Rect(self.width // 2, self.height - 80, 120, 38)
            back_btn = pygame.Rect(self.width // 2 + 130, self.height - 80, 130, 38)

            if step_btn.collidepoint(mouse_pos):
                self.learn_step = min(self.learn_step + 1, len(self.learn_explored))
            elif auto_btn.collidepoint(mouse_pos):
                self.learn_auto_play = not self.learn_auto_play
            elif reset_btn.collidepoint(mouse_pos):
                self.learn_step = 0
                self.learn_auto_play = False
            elif back_btn.collidepoint(mouse_pos):
                self.state = GameState.MENU

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self.learn_step = min(self.learn_step + 1, len(self.learn_explored))
            elif event.key == pygame.K_SPACE:
                self.learn_auto_play = not self.learn_auto_play
            elif event.key == pygame.K_r:
                self.learn_step = 0
                self.learn_auto_play = False
            elif event.key == pygame.K_ESCAPE:
                self.state = GameState.MENU

    def _handle_how_to_play_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
            self.state = GameState.MENU
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            back_btn = pygame.Rect(self.width // 2 - 120, self.height - 90, 240, 44)
            if back_btn.collidepoint(event.pos):
                self.state = GameState.MENU

    # -------------------------------------------------------------------------
    # Updates & Continuous Logic
    # -------------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Update continuous entities, visualizer, hints, and timers."""
        if self.state == GameState.PLAYING:
            self.visualizer.update(dt)
            self.hint_manager.update(dt)

            if self.active_checkpoint and not self.checkpoint_feedback_mode:
                expired = self.checkpoint_timer.update(dt)
                if expired:
                    self.active_checkpoint.attempts += 1
                    cp_cfg = self.config.get("checkpoints", {})
                    cost_pen = cp_cfg.get("cost_penalty_on_failure", 3)
                    score_pen = cp_cfg.get("score_penalty_on_failure", 25)

                    if self.player:
                        self.player.total_cost += cost_pen
                    self.level_manager.deduct_score(score_pen)

                    self.checkpoint_feedback_mode = True
                    self.checkpoint_last_correct = False
                    self.checkpoint_feedback_msg = (
                        f"TIME EXPIRED! Gateway locked. (+{cost_pen} Route Cost penalty, -{score_pen} Score)"
                    )

        elif self.state == GameState.LEARN_ALGORITHM and self.learn_auto_play:
            self.learn_timer += dt
            if self.learn_timer >= 0.25:
                self.learn_timer = 0.0
                if self.learn_step < len(self.learn_explored):
                    self.learn_step += 1
                else:
                    self.learn_auto_play = False

    # -------------------------------------------------------------------------
    # Screen Transition System
    # -------------------------------------------------------------------------

    def _start_transition(self, target_state: GameState) -> None:
        """Initiate a fade-out/fade-in transition to a new state."""
        if self.pending_state is not None:
            return
        self.pending_state = target_state
        self.transition_target = 255
        self.sound_manager.play("click")

    def _update_transition(self, dt: float) -> None:
        """Advance transition overlay alpha and switch state at peak opacity."""
        if self.transition_target == 0 and self.transition_alpha == 0:
            return

        if self.transition_target > self.transition_alpha:
            self.transition_alpha = min(255, self.transition_alpha + int(self.transition_speed * dt))
            if self.transition_alpha >= 255:
                if self.pending_state is not None:
                    self.state = self.pending_state
                    self.pending_state = None
                self.transition_target = 0
        else:
            self.transition_alpha = max(0, self.transition_alpha - int(self.transition_speed * dt))

    def _render_transition_overlay(self) -> None:
        """Draw a semi-transparent overlay for screen transitions."""
        if self.transition_alpha > 0:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((11, 16, 32, self.transition_alpha))
            self.screen.blit(overlay, (0, 0))

    # -------------------------------------------------------------------------
    # Renderers
    # -------------------------------------------------------------------------

    def render(self) -> None:
        """Render frame based on active state."""
        self.screen.fill(self.colors.get("background", pygame.Color("#0B1020")))

        if self.state == GameState.MENU:
            self._render_menu()
        elif self.state == GameState.TOPIC_SELECTION:
            self._render_topic_selection()
        elif self.state == GameState.STUDY_MATERIAL:
            self._render_study_material()
        elif self.state == GameState.PLAYING:
            self._render_playing()
        elif self.state == GameState.LEVEL_COMPLETE:
            self._render_level_complete()
        elif self.state == GameState.SETTINGS:
            self._render_settings()
        elif self.state == GameState.LEARN_ALGORITHM:
            self._render_learn_algorithm()
        elif self.state == GameState.HOW_TO_PLAY:
            self._render_how_to_play()

        if self.status_timer > 0:
            banner_rect = pygame.Rect(self.width // 2 - 270, 16, 540, 36)
            self.ui.draw_panel(
                self.screen,
                banner_rect,
                fill_color=self.colors.get("panel", pygame.Color("#141C2F")),
                border_color=self.colors.get("warning", pygame.Color("#FFBF69")),
                border_radius=6,
            )
            self.ui.draw_text(
                self.screen,
                self.status_message,
                banner_rect.center,
                font_size=13,
                color=self.colors.get("warning", pygame.Color("#FFBF69")),
                bold=True,
                center=True,
            )

        self._render_transition_overlay()
        pygame.display.flip()

    def _render_menu(self) -> None:
        """Render Main Menu."""
        mouse_pos = pygame.mouse.get_pos()

        self.ui.draw_text(
            self.screen,
            "PATHFINDER",
            (self.width // 2, 100),
            font_size=58,
            color=self.colors.get("primary", pygame.Color("#38D9FF")),
            bold=True,
            center=True,
        )

        self.ui.draw_text(
            self.screen,
            "\"Think. Navigate. Conquer.\"",
            (self.width // 2, 155),
            font_size=18,
            color=self.colors.get("secondary", pygame.Color("#8B7CFF")),
            center=True,
        )

        self.ui.draw_text(
            self.screen,
            "Tactical Dijkstra Shortest Path Navigation Engine",
            (self.width // 2, 185),
            font_size=14,
            color=self.colors.get("text_dim", pygame.Color("#8B9AB4")),
            center=True,
        )

        has_save = self.save_manager.has_saved_game()
        save_level = self.save_manager.data.get("current_level", 1)

        continue_text = f"▶  CONTINUE MISSION (L{save_level})" if has_save else "▶  CONTINUE (NO SAVE)"
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, 220, 280, 42), continue_text, mouse_pos, primary=has_save, font_size=16)
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, 270, 280, 42), "❖  SELECT MISSION TOPIC", mouse_pos, font_size=15)
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, 320, 280, 42), "📚  UPLOAD STUDY MATERIAL", mouse_pos, font_size=15)
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, 370, 280, 42), "⚡  LEARN THE ALGORITHM", mouse_pos, font_size=15)
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, 420, 280, 42), "?  HOW TO PLAY", mouse_pos, font_size=15)
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, 470, 280, 42), "⚙  CONSOLE SETTINGS", mouse_pos, font_size=15)
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, 520, 280, 42), "✕  ABORT / QUIT", mouse_pos, font_size=15)

        achievements_count = len(self.save_manager.data.get("achievements", []))
        self.ui.draw_text(
            self.screen,
            f"Achievements: {achievements_count}/3  •  Phase C RAG Pipeline Active",
            (self.width // 2, self.height - 30),
            font_size=12,
            color=self.colors.get("text_dim", pygame.Color("#8B9AB4")),
            center=True,
        )

    def _render_study_material(self) -> None:
        """Render Study Material Ingestion and RAG configuration screen."""
        mouse_pos = pygame.mouse.get_pos()

        panel_w = min(880, self.width - 40)
        panel_h = min(620, self.height - 40)
        panel = pygame.Rect(self.width // 2 - panel_w // 2, self.height // 2 - panel_h // 2, panel_w, panel_h)
        self.ui.draw_panel(self.screen, panel)

        self.ui.draw_text(
            self.screen,
            "STUDY MATERIAL INGESTION // RAG PIPELINE",
            (self.width // 2, panel.y + 24),
            font_size=22,
            color=self.colors.get("primary", pygame.Color("#38D9FF")),
            bold=True,
            center=True,
        )

        self.ui.draw_text(
            self.screen,
            "Upload TXT, Markdown, or PDF notes to generate grounded questions for checkpoint gates.",
            (self.width // 2, panel.y + 54),
            font_size=13,
            color=self.colors.get("text_dim", pygame.Color("#8B9AB4")),
            center=True,
        )

        # File path input card
        input_box = pygame.Rect(panel.x + 40, panel.y + 90, panel.width - 230, 40)
        border_col = self.colors.get("primary", pygame.Color("#38D9FF")) if self.input_active else self.colors.get("panel_border", pygame.Color("#222E4A"))
        self.ui.draw_panel(self.screen, input_box, fill_color=pygame.Color("#0E1628"), border_color=border_col)

        display_text = self.typed_file_path if self.typed_file_path else "Type or paste file path here..."
        text_col = self.colors.get("text", pygame.Color("#F3F6FC")) if self.typed_file_path else self.colors.get("text_dim", pygame.Color("#8B9AB4"))
        self.ui.draw_text(self.screen, display_text, (input_box.x + 14, input_box.y + 11), font_size=14, color=text_col)

        browse_btn = pygame.Rect(panel.right - 170, panel.y + 90, 130, 40)
        self.ui.draw_button(self.screen, browse_btn, "BROWSE...", mouse_pos, font_size=14)

        # Action row
        sample_btn = pygame.Rect(panel.x + 40, panel.y + 142, 210, 36)
        process_btn = pygame.Rect(panel.x + 265, panel.y + 142, 210, 36)
        toggle_btn = pygame.Rect(panel.right - 210, panel.y + 142, 170, 36)

        self.ui.draw_button(self.screen, sample_btn, "LOAD SAMPLE NOTES", mouse_pos, font_size=13)
        self.ui.draw_button(self.screen, process_btn, "PROCESS & GENERATE ▶", mouse_pos, primary=True, font_size=13)

        toggle_label = "IN-GAME: ON" if self.rag_pipeline.use_in_game else "IN-GAME: OFF"
        toggle_primary = self.rag_pipeline.use_in_game and len(self.rag_pipeline.generated_questions) > 0
        self.ui.draw_button(self.screen, toggle_btn, toggle_label, mouse_pos, primary=toggle_primary, font_size=13)

        # Pipeline Telemetry Status Card
        card = pygame.Rect(panel.x + 40, panel.y + 195, panel.width - 80, 320)
        self.ui.draw_panel(self.screen, card, fill_color=pygame.Color("#0E1628"), border_color=self.colors.get("panel_border", pygame.Color("#222E4A")))

        summary = self.rag_pipeline.get_status_summary()
        self.ui.draw_text(self.screen, "RAG PIPELINE TELEMETRY & STATUS", (card.x + 20, card.y + 16), font_size=14, color=self.colors.get("secondary", pygame.Color("#8B7CFF")), bold=True)

        rows = [
            ("Ingested Source Document:", summary["loaded_file"]),
            ("Total Segmented Chunks:", f"{summary['chunk_count']} chunks"),
            ("Active Retrieval Mode:", summary["retrieval_mode"].upper()),
            ("Grounded Questions Generated:", f"{summary['questions_count']} questions"),
            ("Custom Material Gate Status:", "ENABLED (Active in quiz pool)" if summary["use_in_game"] else "DISABLED"),
        ]

        for i, (k, v) in enumerate(rows):
            y_r = card.y + 50 + i * 36
            self.ui.draw_text(self.screen, k, (card.x + 24, y_r), font_size=14, color=self.colors.get("text_dim", pygame.Color("#8B9AB4")))
            val_col = self.colors.get("primary", pygame.Color("#38D9FF")) if "keyword" in v.lower() or "semantic" in v.lower() else self.colors.get("text", pygame.Color("#F3F6FC"))
            if "enabled" in v.lower():
                val_col = self.colors.get("success", pygame.Color("#35D39A"))
            self.ui.draw_text(self.screen, v, (card.x + 280, y_r), font_size=14, color=val_col, bold=True)

        # Honesty disclaimer
        ret_notice = (
            "Honest Architecture Note: Active retrieval uses pure-Python BM25 keyword search. "
            "To activate semantic embeddings, install optional dependencies (pip install -r requirements-rag.txt)."
        )
        self.ui.draw_multiline_text(
            self.screen,
            ret_notice,
            (card.x + 20, card.bottom - 64),
            max_width=card.width - 40,
            font_size=12,
            color=self.colors.get("text_dim", pygame.Color("#8B9AB4")),
        )

        back_btn = pygame.Rect(self.width // 2 - 120, panel.bottom - 46, 240, 36)
        self.ui.draw_button(self.screen, back_btn, "← BACK TO MENU", mouse_pos, font_size=14)

    def _render_topic_selection(self) -> None:
        """Render Topic and Difficulty Selection screen."""
        mouse_pos = pygame.mouse.get_pos()

        panel_w = min(880, self.width - 40)
        panel_h = min(620, self.height - 40)
        panel = pygame.Rect(self.width // 2 - panel_w // 2, self.height // 2 - panel_h // 2, panel_w, panel_h)
        self.ui.draw_panel(self.screen, panel)

        self.ui.draw_text(
            self.screen,
            "MISSION CONFIGURATION // SELECT KNOWLEDGE TOPIC",
            (self.width // 2, panel.y + 24),
            font_size=22,
            color=self.colors.get("primary", pygame.Color("#38D9FF")),
            bold=True,
            center=True,
        )

        self.ui.draw_text(
            self.screen,
            "KNOWLEDGE TOPIC",
            (panel.x + 80, panel.y + 68),
            font_size=14,
            color=self.colors.get("secondary", pygame.Color("#8B7CFF")),
            bold=True,
        )

        self.ui.draw_text(
            self.screen,
            "QUIZ DIFFICULTY",
            (panel.centerx + 40, panel.y + 68),
            font_size=14,
            color=self.colors.get("secondary", pygame.Color("#8B7CFF")),
            bold=True,
        )

        topics = ["All Topics"] + self.question_manager.get_topics()
        difficulties = ["Level Default", "Easy", "Medium", "Hard"]

        for i, topic in enumerate(topics):
            r = pygame.Rect(panel.x + 80, panel.y + 100 + i * 40, 280, 34)
            is_sel = (self.selected_topic == topic)
            self.ui.draw_button(self.screen, r, topic, mouse_pos, selected=is_sel, font_size=13)

        for i, diff in enumerate(difficulties):
            r = pygame.Rect(panel.centerx + 40, panel.y + 100 + i * 40, 280, 34)
            is_sel = (self.selected_difficulty == diff)
            self.ui.draw_button(self.screen, r, diff, mouse_pos, selected=is_sel, font_size=13)

        launch_rect = pygame.Rect(self.width // 2 - 140, panel.bottom - 80, 280, 42)
        self.ui.draw_button(self.screen, launch_rect, "LAUNCH MISSION ▶", mouse_pos, primary=True, font_size=15)

        back_rect = pygame.Rect(self.width // 2 - 140, panel.bottom - 34, 280, 28)
        self.ui.draw_button(self.screen, back_rect, "← BACK TO MENU", mouse_pos, font_size=12)

    def _render_level_complete(self) -> None:
        mouse_pos = pygame.mouse.get_pos()

        panel_w = min(680, self.width - 40)
        panel_h = min(580, self.height - 60)
        panel = pygame.Rect(self.width // 2 - panel_w // 2, self.height // 2 - panel_h // 2, panel_w, panel_h)
        self.ui.draw_panel(self.screen, panel, border_color=self.colors.get("success", pygame.Color("#35D39A")), border_width=2)

        lvl_num = self.level_manager.current_level
        title = f"SECTOR {lvl_num} CONQUERED!" if not self.level_manager.is_max_level else "ALL SECTORS CONQUERED!"
        self.ui.draw_text(self.screen, title, (self.width // 2, panel.y + 30), font_size=28, color=self.colors.get("success", pygame.Color("#35D39A")), bold=True, center=True)

        eff = self.player.calculate_efficiency(self.optimal_cost) if self.player else 100.0
        eff_color = self.colors.get("success", pygame.Color("#35D39A")) if eff >= 90.0 else (self.colors.get("warning", pygame.Color("#FFBF69")) if eff >= 60.0 else self.colors.get("danger", pygame.Color("#FF647C")))

        telemetry = f"Player Route Cost: {self.player.total_cost if self.player else 0}  |  Optimal Cost: {self.optimal_cost}  |  Efficiency: {eff:.1f}%"
        self.ui.draw_text(self.screen, telemetry, (self.width // 2, panel.y + 75), font_size=14, color=eff_color, bold=True, center=True)

        box_rect = pygame.Rect(panel.x + 40, panel.y + 110, panel.width - 80, 240)
        self.ui.draw_panel(self.screen, box_rect, fill_color=pygame.Color("#0E1628"), border_color=self.colors.get("panel_border", pygame.Color("#222E4A")))
        self.ui.draw_text(self.screen, "MISSION REWARDS BREAKDOWN", (box_rect.x + 18, box_rect.y + 16), font_size=14, color=self.colors.get("secondary", pygame.Color("#8B7CFF")), bold=True)

        breakdown = self.last_rewards_summary.get("breakdown", [])
        y_cursor = box_rect.y + 44
        for label, pts in breakdown:
            self.ui.draw_text(self.screen, label, (box_rect.x + 20, y_cursor), font_size=14)
            self.ui.draw_text(self.screen, f"+{pts} PTS", (box_rect.right - 90, y_cursor), font_size=14, color=self.colors.get("primary", pygame.Color("#38D9FF")), bold=True)
            y_cursor += 28

        total_awarded = self.last_rewards_summary.get("total_points", 0)
        pygame.draw.line(self.screen, self.colors.get("panel_border", pygame.Color("#222E4A")), (box_rect.x + 18, y_cursor + 6), (box_rect.right - 18, y_cursor + 6), 1)
        self.ui.draw_text(self.screen, "TOTAL REWARD POINTS", (box_rect.x + 20, y_cursor + 14), font_size=15, color=self.colors.get("text", pygame.Color("#F3F6FC")), bold=True)
        self.ui.draw_text(self.screen, f"+{total_awarded} PTS", (box_rect.right - 100, y_cursor + 14), font_size=16, color=self.colors.get("success", pygame.Color("#35D39A")), bold=True)

        self.ui.draw_text(self.screen, f"CUMULATIVE CAREER SCORE: {self.level_manager.score:05d}", (self.width // 2, panel.y + 370), font_size=17, color=self.colors.get("text", pygame.Color("#F3F6FC")), bold=True, center=True)

        next_btn_label = "NEXT SECTOR ▶" if not self.level_manager.is_max_level else "FINISH CAMPAIGN ▶"
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, panel.bottom - 130, 280, 44), next_btn_label, mouse_pos, primary=True, font_size=16)
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, panel.bottom - 80, 280, 36), "RETRY SECTOR ⟳", mouse_pos, font_size=14)
        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 140, panel.bottom - 38, 280, 30), "MAIN MENU ⌂", mouse_pos, font_size=12)

    def _render_settings(self) -> None:
        mouse_pos = pygame.mouse.get_pos()

        panel_w = min(680, self.width - 40)
        panel_h = min(500, self.height - 60)
        panel = pygame.Rect(self.width // 2 - panel_w // 2, self.height // 2 - panel_h // 2, panel_w, panel_h)
        self.ui.draw_panel(self.screen, panel)

        self.ui.draw_text(self.screen, "CONSOLE CONFIGURATION & SETTINGS", (self.width // 2, panel.y + 35), font_size=22, color=self.colors.get("primary", pygame.Color("#38D9FF")), bold=True, center=True)

        settings_cfg = self.config.get("settings", {})
        sound_on = settings_cfg.get("sound_enabled", True)
        volume = settings_cfg.get("volume", 75)

        self.ui.draw_text(self.screen, "Audio Synthesis Effects:", (panel.x + 60, panel.y + 110), font_size=16)
        sound_label = "ENABLED" if sound_on else "MUTED"
        self.ui.draw_button(self.screen, pygame.Rect(panel.right - 220, panel.y + 104, 160, 36), sound_label, mouse_pos, font_size=14)

        self.ui.draw_text(self.screen, f"Master Volume Level:  {volume}%", (panel.x + 60, panel.y + 165), font_size=16)
        self.ui.draw_button(self.screen, pygame.Rect(panel.right - 220, panel.y + 158, 44, 34), "-", mouse_pos, font_size=18)
        self.ui.draw_button(self.screen, pygame.Rect(panel.right - 104, panel.y + 158, 44, 34), "+", mouse_pos, font_size=18)

        self.ui.draw_text(self.screen, f"Active Knowledge Topic:  {self.selected_topic}", (panel.x + 60, panel.y + 225), font_size=15, color=self.colors.get("secondary", pygame.Color("#8B7CFF")))
        self.ui.draw_text(self.screen, f"Active Difficulty Mode:  {self.selected_difficulty}", (panel.x + 60, panel.y + 260), font_size=15, color=self.colors.get("secondary", pygame.Color("#8B7CFF")))

        self.ui.draw_button(self.screen, pygame.Rect(self.width // 2 - 120, panel.bottom - 60, 240, 42), "← SAVE & RETURN", mouse_pos, primary=True, font_size=15)

    def _render_learn_algorithm(self) -> None:
        mouse_pos = pygame.mouse.get_pos()

        panel_w = min(1040, self.width - 40)
        panel_h = min(640, self.height - 40)
        panel = pygame.Rect(self.width // 2 - panel_w // 2, self.height // 2 - panel_h // 2, panel_w, panel_h)
        self.ui.draw_panel(self.screen, panel)

        self.ui.draw_text(self.screen, "LEARN THE ALGORITHM // DIJKSTRA'S SHORTEST PATH", (self.width // 2, panel.y + 24), font_size=20, color=self.colors.get("primary", pygame.Color("#38D9FF")), bold=True, center=True)

        expl_rect = pygame.Rect(panel.x + 30, panel.y + 60, panel.width - 60, 80)
        self.ui.draw_panel(self.screen, expl_rect, fill_color=pygame.Color("#0E1628"), border_color=self.colors.get("panel_border", pygame.Color("#222E4A")))

        concepts = [
            "1. Greedy Priority Queue: Dijkstra pulls the unsettled node with smallest known distance O(log V).",
            "2. Edge Relaxation: For each neighbor v, if dist[u] + weight(u,v) < dist[v], update dist[v].",
            "3. Key Insight: In weighted graphs, the fewest-steps path (4 steps in swamp = 20) is worse than 6 steps in open (6)!",
        ]
        for idx, text in enumerate(concepts):
            self.ui.draw_text(self.screen, text, (expl_rect.x + 16, expl_rect.y + 12 + idx * 22), font_size=13)

        tile_px = 44
        grid_w_px = tile_px * self.learn_grid.width
        grid_h_px = tile_px * self.learn_grid.height
        grid_origin_x = self.width // 2 - grid_w_px // 2
        grid_origin_y = panel.y + 160

        for gx in range(self.learn_grid.width):
            for gy in range(self.learn_grid.height):
                cell = self.learn_grid.get_cell(gx, gy)
                if not cell:
                    continue

                r = pygame.Rect(grid_origin_x + gx * tile_px, grid_origin_y + gy * tile_px, tile_px, tile_px)
                c = (r.centerx, r.centery)

                if cell.is_wall:
                    pygame.draw.rect(self.screen, self.colors.get("wall", pygame.Color("#162035")), r)
                else:
                    bg = self.colors.get("cell_open", pygame.Color("#0E182A"))
                    glyph = "circle"
                    if cell.cell_type == CellType.SWAMP:
                        bg = self.colors.get("cell_swamp", pygame.Color("#3A1B2D"))
                        glyph = "star"

                    pygame.draw.rect(self.screen, bg, r)
                    pygame.draw.rect(self.screen, pygame.Color("#1A253D"), r, 1)
                    self.ui.draw_shape_glyph(self.screen, c, glyph, size=14, color=self.colors.get("text_dim", pygame.Color("#8B9AB4")), filled=False)
                    self.ui.draw_text(self.screen, str(cell.weight), (c[0] + 8, c[1] - 12), font_size=11, color=self.colors.get("text_dim", pygame.Color("#8B9AB4")))

        revealed = self.learn_explored[:self.learn_step]
        for node in revealed:
            r = pygame.Rect(grid_origin_x + node[0] * tile_px, grid_origin_y + node[1] * tile_px, tile_px, tile_px)
            surf = pygame.Surface((tile_px, tile_px), pygame.SRCALPHA)
            surf.fill((*self.colors.get("primary", pygame.Color("#38D9FF"))[:3], 80))
            self.screen.blit(surf, r.topleft)

        if self.learn_step >= len(self.learn_explored):
            pts = [(grid_origin_x + px * tile_px + tile_px // 2, grid_origin_y + py * tile_px + tile_px // 2) for px, py in self.learn_path]
            pygame.draw.lines(self.screen, self.colors.get("primary", pygame.Color("#38D9FF")), False, pts, width=4)

        start_pt = (grid_origin_x + self.learn_start[0] * tile_px + tile_px // 2, grid_origin_y + self.learn_start[1] * tile_px + tile_px // 2)
        end_pt = (grid_origin_x + self.learn_end[0] * tile_px + tile_px // 2, grid_origin_y + self.learn_end[1] * tile_px + tile_px // 2)
        self.ui.draw_glow_circle(self.screen, start_pt, 8, self.colors.get("player", pygame.Color("#38D9FF")))
        self.ui.draw_glow_circle(self.screen, end_pt, 8, self.colors.get("destination", pygame.Color("#35D39A")))

        self.ui.draw_text(
            self.screen,
            f"Wavefront Expansion: Step {self.learn_step}/{len(self.learn_explored)}  (Optimal Cost: {self.learn_cost})",
            (self.width // 2, grid_origin_y + grid_h_px + 20),
            font_size=15,
            color=self.colors.get("text", pygame.Color("#F3F6FC")),
            bold=True,
            center=True,
        )

        step_btn = pygame.Rect(self.width // 2 - 260, panel.bottom - 55, 120, 38)
        auto_btn = pygame.Rect(self.width // 2 - 130, panel.bottom - 55, 120, 38)
        reset_btn = pygame.Rect(self.width // 2, panel.bottom - 55, 120, 38)
        back_btn = pygame.Rect(self.width // 2 + 130, panel.bottom - 55, 130, 38)

        self.ui.draw_button(self.screen, step_btn, "STEP ▶", mouse_pos, font_size=14)
        auto_label = "PAUSE ⏸" if self.learn_auto_play else "AUTO PLAY ⏯"
        self.ui.draw_button(self.screen, auto_btn, auto_label, mouse_pos, font_size=14)
        self.ui.draw_button(self.screen, reset_btn, "RESET ⟳", mouse_pos, font_size=14)
        self.ui.draw_button(self.screen, back_btn, "← MENU", mouse_pos, primary=True, font_size=14)

    def _render_how_to_play(self) -> None:
        mouse_pos = pygame.mouse.get_pos()

        panel_w = min(880, self.width - 40)
        panel_h = min(630, self.height - 80)
        panel_rect = pygame.Rect(self.width // 2 - panel_w // 2, 40, panel_w, panel_h)
        self.ui.draw_panel(self.screen, panel_rect)

        self.ui.draw_text(
            self.screen,
            "SYSTEM NAVIGATION MANUAL",
            (self.width // 2, panel_rect.y + 35),
            font_size=28,
            color=self.colors.get("primary", pygame.Color("#38D9FF")),
            bold=True,
            center=True,
        )

        lines = [
            ("Core Principle: Dijkstra's Shortest Path", True, self.colors.get("secondary", pygame.Color("#8B7CFF"))),
            ("In PATHFINDER, the fewest number of steps does NOT equal the lowest cost route.", False, self.colors.get("text", pygame.Color("#F3F6FC"))),
            ("Each terrain tile incurs a movement cost strictly upon ENTERING that cell.", False, self.colors.get("text", pygame.Color("#F3F6FC"))),
            ("Your starting cell incurs zero cost (it is free).", False, self.colors.get("text", pygame.Color("#F3F6FC"))),
            ("", False, None),
            ("Checkpoint Security Gates (Hexagons)", True, self.colors.get("secondary", pygame.Color("#8B7CFF"))),
            ("• Checkpoints are placed on unavoidable bottlenecks along your route to the destination.", False, self.colors.get("text", pygame.Color("#F3F6FC"))),
            ("• Stepping onto a checkpoint triggers a timed challenge (DSA, Python, DBMS, OS, Networks, Math, GK).", False, self.colors.get("text", pygame.Color("#F3F6FC"))),
            ("• Clear gates to proceed. Wrong answers or timeouts incur route cost penalties!", False, self.colors.get("text", pygame.Color("#F3F6FC"))),
            ("", False, None),
            ("Tactical Hints & Scoring", True, self.colors.get("secondary", pygame.Color("#8B7CFF"))),
            ("• Press H or click [Route Hint] for the next optimal step (-40 pts).", False, self.colors.get("text", pygame.Color("#F3F6FC"))),
            ("• Click [50/50 Hint] in quiz overlay to eliminate an incorrect option (-30 pts).", False, self.colors.get("text", pygame.Color("#F3F6FC"))),
            ("• Complete sectors without hints for the Tactical Purity bonus (+100 pts)!", False, self.colors.get("text", pygame.Color("#F3F6FC"))),
        ]

        y_offset = panel_rect.y + 85
        for text, bold, col in lines:
            if text:
                self.ui.draw_text(self.screen, text, (panel_rect.x + 36, y_offset), font_size=14, bold=bold, color=col)
            y_offset += 23

        back_btn_rect = pygame.Rect(self.width // 2 - 120, panel_rect.bottom - 50, 240, 42)
        self.ui.draw_button(self.screen, back_btn_rect, "← BACK TO MENU", mouse_pos, font_size=15, primary=True)

    def _render_playing(self) -> None:
        mouse_pos = pygame.mouse.get_pos()

        top_bar_rect = pygame.Rect(0, 0, self.width, 54)
        self.ui.draw_panel(self.screen, top_bar_rect, fill_color=self.colors.get("panel", pygame.Color("#141C2F")), border_radius=0, border_width=0)
        pygame.draw.line(self.screen, self.colors.get("panel_border", pygame.Color("#222E4A")), (0, 54), (self.width, 54), 1)

        lvl_name = self.level_manager.get_name()
        self.ui.draw_text(self.screen, f"{lvl_name.upper()}  //  LEVEL {self.level_manager.current_level}", (24, 18), font_size=18, color=self.colors.get("primary", pygame.Color("#38D9FF")), bold=True)
        self.ui.draw_text(self.screen, f"SCORE: {self.level_manager.score:05d}", (self.width // 2 - 60, 18), font_size=18, color=self.colors.get("text", pygame.Color("#F3F6FC")), bold=True)

        cleared_count = sum(1 for cp in self.checkpoint_manager.checkpoints if cp.is_cleared)
        total_cps = len(self.checkpoint_manager.checkpoints)
        cp_status = f"GATES: {cleared_count}/{total_cps}"
        cp_col = self.colors.get("success", pygame.Color("#35D39A")) if cleared_count == total_cps else self.colors.get("warning", pygame.Color("#FFBF69"))
        self.ui.draw_text(self.screen, cp_status, (self.width - 240, 18), font_size=16, color=cp_col, bold=True)

        if self.grid:
            self._render_maze()

        self._render_side_panel(mouse_pos)

        if self.active_checkpoint is not None:
            self._render_checkpoint_overlay(mouse_pos)

    def _bake_static_maze(self, viewport_rect: pygame.Rect) -> None:
        if not self.grid:
            return

        max_w = viewport_rect.width - 24
        max_h = viewport_rect.height - 24
        self.tile_size = max(16, min(max_w // self.grid.width, max_h // self.grid.height))

        grid_px_w = self.tile_size * self.grid.width
        grid_px_h = self.tile_size * self.grid.height
        self.maze_offset_x = viewport_rect.x + (viewport_rect.width - grid_px_w) // 2
        self.maze_offset_y = viewport_rect.y + (viewport_rect.height - grid_px_h) // 2

        self.maze_surface = pygame.Surface((grid_px_w, grid_px_h))
        self.maze_surface.fill(self.colors.get("background", pygame.Color("#0B1020")))

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get_cell(x, y)
                if not cell:
                    continue

                cell_rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
                center = (x * self.tile_size + self.tile_size // 2, y * self.tile_size + self.tile_size // 2)

                if cell.is_wall:
                    pygame.draw.rect(self.maze_surface, self.colors.get("wall", pygame.Color("#162035")), cell_rect)
                    pygame.draw.rect(self.maze_surface, self.colors.get("wall_border", pygame.Color("#222F4C")), cell_rect, 1)
                else:
                    bg_col = self.colors.get("cell_open", pygame.Color("#0E182A"))
                    glyph_shape = "circle"
                    glyph_color = self.colors.get("text_dim", pygame.Color("#8B9AB4"))

                    if cell.cell_type == CellType.SAND:
                        bg_col = self.colors.get("cell_sand", pygame.Color("#332A15"))
                        glyph_shape = "triangle"
                        glyph_color = self.colors.get("warning", pygame.Color("#FFBF69"))
                    elif cell.cell_type == CellType.FOREST:
                        bg_col = self.colors.get("cell_forest", pygame.Color("#123023"))
                        glyph_shape = "diamond"
                        glyph_color = self.colors.get("success", pygame.Color("#35D39A"))
                    elif cell.cell_type == CellType.SWAMP:
                        bg_col = self.colors.get("cell_swamp", pygame.Color("#3A1B2D"))
                        glyph_shape = "star"
                        glyph_color = self.colors.get("danger", pygame.Color("#FF647C"))

                    pygame.draw.rect(self.maze_surface, bg_col, cell_rect)
                    pygame.draw.rect(self.maze_surface, pygame.Color("#1A253D"), cell_rect, 1)

                    glyph_size = max(6, self.tile_size // 4)
                    self.ui.draw_shape_glyph(self.maze_surface, center, glyph_shape, glyph_size, glyph_color, filled=False)

                    if cell.weight > 1:
                        font_size = max(10, self.tile_size // 3)
                        self.ui.draw_text(self.maze_surface, str(cell.weight), (center[0] + self.tile_size // 4, center[1] - self.tile_size // 3), font_size=font_size, color=glyph_color, bold=True)

    def _render_maze(self) -> None:
        if not self.grid or not self.player:
            return

        viewport_rect = self.get_maze_viewport_rect()
        self.ui.draw_panel(self.screen, viewport_rect)

        if self.maze_surface is None:
            self._bake_static_maze(viewport_rect)

        if self.maze_surface is not None:
            self.screen.blit(self.maze_surface, (self.maze_offset_x, self.maze_offset_y))

        if len(self.player.path) > 1:
            trail_color = self.colors.get("trail", pygame.Color("#1F3D63"))
            trail_pts = [self.grid_to_center(tx, ty) for tx, ty in self.player.path]
            pygame.draw.lines(self.screen, trail_color, False, trail_pts, width=max(2, self.tile_size // 6))
            for pt in trail_pts:
                pygame.draw.circle(self.screen, trail_color, pt, radius=max(2, self.tile_size // 8))

        self.visualizer.draw(self.screen, self.grid_to_rect, self.grid_to_center, self.colors)

        if self.hint_manager.active_path_hint:
            hint_c = self.grid_to_center(*self.hint_manager.active_path_hint)
            hint_col = self.colors.get("hint_glow", pygame.Color("#FFD166"))
            pulse_r = max(8, int(self.tile_size * 0.4 + 3 * math.sin(self.pulse_phase * 2)))
            pygame.draw.circle(self.screen, hint_col, hint_c, pulse_r, width=3)
            self.ui.draw_text(self.screen, "NEXT", (hint_c[0], hint_c[1] - pulse_r - 8), font_size=11, color=hint_col, bold=True, center=True)

        for cp in self.checkpoint_manager.checkpoints:
            cp_center = self.grid_to_center(*cp.pos)
            glyph_color = (
                self.colors.get("success", pygame.Color("#35D39A"))
                if cp.is_cleared
                else self.colors.get("warning", pygame.Color("#FFBF69"))
            )
            if not cp.is_cleared:
                pulse_r = max(6, int(self.tile_size * 0.45 + 2 * math.sin(self.pulse_phase)))
                glow_surf = pygame.Surface((pulse_r * 4, pulse_r * 4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*glyph_color[:3], 50), (pulse_r * 2, pulse_r * 2), pulse_r)
                self.screen.blit(glow_surf, (cp_center[0] - pulse_r * 2, cp_center[1] - pulse_r * 2))

            self.ui.draw_shape_glyph(self.screen, cp_center, "hexagon", size=max(12, int(self.tile_size * 0.7)), color=glyph_color, filled=True)
            self.ui.draw_shape_glyph(self.screen, cp_center, "hexagon", size=max(12, int(self.tile_size * 0.7)), color=pygame.Color("#FFFFFF"), filled=False)
            self.ui.draw_text(self.screen, str(cp.checkpoint_num), cp_center, font_size=max(11, self.tile_size // 3), color=self.colors.get("background", pygame.Color("#0B1020")), bold=True, center=True)

        dest_center = self.grid_to_center(*self.end_pos)
        all_cps_cleared = self.checkpoint_manager.all_cleared()
        dest_color = (
            self.colors.get("destination", pygame.Color("#35D39A"))
            if all_cps_cleared
            else self.colors.get("text_dim", pygame.Color("#8B9AB4"))
        )
        pulse_alpha = int(170 + 70 * (0.5 + 0.5 * math.sin(self.pulse_phase))) if all_cps_cleared else 90
        self.ui.draw_glow_circle(self.screen, dest_center, core_radius=max(5, self.tile_size // 4), color=dest_color, pulse_alpha=pulse_alpha)

        player_center = self.grid_to_center(*self.player.pos)
        player_color = self.colors.get("player", pygame.Color("#38D9FF"))
        self.ui.draw_glow_circle(self.screen, player_center, core_radius=max(5, self.tile_size // 4), color=player_color, pulse_alpha=220)

    def _render_side_panel(self, mouse_pos: Tuple[int, int]) -> None:
        panel_rect = self.get_side_panel_rect()
        self.ui.draw_panel(self.screen, panel_rect)

        self.ui.draw_text(self.screen, "MISSION TELEMETRY", (panel_rect.x + 20, panel_rect.y + 18), font_size=18, color=self.colors.get("primary", pygame.Color("#38D9FF")), bold=True)

        player_cost = self.player.total_cost if self.player else 0
        opt_cost = self.optimal_cost
        eff = self.player.calculate_efficiency(opt_cost) if self.player else 100.0

        if eff >= 90.0:
            eff_color = self.colors.get("success", pygame.Color("#35D39A"))
        elif eff >= 60.0:
            eff_color = self.colors.get("warning", pygame.Color("#FFBF69"))
        else:
            eff_color = self.colors.get("danger", pygame.Color("#FF647C"))

        cards = [
            ("PLAYER ROUTE COST", f"{player_cost}", self.colors.get("text", pygame.Color("#F3F6FC")), 58),
            ("OPTIMAL DIJKSTRA COST", f"{opt_cost}", self.colors.get("primary", pygame.Color("#38D9FF")), 116),
            ("NAVIGATION EFFICIENCY", f"{eff:5.1f}%", eff_color, 174),
        ]

        for title, value, val_col, y_rel in cards:
            card_rect = pygame.Rect(panel_rect.x + 18, panel_rect.y + y_rel, panel_rect.width - 36, 50)
            self.ui.draw_panel(self.screen, card_rect, fill_color=pygame.Color("#0E1628"), border_color=self.colors.get("panel_border", pygame.Color("#222E4A")), border_radius=6)
            self.ui.draw_text(self.screen, title, (card_rect.x + 14, card_rect.y + 7), font_size=11, color=self.colors.get("text_dim", pygame.Color("#8B9AB4")), bold=True)
            self.ui.draw_text(self.screen, value, (card_rect.x + 14, card_rect.y + 22), font_size=20, color=val_col, bold=True)

        buttons = self.get_action_buttons()
        btn_text = "HIDE PATH" if self.visualizer.show_path else "SHOW PATH (P)"
        self.ui.draw_button(self.screen, buttons["show_path"], btn_text, mouse_pos, is_active=self.visualizer.show_path, font_size=13)

        hint_cost = self.hint_manager.get_cost("path")
        self.ui.draw_button(self.screen, buttons["hint_path"], f"HINT (-{hint_cost}P)", mouse_pos, font_size=13)

        self.ui.draw_button(self.screen, buttons["new_maze"], "NEW MAZE", mouse_pos, font_size=13)
        self.ui.draw_button(self.screen, buttons["restart"], "RESET (R)", mouse_pos, font_size=13)

        legend_rect = pygame.Rect(panel_rect.x + 18, panel_rect.y + 334, panel_rect.width - 36, 230)
        self.ui.draw_panel(self.screen, legend_rect, fill_color=pygame.Color("#0E1628"), border_color=self.colors.get("panel_border", pygame.Color("#222E4A")), border_radius=6)
        self.ui.draw_text(self.screen, "TERRAIN & GATE SPECIFICATION", (legend_rect.x + 14, legend_rect.y + 10), font_size=11, color=self.colors.get("secondary", pygame.Color("#8B7CFF")), bold=True)

        legend_items = [
            ("square", "Dense Wall", "Blocked", self.colors.get("wall_border", pygame.Color("#222F4C"))),
            ("circle", "Open Path", "Cost: 1", self.colors.get("text_dim", pygame.Color("#8B9AB4"))),
            ("triangle", "Sand Dune", "Cost: 2", self.colors.get("warning", pygame.Color("#FFBF69"))),
            ("diamond", "Dense Woods", "Cost: 3", self.colors.get("success", pygame.Color("#35D39A"))),
            ("star", "Deep Mire", "Cost: 5", self.colors.get("danger", pygame.Color("#FF647C"))),
            ("hexagon", "Security Gate", "Locked", self.colors.get("warning", pygame.Color("#FFBF69"))),
        ]
        leg_y = legend_rect.y + 30
        for shape, label, cost_str, col in legend_items:
            self.ui.draw_legend_row(self.screen, legend_rect.x + 6, leg_y, shape, label, cost_str, col)
            leg_y += 26

        self.ui.draw_button(self.screen, buttons["menu"], "ESC // MAIN MENU", mouse_pos, font_size=14)

    def _render_checkpoint_overlay(self, mouse_pos: Tuple[int, int]) -> None:
        if not self.active_checkpoint or not self.active_checkpoint.question:
            return

        modal = self.get_checkpoint_modal_rect()
        q = self.active_checkpoint.question

        dim_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim_surf.fill((11, 16, 32, 215))
        self.screen.blit(dim_surf, (0, 0))

        border_col = (
            self.colors.get("success", pygame.Color("#35D39A"))
            if (self.checkpoint_feedback_mode and self.checkpoint_last_correct)
            else (
                self.colors.get("danger", pygame.Color("#FF647C"))
                if (self.checkpoint_feedback_mode and not self.checkpoint_last_correct)
                else self.colors.get("warning", pygame.Color("#FFBF69"))
            )
        )
        self.ui.draw_panel(self.screen, modal, fill_color=self.colors.get("panel", pygame.Color("#141C2F")), border_color=border_col, border_radius=10, border_width=2)

        header_text = f"CHECKPOINT {self.active_checkpoint.checkpoint_num} // SECURITY GATE"
        self.ui.draw_text(self.screen, header_text, (modal.x + 36, modal.y + 20), font_size=20, color=self.colors.get("primary", pygame.Color("#38D9FF")), bold=True)
        badge_text = f"[{q.topic} • {self.active_checkpoint.difficulty.upper()}]"
        self.ui.draw_text(self.screen, badge_text, (modal.right - 220, modal.y + 22), font_size=13, color=self.colors.get("secondary", pygame.Color("#8B7CFF")), bold=True)

        timer_rect = pygame.Rect(modal.x + 36, modal.y + 50, modal.width - 250, 8)
        timer_col = (
            self.colors.get("danger", pygame.Color("#FF647C"))
            if self.checkpoint_timer.time_remaining < 5.0
            else (
                self.colors.get("warning", pygame.Color("#FFBF69"))
                if self.checkpoint_timer.time_remaining < 10.0
                else self.colors.get("primary", pygame.Color("#38D9FF"))
            )
        )
        self.ui.draw_progress_bar(self.screen, timer_rect, self.checkpoint_timer.progress, fill_color=timer_col)
        self.ui.draw_text(self.screen, f"TIME: {self.checkpoint_timer.formatted_time}", (modal.x + 36, modal.y + 64), font_size=12, color=timer_col, bold=True)

        if not self.checkpoint_feedback_mode:
            q_hint_cost = self.hint_manager.get_cost("quiz")
            hint_50_50_rect = pygame.Rect(modal.right - 180, modal.y + 48, 140, 28)
            self.ui.draw_button(self.screen, hint_50_50_rect, f"50/50 HINT (-{q_hint_cost}P)", mouse_pos, font_size=12)

        q_pos = (modal.x + 36, modal.y + 90)
        self.ui.draw_multiline_text(self.screen, q.question, q_pos, max_width=modal.width - 72, font_size=16, color=self.colors.get("text", pygame.Color("#F3F6FC")), bold=True)

        letters = ["A", "B", "C", "D"]
        opt_y = modal.y + 195
        for i, opt_text in enumerate(q.options):
            opt_rect = pygame.Rect(modal.x + 40, opt_y + i * 54, modal.width - 80, 46)
            is_selected = (self.selected_option_index == i)
            is_eliminated = (i in self.eliminated_quiz_options)

            btn_danger = False
            btn_success = False

            if self.checkpoint_feedback_mode:
                if i == q.answer:
                    btn_success = True
                elif is_selected and not self.checkpoint_last_correct:
                    btn_danger = True

            display_opt_text = f"{letters[i]}.  [ELIMINATED BY HINT]" if is_eliminated else f"{letters[i]}.  {opt_text}"
            self.ui.draw_button(self.screen, opt_rect, display_opt_text, mouse_pos, selected=is_selected, danger=btn_danger, success=btn_success, font_size=15)

        action_btn_rect = pygame.Rect(modal.centerx - 130, modal.bottom - 56, 260, 42)

        if not self.checkpoint_feedback_mode:
            can_submit = (self.selected_option_index is not None)
            btn_label = "SUBMIT CREDENTIALS" if can_submit else "SELECT AN OPTION"
            self.ui.draw_button(self.screen, action_btn_rect, btn_label, mouse_pos, primary=can_submit, font_size=15)
        else:
            feed_col = (
                self.colors.get("success", pygame.Color("#35D39A"))
                if self.checkpoint_last_correct
                else self.colors.get("danger", pygame.Color("#FF647C"))
            )
            self.ui.draw_text(self.screen, self.checkpoint_feedback_msg, (modal.centerx, modal.bottom - 110), font_size=14, color=feed_col, bold=True, center=True)
            expl_text = f"Protocol Analysis: {q.explanation}"
            self.ui.draw_multiline_text(self.screen, expl_text, (modal.x + 40, modal.bottom - 90), max_width=modal.width - 80, font_size=13, color=self.colors.get("text_dim", pygame.Color("#8B9AB4")), bold=False)

            if self.checkpoint_last_correct:
                self.ui.draw_button(self.screen, action_btn_rect, "CONTINUE MISSION ▶", mouse_pos, success=True, font_size=15)
            else:
                self.ui.draw_button(self.screen, action_btn_rect, "RETRY GATE CHALLENGE ⟳", mouse_pos, danger=True, font_size=15)
