"""Checkpoint mechanics and cut-vertex placement for PATHFINDER.

Guarantees checkpoints are positioned along unavoidable choke-points of the maze
so they cannot be bypassed without being successfully resolved.
"""

import math
from typing import Dict, List, Optional, Set, Tuple

from src.dijkstra_solver import DijkstraSolver
from src.grid import CellType, Grid
from src.question import Question, QuestionManager


class Checkpoint:
    """Represents a tactical security gate blocking corridor passage until resolved.

    Attributes:
        checkpoint_num: Ordinal sequence number (1, 2, ...).
        pos: Grid coordinates (x, y).
        difficulty: Difficulty tier ('Easy', 'Medium', 'Hard').
        question: Assigned multiple-choice Question instance.
        is_cleared: Whether the gate has been unlocked.
        attempts: Number of answers submitted for this gate.
    """

    def __init__(
        self,
        checkpoint_num: int,
        pos: Tuple[int, int],
        difficulty: str = "Easy",
        question: Optional[Question] = None,
    ) -> None:
        self.checkpoint_num: int = checkpoint_num
        self.pos: Tuple[int, int] = pos
        self.difficulty: str = difficulty
        self.question: Optional[Question] = question
        self.is_cleared: bool = False
        self.attempts: int = 0

    @property
    def x(self) -> int:
        return self.pos[0]

    @property
    def y(self) -> int:
        return self.pos[1]

    def unlock(self) -> None:
        """Unlock the checkpoint gate."""
        self.is_cleared = True

    def reset(self) -> None:
        """Reset gate state for level restart."""
        self.is_cleared = False
        self.attempts = 0


class CheckpointManager:
    """Manages placement, cut-vertex enforcement, and verification of checkpoints."""

    def __init__(self, question_manager: QuestionManager) -> None:
        self.question_manager: QuestionManager = question_manager
        self.checkpoints: List[Checkpoint] = []

    def clear(self) -> None:
        """Clear all active checkpoints."""
        self.checkpoints.clear()

    def get_at_pos(self, pos: Tuple[int, int]) -> Optional[Checkpoint]:
        """Retrieve checkpoint at given coordinate, if present."""
        for cp in self.checkpoints:
            if cp.pos == pos:
                return cp
        return None

    def all_cleared(self) -> bool:
        """Check if all checkpoints on the current level have been unlocked."""
        return all(cp.is_cleared for cp in self.checkpoints)

    def generate_checkpoints(
        self,
        grid: Grid,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        optimal_path: List[Tuple[int, int]],
        count: int = 1,
        difficulty: str = "Easy",
        topic: Optional[str] = None,
    ) -> List[Checkpoint]:
        """Place and validate checkpoints strictly on unavoidable route bottlenecks.

        Guarantees:
        1. Checkpoint lies on reachable open cells between start and destination.
        2. Checkpoint is a cut-vertex: blocking the checkpoint cell disconnects start from end (no bypassing).
        3. Maze remains fully solvable when the checkpoint is open.
        """
        self.clear()
        if count <= 0 or len(optimal_path) <= 2:
            return []

        # Usable path nodes excluding exact start and end coordinates
        candidate_nodes = optimal_path[1:-1]
        if not candidate_nodes:
            return []

        # Find cut-vertices on the optimal path (cells whose removal leaves end unreachable from start)
        cut_vertices: List[Tuple[int, int]] = []
        for pos in candidate_nodes:
            cell = grid.get_cell(*pos)
            if cell is None or cell.is_wall:
                continue

            orig_wall = cell.is_wall
            orig_type = cell.cell_type

            # Temporarily block cell to test if destination is still reachable
            cell.is_wall = True
            test_path, test_cost, _ = DijkstraSolver.solve(grid, start_pos, end_pos)
            cell.is_wall = orig_wall
            cell.cell_type = orig_type

            if math.isinf(test_cost) or not test_path:
                cut_vertices.append(pos)

        # If natural cut-vertices are fewer than requested due to loops, convert path cells into cut-vertices
        # by blocking parallel bypass routes so that the checkpoint cell is unavoidable
        selected_positions: List[Tuple[int, int]] = []

        if len(cut_vertices) >= count:
            # Space them evenly across candidate cut vertices
            step = len(cut_vertices) / float(count)
            for i in range(count):
                idx = int(i * step + step / 2)
                selected_positions.append(cut_vertices[min(idx, len(cut_vertices) - 1)])
        else:
            # Take whatever cut-vertices exist
            selected_positions.extend(cut_vertices)

            # If still need more, pick spaced cells from optimal path and seal bypasses
            needed = count - len(selected_positions)
            remaining_candidates = [p for p in candidate_nodes if p not in selected_positions]
            if remaining_candidates:
                step = len(remaining_candidates) / float(needed)
                for i in range(needed):
                    idx = int(i * step + step / 2)
                    target_pos = remaining_candidates[min(idx, len(remaining_candidates) - 1)]

                    # Seal any bypass paths around target_pos until target_pos is an absolute bottleneck
                    target_cell = grid.get_cell(*target_pos)
                    if target_cell is None:
                        continue

                    # Repeatedly block bypass paths that circumvent target_pos
                    while True:
                        target_cell.is_wall = True
                        bypass_path, bypass_cost, _ = DijkstraSolver.solve(grid, start_pos, end_pos)
                        target_cell.is_wall = False

                        if math.isinf(bypass_cost) or not bypass_path:
                            # Successfully isolated: no bypass route exists!
                            break

                        # Block the middle non-optimal edge of the bypass to close the loop
                        sealed = False
                        for b_pos in bypass_path[1:-1]:
                            if b_pos != target_pos:
                                b_cell = grid.get_cell(*b_pos)
                                if b_cell and not b_cell.is_wall and b_pos not in optimal_path:
                                    b_cell.set_type(CellType.WALL)
                                    sealed = True
                                    break
                        if not sealed:
                            break

                    selected_positions.append(target_pos)

        # Ensure order matches path progression from start to end
        node_order = {pos: idx for idx, pos in enumerate(optimal_path)}
        selected_positions.sort(key=lambda p: node_order.get(p, 0))

        # Instantiate Checkpoints and dispense questions
        assigned_checkpoints: List[Checkpoint] = []
        for i, pos in enumerate(selected_positions, start=1):
            q = self.question_manager.get_question(topic=topic, difficulty=difficulty)
            cp = Checkpoint(
                checkpoint_num=i,
                pos=pos,
                difficulty=difficulty,
                question=q,
            )
            assigned_checkpoints.append(cp)

        self.checkpoints = assigned_checkpoints

        # Final verification: validate solvability and strict non-bypassability
        assert self.validate_solvability(grid, start_pos, end_pos), "Maze must be solvable with open checkpoints"
        for cp in self.checkpoints:
            assert self.validate_unbypassable(grid, start_pos, end_pos, cp.pos), (
                f"Checkpoint at {cp.pos} must be unbypassable"
            )

        return self.checkpoints

    def validate_solvability(
        self,
        grid: Grid,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
    ) -> bool:
        """Validate that the maze remains solvable from start to end."""
        path, cost, _ = DijkstraSolver.solve(grid, start_pos, end_pos)
        return bool(path) and not math.isinf(cost)

    def validate_unbypassable(
        self,
        grid: Grid,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        checkpoint_pos: Tuple[int, int],
    ) -> bool:
        """Verify that blocking this checkpoint cell prevents reaching the destination."""
        cell = grid.get_cell(*checkpoint_pos)
        if cell is None:
            return False

        orig_wall = cell.is_wall
        cell.is_wall = True
        path, cost, _ = DijkstraSolver.solve(grid, start_pos, end_pos)
        cell.is_wall = orig_wall

        # Unbypassable if no path exists without this cell
        return not bool(path) or math.isinf(cost)
