"""UI components and procedural rendering helpers for PATHFINDER.

Renders a warm, hand-made cartographer's table interface:
- Old map, aged parchment, and worn timber tones
- Georgia serif headings and Segoe UI body typography
- Flat buttons with subtle 1px elevation hover shift (no glows or neon halos)
- Crisp mathematical shape glyphs for terrain & solid hexagon checkpoints
- Zero hardcoded hex colors in source; reads dynamically from palette
"""

import math
from typing import Dict, List, Optional, Tuple
import pygame

# Default RGB color tuples for fallback without hardcoded hex strings
DEFAULT_PALETTE = {
    "background": pygame.Color(30, 26, 22),       # dark walnut
    "panel": pygame.Color(42, 36, 30),            # leather brown
    "panel_border": pygame.Color(74, 63, 51),     # worn wood
    "wall": pygame.Color(59, 50, 41),             # dark clay
    "text": pygame.Color(237, 228, 211),          # cream paper
    "text_muted": pygame.Color(168, 156, 136),    # faded parchment
    "text_dim": pygame.Color(168, 156, 136),      # alias
    "accent": pygame.Color(217, 130, 43),         # burnt amber
    "primary": pygame.Color(217, 130, 43),        # alias
    "accent_alt": pygame.Color(168, 116, 90),     # clay rose
    "secondary": pygame.Color(168, 116, 90),      # alias
    "optimal_path": pygame.Color(227, 176, 75),   # mustard
    "success": pygame.Color(127, 166, 80),        # leaf green
    "warning": pygame.Color(227, 176, 75),        # saffron
    "danger": pygame.Color(192, 70, 47),          # brick red
}


class UIManager:
    """Stateless UI rendering helper for cartographer console panels, buttons, and glyphs."""

    def __init__(self, colors: Dict[str, pygame.Color]) -> None:
        """Initialize UIManager with application color palette."""
        self.colors: Dict[str, pygame.Color] = colors
        self._font_cache: Dict[Tuple[int, bool, bool], pygame.font.Font] = {}

    def _color(self, key: str, fallback_key: str = "text") -> pygame.Color:
        """Safely retrieve color from active palette without hardcoded hex strings."""
        if key in self.colors:
            return self.colors[key]
        if fallback_key in self.colors:
            return self.colors[fallback_key]
        return DEFAULT_PALETTE.get(key, DEFAULT_PALETTE.get(fallback_key, pygame.Color(200, 200, 200)))

    def get_font(self, size: int, bold: bool = False, heading: bool = False) -> pygame.font.Font:
        """Retrieve a cached Pygame font with robust system fallback.

        Headings use Georgia (or serif fallback). Body text uses Segoe UI (or system fallback).
        Gracefully falls back to Pygame's default font if system fonts are unavailable.
        """
        key = (size, bold, heading)
        if key in self._font_cache:
            return self._font_cache[key]

        font: Optional[pygame.font.Font] = None
        font_candidates = (
            ("georgia", "timesnewroman", "garamond", "cambria", "serif")
            if heading
            else ("segoeui", "calibri", "helvetica", "arial", "sans-serif")
        )

        try:
            available_fonts = set(pygame.font.get_fonts())
            for font_name in font_candidates:
                if font_name in available_fonts:
                    font = pygame.font.SysFont(font_name, size, bold=bold)
                    break
            if font is None:
                fallback_target = "georgia" if heading else "segoeui"
                font = pygame.font.SysFont(fallback_target, size, bold=bold)
        except Exception:
            font = None

        if font is None:
            font = pygame.font.Font(None, size)

        self._font_cache[key] = font
        return font

    def draw_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        fill_color: Optional[pygame.Color] = None,
        border_color: Optional[pygame.Color] = None,
        border_radius: int = 4,
        border_width: int = 1,
    ) -> None:
        """Render a clean flat panel with slim 1-2px worn wood border."""
        fill = fill_color or self._color("panel", "background")
        border = border_color or self._color("panel_border")

        pygame.draw.rect(surface, fill, rect, border_radius=border_radius)
        if border_width > 0:
            pygame.draw.rect(surface, border, rect, width=border_width, border_radius=border_radius)

    def draw_button(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        text: str,
        mouse_pos: Tuple[int, int],
        is_active: bool = False,
        primary: bool = False,
        font_size: int = 17,
        selected: bool = False,
        danger: bool = False,
        success: bool = False,
        heading: bool = False,
    ) -> bool:
        """Render a tactile flat button with subtle hover shift and no neon glow."""
        is_hover = rect.collidepoint(mouse_pos)
        font = self.get_font(font_size, bold=True, heading=heading)

        # Subtle 1px elevation shift on hover for hand-made physical button feel
        draw_rect = rect.move(0, -1) if is_hover else rect

        if danger:
            base_bg = self._color("danger")
            bg_color = (
                max(0, base_bg.r - 20),
                max(0, base_bg.g - 15),
                max(0, base_bg.b - 15),
            ) if is_hover else base_bg
            border_color = self._color("panel_border")
            text_color = self._color("text")
        elif success:
            base_bg = self._color("success")
            bg_color = (
                max(0, base_bg.r - 20),
                max(0, base_bg.g - 15),
                max(0, base_bg.b - 15),
            ) if is_hover else base_bg
            border_color = self._color("panel_border")
            text_color = self._color("background")
        elif primary:
            base_bg = self._color("accent", "primary")
            bg_color = (
                max(0, base_bg.r - 20),
                max(0, base_bg.g - 15),
                max(0, base_bg.b - 15),
            ) if is_hover else base_bg
            border_color = self._color("panel_border")
            text_color = self._color("background")
        elif selected:
            bg_color = self._color("panel_border")
            border_color = self._color("accent", "primary")
            text_color = self._color("accent", "primary")
        else:
            base_panel = self._color("panel")
            bg_color = (
                max(0, base_panel.r - 12),
                max(0, base_panel.g - 10),
                max(0, base_panel.b - 8),
            ) if is_hover else base_panel
            border_color = (
                self._color("accent", "primary")
                if (is_hover or is_active)
                else self._color("panel_border")
            )
            text_color = (
                self._color("accent", "primary")
                if is_hover
                else self._color("text")
            )

        # Flat button fill (4px radius, no glow halo)
        pygame.draw.rect(surface, bg_color, draw_rect, border_radius=4)
        border_w = 2 if selected else 1
        pygame.draw.rect(surface, border_color, draw_rect, width=border_w, border_radius=4)

        txt_surf = font.render(text, True, text_color)
        txt_rect = txt_surf.get_rect(center=draw_rect.center)
        surface.blit(txt_surf, txt_rect)

        return is_hover

    def draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        pos: Tuple[int, int],
        font_size: int = 16,
        color: Optional[pygame.Color] = None,
        bold: bool = False,
        center: bool = False,
        heading: bool = False,
    ) -> pygame.Rect:
        """Render single line text at specified position with optional serif heading."""
        font = self.get_font(font_size, bold=bold, heading=heading)
        col = color or self._color("text")
        txt_surf = font.render(text, True, col)
        rect = txt_surf.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        surface.blit(txt_surf, rect)
        return rect

    def draw_multiline_text(
        self,
        surface: pygame.Surface,
        text: str,
        pos: Tuple[int, int],
        max_width: int,
        font_size: int = 16,
        color: Optional[pygame.Color] = None,
        line_spacing: int = 4,
        bold: bool = False,
        center: bool = False,
        heading: bool = False,
    ) -> int:
        """Render wrapped multi-line text and return total height consumed."""
        font = self.get_font(font_size, bold=bold, heading=heading)
        col = color or self._color("text")

        words = text.split(" ")
        lines: List[str] = []
        current_line: List[str] = []

        for word in words:
            test_line = " ".join(current_line + [word])
            w, _ = font.size(test_line)
            if w <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        line_h = font.get_linesize()
        curr_y = pos[1]
        for line in lines:
            line_surf = font.render(line, True, col)
            line_rect = line_surf.get_rect()
            if center:
                line_rect.centerx = pos[0]
                line_rect.y = curr_y
            else:
                line_rect.x = pos[0]
                line_rect.y = curr_y
            surface.blit(line_surf, line_rect)
            curr_y += line_h + line_spacing

        return curr_y - pos[1]

    def draw_progress_bar(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        progress: float,
        fill_color: Optional[pygame.Color] = None,
        bg_color: Optional[pygame.Color] = None,
        border_color: Optional[pygame.Color] = None,
    ) -> None:
        """Render a flat parchment progress/countdown bar gauge."""
        bg = bg_color or self._color("background")
        border = border_color or self._color("panel_border")
        fill = fill_color or self._color("accent", "primary")

        pygame.draw.rect(surface, bg, rect, border_radius=4)
        progress_clamped = max(0.0, min(1.0, progress))

        if progress_clamped > 0.0:
            fill_w = max(4, int(rect.width * progress_clamped))
            fill_rect = pygame.Rect(rect.x, rect.y, fill_w, rect.height)
            pygame.draw.rect(surface, fill, fill_rect, border_radius=4)

        pygame.draw.rect(surface, border, rect, width=1, border_radius=4)

    def draw_shape_glyph(
        self,
        surface: pygame.Surface,
        center: Tuple[int, int],
        shape: str,
        size: int,
        color: pygame.Color,
        filled: bool = True,
        outline_color: Optional[pygame.Color] = None,
    ) -> None:
        """Draw mathematical procedural glyphs without Unicode character dependency.

        Supported shapes:
        - 'square': Solid barrier / wall
        - 'circle': Open terrain (weight 1)
        - 'triangle': Sand terrain (weight 2)
        - 'diamond': Forest terrain (weight 3)
        - 'star': Deep mire terrain (weight 5)
        - 'hexagon': Checkpoint gate (solid with dark outline)
        """
        cx, cy = center
        half = size / 2.0
        border_col = outline_color or self._color("panel_border")

        if shape == "square":
            rect = pygame.Rect(int(cx - half), int(cy - half), int(size), int(size))
            if filled:
                pygame.draw.rect(surface, color, rect, border_radius=2)
            else:
                pygame.draw.rect(surface, color, rect, width=1, border_radius=2)

        elif shape == "circle":
            if filled:
                pygame.draw.circle(surface, color, (int(cx), int(cy)), int(half))
            else:
                pygame.draw.circle(surface, color, (int(cx), int(cy)), int(half), width=1)

        elif shape == "triangle":
            pts = [
                (cx, cy - half),
                (cx + half, cy + half * 0.9),
                (cx - half, cy + half * 0.9),
            ]
            if filled:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts])
            else:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts], width=1)

        elif shape == "diamond":
            pts = [
                (cx, cy - half),
                (cx + half, cy),
                (cx, cy + half),
                (cx - half, cy),
            ]
            if filled:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts])
            else:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts], width=1)

        elif shape == "star":
            # 5-pointed star polygon
            pts = []
            outer_r = half
            inner_r = half * 0.45
            for i in range(10):
                angle = (i * math.pi / 5.0) - (math.pi / 2.0)
                r = outer_r if i % 2 == 0 else inner_r
                pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

            if filled:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts])
            else:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts], width=1)

        elif shape == "hexagon":
            # Regular hexagon polygon for checkpoints: solid fill + dark outline
            pts = []
            for i in range(6):
                angle = i * math.pi / 3.0
                pts.append((cx + half * math.cos(angle), cy + half * math.sin(angle)))

            int_pts = [(int(px), int(py)) for px, py in pts]
            if filled:
                pygame.draw.polygon(surface, color, int_pts)
                pygame.draw.polygon(surface, border_col, int_pts, width=1)
            else:
                pygame.draw.polygon(surface, color, int_pts, width=1)

    def draw_legend_row(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        shape: str,
        label: str,
        weight_str: str,
        shape_color: pygame.Color,
    ) -> None:
        """Render a clean legend row with shape glyph, title label, and cost indicator."""
        # Draw glyph centered at (x + 12, y + 10)
        self.draw_shape_glyph(surface, (x + 12, y + 10), shape, size=14, color=shape_color, filled=True)

        # Label text
        self.draw_text(
            surface,
            label,
            (x + 32, y + 2),
            font_size=14,
            color=self._color("text"),
        )

        # Cost tag
        if weight_str == "Cost: 1":
            cost_col = self._color("accent", "primary")
        elif "5" in weight_str:
            cost_col = self._color("danger")
        elif "Wall" in label:
            cost_col = self._color("text_muted", "text_dim")
        else:
            cost_col = self._color("warning")

        self.draw_text(
            surface,
            weight_str,
            (x + 180, y + 2),
            font_size=13,
            color=cost_col,
            bold=True,
        )

    def draw_glow_circle(
        self,
        surface: pygame.Surface,
        center: Tuple[int, int],
        core_radius: int,
        color: pygame.Color,
        pulse_alpha: int = 180,
    ) -> None:
        """Render flat token circle with clean dark outline (no glow/halos)."""
        cx, cy = center
        border_col = self._color("panel_border")

        # Flat solid token with dark border
        pygame.draw.circle(surface, color, (cx, cy), core_radius)
        pygame.draw.circle(surface, border_col, (cx, cy), core_radius, width=1)
