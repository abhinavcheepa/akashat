"""UI components and procedural rendering helpers for PATHFINDER.

Renders dark-navy futuristic console panels, buttons with hover/glow states,
typography with graceful system-font fallbacks, mathematical shape glyphs
(square, circle, triangle, diamond, star polygon) without relying on Unicode fonts,
multi-line wrapped text, and animated countdown gauges.
"""

import math
from typing import Dict, List, Optional, Tuple
import pygame


class UIManager:
    """Stateless UI rendering helper for console panels, buttons, and procedural glyphs."""

    def __init__(self, colors: Dict[str, pygame.Color]) -> None:
        """Initialize UIManager with application color palette."""
        self.colors: Dict[str, pygame.Color] = colors
        self._font_cache: Dict[Tuple[int, bool], pygame.font.Font] = {}

    def get_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        """Retrieve a cached Pygame font with robust system fallback.

        Attempts to load 'segoeui' or 'consolas', then gracefully falls back
        to Pygame's default system font if unavailable.
        """
        key = (size, bold)
        if key in self._font_cache:
            return self._font_cache[key]

        font: Optional[pygame.font.Font] = None
        try:
            # Try futuristic/clean modern system fonts
            for font_name in ("segoeui", "consolas", "helvetica", "arial"):
                if font_name in pygame.font.get_fonts():
                    font = pygame.font.SysFont(font_name, size, bold=bold)
                    break
            if font is None:
                font = pygame.font.SysFont("segoeui", size, bold=bold)
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
        border_radius: int = 8,
        border_width: int = 1,
    ) -> None:
        """Render a rounded console panel with subtle border framing."""
        fill = fill_color or self.colors.get("panel", pygame.Color("#141C2F"))
        border = border_color or self.colors.get("panel_border", pygame.Color("#222E4A"))

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
        font_size: int = 18,
        selected: bool = False,
        danger: bool = False,
        success: bool = False,
    ) -> bool:
        """Render an interactive navigation console button with hover detection.

        Returns True if mouse_pos hovers over the button rect.
        """
        is_hover = rect.collidepoint(mouse_pos)
        font = self.get_font(font_size, bold=True)

        if danger:
            base_bg = self.colors.get("danger", pygame.Color("#FF647C"))
            bg_color = (min(255, base_bg.r + 20), min(255, base_bg.g + 20), min(255, base_bg.b + 20)) if is_hover else base_bg
            border_color = pygame.Color("#FFFFFF") if is_hover else base_bg
            text_color = self.colors.get("text", pygame.Color("#F3F6FC"))
        elif success:
            base_bg = self.colors.get("success", pygame.Color("#35D39A"))
            bg_color = (min(255, base_bg.r + 20), min(255, base_bg.g + 20), min(255, base_bg.b + 20)) if is_hover else base_bg
            border_color = pygame.Color("#FFFFFF") if is_hover else base_bg
            text_color = self.colors.get("background", pygame.Color("#0B1020"))
        elif primary:
            base_bg = self.colors.get("primary", pygame.Color("#38D9FF"))
            bg_color = (min(255, base_bg.r + 20), min(255, base_bg.g + 20), min(255, base_bg.b + 20)) if is_hover else base_bg
            border_color = pygame.Color("#FFFFFF") if is_hover else base_bg
            text_color = self.colors.get("background", pygame.Color("#0B1020"))
        elif selected:
            bg_color = pygame.Color("#1B3354")
            border_color = self.colors.get("primary", pygame.Color("#38D9FF"))
            text_color = self.colors.get("primary", pygame.Color("#38D9FF"))
        else:
            base_panel = self.colors.get("panel", pygame.Color("#141C2F"))
            bg_color = pygame.Color("#1D2840") if is_hover else base_panel
            border_color = self.colors.get("primary", pygame.Color("#38D9FF")) if (is_hover or is_active) else self.colors.get("panel_border", pygame.Color("#222E4A"))
            text_color = self.colors.get("primary", pygame.Color("#38D9FF")) if is_hover else self.colors.get("text", pygame.Color("#F3F6FC"))

        # Outer glow on hover or selected
        if is_hover or selected:
            glow_rect = rect.inflate(4, 4)
            glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(
                glow_surf,
                (*border_color[:3], 45),
                glow_surf.get_rect(),
                border_radius=8,
            )
            surface.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(surface, bg_color, rect, border_radius=6)
        border_w = 2 if selected else 1
        pygame.draw.rect(surface, border_color, rect, width=border_w, border_radius=6)

        txt_surf = font.render(text, True, text_color)
        txt_rect = txt_surf.get_rect(center=rect.center)
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
    ) -> pygame.Rect:
        """Render single line text at specified position."""
        font = self.get_font(font_size, bold=bold)
        col = color or self.colors.get("text", pygame.Color("#F3F6FC"))
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
    ) -> int:
        """Render wrapped multi-line text and return total height consumed."""
        font = self.get_font(font_size, bold=bold)
        col = color or self.colors.get("text", pygame.Color("#F3F6FC"))

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
        """Render a rounded progress/countdown bar gauge."""
        bg = bg_color or pygame.Color("#0D1627")
        border = border_color or self.colors.get("panel_border", pygame.Color("#222E4A"))
        fill = fill_color or self.colors.get("primary", pygame.Color("#38D9FF"))

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
    ) -> None:
        """Draw mathematical procedural glyphs without Unicode character dependency.

        Supported shapes:
        - 'square': Solid barrier / wall
        - 'circle': Open terrain (weight 1)
        - 'triangle': Sand terrain (weight 2)
        - 'diamond': Forest terrain (weight 3)
        - 'star': Deep swamp terrain (weight 5)
        - 'hexagon': Checkpoint gate
        """
        cx, cy = center
        half = size / 2.0

        if shape == "square":
            rect = pygame.Rect(int(cx - half), int(cy - half), int(size), int(size))
            if filled:
                pygame.draw.rect(surface, color, rect, border_radius=2)
            else:
                pygame.draw.rect(surface, color, rect, width=2, border_radius=2)

        elif shape == "circle":
            if filled:
                pygame.draw.circle(surface, color, (int(cx), int(cy)), int(half))
            else:
                pygame.draw.circle(surface, color, (int(cx), int(cy)), int(half), width=2)

        elif shape == "triangle":
            pts = [
                (cx, cy - half),
                (cx + half, cy + half * 0.9),
                (cx - half, cy + half * 0.9),
            ]
            if filled:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts])
            else:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts], width=2)

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
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts], width=2)

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
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts], width=2)

        elif shape == "hexagon":
            # Regular hexagon polygon for checkpoints
            pts = []
            for i in range(6):
                angle = i * math.pi / 3.0
                pts.append((cx + half * math.cos(angle), cy + half * math.sin(angle)))

            if filled:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts])
            else:
                pygame.draw.polygon(surface, color, [(int(px), int(py)) for px, py in pts], width=2)

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
            color=self.colors.get("text", pygame.Color("#F3F6FC")),
        )

        # Cost tag
        cost_col = self.colors.get("primary", pygame.Color("#38D9FF")) if weight_str == "Cost: 1" else self.colors.get("warning", pygame.Color("#FFBF69"))
        if "5" in weight_str:
            cost_col = self.colors.get("danger", pygame.Color("#FF647C"))
        elif "Wall" in label:
            cost_col = self.colors.get("text_dim", pygame.Color("#8B9AB4"))

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
        """Draw layered alpha circles creating a soft futuristic neon glow."""
        cx, cy = center
        glow_surf = pygame.Surface((core_radius * 8, core_radius * 8), pygame.SRCALPHA)
        gcx, gcy = core_radius * 4, core_radius * 4

        # Outer soft halo
        pygame.draw.circle(glow_surf, (*color[:3], int(pulse_alpha * 0.15)), (gcx, gcy), core_radius * 3)
        # Mid halo
        pygame.draw.circle(glow_surf, (*color[:3], int(pulse_alpha * 0.35)), (gcx, gcy), int(core_radius * 2))
        # Inner aura
        pygame.draw.circle(glow_surf, (*color[:3], int(pulse_alpha * 0.7)), (gcx, gcy), int(core_radius * 1.3))

        surface.blit(glow_surf, (cx - gcx, cy - gcy))

        # Core solid
        pygame.draw.circle(surface, color, (cx, cy), core_radius)
        # Highlight pip
        pygame.draw.circle(surface, pygame.Color("#FFFFFF"), (cx - core_radius // 3, cy - core_radius // 3), max(1, core_radius // 3))
