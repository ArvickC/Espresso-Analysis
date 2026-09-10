import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import pygame
import re
import math
from typing import TYPE_CHECKING
import logging
from logging_setup import configure_logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from labeling_handler import ShotDefaults

RES = (320, 240)
HEADER_POS = (-1, 1)
FOOTER_POS = (-1, RES[1] - 22)
SCALE = 3 # scale up for debug viewing
FPS = 30

# Colors
GREEN = (60, 255, 120)
AMBER = (255, 176, 0)
CYAN = (100, 188, 219)
RED = (230, 48, 48)
BG_COLOR = (8, 8, 10)
GRID = (30, 30, 30)

FLOW_WINDOW_S = 0.25 # lookback window for crude flow-rate calculation
DEFAULT_RESET_TIMEOUT = 120 # seconds before idle again
CROSSFADE_DURATION = 0.25 # seconds

GRAPH_SPLIT_FRACTION = 0.5
GRAPH_DOCK_MARGIN = 6 # px

# -- Asset Directories --
BG_DIR = 'assets/bg.png'
IDLE_DIR = 'assets/idle'
BOOT_DIR = 'assets/boot'
BEANS_DIR = 'assets/beans.png'
DOSE_DIR = 'assets/dose'
GRIND_DIR = 'assets/grind'
WDT_DIR = 'assets/wdt'
LEVEL_DIR = 'assets/level'
TAMP_DIR = 'assets/tamp'
TARING_DIR = 'assets/puck_prep'
SPRAY_DIR = 'assets/spray.png'

class State(Enum):
    IDLE = auto()
    BOOT = auto()
    LABELING = auto()
    DOSING = auto()
    PREPPING = auto()
    TARING = auto()
    LOGGING = auto()
    POST_LABELING = auto()
    PREDICTED_LABEL = auto()
    PREDICTED_GRIND = auto()
    SUMMARY = auto()

# States where the graph is docked to the bottom of the screen
GRAPH_DOCKED_STATES = {State.POST_LABELING}

@dataclass
class AppState:
    state: State = State.IDLE
    boot_frame: int = 0
    # (elapsed_s, weight_g, flow_g_s)
    live_points: list[tuple[float, float, float]] = field(default_factory=list)
    dose: float | None = None
    result_label: str | None = None
    result_probs: dict[str, float] | None = None
    previous_grind_rec: str | None = None
    manifest_path: Path | None = None
    gp_result: dict | None = None
    bean_name: str | None = None
    grind_rec: float | None = None
    pred_time: float | None = None
    pred_time_uncertainty: float | None = None
    quit: bool = False
    result_timeout: float = DEFAULT_RESET_TIMEOUT
    start_event: asyncio.Event = field(default_factory=asyncio.Event)
    key_down_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Pre shot label
    form_fields: list = field(default_factory=list)
    form_active_index: int = 0
    form_result: dict | None = None
    form_submit_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Post shot label
    choice_prompt: str = ""
    choice_options: list = field(default_factory=list)
    choice_active_index: int = 0
    choice_result: str | None = None
    choice_submit_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Animations
    bg_image: "pygame.Surface | None" = None
    idle_anim: "Animation | None" = None
    boot_anim: "Animation | None" = None
    dose_anim: "Animation | None" = None
    grind_anim: "Animation | None" = None
    wdt_anim: "Animation | None" = None
    level_anim: "Animation | None" = None
    tamp_anim: "Animation | None" = None
    tare_anim: "Animation | None" = None

    # 0 = spray beans, 1 = grind beans, 2 = wdt, 3 = level, 4 = tamp
    puck_prep_state: int | None = None
    state_entered_at: float | None = None

    def add_point(self, elapsed: float, weight: float) -> None:
        """
        Adds a new point to the live graph, calculating flow rate
        based on the last point that is at least FLOW_WINDOW_S seconds ago.
        :param elapsed: Elapsed time in seconds (x-axis)
        :param weight: Weight in grams (y-axis)
        """
        flow = 0.0
        for t, w, _ in reversed(self.live_points):
            dt = elapsed - t
            if dt >= FLOW_WINDOW_S:
                flow = (weight - w) / dt
                break
        self.live_points.append((elapsed, weight, flow))

class Animation:
    """
    Simple animation class that cycles through a list of frames at a given FPS.
    """
    def __init__(self, frames: list[pygame.Surface], fps: float = 24.0, loop: bool = True):
        self.frames = frames
        self.frame_duration = 1.0/fps
        self.loop = loop
        self.elapsed = 0.0

    def reset(self) -> None:
        self.elapsed = 0.0

    def update(self, dt: float) -> None:
        self.elapsed += dt

    def finished(self) -> bool:
        return not self.loop and self.elapsed >= self.frame_duration * len(self.frames)

    def current_frame(self) -> pygame.Surface:
        """
        Returns the current frame based on elapsed time.
        """
        idx = int(self.elapsed / self.frame_duration)
        idx = idx % len(self.frames) if self.loop else min(idx, len(self.frames) - 1)
        return self.frames[idx]

def text(surface, font, s, pos, color, alpha: int = 255) -> None:
    """
    Renders text on the given surface at the specified position with the given color and alpha transparency.
    """
    rendered = font.render(s, False, color)
    rendered.set_alpha(alpha)
    if pos[0] < 0 and pos[1] < 0: # center along x-axis and y-axis
        rect = rendered.get_rect(midtop=(surface.get_width() // 2, surface.get_height() // 2))
        surface.blit(rendered, rect)
    elif pos[0] < 0: # center along x-axis
        rect = rendered.get_rect(midtop=(surface.get_width() // 2, pos[1]))
        surface.blit(rendered, rect)
    elif pos[1] < 0: # center along y-axis
        rect = rendered.get_rect(midbottom=(pos[0], surface.get_height() // 2))
        surface.blit(rendered, rect)
    else:
        surface.blit(rendered, pos)


def draw_text_outside_circle(surface, font, s, center, radius, color, alpha: int = 255,
                            angle_deg: float = 0.0) -> None:
    """
    Draws text outside a circle at a given angle.
    """
    rendered = font.render(s, False, color)
    rendered.set_alpha(alpha)
    w, h = rendered.get_size()
    angle = math.radians(angle_deg)
    edge_x = center[0] + radius * math.cos(angle)
    edge_y = center[1] + radius * math.sin(angle)

    if math.cos(angle) < 0:
        # left side: keep the text fully outside and make the text's right edge touch the circle
        x = int(edge_x - w)
        y = int(edge_y - h) if math.sin(angle) > 0 else int(edge_y)
    else:
        # right side: keep the text fully outside and make the text's left edge touch the circle
        x = int(edge_x)
        y = int(edge_y - h) if math.sin(angle) > 0 else int(edge_y)

    surface.blit(rendered, (x, y))

def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path.stem)]

def _dim(color, factor=0.35):
    return tuple(int(c * factor) for c in color)

async def request_form(app: AppState, fields: list[tuple[str, str]]) -> dict[str, str] | None:
    """
    Requests a form to be filled out by the user.
    The form consists of a list of (label, default_value) pairs.
    :param app: The application state.
    :param fields: A list of (label, default_value) pairs representing the form fields.
    :return: A dictionary mapping field labels to user-provided values, or None if the form was canceled.
    """
    app.form_fields = [[label, default] for label, default in fields]
    app.form_active_index = 0
    app.form_result = None
    app.form_submit_event.clear()
    app.state = State.LABELING

    await app.form_submit_event.wait()
    return app.form_result

async def request_choice(app: AppState, prompt: str, choices: list[str]) -> str | None:
    """
    Requests the user to make a choice from a list of options.
    :param app: The application state.
    :param prompt: The prompt to display to the user.
    :param choices: A list of choice options.
    :return: The choice selected by the user, or None if the choice was canceled.
    """
    app.choice_prompt = prompt
    app.choice_options = list(choices)
    app.choice_active_index = 0
    app.choice_result = None
    app.choice_submit_event.clear()
    app.state = State.POST_LABELING

    await app.choice_submit_event.wait()
    return app.choice_result

def load_frame_sequence(directory: str, size: tuple[int, int] | None = None) -> list[pygame.Surface]:
    """
    Loads a sequence of PNG frames from a directory and optionally resizes them.
    :param directory: The directory containing PNG frames.
    :param size: Optional tuple specifying the (width, height) to resize the frames to.
    :return: A list of loaded (and optionally resized) pygame.Surface objects.
    """
    paths = sorted(Path(directory).glob("*.png"), key=_natural_key)
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {directory}")

    frames = []
    for p in paths:
        img = pygame.image.load(str(p)).convert_alpha()
        if size is not None:
            img = pygame.transform.scale(img, size)
        frames.append(img)
    return frames

def _draw_animated(surface, app, font, anim: "Animation | None", dt: float,
                    fallback_label: str, fallback_color, fallback_progress: bool = False) -> None:
    """
    Draws an animated sequence on the given surface, or a fallback label and progress
    if the animation is not available.
    :param surface: The pygame surface to draw on.
    :param app: The application state.
    :param font: The font to use for rendering text.
    :param anim: The animation to draw, or None to use the fallback.
    :param dt: The time delta since the last frame.
    :param fallback_label: The label to display if the animation is not available.
    :param fallback_color: The color to use for the fallback label and progress.
    :param fallback_progress: Whether to display a progress bar for the fallback.
    """
    if anim is not None:
        anim.update(dt)
        frame = anim.current_frame()
        surface.blit(frame, ((RES[0] - frame.get_width()) // 2,
                              (RES[1] - frame.get_height()) // 2))
    else:
        # fallback so the app still runs sensibly before assets exist
        text(surface, font, fallback_label, (10, 10), fallback_color)
        if fallback_progress:
            w = min(RES[0], app.boot_frame * 8)
            pygame.draw.rect(surface, fallback_color, (0, RES[1] // 2 - 1, w, 2))
            app.boot_frame += 1

def draw_idle(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    _draw_animated(surface, app, font, app.idle_anim, dt, "READY", GREEN)

def draw_boot(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    _draw_animated(surface, app, font, app.boot_anim, dt, "CONNECTING...", AMBER)

def draw_labeling(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    bg = load_background(BEANS_DIR, RES)
    surface.blit(bg, (0, 0))

    text(surface, font, "LABEL SHOT", HEADER_POS, AMBER)
    text(surface, font, "ENTER: next field", FOOTER_POS, GRID)

    cursor = "_" if int(time.monotonic() * 2) % 2 == 0 else " "  # ~2Hz blink
    y = 30
    for i, (label, value) in enumerate(app.form_fields):
        active = i == app.form_active_index
        color = GREEN if active else AMBER
        shown_value = value + cursor if active else value
        text(surface, font, f"{label}:", (29, y), color)
        text(surface, font, shown_value, (29, y + 18), color)
        y += 48

def draw_dosing(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    _draw_animated(surface, app, font, app.dose_anim, dt, "DOSE", AMBER)
    text(surface, font, f"{app.dose}g", (-1, 160), AMBER) # placeholder

def draw_prepping(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    if app.puck_prep_state is None or app.puck_prep_state < 0 or app.puck_prep_state > 4:
        text(surface, font, "ERR", HEADER_POS, AMBER)
    elif app.puck_prep_state == 0:
        # text(surface, font, "SPRAY BEANS", HEADER_POS, AMBER)
        frame = load_background(SPRAY_DIR, RES)
        surface.blit(frame, (0, 0))
    elif app.puck_prep_state == 1:
        # text(surface, font, "GRIND BEANS", HEADER_POS, AMBER)
        _draw_animated(surface, app, font, app.grind_anim, dt, "GRIND BEANS", AMBER)
        text(surface, font, f"PREVIOUS REC: {app.previous_grind_rec}", (-1, 160), GREEN)
    elif app.puck_prep_state == 2:
        # text(surface, font, "WDT", HEADER_POS, AMBER)
        _draw_animated(surface, app, font, app.wdt_anim, dt, "DISTRIBUTE BEANS", AMBER)
    elif app.puck_prep_state == 3:
        # text(surface, font, "LEVEL", HEADER_POS, AMBER)
        _draw_animated(surface, app, font, app.level_anim, dt, "LEVEL PUCK", AMBER)
    elif app.puck_prep_state == 4:
        # text(surface, font, "TAMP", HEADER_POS, AMBER)
        _draw_animated(surface, app, font, app.tamp_anim, dt, "TAMP PUCK", AMBER)

def draw_tare_scale(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    _draw_animated(surface, app, font, app.tare_anim, dt, "TARE SCALE...", AMBER)
    text(surface, font, "TARE SCALE", HEADER_POS, AMBER)

def draw_logging(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    text(surface, font, "PULLING SHOT", HEADER_POS, AMBER)
    draw_live_graph(surface, app.dose, app.live_points, rect=(26, 27, RES[0] - 53, RES[1] - 53), font=font,
                    fixed_scale=True, tick_labels=False)

def draw_live_graph(surface, dose, points, rect, font, gradient = True,
                    fixed_scale=False, tick_labels = True, grid = True) -> None:
    """
    Draws weight (green, left scale) and flow-rate (cyan, right scale) on
    the same plot area. Each series is auto-scaled independently.
    :param surface: The pygame surface to draw on.
    :param dose: The target dose in grams (optional).
    :param points: A list of (elapsed_s, weight_g, flow_g_s) tuples.
    :param rect: A tuple (x, y, width, height) defining the drawing area.
    :param font: The pygame font to use for text.
    :param gradient: Whether to use a gradient for the lines.
    :param fixed_scale: Whether to use a fixed scale for the axes.
    :param tick_labels: Whether to draw tick labels.
    """
    x0, y0, w, h = rect
    if len(points) < 2:
        return

    ts = [p[0] for p in points]
    weights = [p[1] for p in points]
    flows = [p[2] for p in points]

    if not fixed_scale:
        max_t = max(ts) or 1.0
        max_w = max(weights) or 1.0
        min_f, max_f = min(flows + [0.0]), max(flows + [0.1])
        cur_f, cur_w = flows[-1], weights[-1]
        span_f = (max_f - min_f) or 1.0
    else:
        max_t = max(max(ts), 30.0)
        max_w = max(max(weights), 40.0)
        min_f = min(flows + [0.0])
        max_f = max(max(flows + [1.0]), 2.0)
        cur_f, cur_w = flows[-1], weights[-1]
        span_f = (max_f - min_f) or 1.0

    def to_x(t) -> int:
        """
        Maps a time value to an x-coordinate in the graph area.
        """
        return x0 + int((t / max_t) * w)

    def y_of_w(wt) -> int:
        """
        Maps a weight value to a y-coordinate in the graph area.
        """
        return y0 + h - int((wt / max_w) * h)

    def y_of_f(f) -> int:
        """
        Maps a flow value to a y-coordinate in the graph area.
        """
        return y0 + h - int(((f - min_f) / span_f) * h)

    def _moving_average(values, window=5) -> list[float]:
        """
        Computes the moving average of a list of values.
        :param values: The list of values.
        :param window: The window size for the moving average.
        :return: A list of the moving average values.
        """
        if window <= 1 or len(values) < 2:
            return values[:]
        out = []
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            chunk = values[lo:i + 1]
            out.append(sum(chunk) / len(chunk))
        return out

    def _catmull_rom_points(pts, segments=8) -> list[tuple[float, float]]:
        """
        Computes the Catmull-Rom spline points for a given set of points.
        :param pts: The list of points.
        :param segments: The number of segments between each pair of points.
        :return: A list of the interpolated points.
        """
        if len(pts) < 3:
            return pts

        def catmull(p0, p1, p2, p3, t):
            t2, t3 = t * t, t * t * t
            x = 0.5 * (2 * p1[0] + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * (2 * p1[1] + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            return x, y

        padded = [pts[0]] + pts + [pts[-1]]
        out = []
        for i in range(len(padded) - 3):
            p0, p1, p2, p3 = padded[i:i + 4]
            for s in range(segments):
                t = s / segments
                out.append(catmull(p0, p1, p2, p3, t))
        out.append(pts[-1])
        return out

    def _lerp_color(c1, c2, t) -> tuple[int, ...]:
        """
        Linearly interpolates between two colors.
        """
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def _proximity(value, target, tol) -> float:
        """
        Computes a proximity factor between 0 and 1 based on
        how close value is to target within a tolerance.
        """
        if tol <= 0:
            return 1.0 if value == target else 0.0
        d = abs(target - value)
        return max(0.0, 1 - d / tol)

    def _draw_grad_line_by_y(surface, pts, y_target, y_tol,
                             base_color, full_color, width=1) -> None:
        """
        Draws a line with a color gradient based on the y-coordinate proximity to a target value.
        """
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            avg_y = (y1 + y2) / 2
            t = _proximity(avg_y, y_target, y_tol)
            color = _lerp_color(base_color, full_color, t)
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), width)

    def _frange(start, stop, step):
        """
        Generates a range of floating-point numbers.
        """
        v = start
        while v <= stop + 1e-9:
            yield v
            v += step

    # Colors
    ZERO_W = _dim(GREEN, 0.4)
    ZERO_F = _dim(CYAN, 0.4)
    LABEL_W = _dim(GREEN, 0.4)
    LABEL_F = _dim(CYAN, 0.4)
    GRID_W = _dim(GREEN, 0.2)
    GRID_F = _dim(CYAN, 0.2)
    GRAPH_W = _dim(GREEN, 0.5)
    GRAPH_F = _dim(CYAN, 0.6)

    if dose is not None:
        TARGET_W = float(dose) * 2.0
    else:
        TARGET_W = 38.0 # assume 19g dose

    TARGET_F = TARGET_W / 27.5 # assume ~27.5s shot time
    TOL_W, TOL_F = 6.0, 0.6 # arbitrary tolerances for color gradient
    TIME_GRID_STEP = 5.0 # seconds
    WEIGHT_GRID_STEP = 5.0 # grams
    FLOW_GRID_STEP = 0.5 # g/s

    # Highlight target lines
    pygame.draw.line(surface, ZERO_W, (x0, y_of_w(TARGET_W)), (x0 + w, y_of_w(TARGET_W)), 1)
    pygame.draw.line(surface, ZERO_F, (x0, y_of_f(TARGET_F)), (x0 + w, y_of_f(TARGET_F)), 1)
    pygame.draw.line(surface, GRID, (to_x(27.5), y0), (to_x(27.5), y0 + h), 1)

    if grid:
        # vertical time grid
        for t in _frange(0.0, max_t, TIME_GRID_STEP):
            x = to_x(t)
            pygame.draw.line(surface, _dim(GRID, 0.5), (x, y0), (x, y0 + h), 1)

    weights_smoothed = _moving_average(weights, window=5)
    flows_smoothed = _moving_average(flows, window=5)

    weight_pts = [(to_x(t), y_of_w(wt)) for t, wt in zip(ts, weights_smoothed)]
    flow_pts = [(to_x(t), y_of_f(f)) for t, f in zip(ts, flows_smoothed)]

    weight_curve = _catmull_rom_points(weight_pts, segments=6)
    flow_curve = _catmull_rom_points(flow_pts, segments=6)

    if not gradient:
        pygame.draw.lines(surface, GREEN, False, weight_pts, 1)
        pygame.draw.lines(surface, CYAN, False, flow_pts, 1)
    else:
        _draw_grad_line_by_y(surface, weight_curve, y_of_w(TARGET_W),
                             abs(y_of_w(TARGET_W) - y_of_w(TARGET_W + TOL_W)), GRAPH_W, GREEN)
        _draw_grad_line_by_y(surface, flow_curve, y_of_f(TARGET_F),
                             abs(y_of_f(TARGET_F) - y_of_f(TARGET_F + TOL_F)), GRAPH_F, CYAN)

    w_color = _lerp_color(GRAPH_W, GREEN, _proximity(cur_w, TARGET_W, TOL_W))
    f_color = _lerp_color(GRAPH_F, CYAN, _proximity(cur_f, TARGET_F, TOL_F))

    # Display important values
    avg_f = sum(flows[-5:]) / min(len(flows), 5) if flows else 0.0
    text(surface, font, f"{cur_w:.2f}g", (x0 + 20, FOOTER_POS[1]), w_color)
    text(surface, font, f"{avg_f:.1f}g/s", (x0 + w - 39 - 50, FOOTER_POS[1]), f_color)


def draw_post_labeling(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    """
    Draws the post-shot labeling interface, allowing the user to select from a list of options.
    """
    text(surface, font, app.choice_prompt or "LABEL THIS SHOT", HEADER_POS, AMBER)

    y = 30
    for i, option in enumerate(app.choice_options):
        active = i == app.choice_active_index
        color = GREEN if active else GRID
        prefix = "> " if active else "  "
        text(surface, font, f"{prefix}{option.upper()}", (29, y), color)
        y += 20


def draw_predicted_label(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    if not app.result_label or not app.result_probs:
        app.state = State.PREDICTED_GRIND
        return

    def _draw_circle_outline(surface, color, center, radius, progress, width=1):
        """
        Draws a circular outline (arc) on the given surface.
        :param surface: The surface to draw on.
        :param color: The color of the arc.
        :param center: The center of the circle (x, y).
        :param radius: The radius of the circle.
        :param progress: The progress of the arc (0.0 to 1.0).
        :param width: The width of the arc.
        """
        start_angle = 0.0
        end_angle = progress * 360.0
        pygame.draw.arc(surface, color, (center[0] - radius, center[1] - radius, radius * 2, radius * 2),
                        math.radians(start_angle), math.radians(end_angle), width)

    def _draw_line_segments(surface, color, center, radius, progress, results, width=1):
        """
        Draws line segments from the center to the circumference of a circle,
        representing the results as proportions of the total.
        :param surface: The surface to draw on.
        :param color: The color of the line segments.
        :param center: The center of the circle (x, y).
        :param radius: The radius of the circle.
        :param progress: The progress of the drawing (0.0 to 1.0).
        :param results: A dictionary of result labels and their corresponding values.
        :param width: The width of the line segments.
        :return: The angles of the predicted segment.
        """
        total = sum(results.values())
        predicted = (0, 0)
        start_angle = 0.0
        for lab, p in results.items():
            end_angle = start_angle + p * 360.0 / total
            end_x = center[0] + radius * math.cos(math.radians(start_angle)) * progress
            end_y = center[1] + radius * math.sin(math.radians(start_angle)) * progress
            pygame.draw.line(surface, color, center, (end_x, end_y), width)
            if p > 0.33:
                predicted = (start_angle, end_angle)
            start_angle = end_angle
        return predicted

    def _draw_wedge(surface, color, center, radius, progress,
                    angles, segments=40, width=0):
        """
        Draws a wedge (a filled sector) of a circle.
        :param surface: The surface to draw on.
        :param color: The color of the wedge.
        :param center: The center of the circle (x, y).
        :param radius: The radius of the circle.
        :param progress: The progress of the drawing (0.0 to 1.0).
        :param angles: A tuple of (start_angle, end_angle) in degrees.
        :param segments: The number of segments to approximate the wedge.
        :param width: The width of the wedge border (0 for filled).
        """
        start_angle, end_angle = angles[0], angles[1]
        points = [(center[0], center[1])]
        alpha = round(255 * progress)
        wedge_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for i in range(segments + 1):
            t = i / segments
            angle = start_angle + (end_angle - start_angle) * t
            x = center[0] + radius * math.cos(math.radians(angle))
            y = center[1] + radius * math.sin(math.radians(angle))
            points.append((round(x), round(y)))
        pygame.draw.polygon(wedge_surface, (*color, alpha), points, width)
        surface.blit(wedge_surface, (0, 0))

    def _draw_labels(surface, font, center, radius, progress, results):
        """
        Draws labels outside the circle for each result segment.
        :param surface: The surface to draw on.
        :param font: The font to use for the labels.
        :param center: The center of the circle (x, y).
        :param radius: The radius of the circle.
        :param progress: The progress of the drawing (0.0 to 1.0).
        :param results: A dictionary of result labels and their corresponding values.
        """
        total = sum(results.values())
        start_angle = 0.0
        alpha = round(255 * progress)
        text_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for lab, p in results.items():
            label = lab.upper()
            if label == 'BALANCED':
                label = 'BAL'
            end_angle = start_angle + p * 360.0 / total
            mid_angle = (start_angle + end_angle) / 2
            draw_text_outside_circle(text_surface, font, f"{label}", center,
                                    radius * 1.25, AMBER, alpha, angle_deg=mid_angle)
            start_angle = end_angle
        surface.blit(text_surface, (0, 0))

    RADIUS = 60
    TIME_TO_DRAW = 1.0 # seconds
    elapsed = time.monotonic() - (app.state_entered_at or 0)
    color = GREEN

    if elapsed - 2 * TIME_TO_DRAW > 0: # dim non-predicted segments
        elapsed_shifted = elapsed - 2 * TIME_TO_DRAW
        clamp = 1 - min(1.0, elapsed_shifted / TIME_TO_DRAW)
        factor = min(clamp + 0.25, 1.0)
        color = _dim(GREEN, factor)

    progress = min(1.0, elapsed / TIME_TO_DRAW)
    progress = progress * progress * (3.0 - 2.0 * progress)
    _draw_circle_outline(surface, color, (RES[0] // 2, RES[1] // 2), RADIUS, progress)

    elapsed = elapsed - TIME_TO_DRAW
    progress = min(1.0, elapsed / TIME_TO_DRAW) if elapsed >= 0 else 0.0
    progress = progress * progress * (3.0 - 2.0 * progress)
    angles = _draw_line_segments(surface, color, (RES[0] // 2, RES[1] // 2), RADIUS, progress, app.result_probs or {})

    elapsed = elapsed - TIME_TO_DRAW
    progress = min(1.0, elapsed / TIME_TO_DRAW) if elapsed >= 0 else 0.0
    progress = progress * progress * (3.0 - 2.0 * progress)
    _draw_wedge(surface, GREEN, (RES[0] // 2, RES[1] // 2), RADIUS, progress, angles)

    elapsed = elapsed - TIME_TO_DRAW
    progress = min(1.0, elapsed / TIME_TO_DRAW) if elapsed >= 0 else 0.0
    progress = progress * progress * (3.0 - 2.0 * progress)
    _draw_labels(surface, font, (RES[0] // 2, RES[1] // 2), RADIUS, progress, app.result_probs or {})

def draw_predicted_grind(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    if app.grind_rec is None or app.pred_time is None or app.manifest_path is None or app.gp_result is None:
        app.state = State.SUMMARY
        return

    x0, y0, w, h = 26, 27, RES[0] - 53, RES[1] - 53
    from recommend_grind import _get_manifest

    manifest = _get_manifest(app.manifest_path, bean_name=app.bean_name)
    if not manifest:
        app.state = State.IDLE
        return

    sorted_manifest = sorted(manifest, key=lambda x: x['grind_setting'])
    TIME_TO_DRAW = 0.8  # seconds
    IDEAL_TIME_LOW = 25.0
    IDEAL_TIME_HIGH = 30.0
    IDEAL_TIME_MID = (IDEAL_TIME_LOW + IDEAL_TIME_HIGH) / 2

    def _time_to_y(time_s, min_time, max_time, y0=y0, h=h):
        if max_time == min_time:
            return y0 + h / 2
        return y0 + h - (time_s - min_time) / (max_time - min_time) * h

    def _grind_to_x(grind, min_grind, max_grind, x0=x0, w=w):
        if max_grind == min_grind:
            return x0 + w / 2
        return x0 + (grind - min_grind) / (max_grind - min_grind) * w

    def _draw_manifest_points(surface, manifest, progress, size, x0, y0, w, h):
        if not manifest:
            return

        min_grind_setting = min(entry['grind_setting'] for entry in manifest)
        max_grind_setting = max(entry['grind_setting'] for entry in manifest)
        min_shot_time = min(entry['shot_time_s'] for entry in manifest)
        max_shot_time = max(entry['shot_time_s'] for entry in manifest)
        grind_span = max_grind_setting - min_grind_setting
        shot_span = max_shot_time - min_shot_time

        fade_window = 0.16
        denom = max(1, len(manifest) - 1)
        point_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        for i, entry in enumerate(manifest):
            reveal_start = i / denom
            local = (progress - reveal_start) / fade_window
            if local <= 0.0:
                continue
            local = min(1.0, local)
            local = local * local * (3.0 - 2.0 * local)  # smoothstep
            alpha = int(255 * local)

            shot_time = entry['shot_time_s']
            grind_setting = entry['grind_setting']
            color = GREEN
            if entry['label'] == 'under':
                color = AMBER
            elif entry['label'] == 'over':
                color = RED

            x_norm = 0.5 if grind_span == 0 else (grind_setting - min_grind_setting) / grind_span
            y_norm = 0.5 if shot_span == 0 else (shot_time - min_shot_time) / shot_span
            x = x0 + x_norm * w
            y = y0 + h - y_norm * h
            pygame.draw.circle(point_surface, (*color, alpha), (int(x), int(y)), size)

        surface.blit(point_surface, (0, 0))
        return max_shot_time, min_shot_time, max_grind_setting, min_grind_setting

    def _draw_ideal_time_band(surface, max_shot_time, min_shot_time, progress, x0, y0, w, h):
        band_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        alpha = int(255 * progress / 2)
        band_color = (*_dim(CYAN, 0.5), alpha)

        lower_band_y = _time_to_y(IDEAL_TIME_LOW, min_shot_time, max_shot_time, y0, h)
        upper_band_y = _time_to_y(IDEAL_TIME_HIGH, min_shot_time, max_shot_time, y0, h)
        middle_band_y = _time_to_y(IDEAL_TIME_MID, min_shot_time, max_shot_time, y0, h)

        pygame.draw.line(band_surface, band_color, (x0, lower_band_y), (x0 + w, lower_band_y), 1)
        pygame.draw.line(band_surface, band_color, (x0, upper_band_y), (x0 + w, upper_band_y), 1)
        pygame.draw.line(band_surface, band_color, (x0, middle_band_y), (x0 + w, middle_band_y), 2)

        surface.blit(band_surface, (0, 0))

    def _draw_gp_line(surface, gp_result, progress, x0, y0, w, h,
                      min_shot_time, max_shot_time, z_score=1.96):
        candidates = gp_result.get('all_candidates')
        means = gp_result.get('all_means')
        stds = gp_result.get('all_stds')
        if candidates is None or means is None or stds is None:
            return

        candidates = [float(v) for v in candidates]
        means = [float(v) for v in means]
        stds = [float(v) for v in stds] if stds is not None else None
        if len(candidates) < 2 or len(candidates) != len(means):
            return
        if stds is not None and len(stds) != len(means):
            return

        min_grind = min(candidates)
        max_grind = max(candidates)
        if max_grind == min_grind:
            return

        clip_x = x0 + w * progress
        line_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        line_color = (*CYAN, 255)
        ci_color = (*CYAN, int(255 * 0.2))

        def _clipped_points(values):
            pts = []
            for grind, shot_time in zip(candidates, values):
                x = x0 + (grind - min_grind) / (max_grind - min_grind) * w
                y = _time_to_y(shot_time, min_shot_time, max_shot_time, y0, h)

                if x < x0:
                    x = x0
                elif x > x0 + w:
                    x = x0 + w
                if y < y0:
                    y = y0
                elif y > y0 + h:
                    y = y0 + h

                if x > clip_x:
                    if pts:
                        prev_x, prev_y = pts[-1]
                        if x != prev_x:
                            t = (clip_x - prev_x) / (x - prev_x)
                            interp_y = prev_y + (y - prev_y) * t
                            pts.append((clip_x, interp_y))
                    break
                pts.append((x, y))
            return pts

        if stds is not None:
            upper_vals = [m + z_score * s for m, s in zip(means, stds)]
            lower_vals = [m - z_score * s for m, s in zip(means, stds)]
            upper_pts = _clipped_points(upper_vals)
            lower_pts = _clipped_points(lower_vals)
            if len(upper_pts) >= 2 and len(lower_pts) >= 2:
                band_points = upper_pts + list(reversed(lower_pts))
                pygame.draw.polygon(line_surface, ci_color, band_points)

        points = _clipped_points(means)
        if len(points) >= 2:
            pygame.draw.lines(line_surface, line_color, False, points, 2)
            surface.blit(line_surface, (0, 0))

    def _draw_recommendation(surface, app, font,
                             max_grind, min_grind, progress):
        if app.grind_rec is None or app.pred_time is None or app.pred_time_uncertainty is None:
            return
        alpha = int(255 * progress)
        line_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        line_x = _grind_to_x(app.grind_rec, min_grind, max_grind, x0, w)
        pygame.draw.line(line_surface, (*GREEN, alpha), (line_x, y0), (line_x, y0 + h), 2)
        text(surface, font, f"{app.grind_rec:.2f}", (line_x - 20, FOOTER_POS[1]), GREEN, alpha)
        text(surface, font, f"{app.pred_time:.2f} ± {app.pred_time_uncertainty:.1f}", HEADER_POS, CYAN, alpha)
        surface.blit(line_surface, (0, 0))


    elapsed = time.monotonic() - (app.state_entered_at or 0)
    progress = min(1.0, elapsed / TIME_TO_DRAW)
    progress = progress * progress * (3.0 - 2.0 * progress)
    max_shot_time, min_shot_time, max_grind, min_grind = _draw_manifest_points(surface, sorted_manifest, progress,
                                                                               size=2, x0=x0, y0=y0, w=w, h=h)
    _draw_ideal_time_band(surface, max_shot_time, min_shot_time, progress, x0, y0, w, h)

    elapsed = elapsed - TIME_TO_DRAW
    progress = min(1.0, elapsed / TIME_TO_DRAW) if elapsed >= 0 else 0.0
    progress = progress * progress * (3.0 - 2.0 * progress)
    _draw_gp_line(surface, app.gp_result, progress, z_score=1.28,
                  x0=x0, y0=y0, w=w, h=h, min_shot_time=min_shot_time, max_shot_time=max_shot_time)

    elapsed = elapsed - TIME_TO_DRAW
    progress = min(1.0, elapsed / TIME_TO_DRAW) if elapsed >= 0 else 0.0
    progress = progress * progress * (3.0 - 2.0 * progress)
    _draw_recommendation(surface, app, font, max_grind, min_grind, progress)

def draw_summary(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    bg = load_background(BG_DIR, RES)
    surface.blit(bg, (0, 0))
    text(surface, font, "SUMMARY", HEADER_POS, AMBER)
    y = 30

    if app.result_label is not None:
        text(surface, font, f"SHOT: {app.result_label} ({(app.result_probs[app.result_label] * 100):.0f}%)",
             (29, y), GREEN)
    else:
        text(surface, font, f"SHOT: no model",
             (29, y), GREEN)
    y += 30

    if app.grind_rec is not None:
        text(surface, font, f"NEXT GRIND: {app.grind_rec:.2f}", (29, y), CYAN)
    else:
        text(surface, font, f"NEXT GRIND: no model", (29, y), CYAN)

    y += 30

    if app.pred_time is not None and app.pred_time_uncertainty is not None:
        text(surface, font, f"TIME: {app.pred_time:.1f} ± {app.pred_time_uncertainty:.1f}", (29, y), CYAN)
    else:
        text(surface, font, f"TIME: no model", (29, y), CYAN)

DRAW_FUNCS = {
    State.IDLE : draw_idle,
    State.BOOT: draw_boot,
    State.LABELING: draw_labeling,
    State.DOSING: draw_dosing,
    State.PREPPING: draw_prepping,
    State.TARING: draw_tare_scale,
    State.LOGGING: draw_logging,
    State.POST_LABELING: draw_post_labeling,
    State.PREDICTED_LABEL: draw_predicted_label,
    State.PREDICTED_GRIND: draw_predicted_grind,
    State.SUMMARY: draw_summary,
}

def load_background(path: str, size: tuple[int, int]) -> pygame.Surface | None:
    p = Path(path)
    if not p.exists():
        return None
    img = pygame.image.load(str(p)).convert()
    return pygame.transform.scale(img, size)

def render_scene(surface: pygame.Surface, app: AppState, font, dt: float) -> None:
    if app.bg_image is not None:
        surface.blit(app.bg_image, (0, 0))
    else:
        surface.fill(BG_COLOR)

    if app.state in GRAPH_DOCKED_STATES:
        split_y = int(RES[1] * GRAPH_SPLIT_FRACTION)
        # Top portion: state content
        top = surface.subsurface((0, 0, RES[0], split_y))
        DRAW_FUNCS[app.state](top, app, font, dt)

        # Bottom portion: graph
        graph_rect = (26, split_y + GRAPH_DOCK_MARGIN,
                      RES[0] - 53, RES[1] - split_y - GRAPH_DOCK_MARGIN - 27)
        draw_live_graph(surface, app.dose, app.live_points, rect=graph_rect, font=font,
                        fixed_scale=True, tick_labels=False, grid = False)
    else:
        DRAW_FUNCS[app.state](surface, app, font, dt)

def load_assets(app: AppState) -> None:
    """
    Loads all necessary assets into the application state.
    """
    if app.bg_image is None:
        app.bg_image = load_background(BG_DIR, RES)

    if app.idle_anim is None:
        try:
            frames = load_frame_sequence(IDLE_DIR, size=RES)
            app.idle_anim = Animation(frames, fps=24, loop=True)
        except FileNotFoundError:
            pass

    if app.boot_anim is None:
        try:
            frames = load_frame_sequence(BOOT_DIR, size=RES)
            app.boot_anim = Animation(frames, fps=4, loop=True)
        except FileNotFoundError:
            pass

    if app.dose_anim is None:
        try:
            frames = load_frame_sequence(DOSE_DIR, size=RES)
            app.dose_anim = Animation(frames, fps=30, loop=True)
        except FileNotFoundError:
            pass

    if app.grind_anim is None:
        try:
            frames = load_frame_sequence(GRIND_DIR, size=RES)
            app.grind_anim = Animation(frames, fps=8, loop=True)
        except FileNotFoundError:
            pass

    if app.wdt_anim is None:
        try:
            frames = load_frame_sequence(WDT_DIR, size=RES)
            app.wdt_anim = Animation(frames, fps=30, loop=True)
        except FileNotFoundError:
            pass

    if app.level_anim is None:
        try:
            frames = load_frame_sequence(LEVEL_DIR, size=RES)
            app.level_anim = Animation(frames, fps=30, loop=True)
        except FileNotFoundError:
            pass

    if app.tamp_anim is None:
        try:
            frames = load_frame_sequence(TAMP_DIR, size=RES)
            app.tamp_anim = Animation(frames, fps=50, loop=True)
        except FileNotFoundError:
            pass

    if app.tare_anim is None:
        try:
            frames = load_frame_sequence(TARING_DIR, size=RES)
            app.tare_anim = Animation(frames, fps=24, loop=True)
        except FileNotFoundError:
            pass

async def run_display(app: AppState, fullscreen: bool = False) -> None:
    """
    Runs the main display loop, rendering the application state to the screen.
    :param app: The application state.
    :param fullscreen: Whether to run in fullscreen mode.
    """
    pygame.init()
    flags = pygame.FULLSCREEN if fullscreen else 0
    size = RES if fullscreen else (RES[0] * SCALE, RES[1] * SCALE)
    window = pygame.display.set_mode(size, flags)
    pygame.display.set_caption("espresso analysis")

    internal = pygame.Surface(RES)
    font = pygame.font.SysFont("courier", 20)
    clock = pygame.time.Clock()

    load_assets(app)

    app.live_points = [(0, 0, 0)]

    app.state_entered_at = None
    prev_state = app.state
    dt = 1.0 / FPS  # first-frame estimate

    outgoing_snapshot: pygame.Surface | None = None
    fading = False
    fade_elapsed = 0.0
    next_rec_swap_at: float | None = None

    running = True
    while running and not app.quit:
        for event in pygame.event.get():
            # Quit event
            if event.type == pygame.QUIT:
                running = False

            # Wake up event
            elif event.type == pygame.KEYDOWN and app.state == State.IDLE:
                app.state = State.BOOT
                if app.boot_anim is not None:
                    app.boot_anim.reset()
                app.boot_frame = 0
                app.start_event.set()

            # Log dose event
            elif event.type == pygame.KEYDOWN and app.state == State.DOSING:
                app.key_down_event.set()

            # Prepping beans
            elif event.type == pygame.KEYDOWN and app.state == State.PREPPING:
                if app.puck_prep_state is None:
                    app.key_down_event.set()
                app.puck_prep_state += 1
                if app.puck_prep_state > 4:
                    app.puck_prep_state = None
                    app.key_down_event.set()

            # Tare scale
            elif event.type == pygame.KEYDOWN and app.state == State.TARING:
                app.key_down_event.set()

            # Start / stop logging
            elif event.type == pygame.KEYDOWN and app.state == State.LOGGING:
                app.key_down_event.set()

            # Labeling events
            elif event.type == pygame.KEYDOWN and app.state == State.LABELING:
                i = app.form_active_index
                if event.key == pygame.K_BACKSPACE: # Delete
                    app.form_fields[i][1] = app.form_fields[i][1][:-1]
                elif event.key == pygame.K_RETURN: # Enter
                    if i < len(app.form_fields) - 1:
                        app.form_active_index += 1
                    else:
                        app.form_result = {label: value for label, value in app.form_fields}
                        app.form_submit_event.set()
                elif event.unicode and event.unicode.isprintable():
                    app.form_fields[i][1] += event.unicode

            # Post-shot label events
            elif event.type == pygame.KEYDOWN and app.state == State.POST_LABELING:
                if event.key in (pygame.K_UP, pygame.K_LEFT):
                    app.choice_active_index = (app.choice_active_index - 1) % len(app.choice_options)
                elif event.key in (pygame.K_DOWN, pygame.K_RIGHT):
                    app.choice_active_index = (app.choice_active_index + 1) % len(app.choice_options)
                elif event.key == pygame.K_RETURN:
                    app.choice_result = app.choice_options[app.choice_active_index]
                    app.choice_submit_event.set()
                elif event.unicode:
                    ch = event.unicode.lower()
                    for opt in app.choice_options:
                        if opt.lower().startswith(ch):
                            app.choice_result = opt
                            app.choice_submit_event.set()
                            break

            # Switch to predicted grind recommendation
            elif event.type == pygame.KEYDOWN and app.state == State.PREDICTED_LABEL:
                app.state = State.PREDICTED_GRIND

            # Switch to summary
            elif event.type == pygame.KEYDOWN and app.state == State.PREDICTED_GRIND:
                app.state = State.SUMMARY

            # Force switch to idle
            elif event.type == pygame.KEYDOWN and app.state == State.SUMMARY:
                app.state = State.IDLE

        if (app.state == State.SUMMARY and app.state_entered_at is not None
                and time.monotonic() - app.state_entered_at > app.result_timeout):
            app.state = State.IDLE
            app.state_entered_at = None
            prev_state = State.IDLE

        # State is changing!
        if app.state != prev_state:
            outgoing_snapshot = internal.copy()
            fading = True
            fade_elapsed = 0.0
            if app.state == State.PREDICTED_LABEL or app.state == State.PREDICTED_GRIND or app.state == State.SUMMARY:
                app.state_entered_at = time.monotonic() # Start timeout
            prev_state = app.state

        render_scene(internal, app, font, dt)

        if fading:
            fade_elapsed += dt
            progress = min(fade_elapsed / CROSSFADE_DURATION, 1.0)
            frame_to_show = outgoing_snapshot.copy()
            incoming = internal.copy()
            incoming.set_alpha(int(progress * 255))
            frame_to_show.blit(incoming, (0, 0))
            if progress >= 1.0:
                fading = False
        else:
            frame_to_show = internal

        scaled = pygame.transform.scale(frame_to_show, window.get_size())
        window.blit(scaled, (0, 0))
        pygame.display.flip()

        dt_ms = clock.tick(FPS)
        dt = dt_ms / 1000.0 if dt_ms > 0 else 1.0 / FPS
        await asyncio.sleep(0)  # <-- hand control back to the asyncio loop each frame

    pygame.quit()

async def _demo_fill_pre_shot_form(app: AppState, defaults: "ShotDefaults") -> None:
    """
    Fills the pre-shot labeling form with default values for demo purposes.
    """
    await asyncio.sleep(0.2)
    app.form_fields = [
        ["Bean name", defaults.bean_name],
        ["Roast date", defaults.roast_date],
        ["Bag opened date", defaults.open_date],
        # ["Dose (g)", defaults.dose_g],
        # ["Grind setting", defaults.grind_setting],
    ]
    app.form_active_index = 0
    app.form_result = {
        "Bean name": defaults.bean_name,
        "Roast date": defaults.roast_date,
        "Bag opened date": defaults.open_date,
        # "Dose (g)": defaults.dose_g,
        # "Grind setting": defaults.grind_setting,
    }
    app.state = State.LABELING
    app.previous_grind_rec = f"3.5"  # placeholder for demo
    app.bean_name = defaults.bean_name
    app.form_submit_event.set()


async def _demo(draw_graph = True) -> None:
    """
    Demo mode that replays a synthetic shot and runs the grind recommendation flow
    without requiring a physical scale.
    """
    # lazy imports
    import csv
    import random
    from labeling_handler import ShotDefaults
    from recommend_grind import update_grind_model, load_gp, recommend_next_grind, update_appstate_rec

    app = AppState()
    app.result_timeout = 30.0  # short for the demo; use minutes in real use
    display_task = asyncio.create_task(run_display(app))
    app.state = State.IDLE

    defaults = ShotDefaults(
        grind_setting="3.5",
        dose_g="18.0",
        bean_name="DemoBeans",
        roast_date="",
        open_date="",
    )

    synthetic_dir = Path("./synthetic_shots")
    manifest_path = synthetic_dir / "manifest.csv"

    if not manifest_path.exists():
        from generate_BLE_data import main as generate_synthetic_data
        print("No synthetic dataset found; generating one now...")
        generate_synthetic_data()

    with open(manifest_path, newline="") as f:
        manifest_rows = [
            row for row in csv.DictReader(f)
            if row.get("label") in {"under", "balanced", "over"} and row.get("curve_file")
        ]

    if not manifest_rows:
        raise RuntimeError(f"No labeled synthetic shots found in {manifest_path}")

    # Fit from synthetic manifest (manager.py uses update_grind_model on logged shots).
    gp_path = Path(f"./gp_models/{defaults.bean_name}_gp_synthetic.pkl")
    gp = load_gp(gp_path)
    if gp is None:
        print("No GP model found; fitting from synthetic manifest...")
        gp = update_grind_model(synthetic_dir,
                                gp_path,
                                file_override=f"{defaults.bean_name}_gp_synthetic.pkl",
                                bean_name=defaults.bean_name)
        if gp is None:
            raise RuntimeError("Failed to fit GP model from synthetic manifest")
    print("Loaded GP model for synthetic dataset.")

    shot = random.choice(manifest_rows)
    curve_path = synthetic_dir / shot["curve_file"]
    defaults.grind_setting = shot["grind_setting"]
    app.dose = float(defaults.dose_g)

    print("In idle...")
    app.state = State.IDLE
    await app.start_event.wait()
    app.start_event.clear()

    print("Booting...")
    app.state = State.BOOT
    await asyncio.sleep(1.0)

    print("Pre-shot labeling...")
    await _demo_fill_pre_shot_form(app, defaults)
    await asyncio.sleep(3.0)

    print("Dosing...")
    app.state = State.DOSING
    await app.key_down_event.wait()
    app.key_down_event.clear()

    print("Puck prep...")
    app.puck_prep_state = 0
    app.state = State.PREPPING
    await app.key_down_event.wait()
    app.key_down_event.clear()

    print(f"Replaying synthetic shot: {curve_path.name} (true label={shot['label']}, true grind={shot['grind_setting']})")
    app.state = State.LOGGING

    with open(curve_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                elapsed = float(row["elapsed_s"])
                weight = float(row["weight_g"])
            except (TypeError, ValueError, KeyError):
                continue
            app.add_point(elapsed, weight)
            if draw_graph:
                await asyncio.sleep(0.03)

    print("Requesting post-shot label...")
    label = await request_choice(app, "LABEL THIS SHOT", ["under", "balanced", "over", "discard"])
    print("Picked:", label)

    # Simulated classifier output for demo mode.
    app.result_probs = {"under": 0.1, "balanced": 0.1, "over": 0.1}
    app.result_probs[shot["label"]] = 0.8
    app.result_label = max(app.result_probs, key=app.result_probs.get)
    if not app.result_label:
        app.result_label = label

    if gp is not None:
        result = recommend_next_grind(gp)
        update_appstate_rec(result, app, manifest_path=manifest_path)
    else:
        app.grind_rec = -1
        app.pred_time = -1

    app.state = State.PREDICTED_LABEL
    print(f"({app.result_timeout}s before idle)...")

    await asyncio.sleep(app.result_timeout + 1.0)
    print("Back to idle:", app.state)

    app.quit = True
    await display_task

if __name__ == "__main__":
    configure_logging()
    asyncio.run(_demo(draw_graph = False))