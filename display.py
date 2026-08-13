import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import pygame
import re
import math

RES = (320, 240)
SCALE = 3
FPS = 30

GREEN = (60, 255, 120)
AMBER = (255, 176, 0)
CYAN = (80, 220, 255)
BG_COLOR = (8, 8, 10)
GRID = (30, 30, 30)

FLOW_WINDOW_S = 0.25 # lookback window
DEFAULT_RESET_TIMEOUT = 120 # seconds before idle again
CROSSFADE_DURATION = 0.25 # seconds

GRAPH_SPLIT_FRACTION = 0.5
GRAPH_DOCK_MARGIN = 6 # px

BACKGROUND_DIR = 'assets/bg.png'
IDLE_DIR = 'assets/idle'
BOOT_DIR = 'assets/boot'
BEANS_DIR = 'assets/beans.png'
GRIND_DIR = 'assets/grind'
PUCK_PREP_DIR = 'assets/puck_prep'

class State(Enum):
    IDLE = auto()
    BOOT = auto()
    LABELING = auto()
    GRINDING = auto()
    PREPPING = auto()
    LOGGING = auto()
    POST_LABELING = auto()
    RESULTS = auto()

GRAPH_DOCKED_STATES = {State.POST_LABELING, State.RESULTS}

@dataclass
class AppState:
    state: State = State.IDLE
    boot_frame: int = 0
    # (elapsed_s, weight_g, flow_g_s)
    live_points: list[tuple[float, float, float]] = field(default_factory=list)
    dose: float | None = None
    result_label: str | None = None
    result_probs: dict[str, float] | None = None
    rec: str | None = None # e.g. "grind finer"
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
    grind_anim: "Animation | None" = None
    puck_prep_anim: "Animation | None" = None

    def add_point(self, elapsed: float, weight: float):
        flow = 0.0
        for t, w, _ in reversed(self.live_points):
            dt = elapsed - t
            if dt >= FLOW_WINDOW_S:
                flow = (weight - w) / dt
                break
        self.live_points.append((elapsed, weight, flow))

class Animation:
    def __init__(self, frames: list[pygame.Surface], fps: float = 24.0, loop: bool = True):
        self.frames = frames
        self.frame_duration = 1.0/fps
        self.loop = loop
        self.elapsed = 0.0

    def reset(self):
        self.elapsed = 0.0

    def update(self, dt: float):
        self.elapsed += dt

    def finished(self) -> bool:
        return not self.loop and self.elapsed >= self.frame_duration * len(self.frames)

    def current_frame(self) -> pygame.Surface:
        idx = int(self.elapsed / self.frame_duration)
        idx = idx % len(self.frames) if self.loop else min(idx, len(self.frames) - 1)
        return self.frames[idx]

def text(surface, font, s, pos, color):
    rendered = font.render(s, False, color)
    if pos[0] < 0 and pos[1] < 0:
        rect = rendered.get_rect(midtop=(surface.get_width() // 2, surface.get_height() // 2))
        surface.blit(rendered, rect)
    elif pos[0] < 0:
        rect = rendered.get_rect(midtop=(surface.get_width() // 2, pos[1]))
        surface.blit(rendered, rect)
    elif pos[1] < 0:
        rect = rendered.get_rect(midbottom=(pos[0], surface.get_height() // 2))
        surface.blit(rendered, rect)
    else:
        surface.blit(rendered, pos)

def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path.stem)]

def _dim(color, factor=0.35):
    return tuple(int(c * factor) for c in color)

def _tick_marks(low, hi, target=4):
    span = hi - low
    if span <= 0:
        return [low]
    raw_step = span / target
    mag = 10 ** math.floor(math.log10(raw_step))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw_step:
            break
    start = math.floor(low / step) * step
    ticks, v = [], start
    while v <= hi + step * 1e-6:
        if v >= low - step * 1e-6:
            ticks.append(round(v, 6))
        v += step
    return ticks

async def request_form(app: AppState, fields: list[tuple[str, str]]) -> dict[str, str] | None:
    app.form_fields = [[label, default] for label, default in fields]
    app.form_active_index = 0
    app.form_result = None
    app.form_submit_event.clear()
    app.state = State.LABELING

    await app.form_submit_event.wait()
    return app.form_result

async def request_choice(app: AppState, prompt: str, choices: list[str]) -> str:
    app.choice_prompt = prompt
    app.choice_options = list(choices)
    app.choice_active_index = 0
    app.choice_result = None
    app.choice_submit_event.clear()
    app.state = State.POST_LABELING

    await app.choice_submit_event.wait()
    return app.choice_result

def load_frame_sequence(directory: str, size: tuple[int, int] | None = None) -> list[pygame.Surface]:
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
                    fallback_label: str, fallback_color, fallback_progress: bool = False):
    # surface.fill(BG_COLOR)
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

def draw_idle(surface: pygame.Surface, app: AppState, font, dt: float):
    _draw_animated(surface, app, font, app.idle_anim, dt, "READY", GREEN)

def draw_boot(surface: pygame.Surface, app: AppState, font, dt: float):
    _draw_animated(surface, app, font, app.boot_anim, dt, "CONNECTING...", AMBER)

def draw_labeling(surface: pygame.Surface, app: AppState, font, dt: float):
    bg = load_background(BEANS_DIR, RES)
    surface.blit(bg, (0, 0))

    text(surface, font, "LABELING SHOT", (-1, 6), AMBER)
    text(surface, font, "ENTER: next field  BACKSPACE: edit", (-1, RES[1] - 16), GRID)

    cursor = "_" if int(time.monotonic() * 2) % 2 == 0 else " "  # ~2Hz blink
    y = 32
    for i, (label, value) in enumerate(app.form_fields):
        active = i == app.form_active_index
        color = GREEN if active else AMBER
        shown_value = value + cursor if active else value
        text(surface, font, f"{label}:", (29, y), color)
        text(surface, font, shown_value, (29, y + 12), color)
        y += 28

def draw_grinding(surface: pygame.Surface, app: AppState, font, dt: float):
    # surface.fill(BG_COLOR)
    _draw_animated(surface, app, font, app.grind_anim, dt, "GRINDING...", AMBER)
    text(surface, font, "GRINDING...", (-1, RES[1] - 60), AMBER)

def draw_prepping(surface: pygame.Surface, app: AppState, font, dt: float):
    _draw_animated(surface, app, font, app.puck_prep_anim, dt, "PREPPING...", AMBER)
    text(surface, font, "TARE SCALE...", (-1, RES[1] - 60), AMBER)

def draw_logging(surface: pygame.Surface, app: AppState, font, dt: float):
    # surface.fill(BG_COLOR)
    text(surface, font, "PULLING SHOT", (10, 4), GREEN)
    draw_live_graph(surface, app.dose, app.live_points, rect=(25, 27, RES[0] - 52, RES[1] - 53), font=font)

def draw_live_graph(surface, dose, points, rect, font, gradient = True,
                    emphasize_zero = True, tick_labels = True):
    """
    Draws weight (green, left scale) and flow-rate (cyan, right scale) on
    the same plot area. Each series is auto-scaled independently -- they
    don't share units, so a shared scale would flatten one or the other.
    """
    x0, y0, w, h = rect
    # pygame.draw.rect(surface, GRID, rect, width=1)
    if len(points) < 2:
        return

    ts = [p[0] for p in points]
    weights = [p[1] for p in points]
    flows = [p[2] for p in points]

    max_t = max(ts) or 1.0
    max_w = max(weights) or 1.0
    min_f, max_f = min(flows + [0.0]), max(flows + [0.1])
    cur_f, cur_w = flows[-1], weights[-1]
    span_f = (max_f - min_f) or 1.0

    def to_x(t):
        return x0 + int((t / max_t) * w)

    def y_of_w(wt):
        return y0 + h - int((wt / max_w) * h)

    def y_of_f(f):
        return y0 + h - int(((f - min_f) / span_f) * h)

    def _moving_average(values, window=5):
        if window <= 1 or len(values) < 2:
            return values[:]
        out = []
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            chunk = values[lo:i + 1]
            out.append(sum(chunk) / len(chunk))
        return out

    def _catmull_rom_points(pts, segments=8):
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

    def _lerp_color(c1, c2, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def _proximity(value, target, tol):
        if tol <= 0:
            return 1.0 if value == target else 0.0
        d = abs(target - value)
        return max(0.0, 1 - d / tol)

    def _draw_grad_line_by_y(surface, pts, y_target, y_tol,
                             base_color, full_color, width=1):
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            avg_y = (y1 + y2) / 2
            t = _proximity(avg_y, y_target, y_tol)
            color = _lerp_color(base_color, full_color, t)
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), width)

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
        TARGET_W = 38.0

    TARGET_F = TARGET_W / 27.5
    TOL_W, TOL_F = 6.0, 0.6

    for wt in _tick_marks(0, max_w):
        y = y_of_w(wt)
        color = ZERO_W if (abs(wt) < 1e-9 and emphasize_zero) else GRID_W
        width = 2 if (abs(wt) < 1e-9 and emphasize_zero) else 1
        pygame.draw.line(surface, color, (x0, y), (x0 + w, y), width)
        if tick_labels:
            text(surface, font, f"{wt:g}", (x0 - 4, y - 6), LABEL_W)

    for f in _tick_marks(min_f, max_f):
        y = y_of_f(f)
        color = ZERO_F if (abs(f) < 1e-9 and emphasize_zero) else GRID_F
        width = 2 if (abs(f) < 1e-9 and emphasize_zero) else 1
        pygame.draw.line(surface, color, (x0, y), (x0 + w, y), width)
        if tick_labels:
            text(surface, font, f"{f:g}", (x0 + w + 4, y - 6), LABEL_F)

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
    text(surface, font, f"{cur_w:.2f}g", (x0 + 50, RES[1] - 16), w_color)
    text(surface, font, f"{cur_f:.1f}g/s", (x0 + w - 39 - 50, RES[1] - 16), f_color)


def draw_post_labeling(surface: pygame.Surface, app: AppState, font, dt: float):
    text(surface, font, app.choice_prompt or "LABEL THIS SHOT", (-1, 6), AMBER)
    text(surface, font, "UP/DOWN+ENTER, or press a letter", (-1, surface.get_height() - 16), GRID)

    y = 28
    for i, option in enumerate(app.choice_options):
        active = i == app.choice_active_index
        color = GREEN if active else AMBER
        prefix = "> " if active else "  "
        text(surface, font, f"{prefix}{option.upper()}", (29, y), color)
        y += 14


def draw_result(surface: pygame.Surface, app: AppState, font, dt: float):
    label = app.result_label or "?"
    text(surface, font, f"RESULT: {label.upper()}", (-1, 6), AMBER)
    y = 28
    if app.result_probs:
        for lab, p in app.result_probs.items():
            text(surface, font, f"{lab:9s} {p:.2f}", (29, y), GREEN)
            y += 14
    if app.rec:
        text(surface, font, app.rec, (29, y + 6), AMBER)

DRAW_FUNCS = {
    State.IDLE : draw_idle,
    State.BOOT: draw_boot,
    State.LABELING: draw_labeling,
    State.GRINDING: draw_grinding,
    State.PREPPING: draw_prepping,
    State.LOGGING: draw_logging,
    State.POST_LABELING: draw_post_labeling,
    State.RESULTS: draw_result,
}

def load_background(path: str, size: tuple[int, int]) -> pygame.Surface | None:
    p = Path(path)
    if not p.exists():
        return None
    img = pygame.image.load(str(p)).convert()
    return pygame.transform.scale(img, size)

def render_scene(surface: pygame.Surface, app: AppState, font, dt: float):
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
                        emphasize_zero=False, tick_labels=False)
    else:
        DRAW_FUNCS[app.state](surface, app, font, dt)

async def run_display(app: AppState, fullscreen: bool = False):
    pygame.init()
    flags = pygame.FULLSCREEN if fullscreen else 0
    size = RES if fullscreen else (RES[0] * SCALE, RES[1] * SCALE)
    window = pygame.display.set_mode(size, flags)
    pygame.display.set_caption("shot logger display")

    internal = pygame.Surface(RES)
    font = pygame.font.SysFont("courier", 11)
    clock = pygame.time.Clock()

    if app.bg_image is None:
        app.bg_image = load_background(BACKGROUND_DIR, RES)

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

    if app.grind_anim is None:
        try:
            frames = load_frame_sequence(GRIND_DIR, size=RES)
            app.grind_anim = Animation(frames, fps=8, loop=True)
        except FileNotFoundError:
            pass

    if app.puck_prep_anim is None:
        try:
            frames = load_frame_sequence(PUCK_PREP_DIR, size=RES)
            app.puck_prep_anim = Animation(frames, fps=24, loop=True)
        except FileNotFoundError:
            pass

    app.live_points = [(0, 0, 0)]

    result_entered_at: float | None = None
    prev_state = app.state
    dt = 1.0 / FPS  # first-frame estimate

    outgoing_snapshot: pygame.Surface | None = None
    fading = False
    fade_elapsed = 0.0

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

            # Done grinding beans event
            elif event.type == pygame.KEYDOWN and app.state == State.GRINDING:
                app.key_down_event.set()

            # Tare scale
            elif event.type == pygame.KEYDOWN and app.state == State.PREPPING:
                app.key_down_event.set()

            # Start / stop logging
            elif event.type == pygame.KEYDOWN and app.state == State.LOGGING:
                app.key_down_event.set()

            # Labeling (Keydown) events
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

            # Post-shot label (choice picker) events
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

        # Switch back to idle after timeout
        if (app.state == State.RESULTS and result_entered_at is not None
                and time.monotonic() - result_entered_at > app.result_timeout):
            app.state = State.IDLE
            result_entered_at = None
            prev_state = State.IDLE

        # State is changing!
        if app.state != prev_state:
            outgoing_snapshot = internal.copy()
            fading = True
            fade_elapsed = 0.0
            if app.state == State.RESULTS:
                result_entered_at = time.monotonic()
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


async def _demo(draw_graph = False):
    import random

    app = AppState()
    app.result_timeout = 3.0  # short for the demo; use minutes in real use
    display_task = asyncio.create_task(run_display(app))

    print("In idle...")
    await app.start_event.wait()
    app.start_event.clear()
    print("Boot animation playing...")

    await asyncio.sleep(2)
    print("Label animation playing...")
    app.state = State.LABELING
    await asyncio.sleep(2)
    print("Grinding animation playing...")
    app.state = State.GRINDING
    await asyncio.sleep(2)
    print("Prepping animation playing...")
    app.state = State.PREPPING
    await asyncio.sleep(2)
    if draw_graph:
        print("Logging animation playing...")
        app.state = State.LOGGING

    t, w = 0.0, 0.0
    for _ in range(150):
        t += 0.1
        w += random.uniform(-0.2, 0.6)
        app.add_point(t, w)
        if draw_graph:
            await asyncio.sleep(0.03)

    print("Requesting post-shot label...")
    label = await request_choice(app, "LABEL THIS SHOT", ["under", "balanced", "over", "discard"])
    print("Picked:", label)

    app.result_label = label
    app.result_probs = {"under": 0.12, "balanced": 0.71, "over": 0.17}

    REC_THRESHOLD = 0.0
    diff = app.result_probs['over'] - app.result_probs['under']
    if diff > REC_THRESHOLD:
        app.rec = "grind coarser"
    elif diff < -REC_THRESHOLD:
        app.rec = "grind finer"
    else:
        app.rec = "predicted extraction is balanced"

    app.state = State.RESULTS
    print(f"Result screen ({app.result_timeout}s before idle)...")

    await asyncio.sleep(app.result_timeout + 1.0)
    print("Back to idle:", app.state)

    app.quit = True
    await display_task

if __name__ == "__main__":
    asyncio.run(_demo(draw_graph = False))