import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import pygame
import re

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

BACKGROUND_DIR = 'assets/bg.png'
IDLE_DIR = 'assets/idle'
BOOT_DIR = 'assets/boot'
BEANS_DIR = 'assets/beans.png'

class State(Enum):
    IDLE = auto()
    BOOT = auto()
    LABELING = auto()
    GRINDING = auto()
    LOGGING = auto()
    RESULTS = auto()

@dataclass
class AppState:
    state: State = State.IDLE
    boot_frame: int = 0
    # (elapsed_s, weight_g, flow_g_s)
    live_points: list[tuple[float, float, float]] = field(default_factory=list)
    result_label: str | None = None
    result_probs: dict[str, float] | None = None
    rec: str | None = None # e.g. "grind finer"
    quit: bool = False
    result_timeout: float = DEFAULT_RESET_TIMEOUT
    start_event: asyncio.Event = field(default_factory=asyncio.Event)
    grind_event: asyncio.Event = field(default_factory=asyncio.Event)

    form_fields: list = field(default_factory=list)
    form_active_index: int = 0
    form_result: dict | None = None
    form_submit_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Animations
    bg_image: "pygame.Surface | None" = None
    idle_anim: "Animation | None" = None
    boot_anim: "Animation | None" = None

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


async def request_form(app: AppState, fields: list[tuple[str, str]]) -> dict[str, str]:
    app.form_fields = [[label, default] for label, default in fields]
    app.form_active_index = 0
    app.form_result = None
    app.form_submit_event.clear()
    app.state = State.LABELING

    await app.form_submit_event.wait()
    return app.form_result

def load_frame_sequence(dir: str, size: tuple[int, int] | None = None) -> list[pygame.Surface]:
    paths = sorted(Path(dir).glob("*.png"), key=_natural_key)
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {dir}")

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
    text(surface, font, "ENTER: next field  BACKSPACE: edit", (-1, 224), GRID)

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
    text(surface, font, "GRINDING...", (-1, -1), AMBER)
    # TODO: spinning burr / particle animation

def draw_logging(surface: pygame.Surface, app: AppState, font, dt: float):
    # surface.fill(BG_COLOR)
    text(surface, font, "PULLING SHOT", (10, 4), GREEN)
    draw_live_graph(surface, app.live_points, rect=(10, 20, RES[0] - 20, RES[1] - 40), font=font)

def draw_live_graph(surface, points, rect, font):
    """
    Draws weight (green, left scale) and flow-rate (cyan, right scale) on
    the same plot area. Each series is auto-scaled independently -- they
    don't share units, so a shared scale would flatten one or the other.
    """
    x0, y0, w, h = rect
    pygame.draw.rect(surface, GRID, rect, width=1)
    if len(points) < 2:
        return

    ts = [p[0] for p in points]
    weights = [p[1] for p in points]
    flows = [p[2] for p in points]

    max_t = max(ts) or 1.0
    max_w = max(weights) or 1.0
    min_f, max_f = min(flows + [0.0]), max(flows + [0.1])
    span_f = (max_f - min_f) or 1.0

    def to_x(t):
        return x0 + int((t / max_t) * w)

    weight_pts = [(to_x(t), y0 + h - int((wt / max_w) * h)) for t, wt in zip(ts, weights)]
    flow_pts = [(to_x(t), y0 + h - int(((f - min_f) / span_f) * h)) for t, f in zip(ts, flows)]

    pygame.draw.lines(surface, GREEN, False, weight_pts, 1)
    pygame.draw.lines(surface, CYAN, False, flow_pts, 1)

    text(surface, font, f"{max_w:.0f}g", (x0 + 2, y0 + 1), GREEN)
    text(surface, font, f"{max_f:.3f}g/s", (x0 + w - 42, y0 + 1), CYAN)

def draw_result(surface: pygame.Surface, app: AppState, font, dt: float):
    surface.fill(BG_COLOR)
    label = app.result_label or "?"
    text(surface, font, f"RESULT: {label.upper()}", (10, 10), AMBER)
    y = 24
    if app.result_probs:
        for lab, p in app.result_probs.items():
            text(surface, font, f"{lab:9s} {p:.2f}", (10, y), GREEN)
            y += 10
    if app.rec:
        text(surface, font, app.rec, (10, y + 6), AMBER)

DRAW_FUNCS = {
    State.IDLE : draw_idle,
    State.BOOT: draw_boot,
    State.LABELING: draw_labeling,
    State.GRINDING: draw_grinding,
    State.LOGGING: draw_logging,
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
                app.grind_event.set()

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


async def _demo():
    import random

    app = AppState()
    app.result_timeout = 3.0  # short for the demo; use minutes in real use
    display_task = asyncio.create_task(run_display(app))

    print("Sitting in IDLE -- press any key in the window to start...")
    await app.start_event.wait()
    app.start_event.clear()
    print("Key pressed -- boot animation playing")

    await asyncio.sleep(2)
    app.state = State.WEIGHING
    await asyncio.sleep(2)
    app.state = State.GRINDING
    await asyncio.sleep(2)
    app.state = State.LOGGING

    t, w = 0.0, 0.0
    for _ in range(150):
        t += 0.1
        w += random.uniform(0.3, 0.6)
        app.add_point(t, w)
        await asyncio.sleep(0.03)

    app.state = State.RESULTS
    app.result_label = "balanced"
    app.result_probs = {"under": 0.12, "balanced": 0.71, "over": 0.17}
    app.rec = "grind slightly finer next time"
    print(f"On RESULT -- will auto-return to IDLE after {app.result_timeout}s")

    await asyncio.sleep(app.result_timeout + 1.0)
    print("Back to IDLE:", app.state)

    app.quit = True
    await display_task

if __name__ == "__main__":
    asyncio.run(_demo())