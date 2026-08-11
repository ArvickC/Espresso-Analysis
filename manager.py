import asyncio
from pathlib import Path
from bleak import BLEDevice
from BLE_logger import find_scale
from log_shot import run_shot_session
from labeling_handler import pre_label_shot, label_shot, ShotDefaults, append_manifest
from model import update_model, predict_shot, LABELS
from display import AppState, run_display, State

try:
    from config import KNOWN_ADDRESS, GRIND_SETTING, DOSE_G, BEAN_NAME, ROAST_DATE, OPEN_DATE
except ImportError:
    KNOWN_ADDRESS = None
    GRIND_SETTING = None
    DOSE_G = None
    BEAN_NAME = None
    ROAST_DATE = None
    OPEN_DATE = None

defaults = ShotDefaults.load(fallback=ShotDefaults(GRIND_SETTING, DOSE_G, BEAN_NAME, ROAST_DATE, OPEN_DATE))
model_path = Path("./shot_cnn_{}.pt") # Saved Model

async def pull_shot(app: AppState):
    app.state = State.BOOT
    device = await find_scale(known_address=KNOWN_ADDRESS)
    if device is None:
        print("No varia AKU found.")
        return

    await pre_label_shot(defaults, app)
    app.state = State.GRINDING

    await app.grind_event.wait()
    app.grind_event.clear()

    path = await run_shot_session(device, app=app)

    if model_path.exists():
        # predict shot
        lab, probs = predict_shot(path, model_path)
        app.result_label = lab
        app.result_probs = probs
        print(f"Prediction: {lab}")
        for lab, p in zip(LABELS, probs):
            print(f"  {lab:9s} {p:.3f}")

    row, label = await label_shot(defaults, path)
    append_manifest(row)
    if not app.result_label:
        app.result_label = label

    app.state = State.RESULTS

    if label != 'discard':
        update_model(Path("./shots"))

async def main():
    app = AppState()
    app.result_timeout = 120 # seconds

    asyncio.create_task(run_display(app, fullscreen=False))
    while True:
        await app.start_event.wait()
        app.start_event.clear()
        await pull_shot(app)

if __name__ == "__main__":
    asyncio.run(main())