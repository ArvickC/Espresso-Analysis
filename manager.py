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

REC_THRESHOLD = 0.07

defaults = ShotDefaults.load(fallback=ShotDefaults(GRIND_SETTING, DOSE_G, BEAN_NAME, ROAST_DATE, OPEN_DATE))
model_path = Path("./shot_cnn_{}.pt") # Saved model

async def pull_shot(app: AppState) -> None:
    # Boot and search for scale
    app.state = State.BOOT
    device = await find_scale(known_address=KNOWN_ADDRESS)
    if device is None:
        print("No varia AKU found.")
        return

    # Get pre-shot information
    await pre_label_shot(defaults, app)
    app.state = State.GRINDING
    await app.key_down_event.wait() # wait to continue
    app.key_down_event.clear()

    # Pull shot
    path = await run_shot_session(device, app=app)

    if model_path.exists():
        # Predict shot
        lab, probs = predict_shot(path, model_path)
        app.result_label = lab
        app.result_probs = dict(zip(LABELS, (float(p) for p in probs)))
        print(f"Prediction: {lab}")
        for lab, p in zip(LABELS, probs):
            print(f"  {lab:9s} {p:.3f}")

    # Label shot for model training
    row, label = await label_shot(defaults, path, app)
    print("Label: " + label)
    append_manifest(row)
    if not app.result_label: # if model did not predict
        app.result_label = label

    # Lazy recommendations
    # TODO: implement a more sophisticated recommendation system to adjust grind setting
    if app.result_probs:
        diff = app.result_probs['over'] - app.result_probs['under']
        if diff > REC_THRESHOLD:
            app.rec = "grind coarser"
        elif diff < -REC_THRESHOLD:
            app.rec = "grind finer"
        else:
            app.rec = "extraction is balanced"
    else:
        app.rec = "no recommendation"

    # Display results
    app.state = State.RESULTS

    if label != 'discard': # retrain model
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