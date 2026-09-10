import asyncio
from pathlib import Path
from bleak import BLEDevice, BleakClient
from scale_control import find_scale
from log_shot import run_shot_session, dose_shot
from labeling_handler import pre_label_shot, label_shot, ShotDefaults, append_manifest
from model import update_model, predict_shot, LABELS
from display import AppState, run_display, State
from recommend_grind import update_grind_model, load_gp, recommend_next_grind, update_appstate_rec
from logging_setup import configure_logging
import logging

logger = logging.getLogger(__name__)

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
model_path = Path("./shot_cnn_latest.pt") # Saved model
gp_path = Path(f"./gp_models/{defaults.bean_name}_gp_latest.pkl") # Saved GP model

async def pull_shot(app: AppState) -> None:
    # Boot and search for scale
    app.state = State.BOOT
    device = await find_scale(known_address=KNOWN_ADDRESS)
    if device is None:
        logger.error("Device not found.")
        return

    await pre_label_shot(defaults, app)

    await dose_shot(device, app)

    app.puck_prep_state = 0
    app.state = State.PREPPING
    await app.key_down_event.wait() # wait to continue
    app.key_down_event.clear()

    # Pull shot
    path = await run_shot_session(device, app=app)

    if model_path.exists():
        # Predict shot
        lab, probs = predict_shot(path, model_path)
        app.result_label = lab
        app.result_probs = dict(zip(LABELS, (float(p) for p in probs)))
        logger.info(f"Prediction: {lab}")
        for lab, p in zip(LABELS, probs):
            logger.debug(f"  {lab:9s} {p:.3f}")

    # Label shot for model training
    row, label = await label_shot(defaults, path, app)
    logger.debug("User label: " + label)
    append_manifest(row)

    if not app.result_label: # if model did not predict
        app.result_label = None

    gp = load_gp(gp_path)
    if gp is not None:
        result = recommend_next_grind(gp)
        g = result["grind"]
        defaults.previous_grind_rec = f"{g:.2f}"
        defaults.save()
        update_appstate_rec(result, app)
    else:
        app.grind_rec = None
        app.pred_time = None

    # Display results
    app.state = State.PREDICTED_LABEL

    if label != 'discard': # retrain models
        update_model(Path("./shots"))
        update_grind_model(Path("./shots"))

async def main():
    configure_logging(False)
    app = AppState()
    app.result_timeout = 90 # seconds

    asyncio.create_task(run_display(app, fullscreen=False))
    while True:
        await app.start_event.wait()
        app.start_event.clear()
        await pull_shot(app)

if __name__ == "__main__":
    asyncio.run(main())