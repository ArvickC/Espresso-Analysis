from pathlib import Path
from bleak import BleakClient
from ShotLogger import ShotLogger
from scale_control import *
from display import AppState, State
import logging

logger = logging.getLogger(__name__)

def _expand_uuid(uuid: str) -> str:
    """
    Converts a 16-bit or 32-bit UUID to a full 128-bit UUID string.
    :param uuid: The 16-bit or 32-bit UUID to convert.
    :return: The full 128-bit UUID string.
    """
    return f'0000{uuid.lower()}-0000-1000-8000-00805f9b34fb'

SERVICE_UUID = _expand_uuid("FFF0")
CHAR_UUID = _expand_uuid("FFF1")
CMD_UUID = _expand_uuid("FFF2")

async def dose_shot(device: BLEDevice, app: AppState):
    app.state = State.BOOT
    async with BleakClient(device) as client: # connect to device
        logging.info(f"Connected to {device.name}")

        logger.debug("Instantiating Shot Logger...")
        shot_logger = ShotLogger(app=app)

        logger.debug("Subscribing to weight notifications...")
        await client.start_notify(
            CHAR_UUID,
            lambda sender, data: shot_logger.handle_notification(sender, data, plot=False),
        )

        logger.debug("Place cup on scale. Press Enter to tare...")
        app.state = State.TARING
        await app.key_down_event.wait()
        app.key_down_event.clear()
        await client.write_gatt_char(CMD_UUID, tare_cmd(), response=False)

        logger.debug("Dosing...")
        app.state = State.DOSING
        await app.key_down_event.wait()
        app.key_down_event.clear()

        await client.stop_notify(CHAR_UUID)

async def run_shot_session(device: BLEDevice, app: AppState) -> Path:
    app.state = State.BOOT
    async with BleakClient(device) as client: # connect to device
        logger.info(f"Connected to {device.name}")

        logger.debug("Instantiating Shot Logger...")
        shot_logger = ShotLogger(app=app)

        logger.debug("Subscribing to weight notifications...")
        await client.start_notify(CHAR_UUID, shot_logger.handle_notification)

        logger.debug(f"Resetting timer...")
        await client.write_gatt_char(CMD_UUID, reset_timer_cmd(), response=False)

        # Prep the puck & tare scale
        app.state = State.TARING
        logger.debug("Place cup on scale...")
        await app.key_down_event.wait()
        app.key_down_event.clear()
        await client.write_gatt_char(CMD_UUID, tare_cmd(), response=False)

        logger.debug("Tared. Press to start logging a shot...")
        app.live_points.clear()
        app.state = State.LOGGING

        await app.key_down_event.wait()
        app.key_down_event.clear()

        shot_logger.start_recording()

        await client.write_gatt_char(CMD_UUID, start_timer_cmd(), response=False)
        logger.debug("Logging... press to stop.")
        await app.key_down_event.wait()
        app.key_down_event.clear()

        shot_logger.stop_recording()
        logger.debug("Stopped logging.")
        await client.write_gatt_char(CMD_UUID, stop_timer_cmd(), response=False)
        await client.stop_notify(CHAR_UUID)


        logger.debug("Recording stopped. Clearing Timer...")
        await client.write_gatt_char(CMD_UUID, reset_timer_cmd(), response=False)

        return shot_logger.save()