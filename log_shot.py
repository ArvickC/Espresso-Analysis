import asyncio
from pathlib import Path

from bleak import BleakClient

from ShotLogger import ShotLogger
from BLE_logger import *
from display import AppState, State

def _expand_uuid(uuid: str) -> str:
    return f'0000{uuid.lower()}-0000-1000-8000-00805f9b34fb'

SERVICE_UUID = _expand_uuid("FFF0")
CHAR_UUID = _expand_uuid("FFF1")
CMD_UUID = _expand_uuid("FFF2")

async def wait_for_trigger(_button=None) -> None:
    """
    Adapted from debug_connection.py to support physical buttons
    """
    if _button is not None:
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        _button.when_pressed = lambda: loop.call_soon_threadsafe(event.set)
        await event.wait()
        _button.when_pressed = None
    else:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, input)

async def run_shot_session(device: BLEDevice, app: AppState) -> Path:
    app.state = State.BOOT
    async with BleakClient(device) as client: # Connect to device
        print(f"Connected to {device.name}")

        print("Instantiating Logger...")
        logger = ShotLogger(app=app)

        print("Subscribing to weight notifications...")
        await client.start_notify(CHAR_UUID, logger.handle_notification)

        print(f"Resetting timer...")
        await client.write_gatt_char(CMD_UUID, reset_timer_cmd(), response=False)

        # Prep the puck & tare scale
        app.state = State.PREPPING
        print("Place cup on scale...")
        await app.key_down_event.wait()
        app.key_down_event.clear()
        await client.write_gatt_char(CMD_UUID, tare_cmd(), response=False)

        print("Tared. Press to start logging a shot.")
        app.live_points.clear()
        app.state = State.LOGGING

        await app.key_down_event.wait()
        app.key_down_event.clear()

        logger.start_recording()

        await client.write_gatt_char(CMD_UUID, start_timer_cmd(), response=False)
        print("\nLogging... press to stop.")
        await app.key_down_event.wait()
        app.key_down_event.clear()

        logger.stop_recording()
        await client.write_gatt_char(CMD_UUID, stop_timer_cmd(), response=False)
        await client.stop_notify(CHAR_UUID)


        print("Recording stopped. Clearing Timer.")
        await client.write_gatt_char(CMD_UUID, reset_timer_cmd(), response=False)

        return logger.save()