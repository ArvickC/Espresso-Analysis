"""
A simple script to test the BLE connection and commands to the varia aku scale.
"""

import asyncio
from bleak import BleakClient
from ShotLogger import ShotLogger
from BLE_logger import *
from log_shot import CHAR_UUID, CMD_UUID

try:
    from config import KNOWN_ADDRESS
except ImportError:
    KNOWN_ADDRESS = None

async def wait_for_enter() -> None:
    """
    Wait for Enter without blocking the event loop (BLE notifications
    are dispatched on this loop, so a plain input() call would stall them).
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, input)

async def run_shot_session():
    device = await find_scale(known_address=KNOWN_ADDRESS)
    if device is None:
        return

    async with BleakClient(device) as client:
        print(f"Connected to {device.name}")

        print("Instantiating Logger...")
        logger = ShotLogger()

        print("Subscribing to weight notifications...")
        await client.start_notify(CHAR_UUID, logger.handle_notification)

        print(f"Resetting timer...")
        await client.write_gatt_char(CMD_UUID, reset_timer_cmd(), response=False)

        print("Place cup on scale. Press Enter to tare.")
        await wait_for_enter()
        await client.write_gatt_char(CMD_UUID, tare_cmd(), response=False)

        print("Tared. Press Enter to start recording.")
        await wait_for_enter()
        logger.start_recording()

        await client.write_gatt_char(CMD_UUID, start_timer_cmd(), response=False)
        print("\nLogging... press Enter to stop.")
        await wait_for_enter()

        logger.stop_recording()
        print("Stopped recording.")
        await client.write_gatt_char(CMD_UUID, stop_timer_cmd(), response=False)
        await client.stop_notify(CHAR_UUID)

        # Clear timer
        await client.write_gatt_char(CMD_UUID, reset_timer_cmd(), response=False)

        logger.save('test_')

asyncio.run(run_shot_session())