import asyncio
import logging
from logging_setup import configure_logging
from bleak import BleakClient
from ShotLogger import ShotLogger
from scale_control import find_scale, tare_cmd, start_timer_cmd, stop_timer_cmd, reset_timer_cmd
from log_shot import CHAR_UUID, CMD_UUID

logger = logging.getLogger(__name__)

try:
    from config import KNOWN_ADDRESS
except ImportError:
    KNOWN_ADDRESS = None

async def wait_for_enter() -> None:
    """
    Injects an input() call into the asyncio event loop
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, input)

async def main():
    device = await find_scale(known_address=KNOWN_ADDRESS)
    if device is None:
        logger.error("Device not found.")
        return

    async with BleakClient(device) as client:
        logger.info(f"Connected to {device.name}")

        logger.debug("Instantiating Shot Logger...")
        shot_logger = ShotLogger()

        logger.debug("Subscribing to weight notifications...")
        await client.start_notify(CHAR_UUID, shot_logger.handle_notification)

        logger.debug("Resetting timer...")
        await client.write_gatt_char(CMD_UUID, reset_timer_cmd(), response=False)

        logger.info("Place cup on scale. Press Enter to tare...")
        await wait_for_enter()
        await client.write_gatt_char(CMD_UUID, tare_cmd(), response=False)

        logger.info("Tared. Press Enter to start recording...")
        await wait_for_enter()
        shot_logger.start_recording()

        await client.write_gatt_char(CMD_UUID, start_timer_cmd(), response=False)
        logger.info("Started recording...")
        await wait_for_enter()

        shot_logger.stop_recording()
        logger.info("Stopped recording.")
        await client.write_gatt_char(CMD_UUID, stop_timer_cmd(), response=False)
        await client.stop_notify(CHAR_UUID)

        logger.debug("Clearing timer...")
        await client.write_gatt_char(CMD_UUID, reset_timer_cmd(), response=False)

        shot_logger.save('test_')

if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())