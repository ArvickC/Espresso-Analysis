from bleak import BleakScanner, BLEDevice

DEVICE_NAME_HINTS = ("varia aku", "aku mini", "varia")

def _xor(bytes_: list[int]) -> int:
    return bytes_[0] ^ bytes_[1] ^ bytes_[2]

def parse_weight(raw: bytes) -> float | None:
    """
    Parses the weight from the raw data received from the scale
    :param raw: Bytes received from scale
    :return: Weight in grams
    """
    if len(raw) < 6: return None
    if raw[1] != 0x01: return None

    sign = -1 if (raw[3] & 0x10) else 1
    magnitude = ((raw[3] & 0x0F) << 16) + (raw[4] << 8) + raw[5]
    return sign * magnitude / 100.0

# Courtesy of Beanconqueror
# Various commands to control the varia aku mini scale
def tare_cmd() -> bytes:
    """
    Returns the command to tare the scale
    """
    body = [0x82, 0x01, 0x01]
    return bytes([0xFA, *body, _xor(body)])

def start_timer_cmd() -> bytes:
    """
    Returns the command to start the timer on the scale
    """
    body = [0x88, 0x01, 0x01]
    return bytes([0xFA, *body, _xor(body)])

def stop_timer_cmd() -> bytes:
    """
    Returns the command to stop the timer on the scale
    """
    body = [0x89, 0x01, 0x01]
    return bytes([0xFA, *body, _xor(body)])

def reset_timer_cmd() -> bytes:
    """
    Returns the command to reset the timer on the scale
    """
    body = [0x8A, 0x01, 0x01]
    return bytes([0xFA, *body, _xor(body)])

async def find_scale(timeout: float = 10.0, retries: int = 3,
                     known_address: str | None = None) -> BLEDevice | None:
    """
    Searches for and returns a BLEDevice representing the varia aku mini
    :param timeout: Timeout in seconds
    :param retries: Number of times to retry the scan
    :param known_address: Optionally use a known address
    """
    if known_address:
        print(f"Looking for known device at {known_address}...")
        device = await BleakScanner.find_device_by_address(known_address, timeout=timeout)
        if device is not None:
            print(f"Found scale! [{device.address}]")
            return device
        print(f"Known address not found. Falling back to scan...")

    for i in range(1, retries + 1):
        print(f"Scanning for scale... ({i}/{retries})")

        devices = await BleakScanner.discover(timeout=timeout, scanning_mode='active')
        for d in devices:
            name = (d.name or "").lower()
            if any(hint in name for hint in DEVICE_NAME_HINTS):
                print(f"Found {name} [{d.address}]")
                return d

        if i < retries:
            print(f"No device found. Retrying...")
        else:
            print("No device found. Aborting.")

    return None