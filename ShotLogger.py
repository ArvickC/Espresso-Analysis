import csv
import time
from datetime import datetime
from pathlib import Path
import BLE_logger as varia
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from display import AppState

class ShotLogger:
    """
    A simple logger for recording espresso shot data from a varia aku scale.
    It records the elapsed time and weight during a shot, and can save the
    data to a CSV file. It can also update a GUI application state with live data points if provided.
    """
    def __init__(self, out_dir: str = './shots', app: "AppState | None" = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[tuple[float, float]] = []
        self.t0: float | None = None
        self.recording = False
        self.app = app

    def start_recording(self):
        self.t0 = time.monotonic()
        self.rows.clear()
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def handle_notification(self, _sender, data: bytearray):
        weight = varia.parse_weight(data)

        if weight is None: return
        if not self.recording or self.t0 is None: return

        elapsed = time.monotonic() - self.t0
        self.rows.append((elapsed, weight))
        if self.app is not None:
            self.app.add_point(elapsed, weight)
        # print(f"\r  t={elapsed:6.2f}s  weight={weight:7.2f}g", end="\n", flush=True)

    def save(self, prefix: str = '') -> Path:
        """
        Save the recorded shot data to a CSV file in the output directory.
        The filename will include the current date and time, and an optional prefix.
        :param prefix: Optional prefix for the filename
        :return: Path to the saved CSV file
        """
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"{prefix}shot_{date}.csv"

        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['elapsed_s', 'weight_g'])
            writer.writerows(self.rows)
        print(f'\n Saved {len(self.rows)} rows to {path}')
        return path