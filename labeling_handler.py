import csv
import json
from datetime import datetime
from pathlib import Path

PROMPT_LABELS = {'under', 'balanced', 'over', 'discard'}
MANIFEST_COLUMNS = [
    "shot_id", "timestamp", "curve_file", "label",
    "grind_setting", "dose_g", "bean_name", "roast_date", "open_date", "notes",
]
DEFAULTS_PATH = Path("./shot_defaults.json")

class ShotDefaults:
    def __init__(self,
                 grind_setting = "",
                 dose_g = "16.0",
                 bean_name = "",
                 roast_date = "",
                 open_date = ""
        ):
        self.grind_setting = grind_setting
        self.dose_g = dose_g
        self.bean_name = bean_name
        self.roast_date = roast_date
        self.open_date = open_date

    def save(self, path: Path = DEFAULTS_PATH):
        data = {
            "grind_setting": self.grind_setting,
            "dose_g": self.dose_g,
            "bean_name": self.bean_name,
            "roast_date": self.roast_date,
            "open_date": self.open_date,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path = DEFAULTS_PATH, fallback: "ShotDefaults | None" = None) -> "ShotDefaults":
        """
        Load previously saved defaults if present, else fall back to the
        given ShotDefaults (e.g. built from config.py) or blank defaults.
        """
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        return fallback if fallback is not None else cls()

def _prompt(prompt_text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt_text}{suffix}: ").strip()
    if raw:
        default = raw
        return raw
    return default

def prompt_label() -> str:
    """
    Prompt the user to label a shot, returning one of the PROMPT_LABELS.
    """
    while True:
        raw = input(f"Label this shot {PROMPT_LABELS} (or first letter): ").strip().lower()
        if raw in PROMPT_LABELS:
            return raw
        for lab in PROMPT_LABELS:
            if lab.startswith(raw):
                return lab
        print(f"Didn't recognize '{raw}'. Please try again.")


async def pre_label_shot(defaults: ShotDefaults):
    """
    Informational data collected before the shot is pulled.
    """
    print(f"\n--- Labeling Shot ---")
    defaults.bean_name = _prompt("Bean name", defaults.bean_name)
    defaults.roast_date = _prompt("Roast date (YYYY-MM-DD)", defaults.roast_date)
    defaults.open_date = _prompt("Bag opened date (YYYY-MM-DD)", defaults.open_date)
    defaults.dose_g = _prompt("Dose (g)", defaults.dose_g)
    defaults.grind_setting = _prompt("Grind setting", defaults.grind_setting)
    defaults.save()

async def label_shot(defaults: ShotDefaults, curve_path: Path) -> tuple[dict, str]:
    """
    Labeling intended to be done after the shot is pulled and tasted.
    """
    label = prompt_label()

    row = {
        "shot_id": curve_path.stem,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "curve_file": curve_path.name,
        "label": label,
    }

    if label == "discard":
        row.update({k: "" for k in MANIFEST_COLUMNS if k not in row})
        return row, label

    notes = input("Notes (tasting notes, channeling, etc; optional): ").strip()

    row.update({
        "grind_setting": defaults.grind_setting,
        "dose_g": defaults.dose_g,
        "bean_name": defaults.bean_name,
        "roast_date": defaults.roast_date,
        "open_date": defaults.open_date,
        "notes": notes,
    })

    return row, label

def append_manifest(row: dict, manifest_path: Path = Path("./shots/manifest.csv")):
    """
    Append a row to the manifest CSV file, creating it if it doesn't exist.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = manifest_path.exists()

    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged to {manifest_path} (label={row['label']})")