import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from display import AppState

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

    def save(self, path: Path = DEFAULTS_PATH) -> None:
        """
        Save the current defaults to a JSON file.
        """
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
    """
    Prompt the user for input, returning the default if no input is given.
    """
    warnings.warn(
        "_prompt() is deprecated; use display.request_form() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

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
    warnings.warn(
        "prompt_label() is deprecated; use display.request_choice() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    while True:
        raw = input(f"Label this shot {PROMPT_LABELS} (or first letter): ").strip().lower()
        if raw in PROMPT_LABELS:
            return raw
        for lab in PROMPT_LABELS:
            if lab.startswith(raw):
                return lab
        print(f"Didn't recognize '{raw}'. Please try again.")


async def pre_label_shot(defaults: ShotDefaults, app: "AppState") -> None:
    """
    Informational data collected before the shot is pulled.
    """
    from display import request_form  # lazy import

    fields = [
        ("Bean name", defaults.bean_name),
        ("Roast date (YYYY-MM-DD)", defaults.roast_date),
        ("Bag opened date (YYYY-MM-DD)", defaults.open_date),
        ("Dose (g)", defaults.dose_g),
        ("Grind setting", defaults.grind_setting),
    ]
    result = await request_form(app, fields)

    defaults.bean_name = result["Bean name"]
    defaults.roast_date = result["Roast date (YYYY-MM-DD)"]
    defaults.open_date = result["Bag opened date (YYYY-MM-DD)"]
    defaults.dose_g = result["Dose (g)"]
    app.dose = float(result["Dose (g)"])
    defaults.grind_setting = result["Grind setting"]
    defaults.save() # save to file

async def label_shot(defaults: ShotDefaults, curve_path: Path, app: "AppState") -> tuple[dict, str]:
    """
    Labeling intended to be done after the shot is pulled and tasted.
    """
    from display import State, request_choice # lazy import

    app.state = State.POST_LABELING
    label = await request_choice(app, "LABEL THIS SHOT", list(PROMPT_LABELS))

    row = {
        "shot_id": curve_path.stem,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "curve_file": curve_path.name,
        "label": label,
    }

    if label == "discard":
        row.update({k: "" for k in MANIFEST_COLUMNS if k not in row})
        return row, label

    # notes = input("Notes (tasting notes, channeling, etc; optional): ").strip()
    # TODO: implement a notes input form in the display module
    notes = ""

    row.update({
        "grind_setting": defaults.grind_setting,
        "dose_g": defaults.dose_g,
        "bean_name": defaults.bean_name,
        "roast_date": defaults.roast_date,
        "open_date": defaults.open_date,
        "notes": notes if notes is not None else "",
    })

    return row, label

def append_manifest(row: dict, manifest_path: Path = Path("./shots/manifest.csv")) -> None:
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