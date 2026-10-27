from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import savgol_filter

FLOW_NOISE_FLOOR = 0.05 # g/s
FIRST_DRIP_THRESHOLD = 0.25 # g
LATE_PHASE_START = 0.8 # fraction

def _smooth_weight_and_flow(elapsed_s: npt.NDArray, weight_g: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    n = len(weight_g)
    window = min(7, n if n % 2 else n - 1)
    window = max(window, 5 if n >= 5 else n - (1 - n % 2))
    window = min(window, n if n % 2 else n - 1)
    polyorder = min(3, window - 1)

    dt = float(np.median(np.diff(elapsed_s))) if n > 1 else 1.0
    smoothed = savgol_filter(weight_g, window_length=window, polyorder=polyorder)
    flow = savgol_filter(weight_g, window_length=window, polyorder=polyorder, deriv=1, delta=dt)
    return smoothed, flow


def extract_shot_features(curve: pd.DataFrame, manifest_row: pd.Series) -> dict:
    """
    Compute a flat dict of summary features for a single shot.
    """
    elapsed = curve["elapsed_s"].to_numpy(dtype=float)
    weight = curve["weight_g"].to_numpy(dtype=float)
    smoothed, flow = _smooth_weight_and_flow(elapsed, weight)

    manifest_time = manifest_row.get("shot_time_s")
    total_time_s = (
        float(manifest_time) if pd.notna(manifest_time) else float(elapsed[-1])
    )

    final_yield_g = float(max(smoothed[-1], 0.0))
    avg_flow = final_yield_g / total_time_s if total_time_s > 0 else np.nan

    above_threshold = np.flatnonzero(smoothed >= FIRST_DRIP_THRESHOLD)
    first_drip_idx = int(above_threshold[0]) if len(above_threshold) else len(smoothed) - 1
    time_to_first_drip_s = float(elapsed[first_drip_idx])

    peak_flow_idx = int(np.argmax(flow))
    peak_flow = float(flow[peak_flow_idx])
    time_to_peak_flow_s = float(elapsed[peak_flow_idx])

    main_phase_mask = elapsed >= time_to_first_drip_s
    main_flow = flow[main_phase_mask]
    main_flow = main_flow[main_flow > FLOW_NOISE_FLOOR]
    if len(main_flow) >= 4:
        flow_mean = float(np.mean(main_flow))
        flow_std = float(np.std(main_flow))
        flow_cv = flow_std / flow_mean if flow_mean > 1e-6 else np.nan
    else:
        flow_cv = np.nan

    late_cutoff = elapsed[-1] * LATE_PHASE_START
    late_mask = elapsed >= late_cutoff
    if late_mask.sum() > 1 and final_yield_g > 1e-6:
        late_yield_g = float(smoothed[late_mask][-1] - smoothed[late_mask][0])
        late_flow_frac = late_yield_g / final_yield_g
    else:
        late_flow_frac = np.nan

    dose_g = manifest_row.get("dose_g")
    yield_ratio = (
        final_yield_g / dose_g if pd.notna(dose_g) and dose_g > 0 else np.nan
    )

    return {
        "shot_id": manifest_row["shot_id"],
        "label": manifest_row.get("label"),
        "dose_g": dose_g if pd.notna(dose_g) else np.nan,
        "grind_setting": manifest_row.get("grind_setting"),
        "total_time_s": total_time_s,
        "final_yield_g": final_yield_g,
        "avg_flow_gps": avg_flow,
        "time_to_first_drip_s": time_to_first_drip_s,
        "peak_flow_gps": peak_flow,
        "time_to_peak_flow_s": time_to_peak_flow_s,
        "flow_cv": flow_cv,
        "late_flow_frac": late_flow_frac,
        "yield_ratio": yield_ratio,
    }

def build_feature_table(shots_dir: Path) -> pd.DataFrame:
    shots_dir = Path(shots_dir)
    manifest = pd.read_csv(shots_dir / "manifest.csv")

    rows = []
    for _, row in manifest.iterrows():
        curve_path = shots_dir / row["curve_file"]
        curve = pd.read_csv(curve_path)
        rows.append(extract_shot_features(curve, row))

    return pd.DataFrame(rows)