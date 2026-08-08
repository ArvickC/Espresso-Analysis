from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import savgol_filter

LABELS = ['under', 'balanced', 'over']
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}

TARGET_HZ = 10.0
MAX_DURATION = 35.0     # s
FLOW_RATE_CLIP = 15.0   # g/s
SAVGOL_WINDOW = 5       # samples every 0.5s
SAVGOL_POLYORDER = 3
TARGET_LENGTH = int(MAX_DURATION * TARGET_HZ) # 350 datapoints

def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """
    Load the manifest file as a pandas DataFrame and excludes discarded shots
    """
    df = pd.read_csv(manifest_path)
    df = df[df['label'].isin(LABELS)].reset_index(drop=True)
    return df

def resample(t: npt.NDArray, weight: npt.NDArray) -> npt.NDArray:
    """
    Resample data to a fixed target length using linear interpolation.
    """
    grid = np.linspace(0, MAX_DURATION, TARGET_LENGTH, endpoint=False)
    return np.interp(grid, t, weight)

def weight_to_features(weight: npt.NDArray) -> npt.NDArray:
    """
    Calculate flow-rate and smooth the weight data.
    Returns a 2D array with smoothed weight and flow rate.
    """
    # flow = np.gradient(weight, 1.0 / TARGET_HZ)
    window = min(SAVGOL_WINDOW, len(weight) if len(weight) % 2 else len(weight) - 1)
    smoothed = savgol_filter(weight, window_length=window, polyorder=SAVGOL_POLYORDER)
    flow = savgol_filter(weight, window_length=window, polyorder=SAVGOL_POLYORDER,
                         deriv=1, delta=1.0 / TARGET_HZ)
    flow = np.clip(flow, -FLOW_RATE_CLIP, FLOW_RATE_CLIP)
    return np.stack([smoothed, flow], axis=0)

def build_dataset(shots_dir: Path):
    """
    Build a pandas DataFrame containing all shot curves.
    """
    manifest = load_manifest(shots_dir / 'manifest.csv')

    X = np.zeros((len(manifest), 2, TARGET_LENGTH), dtype=np.float32)
    y = np.zeros(len(manifest), dtype=np.int64)

    for i, row in manifest.iterrows():
        curve_path = shots_dir / row['curve_file']
        curve = pd.read_csv(curve_path)
        resampled = resample(curve['elapsed_s'].to_numpy(), curve['weight_g'].to_numpy())
        X[i] = weight_to_features(resampled)
        y[i] = LABEL_TO_INDEX[row['label']]

    return X, y

def normalize(X: npt.NDArray, mean = None, std = None):
    """
    Normalize the dataset X.
    If mean and std are not provided, they are computed from X.
    """
    if mean is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return (X - mean) / std, mean, std