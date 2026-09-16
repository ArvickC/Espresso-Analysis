from pathlib import Path
import logging

import numpy as np
import pandas as pd

from logging_setup import configure_logging
from anomaly_export_features import export_features
from anomaly_detector import detect_specified_anomalies
from matplotlib import pyplot as plt
from scipy.stats import norm

logger = logging.getLogger(__name__)

SHOTS_DIR = Path("./shots")
THRESHOLD = 2.5 # z-score threshold
DEFAULT_FEATURE = 'total_time_s'

def retrieve_anomaly(current_shot_id: str | None = None,
                     shots_dir: Path = SHOTS_DIR, out_dir: Path = SHOTS_DIR) -> tuple[dict, pd.DataFrame] | None:
    if current_shot_id is None:
        logger.info("No current shot ID provided. Exiting.")
        return None

    shots_df, dist_features = export_features(shots_dir, out_dir)
    flagged = detect_specified_anomalies(shots_dir, threshold=THRESHOLD, shot_id=current_shot_id)

    if flagged.empty:
        return None

    if len(flagged) > 1:
        logger.warning(f"Multiple anomaly rows found for shot {current_shot_id}; using the first one.")

    row = flagged.iloc[0]
    features = np.asarray(row["features"], dtype=object).ravel()
    zscores = np.asarray(row["zscores"], dtype=float).ravel()

    if features.size != zscores.size:
        logger.warning(f"Mismatch in features and z-scores length for shot {current_shot_id}.")
        return None

    shot_row = shots_df.loc[shots_df["shot_id"] == current_shot_id]
    if shot_row.empty:
        logger.warning(f"No shot row found for {current_shot_id}.")
        return None

    shot_values = shot_row.iloc[0]
    output_dict = {}

    for feature, zscore in zip(features, zscores):
        value = shot_values.get(feature, np.nan)
        logger.debug(f"Feature {feature}: Value={value}, Z-score={float(zscore)}")
        output_dict[str(feature)] = {
            'value': value,
            'zscore': float(zscore)
        }

    return output_dict, dist_features

def matplot_anomaly(current_shot_id: str, shots_dir: Path = SHOTS_DIR, out_dir: Path = SHOTS_DIR) -> None:
    result = retrieve_anomaly(current_shot_id, shots_dir, out_dir)

    if result is None:
        logger.debug(f"No anomalies detected for shot {current_shot_id}.")
        return

    output_dict, dist_features = result
    if output_dict is None:
        logger.info(f"No anomalies detected for shot {current_shot_id}.")
        return

    for feature, data in output_dict.items():
        value = data['value']
        zscore = data['zscore']
        dist = dist_features[dist_features['feature_name'] == feature]
        mean, std = dist['median'].values[0], dist['robust_spread'].values[0]
        logger.debug(f"Feature {feature}: Mean={mean}, Std={std}")
        matplot_anomaly_by_feature(feature, value, zscore, mean, std)

def matplot_anomaly_by_feature(feature: str, value: float, zscore: float,
                               mean: float, std: float, threshold: float = THRESHOLD) -> None:
    r = abs(zscore) + 1
    x = np.linspace(mean - r * std, mean + r * std, 100)
    y = norm.pdf(x, mean, std)

    plt.figure()
    plt.plot(x, y, label='Distribution', color='blue')
    plt.fill_between(x, y, color='blue', alpha=0.1)
    plt.axvline(value, color='red', linestyle='--', label=f'Shot Value: {value:.2f}')
    plt.axvline(mean + threshold * std, color='orange', linestyle=':', alpha=0.4,
                label=f'Mean + {threshold}σ: {mean + threshold * std:.2f}')

    plt.title(f"Feature: {feature}")
    plt.xlabel('Values', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    configure_logging()
    synth_dir = Path('./synthetic_shots')
    matplot_anomaly('synth_0026', shots_dir=synth_dir, out_dir=synth_dir)