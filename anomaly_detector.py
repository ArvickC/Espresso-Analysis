from pathlib import Path
import numpy as np
import pandas as pd
from anomaly_shot_features import build_feature_table
import logging
from logging_setup import configure_logging

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "total_time_s",
    "final_yield_g",
    "avg_flow_gps",
    "time_to_first_drip_s",
    "peak_flow_gps",
    "time_to_peak_flow_s",
    "flow_cv",
    "late_flow_frac",
    "yield_ratio",
]

FEATURE_LABELS = {
    "total_time_s": "shot time",
    "final_yield_g": "final yield",
    "avg_flow_gps": "average flow rate",
    "time_to_first_drip_s": "pre-infusion / first-drip time",
    "peak_flow_gps": "peak flow rate",
    "time_to_peak_flow_s": "time to peak flow",
    "flow_cv": "flow rate variability (possible channeling)",
    "late_flow_frac": "share of yield poured late in the shot",
    "yield_ratio": "yield ratio",
}

MAD_TO_STD = 1.4826  # Conversion factor from MAD to standard deviation
MIN_VALID_SAMPLES = 3

def robust_z_scores(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    for col in columns:
        values = df[col]
        valid = values.dropna()
        if len(valid) < MIN_VALID_SAMPLES:
            z[col] = np.nan
            continue

        median = valid.median()
        mad = (valid - median).abs().median()
        if mad < 1e-8:
            std = valid.std()
            if std < 1e-8:
                z[col] = np.nan
                continue
            spread = std
        else:
            spread = mad * MAD_TO_STD

        z[col] = (values - median) / spread
    return z

def flag_shots_by_zscore(feature_df: pd.DataFrame, threshold: float = 2.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = robust_z_scores(feature_df, FEATURE_COLUMNS)
    flagged_mask = (z.abs() > threshold).any(axis=1)

    reports = []
    for idx in feature_df.index[flagged_mask]:
        reasons = []
        features = []
        zscores = []
        for col in FEATURE_COLUMNS:
            zval = z.loc[idx, col]
            if pd.notna(zval) and abs(zval) > threshold:
                direction = "high" if zval > 0 else "low"
                reasons.append(f"{FEATURE_LABELS[col]} is {direction} (z={zval:+.1f})")
                features.append(col)
                zscores.append(zval)
        reports.append({
            "shot_id": feature_df.loc[idx, "shot_id"],
            "label": feature_df.loc[idx, "label"],
            "features": np.asarray(features, dtype=object).ravel(),
            "zscores": np.asarray(zscores, dtype=float).ravel(),
            "reasons": "; ".join(reasons),
        })
    return pd.DataFrame(reports), z


def flag_shots_by_isolation_forest(feature_df: pd.DataFrame, contamination: str | float = "auto", random_state: int = 0) -> pd.Series:
    from sklearn.ensemble import IsolationForest

    X = feature_df[FEATURE_COLUMNS].copy()
    X = X.fillna(X.median(numeric_only=True))
    # If a column is entirely NaN, median() is also NaN -- drop it rather than
    # feed the model an all-zero/all-NaN column.
    X = X.dropna(axis=1, how="all")

    if X.empty or len(X) < 8:
        return pd.Series(False, index=feature_df.index)

    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(X)  # -1 = outlier, 1 = inlier
    return pd.Series(predictions == -1, index=feature_df.index)


def detect_anomalies(shots_dir: Path, threshold: float) -> pd.DataFrame:
    feature_df = build_feature_table(shots_dir)
    flagged_df, _ = flag_shots_by_zscore(feature_df, threshold=threshold)

    logger.info(f"Loaded {len(feature_df)} shots from {shots_dir}")

    if flagged_df.empty:
        logger.info(f"No z-score anomalies flagged (threshold={threshold}).")
    else:
        logger.info(f"Z-score anomalies -- {len(flagged_df)} of {len(feature_df)} shots flagged (threshold={threshold}):")
        for _, row in flagged_df.iterrows():
            logger.debug(f"  {row['shot_id']} [{row['label']}]: {row['reasons']}")

    return flagged_df

def detect_specified_anomalies(shots_dir: Path, threshold: float,
                               shot_id: str | None = None) -> pd.DataFrame:
    if shot_id is None:
        logger.info(f"No shot ID provided. Exiting...")
        return detect_anomalies(shots_dir, threshold)

    feature_df = build_feature_table(shots_dir)
    flagged_df, _ = flag_shots_by_zscore(feature_df, threshold=threshold)
    flagged_df = flagged_df[flagged_df["shot_id"] == shot_id]
    logger.debug(f"Loaded {len(feature_df)} shots from {shots_dir}")

    if flagged_df.empty:
        logger.info(f"No anomalies flagged (threshold={threshold}).")
        return flagged_df

    for _, row in flagged_df.iterrows():
        logger.debug(f"  {row['shot_id']} [{row['label']}]: {row['reasons']}")

    return flagged_df

def main():
    shots_dir = Path('./synthetic_shots')
    threshold = 2.5
    output = shots_dir / 'flagged_shots.csv'

    flagged_df = detect_anomalies(
        shots_dir=shots_dir,
        threshold=threshold,
    )

    if not flagged_df.empty:
        flagged_df.to_csv(output, index=False)
        logger.info(f"Saved flagged-shot details to {output}")


if __name__ == "__main__":
    configure_logging()
    main()