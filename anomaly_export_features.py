from pathlib import Path
import argparse
import logging
from logging_setup import configure_logging
import pandas as pd

from anomaly_shot_features import build_feature_table
from anomaly_detector import FEATURE_COLUMNS, FEATURE_LABELS, robust_z_scores

logger = logging.getLogger(__name__)

# Manifest columns to carry through into export.csv alongside current
# features, if present.
PASSTHROUGH_COLUMNS = [
    "label", "grind_setting", "dose_g", "bean_name",
    "roast_date", "open_date", "timestamp", "notes",
]

def build_shots_csv(feature_df: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"shot_id": feature_df["shot_id"]})
    for col in PASSTHROUGH_COLUMNS:
        if col in feature_df.columns:
            out[col] = feature_df[col]
    for col in FEATURE_COLUMNS:
        out[col] = feature_df[col]
        out[f"{col}_zscore"] = z[col]
    return out

def build_feature_reference_csv(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLUMNS:
        valid = feature_df[col].dropna()
        if len(valid) >= 3:
            median = float(valid.median())
            mad = float((valid - median).abs().median())
            spread = mad * 1.4826 if mad > 1e-9 else float(valid.std())
        else:
            median, spread = None, None
        rows.append(
            {
                "feature_name": col,
                "feature_label": FEATURE_LABELS[col],
                "median": median,
                "robust_spread": spread,
                "n_valid_shots": len(valid),
            }
        )
    return pd.DataFrame(rows)

def export_features(shots_dir: Path, out_dir: Path, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = build_feature_table(shots_dir)
    z = robust_z_scores(feature_df, FEATURE_COLUMNS)

    shots_df = build_shots_csv(feature_df, z)
    reference_df = build_feature_reference_csv(feature_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    shots_path = out_dir / "export.csv"
    reference_path = out_dir / "export_feature_reference.csv"

    if save:
        shots_df.to_csv(shots_path, index=False)
        reference_df.to_csv(reference_path, index=False)

        logger.info(f"Wrote {len(shots_df)} shots to {shots_path}")
        logger.info(f"Wrote {len(reference_df)} feature reference rows to {reference_path}")

    return shots_df, reference_df

def main():
    parser = argparse.ArgumentParser(description="Export shot features to tool-agnostic CSVs.")
    parser.add_argument("--shots-dir", type=Path, default=Path("./synthetic_shots"))
    parser.add_argument("--out-dir", type=Path, default=Path("./synthetic_shots"))
    args = parser.parse_args()

    export_features(args.shots_dir, args.out_dir)


if __name__ == "__main__":
    configure_logging()
    main()