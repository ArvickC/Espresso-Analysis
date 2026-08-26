import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from pathlib import Path
import csv
import pickle
from datetime import datetime
from display import AppState

IDEAL_TIME_LOW = 25.0
IDEAL_TIME_HIGH = 30.0
IDEAL_TIME_MID = (IDEAL_TIME_LOW + IDEAL_TIME_HIGH) / 2
GRIND_MIN, GRIND_MAX = 2.0, 5.0 # fine -> coarse

MIN_SHOTS_FOR_GP = 4
GP_MODEL_PATH = Path("./gp_latest.pkl")

def _get_manifest(manifest_path: Path = Path("./shots/manifest.csv")) -> list[dict]:
    """
    Read the manifest CSV file and return a list of dictionaries representing relevant
    information from each row (grind setting, time elapsed, and label).
    :param manifest_path: Path to the manifest CSV file.
    :return: A list of dictionaries containing relevant information from the manifest.
    """
    manifest = []
    if not manifest_path.exists():
        return manifest

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] in ["under", "balanced", "over"]:
                try:
                    manifest.append({
                        "grind_setting": float(row["grind_setting"]),
                        "shot_time_s": float(row["shot_time_s"]),
                        "label": row["label"],
                    })
                except (ValueError, KeyError):
                    continue  # Skip rows with invalid numeric values
    return manifest

def save_gp(gp: GPR, path: Path = GP_MODEL_PATH):
    """
    Save the Gaussian Process model to a file with a timestamped filename and also to a fixed path.
    """
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = path.parent / f"gp_{date}.pkl"
    with open(timestamped, 'wb') as file:
        pickle.dump(gp, file)
    with open(path, 'wb') as file:
        pickle.dump(gp, file)
    print(f"Saved GP model to {timestamped} and {path}")

def load_gp(path: Path = GP_MODEL_PATH) -> GPR | None:
    """
    Load the Gaussian Process model from a file.
    """
    if not path.exists():
        return None
    with open(path, 'rb') as file:
        gp = pickle.load(file)
    return gp

def update_grind_model(shots_dir: Path = Path("./shots")) -> GPR | None:
    """
    Update the Gaussian Process model based on the manifest data in the specified shots directory.
    """
    manifest = _get_manifest(shots_dir / 'manifest.csv')
    if len(manifest) < MIN_SHOTS_FOR_GP:
        print(f"Not enough data...")
        return None

    gp = fit_gp(manifest)
    save_gp(gp)
    return gp

def fit_gp(manifest: list[dict]) -> GPR:
    """
    Fit a Gaussian Process Regressor to the grind setting and shot time data from the manifest.
    """
    X = np.array([[s['grind_setting']] for s in manifest], dtype=float)
    y = np.array([s['shot_time_s'] for s in manifest], dtype=float)

    # RBF kernel: smooth function, WhiteKernel: models noise
    kernel = (C(1.0, (1e-2, 1e3)) *
              RBF(length_scale=2.0, length_scale_bounds=(0.5, 20)) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 10)))
    gp = GPR(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
    gp.fit(X, y)
    return gp

def recommend_next_grind(gp: GPR, grind_min=GRIND_MIN, grind_max=GRIND_MAX,
                         n_candidates=200, kappa=1.5, exploration_dampening=0.25):
    """
    Recommend the next grind setting based on the fitted Gaussian Process model.
    :param gp: Gaussian Process Regressor fitted to the grind setting and shot time data.
    :param grind_min: Minimum grind setting to consider for recommendation.
    :param grind_max: Maximum grind setting to consider for recommendation.
    :param n_candidates: Number of candidate grind settings to evaluate between grind_min and grind_max.
    :param kappa: Exploration-exploitation trade-off parameter; higher values favor exploration.
    :param exploration_dampening: Dampening factor for the exploration bonus. Reduces the influence of uncertainty.
    :return: Recommendation dictionary containing the best grind setting, predicted shot time and uncertainty.
    """
    candidates = np.linspace(grind_min, grind_max, n_candidates).reshape(-1, 1)
    mean, std = gp.predict(candidates, return_std=True)

    dist_to_band = np.where(
        (mean >= IDEAL_TIME_LOW) & (mean <= IDEAL_TIME_HIGH),
        0.0,
        np.minimum(np.abs(mean - IDEAL_TIME_LOW), np.abs(mean - IDEAL_TIME_HIGH)),
    )

    closeness_score = -dist_to_band

    explore_bonus = exploration_dampening * kappa * std

    acquisition = closeness_score + explore_bonus
    best_index = np.argmax(acquisition)

    return {
        "grind": float(candidates[best_index, 0]),
        "predicted_time": float(mean[best_index]),
        "uncertainty": float(std[best_index]),
        "all_candidates": candidates.flatten(),
        "all_means": mean,
        "all_stds": std,
        "acquisition": acquisition,
    }

def explain(result, app: AppState):
    """
    Update the app state with a human-readable recommendation based on the GP model's output.
    """
    g = result["grind"]
    t = result["predicted_time"]
    u = result["uncertainty"]
    app.rec = f"Grind: {g:.2f}, Pred. Time: {t:.2f} ± {u:.1f}s"

def plot_gp(gp: GPR, manifest: list[dict], result: dict | None = None, save_path: Path | None = None):
    """
    Plot the Gaussian Process regression results along with the manifest data and recommended grind setting.
    :param gp: Fitted Gaussian Process Regressor.
    :param manifest: Manifest data containing grind settings, shot times, and labels.
    :param result: Optional recommendation result from recommend_next_grind; if None, it will be computed.
    :param save_path: Optional path to save the plot; if None, the plot will be displayed instead.
    :return: None
    """
    import matplotlib.pyplot as plt # lazy import

    if result is None:
        result = recommend_next_grind(gp)

    candidates = result["all_candidates"]
    means = result["all_means"]
    stds = result["all_stds"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhspan(IDEAL_TIME_LOW, IDEAL_TIME_HIGH, color="green", alpha=0.1,
               label=f"ideal ({IDEAL_TIME_LOW:.0f}-{IDEAL_TIME_HIGH:.0f}s)")

    ax.fill_between(candidates, means - 1.96 * stds, means + 1.96 * stds,
                    color="steelblue", alpha=0.2, label="95% confidence")
    ax.plot(candidates, means, color="steelblue", label="GP mean")

    label_colors = {"under": "tab:orange", "balanced": "tab:green", "over": "tab:red"}
    for lab, color in label_colors.items():
        xs = [s["grind_setting"] for s in manifest if s["label"] == lab]
        ys = [s["shot_time_s"] for s in manifest if s["label"] == lab]
        if xs:
            ax.scatter(xs, ys, color=color, label=lab, zorder=5, edgecolor="black", linewidth=0.5)

    ax.axvline(result["grind"], color="black", linestyle="--", linewidth=1,
               label=f"recommended: {result['grind']:.2f}")

    ax.set_xlabel("Grind setting (fine -> coarse)")
    ax.set_ylabel("Shot time (s)")
    ax.set_title("Grind vs. shot time -- GP fit")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(GRIND_MIN, GRIND_MAX)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    # quick manual check to fit on ./synthetic_shots/manifest.csv
    manifest = _get_manifest(Path("./synthetic_shots/manifest.csv"))
    if len(manifest) < MIN_SHOTS_FOR_GP:
        print(f"Need >= {MIN_SHOTS_FOR_GP} labeled shots, have {len(manifest)}.")
    else:
        gp = fit_gp(manifest)
        result = recommend_next_grind(gp)
        print(result['grind'], result['predicted_time'], result['uncertainty'])
        plot_gp(gp, manifest, result)