from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from preprocessing import build_dataset, normalize, LABELS, resample, weight_to_features

MIN_SHOTS_PER_CLASS = 5

class ShotCNN(nn.Module):
    def __init__(self, in_channels: int = 2, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

def train(shots_dir: Path, epochs: int = 30, batch_size: int = 16, lr: float = 1e-3, seed: int = 42):
    X, y = build_dataset(shots_dir)
    print(f"Loaded {len(y)} samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    X_train, mean, std = normalize(X_train)
    X_val, _, _ = normalize(X_val, mean, std)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    model = ShotCNN(in_channels=X.shape[1], n_classes=len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
        val_acc = correct / len(val_ds)
        best_val_acc = max(val_acc, best_val_acc)

        if epoch % 5 == 0:
            print(f"epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    print(f"\nBest val acc: {best_val_acc:.4f}")

    date = datetime.now().strftime("%Y%m%d_%H%M%S")

    torch.save({
        'model_state': model.state_dict(),
        'mean': mean,
        'std': std,
        'labels': LABELS,
    }, f"shot_cnn_{date}.pt")
    print(f"Saved model to shot_cnn_{date}.pt")

    return model, best_val_acc


def update_model(shots_dir: Path = Path("./shots"), epochs: int = 15) -> bool:
    manifest_path = shots_dir / "manifest.csv"
    if not manifest_path.exists():
        print("No manifest yet; nothing to train on.")
        return False

    df = pd.read_csv(manifest_path)
    df = df[df["label"].isin(LABELS)]
    counts = df["label"].value_counts()

    missing = [lab for lab in LABELS if counts.get(lab, 0) < MIN_SHOTS_PER_CLASS]
    if missing:
        print(
            f"Not enough data yet to retrain (need >= {MIN_SHOTS_PER_CLASS}/class); "
            f"still short on: {missing}. Current counts: {counts.to_dict()}"
        )
        return False

    print(f"Retraining on {len(df)} shots ({counts.to_dict()})...")
    train(shots_dir, epochs=epochs)
    return True

def load_checkpoint(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = ShotCNN(in_channels=2, n_classes=len(ckpt["labels"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt["labels"]

def predict_shot(curve_path: Path = Path("./shots"), model_path: Path = None):
    if not model_path:
        return None

    model, mean, std, labels = load_checkpoint(model_path)
    curve = pd.read_csv(curve_path)

    resampled = resample(curve['elapsed_s'].to_numpy(), curve['weight_g'].to_numpy())
    features = weight_to_features(resampled)

    x = (features - mean[0]) / std[0]
    x = torch.from_numpy(x).float().unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()

    pred_idx = int(probs.argmax())
    return labels[pred_idx], probs