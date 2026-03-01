"""
Classical Baseline Model
========================
Train classical classifier as baseline for comparison.
Uses Logistic Regression - a linear model that struggles with
complex feature interactions (where quantum excels).

Usage:
    python scripts/classical_baseline.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/test numpy arrays."""
    X_train = np.load(data_dir / "X_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_test = np.load(data_dir / "y_test.npy")

    print(f"Loaded data:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train Logistic Regression classifier."""
    print("\nTraining Logistic Regression classifier...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("Training complete.")
    return model


def evaluate_model(
    model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }

    print("\nClassical Model Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    return metrics


def save_outputs(
    model: LogisticRegression, metrics: dict, models_dir: Path, results_dir: Path
) -> None:
    """Save model and metrics."""
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / "classical_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nSaved model: {model_path}")

    # Save metrics
    metrics_path = results_dir / "classical_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics: {metrics_path}")


def main():
    """Main entry point."""
    try:
        X_train, X_test, y_train, y_test = load_data(DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run prepare_quantum_data.py first.", file=sys.stderr)
        sys.exit(1)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_outputs(model, metrics, MODELS_DIR, RESULTS_DIR)

    print("\nClassical baseline training complete.")


if __name__ == "__main__":
    main()
