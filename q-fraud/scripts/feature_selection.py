"""
Feature Selection for Quantum Fraud Detection
==============================================
Selects the top 4 most discriminating PCA features (V1–V28)
using Cohen's d effect size.

Usage:
    python scripts/feature_selection.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
CONFIG_PATH = BASE_DIR / "config" / "selected_features.json"
OUTPUT_PATH = BASE_DIR / "data" / "quantum_selected_dataset.csv"


def generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic demo data matching creditcard.csv structure."""
    print("⚠️  Dataset empty or missing – generating synthetic demo data...")
    print("   (Place the real Kaggle CSV in data/ for actual results)\n")
    np.random.seed(42)
    n_legit, n_fraud = 284315, 492
    n = n_legit + n_fraud

    # Create PCA-like features with different distributions for fraud/legit
    data = {}
    for i in range(1, 29):
        legit_vals = np.random.randn(n_legit)
        # Some features have stronger separation (simulating real data)
        if i in [14, 17, 12, 10, 4, 11]:
            fraud_vals = np.random.randn(n_fraud) + np.random.choice([-2, 2])
        else:
            fraud_vals = np.random.randn(n_fraud) + np.random.uniform(-0.5, 0.5)
        data[f"V{i}"] = np.concatenate([legit_vals, fraud_vals])

    data["Class"] = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Synthetic dataset created: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def load_data(filepath: Path) -> pd.DataFrame:
    """Load the credit card dataset from CSV."""
    if not filepath.exists():
        return generate_synthetic_data()

    # Check if file is empty
    if filepath.stat().st_size == 0:
        return generate_synthetic_data()

    df = pd.read_csv(filepath)
    if df.empty:
        return generate_synthetic_data()

    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def compute_cohens_d(df: pd.DataFrame, feature: str) -> float:
    """
    Compute Cohen's d effect size for a single feature.

    d = (mean_fraud - mean_legit) / pooled_std
    pooled_std = sqrt((std_fraud^2 + std_legit^2) / 2)
    """
    fraud = df[df["Class"] == 1][feature]
    legit = df[df["Class"] == 0][feature]

    mean_fraud = fraud.mean()
    mean_legit = legit.mean()
    std_fraud = fraud.std()
    std_legit = legit.std()

    pooled_std = np.sqrt((std_fraud ** 2 + std_legit ** 2) / 2)

    if pooled_std == 0:
        return 0.0

    d = (mean_fraud - mean_legit) / pooled_std
    return d


def select_top_features(df: pd.DataFrame, n_top: int = 4) -> list[tuple[str, float]]:
    """
    Rank PCA features (V1–V28) by absolute Cohen's d and return top n.

    Returns:
        List of (feature_name, cohens_d) tuples sorted by |d| descending.
    """
    pca_features = [f"V{i}" for i in range(1, 29)]

    scores = {}
    for feat in pca_features:
        d = compute_cohens_d(df, feat)
        scores[feat] = d

    # Sort by absolute Cohen's d descending
    ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    return ranked[:n_top]


def save_outputs(
    df: pd.DataFrame,
    top_features: list[tuple[str, float]],
    config_path: Path,
    output_path: Path,
) -> None:
    """
    Save selected features JSON and reduced dataset CSV.
    """
    feature_names = [f[0] for f in top_features]

    # Save JSON config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data = {"features": feature_names}
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"Saved config: {config_path}")

    # Save reduced dataset (selected features + Class)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = feature_names + ["Class"]
    df_reduced = df[cols]
    df_reduced.to_csv(output_path, index=False)
    print(f"Saved reduced dataset: {output_path} ({df_reduced.shape[1]} columns)")


def main():
    """Main entry point."""
    df = load_data(DATA_PATH)

    top_features = select_top_features(df, n_top=4)

    print("\nTop 4 features selected for quantum model:")
    for feat, d in top_features:
        print(f"  {feat}: {abs(d):.2f}")

    save_outputs(df, top_features, CONFIG_PATH, OUTPUT_PATH)
    print("\nFeature selection complete.")


if __name__ == "__main__":
    main()
