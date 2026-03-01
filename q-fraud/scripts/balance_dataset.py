"""
Dataset Balancing for Quantum Fraud Detection
==============================================
Balance dataset using undersampling of majority class.

Usage:
    python scripts/balance_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "quantum_selected_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "balanced_creditcard.csv"


def load_data(filepath: Path) -> pd.DataFrame:
    """Load the selected features dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def print_class_distribution(df: pd.DataFrame, label: str) -> None:
    """Print class distribution."""
    counts = df["Class"].value_counts().sort_index()
    print(f"\n{label}:")
    print(f"  Legitimate (0): {counts.get(0, 0):,}")
    print(f"  Fraud      (1): {counts.get(1, 0):,}")
    print(f"  Total:          {len(df):,}")


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Undersample majority class to match minority class count."""
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    n_fraud = len(fraud)
    print(f"\nFraud samples: {n_fraud}")

    # Undersample legitimate class
    legit_undersampled = legit.sample(n=n_fraud, random_state=42)

    # Combine and shuffle
    balanced = pd.concat([fraud, legit_undersampled])
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced


def save_dataset(df: pd.DataFrame, filepath: Path) -> None:
    """Save balanced dataset to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"\nSaved balanced dataset: {filepath}")


def main():
    """Main entry point."""
    try:
        df = load_data(INPUT_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_class_distribution(df, "Before balancing")

    balanced_df = balance_dataset(df)

    print_class_distribution(balanced_df, "After balancing")

    save_dataset(balanced_df, OUTPUT_PATH)

    print("\nDataset balancing complete.")


if __name__ == "__main__":
    main()
