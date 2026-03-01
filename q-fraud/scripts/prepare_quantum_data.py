"""
Quantum Data Preparation
========================
Prepare scaled quantum-ready data with train/test split.
Adds feature interactions that quantum circuits naturally capture
through entanglement (ZZFeatureMap encodes x_i * x_j correlations).

Usage:
    python scripts/prepare_quantum_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "balanced_creditcard.csv"
OUTPUT_DIR = BASE_DIR / "data"


def load_data(filepath: Path) -> pd.DataFrame:
    """Load the balanced dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split dataframe into features and target."""
    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    return X, y


def scale_data(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale features using MinMaxScaler to [0, 2π] range.
    This is optimal for quantum feature maps which use rotations.
    """
    # First standardize
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    
    # Then scale to [0, 2π] for quantum rotations
    minmax_scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    X_train_scaled = minmax_scaler.fit_transform(X_train_std)
    X_test_scaled = minmax_scaler.transform(X_test_std)
    
    print("\nData scaled: StandardScaler → MinMaxScaler [0, 2π]")
    print("  (Optimal range for quantum rotation gates)")
    return X_train_scaled, X_test_scaled


def save_arrays(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Save numpy arrays to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    print(f"\nSaved arrays to {output_dir}:")
    print(f"  X_train.npy: {X_train.shape}")
    print(f"  X_test.npy:  {X_test.shape}")
    print(f"  y_train.npy: {y_train.shape}")
    print(f"  y_test.npy:  {y_test.shape}")


def main():
    """Main entry point."""
    try:
        df = load_data(INPUT_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    X, y = split_features_target(df)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain/Test split (80/20):")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Save arrays
    save_arrays(X_train_scaled, X_test_scaled, y_train, y_test, OUTPUT_DIR)

    print("\nQuantum data preparation complete.")


if __name__ == "__main__":
    main()
