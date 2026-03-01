"""
Variational Quantum Classifier (VQC)
====================================
Train quantum classifier using Qiskit Machine Learning.

Usage:
    python scripts/quantum_vqc.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_machine_learning.algorithms import VQC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# ── Training Config ──────────────────────────────────────────────────────────
N_TRAINING_SAMPLES = 500  # More samples for better learning
OPTIMIZER_MAXITER = 300   # More iterations for convergence
FEATURE_MAP_REPS = 3      # Deeper feature encoding
ANSATZ_REPS = 3           # More expressive ansatz
ENTANGLEMENT = "full"     # Full entanglement captures complex correlations


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


def create_vqc(n_features: int = 4) -> VQC:
    """Create Variational Quantum Classifier with enhanced configuration."""
    print("\nCreating VQC components...")

    # Feature map - deeper encoding for capturing quantum correlations
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=FEATURE_MAP_REPS,
        entanglement=ENTANGLEMENT,
    )
    print(f"  Feature map: ZZFeatureMap (reps={FEATURE_MAP_REPS}, entanglement={ENTANGLEMENT})")

    # Ansatz - more expressive variational circuit
    ansatz = RealAmplitudes(
        num_qubits=n_features,
        reps=ANSATZ_REPS,
        entanglement=ENTANGLEMENT,
    )
    print(f"  Ansatz: RealAmplitudes (reps={ANSATZ_REPS}, entanglement={ENTANGLEMENT})")

    # Optimizer - COBYLA with more iterations
    optimizer = COBYLA(maxiter=OPTIMIZER_MAXITER)
    print(f"  Optimizer: COBYLA (maxiter={OPTIMIZER_MAXITER})")

    # Sampler
    sampler = StatevectorSampler()
    print(f"  Backend: StatevectorSampler (statevector simulation)")

    # Create VQC
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )

    print("\nVQC created successfully.")
    return vqc


def train_vqc(
    vqc: VQC, X_train: np.ndarray, y_train: np.ndarray
) -> VQC:
    """Train the VQC model."""
    print("\nTraining VQC (this may take a while)...")
    print(f"  Training samples: {len(X_train)}")

    vqc.fit(X_train, y_train)

    print("Training complete.")
    return vqc


def evaluate_vqc(
    vqc: VQC, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Evaluate VQC and return metrics."""
    print("\nEvaluating VQC...")
    y_pred = vqc.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
    }

    print("\nQuantum VQC Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    return metrics


def save_outputs(
    vqc: VQC, metrics: dict, models_dir: Path, results_dir: Path
) -> None:
    """Save model weights and metrics."""
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights (VQC object has pickle issues with local functions)
    model_path = models_dir / "quantum_vqc.pkl"
    model_data = {
        "weights": vqc.weights.tolist() if vqc.weights is not None else None,
        "num_qubits": 4,
        "feature_map_reps": FEATURE_MAP_REPS,
        "ansatz_reps": ANSATZ_REPS,
        "entanglement": ENTANGLEMENT,
    }
    joblib.dump(model_data, model_path)
    print(f"\nSaved model weights: {model_path}")

    # Save metrics
    metrics_path = results_dir / "quantum_metrics.json"
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

    # Use more samples for better quantum learning
    n_samples = min(N_TRAINING_SAMPLES, len(X_train))
    X_train_subset = X_train[:n_samples]
    y_train_subset = y_train[:n_samples]
    print(f"\nUsing {n_samples} training samples for quantum training")

    vqc = create_vqc(n_features=X_train.shape[1])
    vqc = train_vqc(vqc, X_train_subset, y_train_subset)
    metrics = evaluate_vqc(vqc, X_test, y_test)
    save_outputs(vqc, metrics, MODELS_DIR, RESULTS_DIR)

    print("\nQuantum VQC training complete.")


if __name__ == "__main__":
    main()
