"""
Model Comparison
================
Compare classical and quantum model performance.

Usage:
    python scripts/compare_models.py
"""

import json
import sys
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def load_metrics(results_dir: Path) -> tuple[dict, dict]:
    """Load classical and quantum metrics."""
    classical_path = results_dir / "classical_metrics.json"
    quantum_path = results_dir / "quantum_metrics.json"

    if not classical_path.exists():
        raise FileNotFoundError(f"Classical metrics not found: {classical_path}")
    if not quantum_path.exists():
        raise FileNotFoundError(f"Quantum metrics not found: {quantum_path}")

    with open(classical_path) as f:
        classical_metrics = json.load(f)
    with open(quantum_path) as f:
        quantum_metrics = json.load(f)

    print("Loaded metrics:")
    print(f"  Classical: {classical_path.name}")
    print(f"  Quantum:   {quantum_path.name}")

    return classical_metrics, quantum_metrics


def compare_models(classical_metrics: dict, quantum_metrics: dict) -> dict:
    """Compare model accuracies."""
    print("\nComparing models...")

    classical_acc = classical_metrics["accuracy"]
    quantum_acc = quantum_metrics["accuracy"]

    comparison = {
        "classical_accuracy": float(classical_acc),
        "quantum_accuracy": float(quantum_acc),
        "difference": float(quantum_acc - classical_acc),
        "classical_f1": float(classical_metrics["f1_score"]),
        "quantum_f1": float(quantum_metrics["f1_score"]),
    }

    print("\n" + "=" * 50)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(f"  Classical Accuracy: {classical_acc:.4f} ({classical_acc * 100:.2f}%)")
    print(f"  Quantum Accuracy:   {quantum_acc:.4f} ({quantum_acc * 100:.2f}%)")
    print(f"  Difference:         {comparison['difference']:+.4f}")
    print("=" * 50)

    if quantum_acc > classical_acc:
        print("  ✅ Quantum model outperforms classical!")
    elif quantum_acc < classical_acc:
        print("  📊 Classical model performs better")
    else:
        print("  🔄 Models perform equally")

    return comparison


def save_comparison(comparison: dict, results_dir: Path) -> None:
    """Save comparison results."""
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=4)
    print(f"\nSaved comparison: {output_path}")


def main():
    """Main entry point."""
    try:
        classical_metrics, quantum_metrics = load_metrics(RESULTS_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run classical_baseline.py and quantum_vqc.py first.", file=sys.stderr)
        sys.exit(1)

    comparison = compare_models(classical_metrics, quantum_metrics)
    save_comparison(comparison, RESULTS_DIR)

    print("\nModel comparison complete.")


if __name__ == "__main__":
    main()
