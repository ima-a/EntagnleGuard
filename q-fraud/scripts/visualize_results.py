"""
Results Visualization
=====================
Create hackathon visualizations for demo.

Usage:
    python scripts/visualize_results.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import RealAmplitudes


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CIRCUITS_DIR = BASE_DIR / "circuits"


# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#ccc",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "text.color": "#eee",
    "grid.color": "#2a2d3a",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})

CLASSICAL_COLOR = "#4cc9f0"
QUANTUM_COLOR = "#f72585"


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
    print(f"  Classical accuracy: {classical_metrics['accuracy']:.4f}")
    print(f"  Quantum accuracy:   {quantum_metrics['accuracy']:.4f}")

    return classical_metrics, quantum_metrics


def create_accuracy_comparison(
    classical_metrics: dict, quantum_metrics: dict, output_dir: Path
) -> None:
    """Create bar chart comparing classical vs quantum accuracy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["Classical\n(Random Forest)", "Quantum\n(VQC)"]
    accuracies = [classical_metrics["accuracy"], quantum_metrics["accuracy"]]
    colors = [CLASSICAL_COLOR, QUANTUM_COLOR]

    bars = ax.bar(models, accuracies, color=colors, width=0.5, edgecolor="#444", linewidth=2)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2%}",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Classical vs Quantum Model Accuracy\nCredit Card Fraud Detection", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="#ffd166", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend(loc="upper right")

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    output_path = output_dir / "accuracy_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)

    print(f"\nSaved: {output_path}")


def create_metrics_comparison(
    classical_metrics: dict, quantum_metrics: dict, output_dir: Path
) -> None:
    """Create grouped bar chart for all metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    classical_values = [
        classical_metrics["accuracy"],
        classical_metrics["precision"],
        classical_metrics["recall"],
        classical_metrics["f1_score"],
    ]
    quantum_values = [
        quantum_metrics["accuracy"],
        quantum_metrics["precision"],
        quantum_metrics["recall"],
        quantum_metrics["f1_score"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, classical_values, width, label="Classical", color=CLASSICAL_COLOR, edgecolor="#444")
    bars2 = ax.bar(x + width / 2, quantum_values, width, label="Quantum", color=QUANTUM_COLOR, edgecolor="#444")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=9, color="white")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=9, color="white")

    output_path = output_dir / "metrics_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)

    print(f"Saved: {output_path}")


def create_ansatz_visualization(output_dir: Path) -> None:
    """Create ansatz circuit visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ansatz = RealAmplitudes(
        num_qubits=4,
        reps=2,
        entanglement="linear",
    )

    fig = ansatz.decompose().draw(
        output="mpl",
        style="iqp",
        fold=-1,
    )

    output_path = output_dir / "ansatz.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    """Main entry point."""
    try:
        classical_metrics, quantum_metrics = load_metrics(RESULTS_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run classical_baseline.py and quantum_vqc.py first.", file=sys.stderr)
        sys.exit(1)

    create_accuracy_comparison(classical_metrics, quantum_metrics, RESULTS_DIR)
    create_metrics_comparison(classical_metrics, quantum_metrics, RESULTS_DIR)
    create_ansatz_visualization(CIRCUITS_DIR)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
