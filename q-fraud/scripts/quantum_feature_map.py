"""
Quantum Feature Map Visualization
=================================
Create ZZFeatureMap circuit visualization.

Usage:
    python scripts/quantum_feature_map.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CIRCUITS_DIR = BASE_DIR / "circuits"


def create_feature_map(n_qubits: int = 4, reps: int = 2) -> ZZFeatureMap:
    """Create ZZFeatureMap circuit."""
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement="linear",
    )
    print(f"ZZFeatureMap created:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Reps: {reps}")
    print(f"  Entanglement: linear")
    print(f"  Depth: {feature_map.depth()}")
    return feature_map


def save_circuit_image(circuit: ZZFeatureMap, output_dir: Path) -> None:
    """Save circuit visualization as PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "zz_feature_map.png"

    fig = circuit.decompose().draw(
        output="mpl",
        style="iqp",
        fold=-1,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nSaved circuit image: {output_path}")


def main():
    """Main entry point."""
    try:
        feature_map = create_feature_map(n_qubits=4, reps=2)
        save_circuit_image(feature_map, CIRCUITS_DIR)
        print("\nQuantum feature map visualization complete.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
