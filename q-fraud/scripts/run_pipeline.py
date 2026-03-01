#!/usr/bin/env python3
"""
Run Pipeline - Execute all steps with smart caching.
Skips training if models already exist.

Usage:
    python scripts/run_pipeline.py           # Skip training if models exist
    python scripts/run_pipeline.py --force   # Force retrain everything
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def run_script(name: str):
    """Run a script and return success status."""
    script_path = SCRIPTS_DIR / f"{name}.py"
    print(f"\n{'='*60}")
    print(f"  Running: {name}.py")
    print('='*60)
    result = subprocess.run([sys.executable, str(script_path)], cwd=BASE_DIR)
    return result.returncode == 0


def main():
    force = "--force" in sys.argv
    
    # Check what already exists
    classical_exists = (MODELS_DIR / "classical_model.pkl").exists()
    quantum_exists = (MODELS_DIR / "quantum_vqc.pkl").exists()
    data_exists = (DATA_DIR / "X_train.npy").exists()
    
    print("\n" + "="*60)
    print("  Q-FRAUD PIPELINE")
    print("="*60)
    print(f"  Force retrain: {force}")
    print(f"  Data prepared: {data_exists}")
    print(f"  Classical model: {'✓' if classical_exists else '✗'}")
    print(f"  Quantum model: {'✓' if quantum_exists else '✗'}")
    
    # Step 1-3: Data preparation (always run if data missing)
    if not data_exists or force:
        run_script("feature_selection")
        run_script("balance_dataset")
        run_script("prepare_quantum_data")
    else:
        print("\n⏭️  Skipping data preparation (already exists)")
    
    # Step 4: Classical baseline
    if not classical_exists or force:
        run_script("classical_baseline")
    else:
        print("\n⏭️  Skipping classical training (model exists)")
    
    # Step 5: Quantum feature map (always regenerate - fast)
    run_script("quantum_feature_map")
    
    # Step 6: Quantum VQC (skip if exists - slow)
    if not quantum_exists or force:
        run_script("quantum_vqc")
    else:
        print("\n⏭️  Skipping VQC training (model exists)")
    
    # Step 7-8: Comparison and visualization (always run)
    run_script("compare_models")
    run_script("visualize_results")
    
    print("\n" + "="*60)
    print("  ✅ PIPELINE COMPLETE")
    print("="*60)
    print("\n  Launch dashboard:")
    print("    streamlit run dashboard.py")
    print()


if __name__ == "__main__":
    main()
