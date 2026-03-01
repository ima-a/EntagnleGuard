# Q-Fraud: Quantum Credit Card Fraud Detection

A quantum-classical hybrid machine learning system for detecting fraudulent credit card transactions using Variational Quantum Classifiers (VQC).

**Team: Volunteers**

## Overview

Q-Fraud leverages quantum computing to identify patterns in financial transaction data that may be difficult for classical algorithms to detect. The system uses Qiskit's VQC implementation with ZZFeatureMap encoding and RealAmplitudes ansatz to classify transactions as fraudulent or legitimate.

## Features

- **Quantum VQC Model**: 4-qubit variational classifier with ZZ entanglement
- **Classical Baseline**: Logistic Regression for performance comparison
- **Interactive Dashboard**: Streamlit-based UI for real-time predictions
- **Automated Pipeline**: End-to-end data processing and model training
- **Visualization**: Quantum circuit diagrams and performance charts

## Project Structure

```
q-fraud/
├── data/                   # Datasets and processed data
├── scripts/                # Pipeline scripts
│   ├── feature_selection.py
│   ├── balance_dataset.py
│   ├── prepare_quantum_data.py
│   ├── classical_baseline.py
│   ├── quantum_feature_map.py
│   ├── quantum_vqc.py
│   ├── compare_models.py
│   └── visualize_results.py
├── models/                 # Trained model files
├── circuits/               # Quantum circuit visualizations
├── results/                # Performance metrics and charts
├── config/                 # Configuration files
├── docs/                   # Documentation
│   ├── architecture.md
│   ├── technologies.md
│   └── workflow.md
├── eda/                    # Exploratory data analysis
└── dashboard.py            # Streamlit application
```

## Quick Start

### 1. Install Dependencies

```bash
pip install qiskit qiskit-machine-learning qiskit-algorithms
pip install scikit-learn pandas numpy matplotlib streamlit kaggle joblib
```

### 2. Download Dataset

```bash
kaggle datasets download mlg-ulb/creditcardfraud -p q-fraud/data --unzip
```

### 3. Run Pipeline

```bash
cd q-fraud
python scripts/feature_selection.py
python scripts/balance_dataset.py
python scripts/prepare_quantum_data.py
python scripts/classical_baseline.py
python scripts/quantum_feature_map.py
python scripts/quantum_vqc.py
python scripts/compare_models.py
python scripts/visualize_results.py
```

### 4. Launch Dashboard

```bash
streamlit run dashboard.py
```

## Methodology

### Feature Selection
Uses Cohen's d statistic to identify the 4 most discriminative PCA features (V14, V4, V11, V12) from the original 28.

### Data Preprocessing
1. **Balancing**: Undersample majority class (492 fraud + 492 legit)
2. **Scaling**: StandardScaler → MinMaxScaler [0, 2π] for quantum rotation gates

### Quantum Model
- **Feature Map**: ZZFeatureMap (4 qubits, reps=3, full entanglement)
- **Ansatz**: RealAmplitudes (4 qubits, reps=3, full entanglement)
- **Optimizer**: COBYLA (300 iterations)
- **Sampler**: StatevectorSampler (exact simulation)

### Classical Baseline
- Logistic Regression with default parameters

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Classical (LR) | ~90% | ~90% | ~90% | ~90% |
| Quantum (VQC) | See results/quantum_metrics.json | | | |

## Documentation

- [Architecture](docs/architecture.md) - System design and components
- [Technologies](docs/technologies.md) - Tech stack details
- [Workflow](docs/workflow.md) - Step-by-step execution guide

## Dataset

**Kaggle Credit Card Fraud Detection Dataset**
- 284,807 European cardholder transactions (September 2013)
- 492 frauds (0.172% of all transactions)
- Features V1-V28: PCA-transformed for confidentiality
- [Source](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Requirements

- Python 3.10+
- Qiskit 2.3.0
- 4GB+ RAM

## License

MIT License

---

*Built with ❤️ by Team Volunteers*
