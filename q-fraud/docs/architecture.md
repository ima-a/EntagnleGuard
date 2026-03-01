# System Architecture

## Overview

Q-Fraud is a quantum-classical hybrid fraud detection system that leverages Variational Quantum Classifiers (VQC) to identify fraudulent credit card transactions.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Q-Fraud Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Raw Data   │───▶│   Feature    │───▶│   Dataset Balancing      │  │
│  │  (284,807)   │    │  Selection   │    │  (Undersampling)         │  │
│  └──────────────┘    │  Cohen's d   │    │  492 fraud + 492 legit   │  │
│                      └──────────────┘    └──────────────────────────┘  │
│                             │                        │                  │
│                             ▼                        ▼                  │
│                      ┌──────────────┐    ┌──────────────────────────┐  │
│                      │ Top 4 PCA    │    │   Data Preparation       │  │
│                      │ Features     │    │   StandardScaler ──▶     │  │
│                      │ V14,V4,V11,  │    │   MinMaxScaler [0, 2π]   │  │
│                      │ V12          │    └──────────────────────────┘  │
│                      └──────────────┘               │                  │
│                                                     ▼                  │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                    Model Training                               │   │
│  │  ┌─────────────────────┐    ┌─────────────────────────────┐    │   │
│  │  │  Classical Model    │    │     Quantum VQC Model       │    │   │
│  │  │  Logistic Regression│    │  ZZFeatureMap + RealAmp     │    │   │
│  │  │  scikit-learn       │    │  Qiskit + StatevectorSampler│    │   │
│  │  └─────────────────────┘    └─────────────────────────────┘    │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                             │                        │                  │
│                             ▼                        ▼                  │
│                      ┌─────────────────────────────────────────────┐   │
│                      │           Model Comparison                   │   │
│                      │   Accuracy, Precision, Recall, F1 Score     │   │
│                      └─────────────────────────────────────────────┘   │
│                                          │                              │
│                                          ▼                              │
│                      ┌─────────────────────────────────────────────┐   │
│                      │        Streamlit Dashboard                   │   │
│                      │  • Real-time fraud prediction                │   │
│                      │  • Quantum circuit visualization             │   │
│                      │  • Model performance comparison              │   │
│                      └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Layer (`data/`)
- `creditcard.csv` - Raw Kaggle dataset (284,807 transactions)
- `quantum_selected_dataset.csv` - Dataset with selected features
- `balanced_creditcard.csv` - Balanced 50/50 dataset
- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` - Processed arrays

### 2. Processing Scripts (`scripts/`)
| Script | Function |
|--------|----------|
| `feature_selection.py` | Cohen's d ranking, selects V14, V4, V11, V12 |
| `balance_dataset.py` | Undersamples majority class |
| `prepare_quantum_data.py` | Scales data to [0, 2π] for quantum gates |
| `classical_baseline.py` | Trains Logistic Regression |
| `quantum_feature_map.py` | Creates ZZFeatureMap visualization |
| `quantum_vqc.py` | Trains VQC with COBYLA optimizer |
| `compare_models.py` | Generates comparison metrics |
| `visualize_results.py` | Creates performance charts |

### 3. Quantum Circuits (`circuits/`)
- `zz_feature_map.png` - ZZFeatureMap circuit diagram
- `ansatz.png` - RealAmplitudes ansatz diagram

### 4. Models (`models/`)
- `classical_model.pkl` - Trained Logistic Regression
- `quantum_vqc.pkl` - VQC weights and configuration

### 5. Results (`results/`)
- `classical_metrics.json` / `quantum_metrics.json` - Performance metrics
- `comparison.json` - Side-by-side comparison
- `accuracy_comparison.png` / `metrics_comparison.png` - Visualizations

### 6. Dashboard (`dashboard.py`)
Four-page Streamlit application:
1. **Fraud Predictor** - Real-time prediction interface
2. **Quantum Circuits** - Interactive circuit visualizations
3. **Model Comparison** - Performance analytics
4. **About** - Project information

## Quantum Circuit Architecture

### ZZFeatureMap (Data Encoding)
```
|0⟩ ─── H ─── P(2x₁) ─── ZZ(x₁x₂) ─── ... ─── (repeat 3x)
|0⟩ ─── H ─── P(2x₂) ─── ZZ(x₁x₃) ─── ... ─── (repeat 3x)
|0⟩ ─── H ─── P(2x₃) ─── ZZ(x₂x₃) ─── ... ─── (repeat 3x)
|0⟩ ─── H ─── P(2x₄) ─── ZZ(x₃x₄) ─── ... ─── (repeat 3x)
```

### RealAmplitudes Ansatz (Variational Layer)
```
|q₀⟩ ─── RY(θ₀) ─── CX ─── RY(θ₄) ─── CX ─── ... ─── (repeat 3x)
|q₁⟩ ─── RY(θ₁) ─── ●  ─── RY(θ₅) ─── ●  ─── ... ─── (repeat 3x)
|q₂⟩ ─── RY(θ₂) ─── CX ─── RY(θ₆) ─── CX ─── ... ─── (repeat 3x)
|q₃⟩ ─── RY(θ₃) ─── ●  ─── RY(θ₇) ─── ●  ─── ... ─── (repeat 3x)
```

## Data Flow

1. **Input**: 284,807 credit card transactions (V1-V28 PCA features + Class)
2. **Feature Selection**: Cohen's d identifies 4 most discriminative features
3. **Balancing**: Undersample to 984 samples (492 per class)
4. **Scaling**: StandardScaler → MinMaxScaler [0, 2π]
5. **Training**: Parallel classical/quantum model training
6. **Evaluation**: Compare metrics on held-out test set
7. **Visualization**: Dashboard + static charts
